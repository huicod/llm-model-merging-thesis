"""Compact MaskRankMean-style model merging utilities.

This file is a cleaned reference implementation of the thesis idea. It does
not copy upstream RankMean code. It works on PyTorch state dictionaries and
keeps the algorithm small enough to audit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass(frozen=True)
class MergeConfig:
    base_mask_rate: float = 0.9
    alpha: float = 0.5
    mask_strategy: str = "magnitude"
    use_rescale: bool = True


def layer_id_from_name(param_name: str) -> int | None:
    match = _LAYER_RE.search(param_name)
    if match is None:
        return None
    return int(match.group(1))


def infer_num_layers(state_dicts: Sequence[Mapping[str, torch.Tensor]]) -> int:
    max_layer = -1
    for state_dict in state_dicts:
        for name in state_dict:
            layer_id = layer_id_from_name(name)
            if layer_id is not None:
                max_layer = max(max_layer, layer_id)
    if max_layer < 0:
        raise ValueError("No transformer layer parameters were found.")
    return max_layer + 1


def compute_layer_delta_scores(
    base_state: Mapping[str, torch.Tensor],
    tuned_states: Sequence[Mapping[str, torch.Tensor]],
) -> torch.Tensor:
    """Return an [num_models, num_layers] tensor of L1 delta scores."""

    num_layers = infer_num_layers([base_state, *tuned_states])
    scores = torch.zeros((len(tuned_states), num_layers), dtype=torch.float64)

    for model_index, tuned_state in enumerate(tuned_states):
        for name, base_tensor in base_state.items():
            layer_id = layer_id_from_name(name)
            if layer_id is None or name not in tuned_state:
                continue
            tuned_tensor = tuned_state[name]
            if not torch.is_floating_point(base_tensor):
                continue
            delta = tuned_tensor.detach().float().cpu() - base_tensor.detach().float().cpu()
            scores[model_index, layer_id] += delta.abs().sum().item()

    return scores


def rank_layers(layer_scores: torch.Tensor) -> torch.Tensor:
    """Rank layers for each model.

    Lower delta layers get smaller ranks. Higher delta layers get larger ranks,
    which gives them larger merge weights and lower mask rates.
    """

    ranks = torch.zeros_like(layer_scores, dtype=torch.float64)
    for model_index in range(layer_scores.shape[0]):
        order = torch.argsort(layer_scores[model_index], descending=False)
        for rank, layer_id in enumerate(order.tolist(), start=1):
            ranks[model_index, layer_id] = rank
    return ranks


def mask_delta(
    delta: torch.Tensor,
    drop_rate: float,
    strategy: str,
    use_rescale: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Drop part of a task delta and optionally rescale the kept values."""

    if drop_rate <= 0:
        return delta

    flat = delta.reshape(-1)
    num_drop = int(flat.numel() * min(drop_rate, 1.0))
    if num_drop <= 0:
        return delta
    if num_drop >= flat.numel():
        return torch.zeros_like(delta)

    keep_mask = torch.ones(flat.numel(), dtype=torch.bool, device=flat.device)
    if strategy == "random":
        drop_indices = torch.randperm(flat.numel(), device=flat.device, generator=generator)[:num_drop]
    elif strategy == "magnitude":
        drop_indices = torch.topk(flat.abs(), k=num_drop, largest=False).indices
    else:
        raise ValueError(f"Unsupported mask strategy: {strategy}")

    keep_mask[drop_indices] = False
    masked = flat * keep_mask.to(flat.dtype)

    if use_rescale:
        keep_rate = 1.0 - drop_rate
        if keep_rate > 0:
            masked = masked / keep_rate

    return masked.reshape_as(delta)


def merge_state_dicts(
    base_state: Mapping[str, torch.Tensor],
    tuned_states: Sequence[Mapping[str, torch.Tensor]],
    config: MergeConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Merge fine-tuned model state dicts with layer ranks and delta masking."""

    if not tuned_states:
        raise ValueError("At least one tuned model is required.")

    config = config or MergeConfig()
    layer_scores = compute_layer_delta_scores(base_state, tuned_states)
    ranks = rank_layers(layer_scores)
    num_layers = ranks.shape[1]
    merged_state: dict[str, torch.Tensor] = {}

    for name, base_tensor in base_state.items():
        if not torch.is_floating_point(base_tensor):
            merged_state[name] = base_tensor.detach().clone()
            continue

        available = [state[name] for state in tuned_states if name in state]
        if len(available) != len(tuned_states):
            merged_state[name] = base_tensor.detach().clone()
            continue

        layer_id = layer_id_from_name(name)
        if layer_id is None:
            merged_state[name] = torch.stack([tensor.to(base_tensor.dtype) for tensor in available]).mean(dim=0)
            continue

        masked_deltas = []
        weights = []
        for model_index, tuned_tensor in enumerate(available):
            rank = float(ranks[model_index, layer_id].item())
            normalized_rank = rank / float(num_layers)
            drop_rate = config.base_mask_rate * (1.0 - normalized_rank)
            drop_rate = max(0.0, min(drop_rate, 0.99))

            delta = tuned_tensor.to(base_tensor.dtype) - base_tensor
            masked_deltas.append(
                mask_delta(
                    delta=delta,
                    drop_rate=drop_rate,
                    strategy=config.mask_strategy,
                    use_rescale=config.use_rescale,
                )
            )

            penalty = max(0.0, 1.0 - config.alpha * drop_rate)
            weights.append(rank * penalty)

        weight_tensor = torch.tensor(weights, dtype=base_tensor.dtype, device=base_tensor.device)
        weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(torch.finfo(weight_tensor.dtype).eps)
        merged_delta = sum(delta * weight for delta, weight in zip(masked_deltas, weight_tensor))
        merged_state[name] = base_tensor + merged_delta

    return merged_state
