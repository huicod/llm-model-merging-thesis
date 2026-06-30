# LLM Model Merging Thesis Project

This repository is a cleaned public version of my undergraduate thesis project on merging task-specialized large language models.

The original local workspace contained upstream baseline code, model checkpoints, datasets, TensorBoard logs, raw lm-eval outputs, and temporary experiment files. Those files are intentionally excluded here. This repository keeps only my contribution summary, reproducible workflow notes, sanitized configuration templates, and lightweight reference scripts.

## Project Summary

The project studies whether several LoRA/SFT models trained from the same base model can be merged into one model that keeps multiple task abilities.

Main experiment setting:

- Base model: Qwen2-1.5B.
- Task-specialized SFT models: CodeAlpaca-20k, MathInstruct subset, and SciQ.
- Merge groups: code+math, code+science, math+science, and code+math+science.
- Evaluation tasks: HumanEval, GSM8K, and SciQ through lm-evaluation-harness.
- Baselines and references: average merging, DARE-style masking, RankMean-style layer ranking, and combined variants.

## My Contributions

- Built the training pipeline for task-specific Qwen2-1.5B LoRA/SFT models with LLaMA-Factory.
- Organized export, merge, and evaluation scripts for pairwise and three-model merge experiments.
- Implemented and tested layer-aware merge variants that combine RankMean-style layer ranking with delta masking and rescaling. In this cleaned repository, the core idea is provided as a standalone reference implementation in `scripts/mask_rankmean_core.py`.
- Added diagnostic scripts to compare layer-wise parameter changes between the base model and fine-tuned models.
- Designed the experiment workflow for comparing merged models on code, math, and science benchmarks.

## Repository Layout

```text
.
|-- README.md
|-- configs/
|   |-- qwen2_1_5b_code_lora.yaml
|   |-- qwen2_1_5b_math_lora.yaml
|   |-- qwen2_1_5b_sciq_lora.yaml
|   `-- export_lora_example.yaml
|-- docs/
|   |-- contribution_summary.md
|   |-- experiment_workflow.md
|   `-- excluded_from_release.md
|-- scripts/
|   |-- analyze_layer_deltas.py
|   |-- convert_parquet_to_jsonl.py
|   |-- mask_rankmean_core.py
|   |-- merge_models.py
|   |-- prepare_mathinstruct_subset.py
|   |-- run_eval.sh
|   |-- run_merge.sh
|   `-- run_training.sh
`-- requirements.txt
```

## What Is Not Included

The following are not part of this public repository:

- Upstream RankMean source code.
- Model checkpoints, merged models, LoRA adapters, and tokenizer artifacts.
- Original datasets or processed dataset files.
- TensorBoard logs and raw evaluation JSON files.
- Temporary notebooks, cache directories, and generated figures.

See `docs/excluded_from_release.md` for the full release filter.

## Reproduction Outline

1. Install the Python dependencies in `requirements.txt`.
2. Install external CLIs used by the workflow, especially LLaMA-Factory and lm-evaluation-harness.
3. Download the base model and datasets according to their own licenses.
4. Edit the path placeholders in `configs/`.
5. Run LoRA/SFT training with `scripts/run_training.sh`.
6. Export LoRA adapters into standalone model directories.
7. Merge selected task models with `scripts/run_merge.sh` or `scripts/merge_models.py`.
8. Evaluate merged models with `scripts/run_eval.sh`.
9. Use `scripts/analyze_layer_deltas.py` for layer-wise parameter-change diagnostics.

This repository is intended as a thesis portfolio and reproducibility guide, not as a full redistribution of every experiment artifact.
