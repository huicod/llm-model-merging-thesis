# Experiment Workflow

This document summarizes the thesis experiment pipeline without publishing private data, logs or third-party code.

## 1. Base Model

The experiments used Qwen2-1.5B as the common base model. All task models were produced from the same base so that parameter deltas could be compared and merged.

## 2. Task Fine-Tuning

Three task directions were used:

- Code: CodeAlpaca-20k.
- Math: a 22k subset of MathInstruct.
- Science QA: SciQ.

The fine-tuning setup used LLaMA-Factory SFT with LoRA:

- `finetuning_type: lora`
- `lora_rank: 16`
- LoRA target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- `cutoff_len: 1024`
- cosine learning-rate schedule
- gradient checkpointing enabled

The public config files under `configs/` preserve these settings with local paths replaced by placeholders.

## 3. Export

Each trained LoRA checkpoint was exported into a model directory that could be loaded directly by Hugging Face Transformers or the merge script.

The original project compared checkpoints such as `checkpoint-600` and `checkpoint-1000`; raw checkpoint folders are not included in this repository.

## 4. Model Merging

The experiments merged task models in both three-task and two-task combinations:

- `CMS`: Code + Math + SciQ
- `CM`: Code + Math
- `CS`: Code + SciQ
- `MS`: Math + SciQ

Baseline and comparison methods included:

- simple average merging
- DARE + average merging
- RankMean-style layer-rank weighted merging
- DARE + RankMean-style merging

Additional tuning and ablation variants explored:

- dynamic mask rate by layer rank
- magnitude mask versus random mask
- weight rescaling enabled versus disabled
- reverse rank weighting
- no-penalty weighting
- fixed mask-rate variants such as 0.75 and 0.95

The original RankMean implementation and edited copies are not included. `scripts/merge_all.sh` records the CLI-level workflow only.

## 5. Evaluation

Evaluation used `lm-evaluation-harness`:

- HumanEval for code generation
- GSM8K for math reasoning
- SciQ for science QA

Important evaluation settings used in the experiments:

- HumanEval was run with code execution explicitly enabled.
- Some GSM8K runs used 0-shot, while early base/SFT checks also included 1-shot settings.
- Some development runs used `--limit 0.5` to reduce evaluation cost.
- Generation evaluations used `max_new_tokens=512` in batch merge comparisons.

Raw result JSON files are excluded from the public repository.

## 6. Analysis

The analysis scripts inspected how much each task fine-tuning changed model parameters:

- average L1 parameter change per transformer layer
- separate MLP and self-attention module changes
- comparison between normal merging and rescale-disabled variants
- layer-wise merge weight proportions for RankMean-style methods

Only reusable plotting code is kept here. Concrete plots and weight JSON files are excluded because they are experiment outputs.
