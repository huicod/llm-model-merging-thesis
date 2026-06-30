# Contribution Summary

## Problem

Fine-tuning a language model on one domain often improves that domain while weakening other abilities. This project explored whether multiple task-specialized models can be merged into a single model that keeps useful behavior across code, math, and science tasks.

## Experiment Design

The workflow starts from one base model and trains separate LoRA/SFT variants:

- Code model: trained on CodeAlpaca-20k style instruction data.
- Math model: trained on a MathInstruct subset.
- Science model: trained on SciQ.

The trained models are exported and merged in two-model and three-model groups:

- CM: code + math.
- CS: code + science.
- MS: math + science.
- CMS: code + math + science.

The merged models are evaluated on:

- HumanEval for code generation.
- GSM8K for math reasoning.
- SciQ for science QA.

## Method Work

The thesis work included testing standard merge baselines and developing layer-aware variants:

- Average merging: direct parameter averaging.
- DARE-style masking: sparse task-vector merging with optional rescaling.
- RankMean-style weighting: rank transformer layers by how much each fine-tuned model differs from the base model.
- MaskRankMean-style variant: combine layer ranking with magnitude-based delta masking and rescaling so that more important layer deltas are preserved more strongly.

The cleaned reference implementation in `scripts/mask_rankmean_core.py` keeps this core idea without copying upstream RankMean code.

## Engineering Work

The local project also contained infrastructure work:

- Batch training scripts for multiple task models.
- Export templates for LoRA checkpoints.
- Batch merge scripts for all merge groups.
- lm-eval scripts for task-specific and merged models.
- Layer-wise L1 diagnostic plots for understanding where fine-tuning changed the base model.

Raw data, logs, model files, and external baseline code were removed from the public release.
