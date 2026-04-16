# Detection Validation Module

This directory contains the manuscript-facing detection code for the MekaNet release.

## Scope

The detection module covers:

- YOLOv8 + SAHI inference
- tiling-aware megakaryocyte detection
- semi-supervised training utilities
- institutional validation helpers
- manuscript-oriented evaluation wrappers

## Manuscript-Aligned Counts

According to the private manuscript:

- Detection training used `100` partially labeled B-hospital MPN images.
- Internal fully labeled evaluation used `9` images.
- External validation used `21` S-hospital test sets.

The public repository does **not** include the raw image cohort or the ID manifest for the 100-image training subset.

## Entry Points

```bash
cd experiments/detection
python inference_demo.py --image_path /path/to/image.png --model_path ../../weights/epoch60.pt
python ../../run_paper_reproduction.py --quick --dry-run
```

## Public-Release Limitations

- No production weights are bundled.
- No public image dataset is bundled.
- Validation scripts can be imported and dry-run tested, but real execution requires user-supplied images and weights.

## Core Files

- `tessd_framework.py`: high-level detection wrapper
- `sahi_inference_module.py`: sliced inference utilities
- `institutional_validator.py`: cross-institution evaluation
- `semi_supervised_trainer.py`: training loop and pseudo-label flow
- `paper_reproduction_runner.py`: manuscript-style orchestration
- `configs/paper_reproduction_quick.yaml`: lightweight reproducibility config
- `configs/paper_reproduction_full.yaml`: full reproducibility config

## Notes

Older deployment guides, benchmark helpers, and config variants that were not part of the canonical reproduction path were removed. Keep manuscript-facing count claims synchronized with the private manuscript, not this public repository.
