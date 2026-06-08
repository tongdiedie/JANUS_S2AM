# JANUS-S²AM Change Log

## Core method

- Added foreground-background mutual prompting from support foreground/background prototypes.
- Added prototype-confusion hard background score: `S_fg * S_bg` plus boundary/curvature terms.
- Added shape-aware background prompt count allocation using compactness `P²/(4πA)`.
- Added curvature-aware prompt distribution using contour tangent change density.
- Added SAM-induced second-pass hard background corrective prompt mining.
- Added hard-background contrastive loss for training.

## Engineering changes

- `FewShotSeg.forward(..., train=False)` now returns a prompt dictionary instead of only `(neg_point, pos_point)`.
- `SAM.forward()` accepts this prompt dictionary and remains backward-compatible with the old tuple output.
- Removed hard-coded encoder checkpoint paths; missing encoder checkpoint is now a warning.
- SAM checkpoint is configurable through `sam_checkpoint` or `SAM_CKPT` in scripts.
- Added detailed training, testing, and ablation scripts under `scripts/janus/`.
- Added `docs/EXPERIMENTS_AND_ABLATIONS.md` and a full README.

## Validation performed here

- Python syntax compilation for modified project files passed.
- Bash syntax check for all `scripts/janus/*.sh` passed.
- Lightweight prompt utility smoke test passed.

Full dataset training/evaluation was not executed here because the medical datasets and SAM checkpoint are not present in this runtime.
