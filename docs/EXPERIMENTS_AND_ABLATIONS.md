# JANUS-S²AM Experiment and Ablation Plan

This file gives the exact experiment matrix corresponding to the implemented code.

## Main experiments

Run full JANUS-S²AM on each dataset:

```bash
bash scripts/janus/train_CHAOST2_full.sh
bash scripts/janus/test_CHAOST2_full.sh

bash scripts/janus/train_SABS_full.sh
bash scripts/janus/test_SABS_full.sh

bash scripts/janus/train_isic_setting1_full.sh
bash scripts/janus/test_isic_setting1_full.sh

bash scripts/janus/train_isic_setting2_full.sh
bash scripts/janus/test_isic_setting2_full.sh
```

Required environment variables for test:

```bash
export RELOAD_MODEL_PATH=/path/to/snapshot.pth
export SAM_CKPT=/path/to/sam_vit_h_4b8939.pth
```

## Inference ablation matrix

Use one trained checkpoint and toggle prompt modules.

| ID | Mutual FG/BG | Hard BG | Curvature allocation | SAM second pass | Expected interpretation |
|---|---:|---:|---:|---:|---|
| A0 | 0 | 0 | 0 | 0 | FoB-compatible baseline |
| A1 | 1 | 0 | 0 | 0 | value of foreground-background mutual prompts |
| A2 | 1 | 1 | 0 | 0 | value of prototype-confusion hard BG mining |
| A3 | 1 | 1 | 1 | 0 | value of curvature-aware allocation/distribution |
| A4 | 1 | 1 | 1 | 1 | full closed-loop JANUS-S²AM inference |

Run:

```bash
FOLD=0 SUPP_IDX=2 DATA_DIR=/path/to/CHAOST2 bash scripts/janus/ablate_inference_CHAOST2.sh
FOLD=0 SUPP_IDX=2 DATA_DIR=/path/to/SABS bash scripts/janus/ablate_inference_SABS.sh
FOLD=1 DATA_DIR=/path/to/isic/combine bash scripts/janus/ablate_inference_isic_setting2.sh
```

## Training ablation matrix

The core train-time ablation is the hard-background contrastive loss:

| ID | Full inference | HBG loss weight | Purpose |
|---|---:|---:|---|
| T0 | 1 | 0.00 | no train-time hard-background discrimination |
| T1 | 1 | 0.10 | full train-time JANUS-S²AM |

Run:

```bash
FOLD=0 DATA_DIR=/path/to/CHAOST2 bash scripts/janus/ablate_training_hbg_loss_CHAOST2.sh
```

Then test both snapshots using `scripts/janus/test_CHAOST2_full.sh`.

## Curvature sensitivity study

For a deeper study, keep the same checkpoint and test these settings:

```bash
# Fixed K=10, no curvature distribution
python3 test.py with janus_curvature_allocation=False janus_bg_points_mid=10 ...

# Conservative curvature allocation
python3 test.py with janus_curvature_allocation=True janus_bg_points_low=4 janus_bg_points_mid=8 janus_bg_points_high=12 ...

# Default allocation
python3 test.py with janus_curvature_allocation=True janus_bg_points_low=6 janus_bg_points_mid=10 janus_bg_points_high=16 ...

# Aggressive allocation
python3 test.py with janus_curvature_allocation=True janus_bg_points_low=8 janus_bg_points_mid=14 janus_bg_points_high=20 ...
```

Recommended reporting:

| Setting | Low K | Mid K | High K | Dice | IoU | Notes |
|---|---:|---:|---:|---:|---:|---|
| fixed | 10 | 10 | 10 | | | no shape adaptation |
| conservative | 4 | 8 | 12 | | | fewer negative prompts |
| default | 6 | 10 | 16 | | | recommended |
| aggressive | 8 | 14 | 20 | | | useful for very irregular lesions |

## Prompt-count ablation

Evaluate whether improvements come from smarter prompts rather than simply more prompts.

```bash
# Same total K, hard BG off
python3 test.py with janus_hard_background=False janus_curvature_allocation=False janus_base_bg_points=16 ...

# Same total K, hard BG on
python3 test.py with janus_hard_background=True janus_curvature_allocation=False janus_base_bg_points=16 janus_hard_bg_ratio=0.40 ...
```

If the second setting outperforms the first, the gain is attributable to prototype-confusion hard-background selection rather than only point count.

## Qualitative visualization suggestion

For a paper figure, visualize these maps from the prompt bundle:

```python
prompts['fg_score']
prompts['bg_score']
prompts['p_fg_score']
prompts['hbg_score']
prompts['curvature_score']
prompts['pos_points']
prompts['base_neg_points']
prompts['hard_neg_points']
```

A recommended figure layout:

```text
query image | GT | SAM mask0 | final mask | P_fg | H_bg | curvature | prompts overlay
```
