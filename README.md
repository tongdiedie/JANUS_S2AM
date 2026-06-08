# JANUS-S²AM: Curvature-aware Foreground/Hard-Background Prompting on FoB-SAM

This repository is a modified FoB-SAM codebase.  The original background-centric FoB prompt generator is extended into **JANUS-S²AM**, a foreground-background mutually calibrated prompt framework for few-shot medical image segmentation.

The implementation focuses on the requested direction:

> **curvature-aware prompt allocation, with prompt distribution adapting to target shape.**

JANUS-S²AM keeps FoB's learned background prompt localization branch, but changes the final prompt construction and SAM inference loop:

```
support mask -> foreground/background prototypes
             -> query foreground score S_fg and background score S_bg
             -> positive foreground prompts from P_fg = S_fg - S_bg
             -> hard background prompts from prototype confusion + boundary + curvature
             -> SAM first mask
             -> SAM-induced hard background corrective prompts
             -> SAM refined mask
```

---

## 1. What changed

### 1.1 Foreground-Background Mutual Prompting

FoB mainly predicts background prompts.  JANUS-S²AM additionally computes foreground and background prototypes from the support episode:

```python
S_fg(x) = cos(F_q(x), p_fg)
S_bg(x) = cos(F_q(x), p_bg)
P_fg(x) = S_fg(x) - S_bg(x)
```

Positive foreground prompts are selected from the top local maxima of `P_fg`, restricted to regions with high foreground score and supported by the coarse query mask.  This gives SAM a stable foreground core instead of only telling SAM where not to segment.

Implemented in:

- `models/FoB.py::_build_prompt_bundle`
- `models/janus_s2am.py::compute_mutual_similarity_maps`
- `models/janus_s2am.py::foreground_core_points`

### 1.2 Prototype-confusion Hard Background Mining

Hard negative prompts are not uniformly sampled from all background.  JANUS-S²AM mines regions that are likely to confuse SAM:

```python
H_bg(x) = S_fg_norm(x) * S_bg_norm(x)
```

The implementation augments this score with background confidence, coarse-mask boundary, and curvature:

```python
score = S_fg * S_bg + w_bg * S_bg + w_boundary * boundary + w_curvature * curvature - foreground_core_penalty
```

Implemented in:

- `models/janus_s2am.py::build_hard_background_score`
- `models/janus_s2am.py::hard_background_points`

### 1.3 Curvature-aware / Shape-aware Background Prompt Allocation

JANUS-S²AM computes support-mask compactness:

```python
C = P^2 / (4 * pi * A)
```

Default allocation:

```text
C < 1.5       -> K_bg = 6
1.5 <= C < 3 -> K_bg = 10
C >= 3       -> K_bg = 16
```

The distribution of hard background points is further biased toward high-curvature query/coarse-mask boundary regions.  This is designed for irregular lesions, concave organs, and adjacent tissue boundaries where SAM over-segmentation is most frequent.

Implemented in:

- `models/janus_s2am.py::allocate_background_points`
- `models/janus_s2am.py::curvature_score`
- `models/FoB.py::_build_prompt_bundle`

### 1.4 SAM-induced Closed-loop Corrective Prompting

At inference, `SAM.py` now supports a two-pass loop:

```text
JANUS prompt bundle -> SAM mask0
mask0 + H_bg score  -> mine corrective hard negative points
pos + neg + corrective neg + mask_input -> SAM mask1
```

Implemented in:

- `SAM.py::forward`
- `models/janus_s2am.py::mine_sam_induced_hard_background`

### 1.5 Hard Background Contrastive Loss

During training, JANUS-S²AM adds a hard-background contrastive loss:

```python
L_hbg = max(0, sim(f_hbg, p_fg) - sim(f_hbg, p_bg) + margin)
```

Hard background pixels are approximated by high prototype-confusion areas in the coarse predicted foreground but ground-truth background.  The loss is weighted by `janus_hbg_loss_weight`.

Implemented in:

- `models/FoB.py::_hard_background_contrastive_loss`
- `train.py` logging keys: `hbg_loss`, `foreground_loss`, `contrastive_loss`, `prompt_loss`, `total_loss`

---

## 2. Modified / added files

```text
models/janus_s2am.py                 # New rule-based JANUS prompt mining utilities
models/FoB.py                        # Modified FewShotSeg; returns JANUS prompt bundle at test time
SAM.py                               # Modified SAM wrapper; supports second-pass corrective prompting
models/encoder.py                    # Removed hard-coded checkpoint crash; supports config/env paths
config.py                            # Added JANUS/SAM/encoder config options
train.py                             # Supports dict losses and logs hard-background loss
test.py                              # Consumes JANUS prompt bundle directly
requirements.txt                     # Dependency list
scripts/janus/*.sh                   # Full training/testing and ablation scripts
```

Backward compatibility is preserved at the SAM-wrapper level: if a model still returns the old `(neg_point, pos_point)` tuple, `SAM.forward()` can parse it.

---

## 3. Environment

Create an environment compatible with your CUDA version.  The original FoB code was developed around PyTorch 1.10/torchvision 0.11; newer versions can work, but keep `torch` and `torchvision` mutually compatible.

```bash
conda create -n janus_s2am python=3.9 -y
conda activate janus_s2am
pip install -r requirements.txt
```

Install `segment-anything` if your package mirror does not provide it through `requirements.txt`:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

---

## 4. Checkpoints

### 4.1 SAM checkpoint

Download `sam_vit_h_4b8939.pth` and put it under:

```text
./checkpoints/sam_vit_h_4b8939.pth
```

or pass it explicitly:

```bash
export SAM_CKPT=/absolute/path/to/sam_vit_h_4b8939.pth
```

### 4.2 Encoder checkpoint

The encoder supports these options:

```bash
encoder_pretrained_weights=COCO        # default; reads ./checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth
encoder_pretrained_weights=resnet101   # reads ./checkpoints/resnet101-63fe2227.pth
encoder_pretrained_weights=/path/to/your_encoder.pth
encoder_pretrained_weights=none        # random init, useful only for smoke/debug
```

Environment variables are also supported:

```bash
export DEEPLABV3_RESNET101_COCO=/path/to/deeplabv3_resnet101_coco-586e9e4e.pth
export RESNET101_IMAGENET=/path/to/resnet101-63fe2227.pth
```

Missing encoder weights now raise a warning and continue with random initialization.  Missing SAM weights raise an error because SAM inference cannot run without the SAM checkpoint.

---

## 5. Dataset layout

Keep the original FoB-SAM layout:

```text
data/CHAOST2
data/SABS
data/isic/combine
data/ISIC_setting_1
```

You can override paths in scripts:

```bash
DATA_DIR=/path/to/CHAOST2 bash scripts/janus/train_CHAOST2_full.sh
DATA_DIR=/path/to/SABS    bash scripts/janus/train_SABS_full.sh
```

For 3D datasets, compile the original supervoxel extension before training:

```bash
cd data/supervoxels
python setup.py build_ext --inplace
cd ../..
python data/supervoxels/generate_supervoxels.py
```

---

## 6. Smoke check

Run the lightweight syntax/prompt-utility check:

```bash
bash scripts/janus/smoke_compile.sh
```

This checks Python syntax for the modified files and verifies that curvature allocation, foreground point selection, and hard-background point selection produce valid point arrays.

---

## 7. Full training scripts

### CHAOST2 / Abd-MRI

```bash
export SAM_CKPT=/path/to/sam_vit_h_4b8939.pth
export ENCODER_WEIGHTS=COCO
export DATA_DIR=/path/to/CHAOST2
bash scripts/janus/train_CHAOST2_full.sh
```

Useful overrides:

```bash
FOLDS="0" N_STEPS=1000 SAVE_SNAPSHOT_EVERY=1000 bash scripts/janus/train_CHAOST2_full.sh
```

### SABS / Abd-CT

```bash
export SAM_CKPT=/path/to/sam_vit_h_4b8939.pth
export DATA_DIR=/path/to/SABS
bash scripts/janus/train_SABS_full.sh
```

### ISIC Setting 1

```bash
export SAM_CKPT=/path/to/sam_vit_h_4b8939.pth
export DATA_DIR=/path/to/ISIC_setting_1
bash scripts/janus/train_isic_setting1_full.sh
```

### ISIC Setting 2

```bash
export SAM_CKPT=/path/to/sam_vit_h_4b8939.pth
export DATA_DIR=/path/to/isic/combine
bash scripts/janus/train_isic_setting2_full.sh
```

---

## 8. Full testing scripts

Testing requires a trained FoB/JANUS checkpoint:

```bash
export RELOAD_MODEL_PATH=/path/to/snapshot.pth
export SAM_CKPT=/path/to/sam_vit_h_4b8939.pth
```

### CHAOST2

```bash
DATA_DIR=/path/to/CHAOST2 bash scripts/janus/test_CHAOST2_full.sh
```

### SABS

```bash
DATA_DIR=/path/to/SABS bash scripts/janus/test_SABS_full.sh
```

### ISIC Setting 1

```bash
DATA_DIR=/path/to/ISIC_setting_1 bash scripts/janus/test_isic_setting1_full.sh
```

### ISIC Setting 2

```bash
DATA_DIR=/path/to/isic/combine bash scripts/janus/test_isic_setting2_full.sh
```

---

## 9. Ablation experiments

### 9.1 Inference-time ablation using one trained checkpoint

These scripts evaluate the same checkpoint with different prompt modules disabled/enabled.

```bash
export RELOAD_MODEL_PATH=/path/to/snapshot.pth
export SAM_CKPT=/path/to/sam_vit_h_4b8939.pth
FOLD=0 SUPP_IDX=2 DATA_DIR=/path/to/CHAOST2 bash scripts/janus/ablate_inference_CHAOST2.sh
```

For SABS:

```bash
FOLD=0 SUPP_IDX=2 DATA_DIR=/path/to/SABS bash scripts/janus/ablate_inference_SABS.sh
```

For ISIC Setting 2:

```bash
FOLD=1 DATA_DIR=/path/to/isic/combine bash scripts/janus/ablate_inference_isic_setting2.sh
```

The variants are:

| Variant | Purpose | Main flags |
|---|---|---|
| `A0_fob_compatible` | Original-style FoB prompts | `janus_enabled=False` |
| `A1_mutual_prompting` | Add foreground-background mutual positive prompts | `mutual=True, hard_bg=False, curvature=False, sam_refine=False` |
| `A2_hard_background_fixedK` | Add prototype-confusion hard BG with fixed K | `hard_bg=True, curvature=False, sam_refine=False` |
| `A3_curvature_allocation` | Add shape/curvature-aware background distribution | `curvature=True, sam_refine=False` |
| `A4_full_inference` | Full JANUS-S²AM inference | `mutual=True, hard_bg=True, curvature=True, sam_refine=True` |

Recommended table columns:

```text
Variant | Dice | IoU | K_bg avg | compactness bucket | notes
```

`K_bg` and compactness are available inside each prompt bundle.  For large-scale logging, add a small logger in `test.py` around the returned `prompts` dictionary.

### 9.2 Training-time ablation for Hard Background Contrastive Loss

```bash
FOLD=0 DATA_DIR=/path/to/CHAOST2 bash scripts/janus/ablate_training_hbg_loss_CHAOST2.sh
```

This runs two training jobs:

```text
no_hbg_loss:    janus_hbg_loss_weight=0.0
with_hbg_loss:  janus_hbg_loss_weight=0.10
```

After training, test each produced snapshot with the same full inference script:

```bash
export RELOAD_MODEL_PATH=/path/to/no_hbg_loss/snapshot.pth
bash scripts/janus/test_CHAOST2_full.sh

export RELOAD_MODEL_PATH=/path/to/with_hbg_loss/snapshot.pth
bash scripts/janus/test_CHAOST2_full.sh
```

---

## 10. Important JANUS config switches

All can be passed through Sacred CLI: `python train.py with key=value` or `python test.py with key=value`.

```text
janus_enabled=True
janus_mutual_prompting=True
janus_hard_background=True
janus_curvature_allocation=True
janus_sam_refinement=True
```

Foreground prompts:

```text
janus_fg_points=6
janus_fg_prob_threshold=0.96
janus_fg_min_distance=18
```

Curvature-aware background allocation:

```text
janus_bg_points_low=6
janus_bg_points_mid=10
janus_bg_points_high=16
janus_compactness_low=1.5
janus_compactness_high=3.0
janus_curvature_radius=7
janus_curvature_blur=7
```

Hard background scoring:

```text
janus_hard_bg_ratio=0.40
janus_hard_bg_max_points=8
janus_boundary_weight=0.35
janus_curvature_weight=0.35
janus_bg_score_weight=0.25
janus_fg_core_penalty=0.50
```

SAM-induced refinement:

```text
janus_sam_refinement=True
janus_sam_refine_points=4
janus_sam_mined_points=4
```

Training loss:

```text
janus_hbg_loss_weight=0.10
janus_hbg_margin=0.20
janus_hbg_train_pred_threshold=0.50
janus_hbg_train_top_quantile=0.90
```

---

## 11. Expected prompt behavior

For a compact/round support mask:

```text
compactness < 1.5 -> fewer background prompts, default K_bg=6
```

For a moderately irregular target:

```text
1.5 <= compactness < 3 -> default K_bg=10
```

For a complex or highly concave target:

```text
compactness >= 3 -> more negative prompts, default K_bg=16
```

Within the selected budget, points are not uniformly placed.  Hard negative prompts are biased toward:

```text
high S_fg * S_bg prototype-confusion regions
coarse-mask boundary
high-curvature boundary areas
regions selected by SAM's first-pass over-segmentation tendency
```

---

## 12. Notes on reproducibility

Use these defaults for paper-style experiments:

```bash
SEED=2025
N_STEPS=39001
MAX_ITERS_PER_LOAD=3000
SAVE_SNAPSHOT_EVERY=1000
NUM_WORKERS=16
```

Use this for a quick debug pass:

```bash
FOLDS="0" N_STEPS=20 MAX_ITERS_PER_LOAD=20 SAVE_SNAPSHOT_EVERY=20 NUM_WORKERS=0 bash scripts/janus/train_CHAOST2_full.sh
```

---

## 13. Original FoB-SAM acknowledgement

This repository is modified from FoB-SAM: **Focus on the Background: Exploring SAM's Potential in Few-shot Medical Image Segmentation with Background-centric Prompting**.  The original codebase also builds on ALPNet, ADNet, Segment Anything, and ProtoSAM.
