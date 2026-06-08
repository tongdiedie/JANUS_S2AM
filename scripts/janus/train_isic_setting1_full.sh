#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# JANUS-S2AM training script: train_isic_setting1_full.sh
#
# Usage:
#   bash scripts/janus/train_isic_setting1_full.sh
#   FOLD=0 bash scripts/janus/train_isic_setting1_full.sh
#   FOLDS="0 3 4" bash scripts/janus/train_isic_setting1_full.sh
#   FOLD=0 N_STEPS=1000 bash scripts/janus/train_isic_setting1_full.sh
# ============================================================

GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}

SAM_CKPT=${SAM_CKPT:-./checkpoints/sam_vit_h_4b8939.pth}
ENCODER_WEIGHTS=${ENCODER_WEIGHTS:-COCO}
DATA_DIR=${DATA_DIR:-./data/ISIC_setting_1}
LOGDIR=${LOGDIR:-./runs_janus/train_isic_setting1_full}

NUM_WORKERS=${NUM_WORKERS:-16}
N_STEPS=${N_STEPS:-39001}
MAX_ITERS_PER_LOAD=${MAX_ITERS_PER_LOAD:-3000}
SAVE_SNAPSHOT_EVERY=${SAVE_SNAPSHOT_EVERY:-1000}
LR_STEP_GAMMA=${LR_STEP_GAMMA:-0.98}
SEED=${SEED:-2025}
JANUS_HBG_LOSS_WEIGHT=${JANUS_HBG_LOSS_WEIGHT:-0.10}

# Priority:
#   FOLD=2        -> single fold
#   FOLDS="0 1"   -> selected folds
#   neither set   -> all default folds
if [[ -n "${FOLD:-}" ]]; then
  FOLDS_STR="$FOLD"
else
  FOLDS_STR=${FOLDS:-"1 2 3 4 5"}
fi

read -ra FOLDS_ARR <<< "$FOLDS_STR"

echo "============================================================"
echo "JANUS-S2AM Training"
echo "SCRIPT:              $0"
echo "DATA_DIR:            $DATA_DIR"
echo "SAM_CKPT:            $SAM_CKPT"
echo "ENCODER_WEIGHTS:     $ENCODER_WEIGHTS"
echo "LOGDIR:              $LOGDIR"
echo "GPU_ID:              $GPU_ID"
echo "CUDA_VISIBLE_DEVICES:$CUDA_VISIBLE_DEVICES"
echo "FOLDS:               ${FOLDS_ARR[*]}"
echo "N_STEPS:             $N_STEPS"
echo "MAX_ITERS_PER_LOAD:  $MAX_ITERS_PER_LOAD"
echo "SAVE_SNAPSHOT_EVERY: $SAVE_SNAPSHOT_EVERY"
echo "SEED:                $SEED"
echo "JANUS_HBG_LOSS_WEIGHT:$JANUS_HBG_LOSS_WEIGHT"
echo "============================================================"

for CV_FOLD in "${FOLDS_ARR[@]}"; do
  echo ""
  echo "==================== Training fold $CV_FOLD ===================="

  python3 train.py with \
    mode=train \
    dataset=isic \\
    isic_setting=1 \
    gpu_id=$GPU_ID \
    num_workers=$NUM_WORKERS \
    n_steps=$N_STEPS \
    max_iters_per_load=$MAX_ITERS_PER_LOAD \
    save_snapshot_every=$SAVE_SNAPSHOT_EVERY \
    lr_step_gamma=$LR_STEP_GAMMA \
    eval_fold=$CV_FOLD \
    test_label=None \
    exclude_label=None \
    use_gt=False \
    seed=$SEED \
    sam_checkpoint=$SAM_CKPT \
    encoder_pretrained_weights=$ENCODER_WEIGHTS \
    janus_enabled=True \
    janus_mutual_prompting=True \
    janus_hard_background=True \
    janus_curvature_allocation=True \
    janus_sam_refinement=True \
    janus_hbg_loss_weight=$JANUS_HBG_LOSS_WEIGHT \
    isic_setting_1_base_path=$DATA_DIR \
    path.log_dir=$LOGDIR
done
