#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# JANUS-S2AM CHAOST2 sparse retraining script
#
# Default:
#   run all folds: 0 1 2 3 4
#
# Examples:
#   bash scripts/janus/train_CHAOST2_sparse_full.sh
#   FOLD=0 bash scripts/janus/train_CHAOST2_sparse_full.sh
#   FOLDS="2 3 4" bash scripts/janus/train_CHAOST2_sparse_full.sh
#   FOLD=0 N_STEPS=1000 bash scripts/janus/train_CHAOST2_sparse_full.sh
#
# This script avoids passing new config keys. It only uses JANUS
# config entries already present in your current project.
# ============================================================

GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}

DATA_DIR=${DATA_DIR:-./data/CHAOST2}
LOGDIR=${LOGDIR:-./runs_janus/train_CHAOST2_sparse_full}

SAM_CKPT=${SAM_CKPT:-./checkpoints/sam_vit_h_4b8939.pth}
ENCODER_WEIGHTS=${ENCODER_WEIGHTS:-COCO}

NUM_WORKERS=${NUM_WORKERS:-16}
N_STEPS=${N_STEPS:-39001}
MAX_ITERS_PER_LOAD=${MAX_ITERS_PER_LOAD:-3000}
SAVE_SNAPSHOT_EVERY=${SAVE_SNAPSHOT_EVERY:-1000}

# FoB paper uses 0.95 decay per 1K steps. Use 0.95 by default for retraining.
LR_STEP_GAMMA=${LR_STEP_GAMMA:-0.95}
SEED=${SEED:-2025}

# Sparse JANUS training defaults.
# Lower hbg loss avoids over-suppressing large organs such as liver.
JANUS_HBG_LOSS_WEIGHT=${JANUS_HBG_LOSS_WEIGHT:-0.05}
JANUS_FG_POINTS=${JANUS_FG_POINTS:-1}
JANUS_BASE_BG_POINTS=${JANUS_BASE_BG_POINTS:-2}
JANUS_HARD_BACKGROUND=${JANUS_HARD_BACKGROUND:-True}
JANUS_HARD_BG_RATIO=${JANUS_HARD_BG_RATIO:-0.20}
JANUS_HARD_BG_MAX_POINTS=${JANUS_HARD_BG_MAX_POINTS:-2}
JANUS_CURVATURE_ALLOCATION=${JANUS_CURVATURE_ALLOCATION:-False}
JANUS_SAM_REFINEMENT=${JANUS_SAM_REFINEMENT:-True}
JANUS_SAM_REFINE_POINTS=${JANUS_SAM_REFINE_POINTS:-1}
JANUS_SAM_MINED_POINTS=${JANUS_SAM_MINED_POINTS:-1}
JANUS_SAM_MINED_MIN_DISTANCE=${JANUS_SAM_MINED_MIN_DISTANCE:-20}
JANUS_SAM_MINED_AVOID_RADIUS=${JANUS_SAM_MINED_AVOID_RADIUS:-28}

# Fold selection priority:
#   FOLD=2        -> single fold
#   FOLDS="0 1"   -> selected folds
#   neither set   -> all default folds
if [[ -n "${FOLD:-}" ]]; then
  FOLDS_STR="$FOLD"
else
  FOLDS_STR=${FOLDS:-"0 1 2 3 4"}
fi

read -ra FOLDS_ARR <<< "$FOLDS_STR"

echo "============================================================"
echo "JANUS-S2AM CHAOST2 Sparse Retraining"
echo "DATA_DIR:              $DATA_DIR"
echo "LOGDIR:                $LOGDIR"
echo "SAM_CKPT:              $SAM_CKPT"
echo "ENCODER_WEIGHTS:       $ENCODER_WEIGHTS"
echo "GPU_ID:                $GPU_ID"
echo "CUDA_VISIBLE_DEVICES:  $CUDA_VISIBLE_DEVICES"
echo "FOLDS:                 ${FOLDS_ARR[*]}"
echo "N_STEPS:               $N_STEPS"
echo "LR_STEP_GAMMA:         $LR_STEP_GAMMA"
echo "JANUS_HBG_LOSS_WEIGHT: $JANUS_HBG_LOSS_WEIGHT"
echo "JANUS_FG_POINTS:       $JANUS_FG_POINTS"
echo "JANUS_BASE_BG_POINTS:  $JANUS_BASE_BG_POINTS"
echo "JANUS_HARD_BACKGROUND: $JANUS_HARD_BACKGROUND"
echo "============================================================"

for CV_FOLD in "${FOLDS_ARR[@]}"; do
  echo ""
  echo "==================== Training CHAOST2 fold ${CV_FOLD} ===================="

  python3 train.py with \
    mode=train \
    dataset=CHAOST2 \
    gpu_id=$GPU_ID \
    num_workers=$NUM_WORKERS \
    n_steps=$N_STEPS \
    max_iters_per_load=$MAX_ITERS_PER_LOAD \
    save_snapshot_every=$SAVE_SNAPSHOT_EVERY \
    lr_step_gamma=$LR_STEP_GAMMA \
    eval_fold=$CV_FOLD \
    test_label=[1,2,3,4] \
    exclude_label=None \
    use_gt=False \
    seed=$SEED \
    sam_checkpoint=$SAM_CKPT \
    encoder_pretrained_weights=$ENCODER_WEIGHTS \
    janus_enabled=True \
    janus_mutual_prompting=True \
    janus_fg_points=$JANUS_FG_POINTS \
    janus_base_bg_points=$JANUS_BASE_BG_POINTS \
    janus_hard_background=$JANUS_HARD_BACKGROUND \
    janus_hard_bg_ratio=$JANUS_HARD_BG_RATIO \
    janus_hard_bg_max_points=$JANUS_HARD_BG_MAX_POINTS \
    janus_curvature_allocation=$JANUS_CURVATURE_ALLOCATION \
    janus_sam_refinement=$JANUS_SAM_REFINEMENT \
    janus_sam_refine_points=$JANUS_SAM_REFINE_POINTS \
    janus_sam_mined_points=$JANUS_SAM_MINED_POINTS \
    janus_sam_mined_min_distance=$JANUS_SAM_MINED_MIN_DISTANCE \
    janus_sam_mined_avoid_radius=$JANUS_SAM_MINED_AVOID_RADIUS \
    janus_hbg_loss_weight=$JANUS_HBG_LOSS_WEIGHT \
    path.CHAOST2.data_dir=$DATA_DIR \
    path.log_dir=$LOGDIR
done
