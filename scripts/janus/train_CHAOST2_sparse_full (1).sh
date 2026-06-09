#!/usr/bin/env bash
set -euo pipefail

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
LR_STEP_GAMMA=${LR_STEP_GAMMA:-0.95}
SEED=${SEED:-2025}
JANUS_HBG_LOSS_WEIGHT=${JANUS_HBG_LOSS_WEIGHT:-0.05}

if [[ -n "${FOLD:-}" ]]; then
  FOLDS_STR="$FOLD"
else
  FOLDS_STR=${FOLDS:-"0 1 2 3 4"}
fi
read -ra FOLDS_ARR <<< "$FOLDS_STR"

for CV_FOLD in "${FOLDS_ARR[@]}"; do
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
    janus_fg_points=1 \
    janus_base_bg_points=2 \
    janus_hard_background=True \
    janus_hard_bg_ratio=0.20 \
    janus_hard_bg_max_points=2 \
    janus_curvature_allocation=False \
    janus_sam_refinement=True \
    janus_sam_refine_points=1 \
    janus_sam_mined_points=1 \
    janus_sam_mined_min_distance=20 \
    janus_sam_mined_avoid_radius=28 \
    janus_hbg_loss_weight=$JANUS_HBG_LOSS_WEIGHT \
    path.CHAOST2.data_dir=$DATA_DIR \
    path.log_dir=$LOGDIR
done
