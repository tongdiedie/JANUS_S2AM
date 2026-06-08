#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}
SAM_CKPT=${SAM_CKPT:-./checkpoints/sam_vit_h_4b8939.pth}
ENCODER_WEIGHTS=${ENCODER_WEIGHTS:-COCO}
DATA_DIR=${DATA_DIR:-./data/ISIC_setting_1}
LOGDIR=${LOGDIR:-./runs_janus/train_isic_setting1_full}
FOLDS_STR=${FOLDS:-"1 2 3 4 5"}
read -ra FOLDS_ARR <<< "$FOLDS_STR"

for FOLD in "${FOLDS_ARR[@]}"; do
  python3 train.py with \
    mode=train \
    dataset=isic \
    isic_setting=1 \
    gpu_id=$GPU_ID \
    num_workers=${NUM_WORKERS:-16} \
    n_steps=${N_STEPS:-39001} \
    max_iters_per_load=${MAX_ITERS_PER_LOAD:-3000} \
    save_snapshot_every=${SAVE_SNAPSHOT_EVERY:-1000} \
    lr_step_gamma=${LR_STEP_GAMMA:-0.98} \
    eval_fold=$FOLD \
    test_label=None \
    exclude_label=None \
    use_gt=False \
    seed=${SEED:-2025} \
    sam_checkpoint=$SAM_CKPT \
    encoder_pretrained_weights=$ENCODER_WEIGHTS \
    janus_enabled=True \
    janus_mutual_prompting=True \
    janus_hard_background=True \
    janus_curvature_allocation=True \
    janus_sam_refinement=True \
    janus_hbg_loss_weight=${JANUS_HBG_LOSS_WEIGHT:-0.10} \
    isic_setting_1_base_path=$DATA_DIR \
    path.log_dir=$LOGDIR
done
