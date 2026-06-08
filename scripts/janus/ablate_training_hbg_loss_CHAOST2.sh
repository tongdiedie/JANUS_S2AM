#!/usr/bin/env bash
set -euo pipefail

# Training-time ablation for the Hard Background Contrastive Loss.
# The two runs differ only by janus_hbg_loss_weight.
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}
SAM_CKPT=${SAM_CKPT:-./checkpoints/sam_vit_h_4b8939.pth}
ENCODER_WEIGHTS=${ENCODER_WEIGHTS:-COCO}
DATA_DIR=${DATA_DIR:-./data/CHAOST2}
FOLD=${FOLD:-0}
COMMON=(mode=train dataset=CHAOST2 gpu_id=$GPU_ID num_workers=${NUM_WORKERS:-16} n_steps=${N_STEPS:-39001} max_iters_per_load=${MAX_ITERS_PER_LOAD:-3000} save_snapshot_every=${SAVE_SNAPSHOT_EVERY:-1000} lr_step_gamma=${LR_STEP_GAMMA:-0.98} eval_fold=$FOLD test_label=[1,2,3,4] exclude_label=None use_gt=False seed=${SEED:-2025} sam_checkpoint=$SAM_CKPT encoder_pretrained_weights=$ENCODER_WEIGHTS janus_enabled=True janus_mutual_prompting=True janus_hard_background=True janus_curvature_allocation=True janus_sam_refinement=True path.CHAOST2.data_dir=$DATA_DIR)
python3 train.py with "${COMMON[@]}" janus_hbg_loss_weight=0.0 path.log_dir=${BASE_LOGDIR:-./runs_janus/ablation_train}/no_hbg_loss
python3 train.py with "${COMMON[@]}" janus_hbg_loss_weight=0.10 path.log_dir=${BASE_LOGDIR:-./runs_janus/ablation_train}/with_hbg_loss
