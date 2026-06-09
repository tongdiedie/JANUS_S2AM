#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}

SAM_CKPT=${SAM_CKPT:-./checkpoints/sam_vit_h_4b8939.pth}
ENCODER_WEIGHTS=${ENCODER_WEIGHTS:-COCO}
DATA_DIR=${DATA_DIR:-./data/CHAOST2}
LOGDIR=${LOGDIR:-./runs_janus/test_CHAOST2_class_adaptive}
RELOAD_MODEL_PATH=${RELOAD_MODEL_PATH:?Please export RELOAD_MODEL_PATH=/path/to/fold_snapshot.pth}

NUM_WORKERS=${NUM_WORKERS:-16}
N_PART=${N_PART:-3}
SEED=${SEED:-2025}
FOLD=${FOLD:-0}
SUPP_IDX=${SUPP_IDX:-2}

echo "==================== Fold ${FOLD}, support ${SUPP_IDX}: LIVER policy ===================="
python3 test.py with \
  mode=test dataset=CHAOST2 gpu_id=$GPU_ID num_workers=$NUM_WORKERS \
  eval_fold=$FOLD supp_idx=$SUPP_IDX test_label=[1] n_part=$N_PART seed=$SEED \
  reload_model_path=$RELOAD_MODEL_PATH sam_checkpoint=$SAM_CKPT encoder_pretrained_weights=$ENCODER_WEIGHTS \
  janus_enabled=True janus_mutual_prompting=True \
  janus_fg_points=1 janus_base_bg_points=3 \
  janus_hard_background=False janus_curvature_allocation=False \
  janus_sam_refinement=True janus_sam_refine_points=1 janus_sam_mined_points=1 \
  janus_sam_mined_min_distance=28 janus_sam_mined_avoid_radius=40 \
  path.CHAOST2.data_dir=$DATA_DIR \
  path.log_dir=${LOGDIR}/fold${FOLD}_supp${SUPP_IDX}_liver

echo "==================== Fold ${FOLD}, support ${SUPP_IDX}: small-organ policy ===================="
python3 test.py with \
  mode=test dataset=CHAOST2 gpu_id=$GPU_ID num_workers=$NUM_WORKERS \
  eval_fold=$FOLD supp_idx=$SUPP_IDX test_label=[2,3,4] n_part=$N_PART seed=$SEED \
  reload_model_path=$RELOAD_MODEL_PATH sam_checkpoint=$SAM_CKPT encoder_pretrained_weights=$ENCODER_WEIGHTS \
  janus_enabled=True janus_mutual_prompting=True \
  janus_fg_points=1 janus_base_bg_points=2 \
  janus_hard_background=True janus_hard_bg_ratio=0.20 janus_hard_bg_max_points=2 \
  janus_curvature_allocation=False \
  janus_sam_refinement=True janus_sam_refine_points=1 janus_sam_mined_points=1 \
  janus_sam_mined_min_distance=20 janus_sam_mined_avoid_radius=28 \
  path.CHAOST2.data_dir=$DATA_DIR \
  path.log_dir=${LOGDIR}/fold${FOLD}_supp${SUPP_IDX}_small_organs
