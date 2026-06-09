#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}

DATA_DIR=${DATA_DIR:-./data/CHAOST2}
SAM_CKPT=${SAM_CKPT:-./checkpoints/sam_vit_h_4b8939.pth}
ENCODER_WEIGHTS=${ENCODER_WEIGHTS:-COCO}
RELOAD_MODEL_PATH=${RELOAD_MODEL_PATH:?Please export RELOAD_MODEL_PATH=/path/to/snapshot.pth}

FOLD=${FOLD:-0}
SUPP_IDX=${SUPP_IDX:-2}
N_PART=${N_PART:-3}
THRESH=${THRESH:-0.08}
LOGDIR=${LOGDIR:-./runs_janus/debug_scale_aware_CHAOST2}

python3 test.py with \
  mode=test \
  dataset=CHAOST2 \
  gpu_id=$GPU_ID \
  eval_fold=$FOLD \
  supp_idx=$SUPP_IDX \
  test_label=[1,2,3,4] \
  n_part=$N_PART \
  reload_model_path=$RELOAD_MODEL_PATH \
  sam_checkpoint=$SAM_CKPT \
  encoder_pretrained_weights=$ENCODER_WEIGHTS \
  janus_enabled=True \
  janus_scale_aware_policy=True \
  janus_debug_policy=True \
  janus_large_area_thresh=$THRESH \
  path.CHAOST2.data_dir=$DATA_DIR \
  path.log_dir=${LOGDIR}/fold${FOLD}_supp${SUPP_IDX}_t${THRESH}
