#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}
SAM_CKPT=${SAM_CKPT:-./checkpoints/sam_vit_h_4b8939.pth}
ENCODER_WEIGHTS=${ENCODER_WEIGHTS:-COCO}
DATA_DIR=${DATA_DIR:-./data/SABS}
LOGDIR=${LOGDIR:-./runs_janus/test_SABS_full}
RELOAD_MODEL_PATH=${RELOAD_MODEL_PATH:?Please export RELOAD_MODEL_PATH=/path/to/trained_snapshot.pth}
FOLDS_STR=${FOLDS:-"0 1 2 3 4"}
SUPPORTS_STR=${SUPPORTS:-"2"}
read -ra FOLDS_ARR <<< "$FOLDS_STR"
read -ra SUPPORTS_ARR <<< "$SUPPORTS_STR"

for FOLD in "${FOLDS_ARR[@]}"; do
  for SUPP in "${SUPPORTS_ARR[@]}"; do
    python3 test.py with \
      mode=test \
      dataset=SABS \
      gpu_id=$GPU_ID \
      num_workers=${NUM_WORKERS:-16} \
      n_steps=${N_STEPS:-39001} \
      max_iters_per_load=${MAX_ITERS_PER_LOAD:-3000} \
      save_snapshot_every=${SAVE_SNAPSHOT_EVERY:-3000} \
      lr_step_gamma=${LR_STEP_GAMMA:-0.98} \
      eval_fold=$FOLD \
      supp_idx=$SUPP \
      test_label=[1,2,3,6] \
      n_part=${N_PART:-3} \
      seed=${SEED:-2025} \
      reload_model_path=$RELOAD_MODEL_PATH \
      sam_checkpoint=$SAM_CKPT \
      encoder_pretrained_weights=$ENCODER_WEIGHTS \
      janus_enabled=True \
      janus_mutual_prompting=True \
      janus_hard_background=True \
      janus_curvature_allocation=True \
      janus_sam_refinement=True \
      path.SABS.data_dir=$DATA_DIR \
      path.log_dir=$LOGDIR
  done
done
