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
THRESHOLDS=${THRESHOLDS:-"0.02 0.04 0.06 0.08 0.10 0.12 0.16 0.20"}
LOGDIR=${LOGDIR:-./runs_janus/sweep_scale_thresh_CHAOST2}

for T in $THRESHOLDS; do
  echo "================ threshold ${T} ================"
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
    janus_large_area_thresh=$T \
    path.CHAOST2.data_dir=$DATA_DIR \
    path.log_dir=${LOGDIR}/fold${FOLD}_supp${SUPP_IDX}_t${T}
done

echo ""
echo "Summary:"
grep -R "Whole mean Dice\|Wholemean Dice" "$LOGDIR" || true
