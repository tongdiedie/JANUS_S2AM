#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# JANUS-S2AM testing script: test_SABS_full.sh
#
# Usage:
#   RELOAD_MODEL_PATH=/path/to/model.pth bash scripts/janus/test_SABS_full.sh
#   FOLD=0 RELOAD_MODEL_PATH=/path/to/model.pth bash scripts/janus/test_SABS_full.sh
#   FOLDS="0 3" RELOAD_MODEL_PATH=/path/to/model.pth bash scripts/janus/test_SABS_full.sh
#   FOLD=0 SUPP_IDX=2 RELOAD_MODEL_PATH=/path/to/model.pth bash scripts/janus/test_SABS_full.sh
# ============================================================

GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}

SAM_CKPT=${SAM_CKPT:-./checkpoints/sam_vit_h_4b8939.pth}
ENCODER_WEIGHTS=${ENCODER_WEIGHTS:-COCO}
DATA_DIR=${DATA_DIR:-./data/SABS}
LOGDIR=${LOGDIR:-./runs_janus/test_SABS_full}
RELOAD_MODEL_PATH=${RELOAD_MODEL_PATH:?Please export RELOAD_MODEL_PATH=/path/to/trained_snapshot.pth}

NUM_WORKERS=${NUM_WORKERS:-16}
N_STEPS=${N_STEPS:-39001}
MAX_ITERS_PER_LOAD=${MAX_ITERS_PER_LOAD:-3000}
SAVE_SNAPSHOT_EVERY=${SAVE_SNAPSHOT_EVERY:-3000}
LR_STEP_GAMMA=${LR_STEP_GAMMA:-0.98}
N_PART=${N_PART:-3}
SEED=${SEED:-2025}

# Priority:
#   FOLD=2        -> single fold
#   FOLDS="0 1"   -> selected folds
#   neither set   -> all default folds
if [[ -n "${FOLD:-}" ]]; then
  FOLDS_STR="$FOLD"
else
  FOLDS_STR=${FOLDS:-"0 1 2 3 4"}
fi

read -ra FOLDS_ARR <<< "$FOLDS_STR"

# Priority:
#   SUPP_IDX=2        -> single support index
#   SUPPORTS="0 1 2"  -> selected support indices
#   neither set       -> default support 2
if [[ -n "${SUPP_IDX:-}" ]]; then
  SUPPORTS_STR="$SUPP_IDX"
else
  SUPPORTS_STR=${SUPPORTS:-"2"}
fi

read -ra SUPPORTS_ARR <<< "$SUPPORTS_STR"

echo "============================================================"
echo "JANUS-S2AM Testing"
echo "SCRIPT:              $0"
echo "DATA_DIR:            $DATA_DIR"
echo "SAM_CKPT:            $SAM_CKPT"
echo "ENCODER_WEIGHTS:     $ENCODER_WEIGHTS"
echo "RELOAD_MODEL_PATH:   $RELOAD_MODEL_PATH"
echo "LOGDIR:              $LOGDIR"
echo "GPU_ID:              $GPU_ID"
echo "CUDA_VISIBLE_DEVICES:$CUDA_VISIBLE_DEVICES"
echo "FOLDS:               ${FOLDS_ARR[*]}"
echo "SUPPORTS:            ${SUPPORTS_ARR[*]}"
echo "N_PART:              $N_PART"
echo "SEED:                $SEED"
echo "============================================================"

for CV_FOLD in "${FOLDS_ARR[@]}"; do
  for SUPP in "${SUPPORTS_ARR[@]}"; do
    echo ""
    echo "==================== Testing fold $CV_FOLD support $SUPP ===================="

    python3 test.py with \
      mode=test \
      dataset=SABS \
      gpu_id=$GPU_ID \
      num_workers=$NUM_WORKERS \
      n_steps=$N_STEPS \
      max_iters_per_load=$MAX_ITERS_PER_LOAD \
      save_snapshot_every=$SAVE_SNAPSHOT_EVERY \
      lr_step_gamma=$LR_STEP_GAMMA \
      eval_fold=$CV_FOLD \
    supp_idx=$SUPP \
      test_label=[1,2,3,6] \
      n_part=$N_PART \
      seed=$SEED \
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
