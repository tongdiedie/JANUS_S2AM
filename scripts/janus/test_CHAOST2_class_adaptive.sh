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

if [[ -n "${FOLD:-}" ]]; then
  FOLDS_STR="$FOLD"
else
  FOLDS_STR=${FOLDS:-"0 1 2 3 4"}
fi

if [[ -n "${SUPP_IDX:-}" ]]; then
  SUPPORTS_STR="$SUPP_IDX"
else
  SUPPORTS_STR=${SUPPORTS:-"2"}
fi

read -ra FOLDS_ARR <<< "$FOLDS_STR"
read -ra SUPPORTS_ARR <<< "$SUPPORTS_STR"

echo "============================================================"
echo "JANUS-S2AM CHAOST2 Class-Adaptive Testing"
echo "DATA_DIR:          $DATA_DIR"
echo "RELOAD_MODEL_PATH: $RELOAD_MODEL_PATH"
echo "FOLDS:             ${FOLDS_ARR[*]}"
echo "SUPPORTS:          ${SUPPORTS_ARR[*]}"
echo "LOGDIR:            $LOGDIR"
echo "============================================================"

for CV_FOLD in "${FOLDS_ARR[@]}"; do
  for SUPP in "${SUPPORTS_ARR[@]}"; do

    echo ""
    echo "==================== Fold ${CV_FOLD}, support ${SUPP}: LIVER policy ===================="

    python3 test.py with \
      mode=test \
      dataset=CHAOST2 \
      gpu_id=$GPU_ID \
      num_workers=$NUM_WORKERS \
      eval_fold=$CV_FOLD \
      supp_idx=$SUPP \
      test_label=[1] \
      n_part=$N_PART \
      seed=$SEED \
      reload_model_path=$RELOAD_MODEL_PATH \
      sam_checkpoint=$SAM_CKPT \
      encoder_pretrained_weights=$ENCODER_WEIGHTS \
      janus_enabled=True \
      janus_mutual_prompting=True \
      janus_fg_points=1 \
      janus_base_bg_points=3 \
      janus_hard_background=False \
      janus_curvature_allocation=False \
      janus_sam_refinement=True \
      janus_sam_refine_points=1 \
      janus_sam_mined_points=1 \
      janus_sam_mined_min_distance=28 \
      janus_sam_mined_avoid_radius=40 \
      path.CHAOST2.data_dir=$DATA_DIR \
      path.log_dir=${LOGDIR}/fold${CV_FOLD}_supp${SUPP}_liver

    echo ""
    echo "==================== Fold ${CV_FOLD}, support ${SUPP}: small-organ policy ===================="

    python3 test.py with \
      mode=test \
      dataset=CHAOST2 \
      gpu_id=$GPU_ID \
      num_workers=$NUM_WORKERS \
      eval_fold=$CV_FOLD \
      supp_idx=$SUPP \
      test_label=[2,3,4] \
      n_part=$N_PART \
      seed=$SEED \
      reload_model_path=$RELOAD_MODEL_PATH \
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
      path.CHAOST2.data_dir=$DATA_DIR \
      path.log_dir=${LOGDIR}/fold${CV_FOLD}_supp${SUPP}_small_organs

  done
done
