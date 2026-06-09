#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}

SAM_CKPT=${SAM_CKPT:-./checkpoints/sam_vit_h_4b8939.pth}
ENCODER_WEIGHTS=${ENCODER_WEIGHTS:-COCO}
DATA_DIR=${DATA_DIR:-./data/CHAOST2}
RELOAD_MODEL_PATH=${RELOAD_MODEL_PATH:?Please export RELOAD_MODEL_PATH=/path/to/trained_snapshot.pth}
BASE_LOGDIR=${BASE_LOGDIR:-./runs_janus/ablation_CHAOST2_inference_light}

NUM_WORKERS=${NUM_WORKERS:-16}
N_STEPS=${N_STEPS:-39001}
MAX_ITERS_PER_LOAD=${MAX_ITERS_PER_LOAD:-3000}
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
echo "JANUS-S2AM Light Inference Ablation: CHAOST2"
echo "DATA_DIR:          $DATA_DIR"
echo "RELOAD_MODEL_PATH: $RELOAD_MODEL_PATH"
echo "FOLDS:             ${FOLDS_ARR[*]}"
echo "SUPPORTS:          ${SUPPORTS_ARR[*]}"
echo "BASE_LOGDIR:       $BASE_LOGDIR"
echo "============================================================"

run_variant () {
  local name=$1
  shift

  echo ""
  echo "========== $name =========="

  python3 test.py with "${COMMON[@]}" \
    path.log_dir=${RUN_LOGDIR}/${name} \
    "$@"
}

for CV_FOLD in "${FOLDS_ARR[@]}"; do
  for SUPP in "${SUPPORTS_ARR[@]}"; do
    RUN_LOGDIR="${BASE_LOGDIR}/fold${CV_FOLD}_supp${SUPP}"

    COMMON=(
      mode=test
      dataset=CHAOST2
      gpu_id=$GPU_ID
      num_workers=$NUM_WORKERS
      n_steps=$N_STEPS
      max_iters_per_load=$MAX_ITERS_PER_LOAD
      eval_fold=$CV_FOLD
      supp_idx=$SUPP
      test_label=[1,2,3,4]
      n_part=$N_PART
      seed=$SEED
      reload_model_path=$RELOAD_MODEL_PATH
      sam_checkpoint=$SAM_CKPT
      encoder_pretrained_weights=$ENCODER_WEIGHTS
      path.CHAOST2.data_dir=$DATA_DIR
    )

    run_variant A0_fob_compatible \
      janus_enabled=False \
      janus_mutual_prompting=False \
      janus_hard_background=False \
      janus_curvature_allocation=False \
      janus_sam_refinement=False

    run_variant A1_light_mutual \
      janus_enabled=True \
      janus_mutual_prompting=True \
      janus_fg_points=1 \
      janus_base_bg_points=3 \
      janus_hard_background=False \
      janus_curvature_allocation=False \
      janus_sam_refinement=False

    run_variant A2_light_hard_background \
      janus_enabled=True \
      janus_mutual_prompting=True \
      janus_fg_points=1 \
      janus_base_bg_points=3 \
      janus_hard_background=True \
      janus_hard_bg_ratio=0.20 \
      janus_hard_bg_max_points=2 \
      janus_curvature_allocation=False \
      janus_sam_refinement=False

    run_variant A3_light_sam_refine \
      janus_enabled=True \
      janus_mutual_prompting=True \
      janus_fg_points=1 \
      janus_base_bg_points=3 \
      janus_hard_background=False \
      janus_curvature_allocation=False \
      janus_sam_refinement=True \
      janus_sam_refine_points=1 \
      janus_sam_mined_points=1 \
      janus_sam_mined_min_distance=20 \
      janus_sam_mined_avoid_radius=28

    run_variant A4_light_hbg_refine \
      janus_enabled=True \
      janus_mutual_prompting=True \
      janus_fg_points=1 \
      janus_base_bg_points=3 \
      janus_hard_background=True \
      janus_hard_bg_ratio=0.20 \
      janus_hard_bg_max_points=2 \
      janus_curvature_allocation=False \
      janus_sam_refinement=True \
      janus_sam_refine_points=1 \
      janus_sam_mined_points=1 \
      janus_sam_mined_min_distance=20 \
      janus_sam_mined_avoid_radius=28

    run_variant A5_light_curvature \
      janus_enabled=True \
      janus_mutual_prompting=True \
      janus_fg_points=1 \
      janus_base_bg_points=3 \
      janus_hard_background=False \
      janus_curvature_allocation=True \
      janus_bg_points_low=3 \
      janus_bg_points_mid=4 \
      janus_bg_points_high=5 \
      janus_sam_refinement=False
  done
done
