#!/usr/bin/env bash
set -euo pipefail

# Inference-time ablation on CHAOST2 using the same trained checkpoint.
# Required: export RELOAD_MODEL_PATH=/path/to/snapshot.pth
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}
SAM_CKPT=${SAM_CKPT:-./checkpoints/sam_vit_h_4b8939.pth}
ENCODER_WEIGHTS=${ENCODER_WEIGHTS:-COCO}
DATA_DIR=${DATA_DIR:-./data/CHAOST2}
RELOAD_MODEL_PATH=${RELOAD_MODEL_PATH:?Please export RELOAD_MODEL_PATH=/path/to/trained_snapshot.pth}
FOLD=${FOLD:-0}
SUPP_IDX=${SUPP_IDX:-2}
BASE_LOGDIR=${BASE_LOGDIR:-./runs_janus/ablation_CHAOST2_inference}

COMMON=(
  mode=test dataset=CHAOST2 gpu_id=$GPU_ID num_workers=${NUM_WORKERS:-16}
  n_steps=${N_STEPS:-39001} max_iters_per_load=${MAX_ITERS_PER_LOAD:-3000}
  eval_fold=$FOLD supp_idx=$SUPP_IDX test_label=[1,2,3,4] n_part=${N_PART:-3}
  seed=${SEED:-2025} reload_model_path=$RELOAD_MODEL_PATH
  sam_checkpoint=$SAM_CKPT encoder_pretrained_weights=$ENCODER_WEIGHTS
  path.CHAOST2.data_dir=$DATA_DIR
)

run_variant () {
  local name=$1; shift
  echo "========== ${name} =========="
  python3 test.py with "${COMMON[@]}" path.log_dir=${BASE_LOGDIR}/${name} "$@"
}

# A0: original FoB-style prompts only: learned background prompts + coarse positive points.
run_variant A0_fob_compatible \
  janus_enabled=False janus_mutual_prompting=False janus_hard_background=False \
  janus_curvature_allocation=False janus_sam_refinement=False

# A1: foreground-background mutual prompting only.
run_variant A1_mutual_prompting \
  janus_enabled=True janus_mutual_prompting=True janus_hard_background=False \
  janus_curvature_allocation=False janus_sam_refinement=False

# A2: add prototype-confusion hard background prompts, fixed K.
run_variant A2_hard_background_fixedK \
  janus_enabled=True janus_mutual_prompting=True janus_hard_background=True \
  janus_curvature_allocation=False janus_sam_refinement=False

# A3: add curvature-aware/shape-aware allocation, no SAM second pass.
run_variant A3_curvature_allocation \
  janus_enabled=True janus_mutual_prompting=True janus_hard_background=True \
  janus_curvature_allocation=True janus_sam_refinement=False

# A4: full inference pipeline with SAM-induced hard-background corrective prompts.
run_variant A4_full_inference \
  janus_enabled=True janus_mutual_prompting=True janus_hard_background=True \
  janus_curvature_allocation=True janus_sam_refinement=True
