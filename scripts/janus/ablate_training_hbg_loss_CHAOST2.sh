#!/usr/bin/env bash
set -euo pipefail

# Training-time ablation for Hard Background Contrastive Loss on CHAOST2.
# Default: all folds. Use FOLD=2 or FOLDS="0 1" to restrict folds.

GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}

SAM_CKPT=${SAM_CKPT:-./checkpoints/sam_vit_h_4b8939.pth}
ENCODER_WEIGHTS=${ENCODER_WEIGHTS:-COCO}
DATA_DIR=${DATA_DIR:-./data/CHAOST2}
BASE_LOGDIR=${BASE_LOGDIR:-./runs_janus/ablation_train_hbg_loss_CHAOST2}

NUM_WORKERS=${NUM_WORKERS:-16}
N_STEPS=${N_STEPS:-39001}
MAX_ITERS_PER_LOAD=${MAX_ITERS_PER_LOAD:-3000}
SAVE_SNAPSHOT_EVERY=${SAVE_SNAPSHOT_EVERY:-1000}
LR_STEP_GAMMA=${LR_STEP_GAMMA:-0.98}
SEED=${SEED:-2025}

if [[ -n "${FOLD:-}" ]]; then
  FOLDS_STR="$FOLD"
else
  FOLDS_STR=${FOLDS:-"0 1 2 3 4"}
fi
read -ra FOLDS_ARR <<< "$FOLDS_STR"

echo "============================================================"
echo "JANUS-S2AM HBG Loss Training Ablation: CHAOST2"
echo "DATA_DIR:          $DATA_DIR"
echo "FOLDS:             ${FOLDS_ARR[*]}"
echo "N_STEPS:           $N_STEPS"
echo "BASE_LOGDIR:       $BASE_LOGDIR"
echo "============================================================"

for CV_FOLD in "${FOLDS_ARR[@]}"; do
  COMMON=(
    mode=train
    dataset=CHAOST2
    gpu_id=$GPU_ID
    num_workers=$NUM_WORKERS
    n_steps=$N_STEPS
    max_iters_per_load=$MAX_ITERS_PER_LOAD
    save_snapshot_every=$SAVE_SNAPSHOT_EVERY
    lr_step_gamma=$LR_STEP_GAMMA
    eval_fold=$CV_FOLD
    test_label=[1,2,3,4]
    exclude_label=None
    use_gt=False
    seed=$SEED
    sam_checkpoint=$SAM_CKPT
    encoder_pretrained_weights=$ENCODER_WEIGHTS
    janus_enabled=True
    janus_mutual_prompting=True
    janus_hard_background=True
    janus_curvature_allocation=True
    janus_sam_refinement=True
    path.CHAOST2.data_dir=$DATA_DIR
  )

  echo "========== fold ${CV_FOLD}: no_hbg_loss =========="
  python3 train.py with "${COMMON[@]}"     janus_hbg_loss_weight=0.0     path.log_dir=${BASE_LOGDIR}/fold${CV_FOLD}/no_hbg_loss

  echo "========== fold ${CV_FOLD}: with_hbg_loss =========="
  python3 train.py with "${COMMON[@]}"     janus_hbg_loss_weight=0.10     path.log_dir=${BASE_LOGDIR}/fold${CV_FOLD}/with_hbg_loss
done
