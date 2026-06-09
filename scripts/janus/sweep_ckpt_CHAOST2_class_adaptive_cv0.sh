#!/usr/bin/env bash
set -euo pipefail

FOLD=${FOLD:-0}
SUPP_IDX=${SUPP_IDX:-2}
DATA_DIR=${DATA_DIR:-./data/CHAOST2}
TRAIN_DIR=${TRAIN_DIR:-./runs_janus/train_CHAOST2_safe_base_for_janus}
LOGDIR=${LOGDIR:-./runs_janus/sweep_ckpt_CHAOST2_class_adaptive}

SNAP_DIR=$(find "$TRAIN_DIR" -path "*cv${FOLD}*/snapshots" -type d | tail -1)
if [[ -z "$SNAP_DIR" ]]; then
  echo "Cannot find snapshots for fold ${FOLD} under ${TRAIN_DIR}"
  exit 1
fi

echo "Using SNAP_DIR=$SNAP_DIR"

for STEP in 1000 3000 5000 7000 9000 11000 13000 15000 17000 19000 21000 23000 25000 27000 29000 31000 33000 35000 37000 39000; do
  CKPT="${SNAP_DIR}/${STEP}.pth"
  if [[ ! -f "$CKPT" ]]; then
    continue
  fi

  echo ""
  echo "==================== Testing checkpoint ${STEP} ===================="

  FOLD=$FOLD SUPP_IDX=$SUPP_IDX \
  RELOAD_MODEL_PATH=$CKPT \
  DATA_DIR=$DATA_DIR \
  LOGDIR=${LOGDIR}/step${STEP} \
  bash scripts/janus/test_CHAOST2_class_adaptive.sh
done

echo ""
echo "Summary:"
grep -R "Whole mean Dice\|Wholemean Dice\|Mean Dice" "$LOGDIR" || true
