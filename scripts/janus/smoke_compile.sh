#!/usr/bin/env bash
set -euo pipefail

python3 -m py_compile   config.py   train.py   test.py   SAM.py   models/FoB.py   models/encoder.py   models/janus_s2am.py

python3 - <<'PY_SMOKE'
import numpy as np
from models.janus_s2am import (
    allocate_background_points,
    curvature_score,
    build_hard_background_score,
    foreground_core_points,
    hard_background_points,
)

mask = np.zeros((128, 128), dtype=np.float32)
mask[30:100, 40:90] = 1

k, c = allocate_background_points(mask)
curv = curvature_score(mask)
score = build_hard_background_score(
    np.random.rand(128, 128),
    np.random.rand(128, 128),
    mask,
    curv,
)
pos = foreground_core_points(score, score, mask, num_points=3)
neg = hard_background_points(score, num_points=3, coarse_mask=mask, foreground_points=pos)

print({
    "allocated_bg": k,
    "compactness": round(c, 3),
    "pos_shape": pos.shape,
    "neg_shape": neg.shape,
})
PY_SMOKE
