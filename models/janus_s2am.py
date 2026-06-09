"""JANUS-S²AM prompt utilities.

This module is intentionally dependency-light and contains no learnable layers.  It
implements the geometry/prototype rules used by JANUS-S²AM:

1. foreground-background mutual similarity maps;
2. prototype-confusion hard background scoring;
3. curvature/shape-aware background prompt allocation;
4. SAM-induced corrective negative point mining.

All point coordinates returned by this file follow SAM convention: ``[x, y]`` in
pixel coordinates of the 256x256 image used by FoB-SAM.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a value from a dict/Sacred config/object without assuming a type."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        if key in cfg:  # Sacred ConfigDict supports this in most versions.
            return cfg[key]
    except Exception:
        pass
    return getattr(cfg, key, default)


def to_numpy_2d(x: Any) -> np.ndarray:
    """Convert a tensor/array with possible singleton dimensions to a 2-D array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x)
    x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2-D map after squeezing, got shape {x.shape}.")
    return x


def normalize_np(x: Any, eps: float = 1e-6) -> np.ndarray:
    """Min-max normalize a 2-D numpy array to [0, 1]."""
    x = to_numpy_2d(x).astype(np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)


def normalize_tensor_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample min-max normalization for [B,1,H,W] or [B,H,W] maps."""
    if x.dim() == 3:
        x = x.unsqueeze(1)
    b = x.shape[0]
    flat = x.reshape(b, -1)
    mn = flat.min(dim=1)[0].view(b, 1, 1, 1)
    mx = flat.max(dim=1)[0].view(b, 1, 1, 1)
    return (x - mn) / (mx - mn + eps)


def ensure_points_array(points: Any) -> np.ndarray:
    """Return a SAM-ready prompt array with shape [1, K, 2]."""
    if points is None:
        return np.zeros((1, 0, 2), dtype=np.float32)
    arr = np.asarray(points, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((1, 0, 2), dtype=np.float32)
    if arr.ndim == 2 and arr.shape[-1] == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3 and arr.shape[-1] == 2:
        pass
    else:
        arr = arr.reshape(1, -1, 2)
    return arr.astype(np.float32)


def squeeze_points(points: Any) -> np.ndarray:
    """Return prompt points as [K, 2]."""
    arr = ensure_points_array(points)
    return arr.reshape(-1, 2).astype(np.float32)


def merge_points(
    *point_groups: Any, image_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Merge multiple point groups into one [1,K,2] prompt array."""
    pts: List[np.ndarray] = []
    for group in point_groups:
        arr = squeeze_points(group)
        if arr.size > 0:
            pts.append(arr)
    if not pts:
        return np.zeros((1, 0, 2), dtype=np.float32)
    merged = np.concatenate(pts, axis=0).astype(np.float32)
    if image_shape is not None:
        h, w = int(image_shape[0]), int(image_shape[1])
        merged[:, 0] = np.clip(merged[:, 0], 0, max(w - 1, 0))
        merged[:, 1] = np.clip(merged[:, 1], 0, max(h - 1, 0))
    return merged[None, ...]


def subsample_points(points: Any, num_points: int) -> np.ndarray:
    """Evenly subsample point list while preserving deterministic order."""
    pts = squeeze_points(points)
    if num_points <= 0 or pts.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(pts) <= num_points:
        return pts.astype(np.float32)
    idx = np.linspace(0, len(pts) - 1, num_points).round().astype(np.int64)
    return pts[idx].astype(np.float32)


def compute_mutual_similarity_maps(
    query_feats: torch.Tensor,
    fg_proto: torch.Tensor,
    bg_proto: torch.Tensor,
    out_size: Tuple[int, int],
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Compute foreground/background prototype competition maps.

    Args:
        query_feats: Query feature map, [B,C,h,w].
        fg_proto: Foreground prototype, [1,C] or [B,C].
        bg_proto: Background prototype, [1,C] or [B,C].
        out_size: Output spatial size (H,W).
    """
    if query_feats.dim() != 4:
        raise ValueError(f"query_feats must be [B,C,H,W], got {query_feats.shape}")

    b, c, _, _ = query_feats.shape
    fg_proto = fg_proto.reshape(-1, c)
    bg_proto = bg_proto.reshape(-1, c)
    if fg_proto.shape[0] == 1 and b > 1:
        fg_proto = fg_proto.expand(b, -1)
    if bg_proto.shape[0] == 1 and b > 1:
        bg_proto = bg_proto.expand(b, -1)

    q = F.normalize(query_feats, dim=1, eps=eps)
    fg = F.normalize(fg_proto, dim=1, eps=eps).view(b, c, 1, 1)
    bg = F.normalize(bg_proto, dim=1, eps=eps).view(b, c, 1, 1)

    s_fg = torch.sum(q * fg, dim=1, keepdim=True)
    s_bg = torch.sum(q * bg, dim=1, keepdim=True)
    s_fg = F.interpolate(s_fg, size=out_size, mode="bilinear", align_corners=True)
    s_bg = F.interpolate(s_bg, size=out_size, mode="bilinear", align_corners=True)

    s_fg_norm = normalize_tensor_map(s_fg)
    s_bg_norm = normalize_tensor_map(s_bg)
    p_fg = s_fg_norm - s_bg_norm
    h_bg = s_fg_norm * s_bg_norm
    return {
        "s_fg": s_fg,
        "s_bg": s_bg,
        "s_fg_norm": s_fg_norm,
        "s_bg_norm": s_bg_norm,
        "p_fg": p_fg,
        "h_bg": h_bg,
    }


def binary_boundary(mask: Any, kernel_size: int = 5) -> np.ndarray:
    """Return a normalized morphological boundary map."""
    mask_np = (normalize_np(mask) > 0.5).astype(np.uint8)
    if mask_np.sum() == 0:
        return np.zeros_like(mask_np, dtype=np.float32)
    k = max(3, int(kernel_size) | 1)
    kernel = np.ones((k, k), np.uint8)
    dilated = cv2.dilate(mask_np, kernel, iterations=1)
    eroded = cv2.erode(mask_np, kernel, iterations=1)
    boundary = (dilated - eroded).astype(np.float32)
    return normalize_np(boundary)


def largest_contour(mask: Any) -> Optional[np.ndarray]:
    """Return the largest external contour as [N,1,2], or None."""
    mask_np = (normalize_np(mask) > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def shape_complexity(mask: Any, eps: float = 1e-6) -> float:
    """Compactness C = P^2/(4*pi*A). C≈1 for round objects, larger for irregular objects."""
    mask_np = (normalize_np(mask) > 0.5).astype(np.uint8)
    area = float(mask_np.sum())
    if area < 1.0:
        return 1.0
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    return float((perimeter * perimeter) / (4.0 * math.pi * area + eps))


def allocate_background_points(
    mask: Any,
    enabled: bool = True,
    low: int = 6,
    mid: int = 10,
    high: int = 16,
    threshold_low: float = 1.5,
    threshold_high: float = 3.0,
    fallback: int = 10,
) -> Tuple[int, float]:
    """Piecewise shape-aware allocation based on compactness."""
    complexity = shape_complexity(mask)
    if not enabled:
        return int(fallback), float(complexity)
    if complexity < threshold_low:
        return int(low), float(complexity)
    if complexity < threshold_high:
        return int(mid), float(complexity)
    return int(high), float(complexity)


def curvature_score(mask: Any, radius: int = 7, blur: int = 7) -> np.ndarray:
    """Build a curvature density map from the largest contour of a binary mask.

    The score is high near boundary points with strong tangent-direction changes.
    """
    mask_np = (normalize_np(mask) > 0.5).astype(np.uint8)
    h, w = mask_np.shape
    score = np.zeros((h, w), dtype=np.float32)
    contour = largest_contour(mask_np)
    if contour is None or len(contour) < 5:
        return score

    pts = contour[:, 0, :].astype(np.float32)
    n = len(pts)
    step = max(2, min(12, n // 60))
    for i in range(n):
        p0 = pts[(i - step) % n]
        p1 = pts[i]
        p2 = pts[(i + step) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cos_angle = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        angle = math.acos(cos_angle)  # 0 on straight lines, high on sharp turns.
        x, y = int(round(p1[0])), int(round(p1[1]))
        cv2.circle(score, (x, y), int(radius), float(angle), thickness=-1)

    if blur and blur > 1:
        k = int(blur) | 1
        score = cv2.GaussianBlur(score, (k, k), 0)
    return normalize_np(score)


def distance_suppression_mask(
    shape: Tuple[int, int],
    avoid_points: Optional[Any],
    radius: float,
) -> np.ndarray:
    """Return True where points are far enough from avoid_points."""
    h, w = int(shape[0]), int(shape[1])
    keep = np.ones((h, w), dtype=bool)
    pts = (
        squeeze_points(avoid_points)
        if avoid_points is not None
        else np.zeros((0, 2), dtype=np.float32)
    )
    if pts.size == 0 or radius <= 0:
        return keep
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = float(radius * radius)
    for x, y in pts:
        keep &= ((xx - float(x)) ** 2 + (yy - float(y)) ** 2) >= r2
    return keep


def nms_topk_points(
    score_map: Any,
    k: int,
    min_distance: int = 12,
    valid_mask: Optional[Any] = None,
    avoid_points: Optional[Any] = None,
    avoid_radius: int = 16,
    threshold_abs: Optional[float] = None,
    fallback_to_argmax: bool = True,
) -> np.ndarray:
    """Select top-K local maxima with deterministic NMS.

    Args:
        score_map: 2-D map where high values are preferred.
        valid_mask: Optional 2-D boolean/float mask restricting candidates.
        avoid_points: Optional [K,2] or [1,K,2] points to suppress locally.
    """
    k = int(k)
    if k <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    score = normalize_np(score_map)
    h, w = score.shape
    finite = np.isfinite(score)
    candidate = finite.copy()
    if valid_mask is not None:
        candidate &= to_numpy_2d(valid_mask).astype(bool)
    candidate &= distance_suppression_mask((h, w), avoid_points, avoid_radius)

    if threshold_abs is not None:
        candidate &= score >= float(threshold_abs)

    if not np.any(candidate):
        if not fallback_to_argmax:
            return np.zeros((0, 2), dtype=np.float32)
        candidate = finite & distance_suppression_mask(
            (h, w), avoid_points, max(0, avoid_radius // 2)
        )
        if not np.any(candidate):
            candidate = finite

    # Local maxima first; fall back to all candidates if the map is too flat.
    kernel_size = max(3, int(min_distance) | 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(score, kernel, iterations=1)
    local_max = (score >= dilated - 1e-6) & candidate
    if np.sum(local_max) < k:
        local_max = candidate

    ys, xs = np.nonzero(local_max)
    vals = score[ys, xs]
    order = np.lexsort((xs, ys, -vals))

    selected: List[Tuple[float, float]] = []
    min_d2 = float(min_distance * min_distance)
    for idx in order:
        x, y = float(xs[idx]), float(ys[idx])
        if all((x - sx) ** 2 + (y - sy) ** 2 >= min_d2 for sx, sy in selected):
            selected.append((x, y))
        if len(selected) >= k:
            break

    # If strict NMS leaves too few points, relax distance once.
    if len(selected) < k:
        relaxed_d2 = float((max(1, min_distance // 2)) ** 2)
        for idx in order:
            x, y = float(xs[idx]), float(ys[idx])
            if all((x - sx) ** 2 + (y - sy) ** 2 >= relaxed_d2 for sx, sy in selected):
                selected.append((x, y))
            if len(selected) >= k:
                break

    return np.asarray(selected, dtype=np.float32).reshape(-1, 2)


def build_hard_background_score(
    fg_score: Any,
    bg_score: Any,
    coarse_mask: Optional[Any] = None,
    curvature: Optional[Any] = None,
    boundary_weight: float = 0.35,
    curvature_weight: float = 0.35,
    bg_weight: float = 0.25,
    fg_core_penalty: float = 0.50,
) -> np.ndarray:
    """Prototype-confusion score used for hard background prompt mining."""
    fg = normalize_np(fg_score)
    bg = normalize_np(bg_score)
    score = fg * bg + bg_weight * bg

    if coarse_mask is not None:
        boundary = binary_boundary(coarse_mask, kernel_size=7)
        score = score + boundary_weight * boundary

    if curvature is not None:
        score = score + curvature_weight * normalize_np(curvature)

    # Protect the most confident foreground core from being selected as negative.
    fg_core = fg >= np.quantile(fg, 0.90)
    score = score - fg_core_penalty * fg_core.astype(np.float32)
    return normalize_np(score)


def foreground_core_points(
    fg_minus_bg: Any,
    fg_score: Optional[Any] = None,
    coarse_mask: Optional[Any] = None,
    num_points: int = 6,
    min_distance: int = 18,
) -> np.ndarray:
    """Select positive foreground prompts from P_fg=S_fg-S_bg."""
    p_fg = normalize_np(fg_minus_bg)
    valid = np.ones_like(p_fg, dtype=bool)
    if coarse_mask is not None:
        cm = normalize_np(coarse_mask)
        if np.any(cm > 0.5):
            valid &= cm > 0.15
    if fg_score is not None:
        fg = normalize_np(fg_score)
        valid &= fg >= np.quantile(fg, 0.70)

    pts = nms_topk_points(
        p_fg,
        k=num_points,
        min_distance=min_distance,
        valid_mask=valid,
        avoid_points=None,
        avoid_radius=0,
        fallback_to_argmax=True,
    )
    return pts


def hard_background_points(
    hbg_score: Any,
    num_points: int,
    coarse_mask: Optional[Any] = None,
    foreground_points: Optional[Any] = None,
    min_distance: int = 14,
    avoid_radius: int = 18,
    prefer_mask_boundary: bool = True,
) -> np.ndarray:
    """Select negative prompts from confusion/boundary/high-curvature background regions."""
    score = normalize_np(hbg_score)
    valid = np.ones_like(score, dtype=bool)
    if coarse_mask is not None and prefer_mask_boundary:
        cm = normalize_np(coarse_mask)
        boundary = binary_boundary(cm, kernel_size=9)
        # Keep boundary-near points if available; otherwise keep all valid pixels.
        boundary_valid = boundary > 0.05
        if np.any(boundary_valid):
            valid &= boundary_valid | (score >= np.quantile(score, 0.85))
    return nms_topk_points(
        score,
        k=num_points,
        min_distance=min_distance,
        valid_mask=valid,
        avoid_points=foreground_points,
        avoid_radius=avoid_radius,
        threshold_abs=None,
        fallback_to_argmax=True,
    )


def mine_sam_induced_hard_background(
    initial_mask: Any,
    prompt_meta: Optional[Dict[str, Any]],
    num_points: int = 4,
    min_distance: int = 14,
    avoid_radius: int = 18,
) -> np.ndarray:
    """Mine corrective negative prompts from SAM's first-pass prediction.

    At inference time there is no ground truth. We therefore expose the most suspicious
    area inside SAM's predicted foreground: high prototype-confusion score, high
    background similarity, and not close to foreground positive prompts.
    """
    if prompt_meta is None or num_points <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    mask = (normalize_np(initial_mask) > 0.5).astype(np.uint8)
    hbg = prompt_meta.get("hbg_score", None)
    if hbg is None:
        hbg = prompt_meta.get("sam_hard_bg_score", None)
    if hbg is None:
        return np.zeros((0, 2), dtype=np.float32)

    score = normalize_np(hbg)
    if mask.shape != score.shape:
        mask = cv2.resize(
            mask, (score.shape[1], score.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    fg_score = prompt_meta.get("fg_score", None)
    bg_score = prompt_meta.get("bg_score", None)
    if fg_score is not None and bg_score is not None:
        fg = normalize_np(fg_score)
        bg = normalize_np(bg_score)
        score = normalize_np(score + 0.40 * bg - 0.25 * fg)

    valid = mask.astype(bool)
    if np.sum(valid) < 5:
        valid = np.ones_like(score, dtype=bool)

    pts = nms_topk_points(
        score,
        k=num_points,
        min_distance=min_distance,
        valid_mask=valid,
        avoid_points=prompt_meta.get("pos_points", None),
        avoid_radius=avoid_radius,
        fallback_to_argmax=True,
    )
    return pts
