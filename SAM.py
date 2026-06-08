# Code modified from github repository: segment-anything.
# JANUS-S²AM extends the wrapper with optional closed-loop prompt refinement.
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from segment_anything import SamPredictor, sam_model_registry

from models.janus_s2am import (
    cfg_get,
    ensure_points_array,
    merge_points,
    mine_sam_induced_hard_background,
    squeeze_points,
)


class SAM(nn.Module):
    def __init__(
        self,
        sam_pretrained_path: str = ".../sam_vit_h_4b8939.pth",
        model_type: str = "vit_h",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_sam(sam_pretrained_path, model_type=model_type)

    def get_sam(self, checkpoint_path: str, model_type: str = "vit_h"):
        if checkpoint_path is None or str(checkpoint_path).strip() == "":
            raise ValueError("sam_checkpoint is empty. Please pass sam_checkpoint=/path/to/sam_vit_h_4b8939.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"SAM checkpoint not found: {checkpoint_path}. Set sam_checkpoint in config or scripts."
            )
        print(f"Using SAM model type {model_type}")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path).eval().to(self.device)
        self.predictor = SamPredictor(self.sam)
        self.sam.requires_grad_(False)

    @staticmethod
    def _best_mask_index(scores: np.ndarray, config: Optional[Dict[str, Any]] = None) -> int:
        dataset = cfg_get(config, "dataset", "")
        if dataset == "isic" and len(scores) > 1:
            # FoB's original evaluation uses the second SAM mask for Skin-DS.
            return 1
        return int(np.argmax(scores)) if scores is not None and len(scores) > 0 else 0

    @staticmethod
    def _stack_point_prompts(pos_points=None, neg_points=None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        all_points = []
        all_labels = []

        pos = squeeze_points(pos_points) if pos_points is not None else np.zeros((0, 2), dtype=np.float32)
        neg = squeeze_points(neg_points) if neg_points is not None else np.zeros((0, 2), dtype=np.float32)
        if len(pos) > 0:
            all_points.append(pos)
            all_labels.extend([1] * len(pos))
        if len(neg) > 0:
            all_points.append(neg)
            all_labels.extend([0] * len(neg))

        if not all_points:
            return None, None
        points = np.vstack(all_points).astype(np.float32)
        labels = np.asarray(all_labels, dtype=np.int64)
        return points, labels

    def _predict_once(
        self,
        pos_points=None,
        neg_points=None,
        config=None,
        mask_input: Optional[np.ndarray] = None,
        return_logits: bool = False,
    ):
        point_coords, point_labels = self._stack_point_prompts(pos_points, neg_points)
        masks, scores, low_res_logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=None,
            mask_input=mask_input,
            return_logits=return_logits,
            multimask_output=True,
        )
        best_idx = min(self._best_mask_index(scores, config), len(masks) - 1)
        return masks[best_idx], scores[best_idx], low_res_logits[best_idx], best_idx

    def predict_w_points_bbox(self, sam_input_points, bboxes, sam_neg_input_points, qry_img, config=None, return_logits=False):
        """Backward-compatible API used by the original FoB code."""
        assert qry_img.max() <= 255 and qry_img.min() >= 0 and qry_img.dtype == np.uint8
        self.predictor.set_image(qry_img)
        mask, score, _, _ = self._predict_once(
            pos_points=sam_input_points,
            neg_points=sam_neg_input_points,
            config=config,
            mask_input=None,
            return_logits=return_logits,
        )
        return [mask], [score]

    def pre_process(self, image):
        """Convert a normalized CHW tensor to uint8 RGB/3-channel numpy image for SAM."""
        if isinstance(image, torch.Tensor):
            image = image.detach().permute(1, 2, 0).cpu().numpy()
        image = np.asarray(image)
        denom = float(image.max() - image.min())
        if denom < 1e-8:
            return np.zeros_like(image, dtype=np.uint8)
        image = ((image - image.min()) / denom * 255.0).clip(0, 255).astype(np.uint8)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        return image

    def _parse_prompt_input(self, pos_point, neg_point):
        """Accept JANUS prompt dict, original tuple, or explicit pos/neg arrays."""
        prompt_meta = None
        if isinstance(pos_point, dict):
            prompt_meta = pos_point
            pos = prompt_meta.get("pos_points", None)
            neg = prompt_meta.get("neg_points", None)
        elif isinstance(pos_point, (tuple, list)) and len(pos_point) == 2 and neg_point is None:
            # Original FewShotSeg returned (neg_point, pos_point).
            neg, pos = pos_point
        else:
            pos, neg = pos_point, neg_point
        return ensure_points_array(pos), ensure_points_array(neg), prompt_meta

    def forward(self, query_image, pos_point=None, neg_point=None, config=None, return_logits=False):
        """Run SAM with optional JANUS-S²AM two-pass corrective prompting.

        ``pos_point`` may be a JANUS prompt dict returned by ``FewShotSeg``.  In that
        case, SAM first predicts a mask, then mines additional hard-background
        negative prompts from the first mask and re-runs SAM with ``mask_input``.
        """
        query_image_np = self.pre_process(query_image)
        assert query_image_np.max() <= 255 and query_image_np.min() >= 0 and query_image_np.dtype == np.uint8
        self.predictor.set_image(query_image_np)

        pos_points, neg_points, prompt_meta = self._parse_prompt_input(pos_point, neg_point)
        mask0, score0, low_res0, _ = self._predict_once(
            pos_points=pos_points,
            neg_points=neg_points,
            config=config,
            mask_input=None,
            return_logits=False,
        )

        use_refine = bool(cfg_get(config, "janus_sam_refinement", True)) and prompt_meta is not None
        if not use_refine:
            return mask0

        mined_points = mine_sam_induced_hard_background(
            initial_mask=mask0,
            prompt_meta=prompt_meta,
            num_points=int(cfg_get(config, "janus_sam_mined_points", 4)),
            min_distance=int(cfg_get(config, "janus_sam_mined_min_distance", 14)),
            avoid_radius=int(cfg_get(config, "janus_sam_mined_avoid_radius", 18)),
        )
        corrective_points = merge_points(prompt_meta.get("sam_refine_points", None), mined_points, image_shape=mask0.shape)
        if corrective_points.shape[1] == 0:
            return mask0

        refined_neg_points = merge_points(neg_points, corrective_points, image_shape=mask0.shape)
        mask_input = low_res0[None, :, :]
        mask1, score1, _, _ = self._predict_once(
            pos_points=pos_points,
            neg_points=refined_neg_points,
            config=config,
            mask_input=mask_input,
            return_logits=return_logits,
        )
        return mask1
