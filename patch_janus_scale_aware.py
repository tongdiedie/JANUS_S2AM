#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


def backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak_before_scale_aware_patch")
    if not bak.exists():
        bak.write_text(path.read_text())


def patch_config() -> None:
    p = Path("config.py")
    backup(p)
    s = p.read_text()

    if "janus_scale_aware_policy" not in s:
        marker = "janus_hbg_train_top_quantile = 0.90"
        insert = """
    # Scale-aware sparse JANUS policy.
    janus_scale_aware_policy = False
    janus_large_area_thresh = 0.08

    janus_large_fg_points = 1
    janus_large_base_bg_points = 3
    janus_large_hard_background = False
    janus_large_hard_bg_ratio = 0.0
    janus_large_hard_bg_max_points = 0
    janus_large_sam_refine_points = 1
    janus_large_sam_mined_points = 1
    janus_large_sam_mined_min_distance = 28
    janus_large_sam_mined_avoid_radius = 40

    janus_small_fg_points = 1
    janus_small_base_bg_points = 2
    janus_small_hard_background = True
    janus_small_hard_bg_ratio = 0.20
    janus_small_hard_bg_max_points = 2
    janus_small_sam_refine_points = 1
    janus_small_sam_mined_points = 1
    janus_small_sam_mined_min_distance = 20
    janus_small_sam_mined_avoid_radius = 28
"""
        if marker in s:
            s = s.replace(marker, marker + insert)
        else:
            s += "\n" + insert

    replacements = {
        "janus_curvature_allocation = True": "janus_curvature_allocation = False",
        "janus_fg_points = 6": "janus_fg_points = 1",
        "janus_base_bg_points = 10": "janus_base_bg_points = 2",
        "janus_hard_bg_ratio = 0.40": "janus_hard_bg_ratio = 0.20",
        "janus_hard_bg_max_points = 8": "janus_hard_bg_max_points = 2",
        "janus_sam_refine_points = 4": "janus_sam_refine_points = 1",
        "janus_sam_mined_points = 4": "janus_sam_mined_points = 1",
        "janus_sam_mined_min_distance = 14": "janus_sam_mined_min_distance = 20",
        "janus_sam_mined_avoid_radius = 18": "janus_sam_mined_avoid_radius = 28",
        "janus_hbg_loss_weight = 0.10": "janus_hbg_loss_weight = 0.05",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)

    p.write_text(s)
    print("patched config.py")


def patch_fob() -> None:
    p = Path("models/FoB.py")
    backup(p)
    s = p.read_text()

    if "_resolve_janus_policy" not in s:
        method = """
    def _resolve_janus_policy(self, support_mask, img_size):
        # Resolve sparse JANUS prompt policy from support-mask scale.
        default_policy = {
            "policy_name": "default_sparse",
            "fg_points": int(cfg_get(self.args, "janus_fg_points", 1)),
            "base_bg_points": int(cfg_get(self.args, "janus_base_bg_points", 2)),
            "hard_background": bool(cfg_get(self.args, "janus_hard_background", True)),
            "hard_bg_ratio": float(cfg_get(self.args, "janus_hard_bg_ratio", 0.20)),
            "hard_bg_max_points": int(cfg_get(self.args, "janus_hard_bg_max_points", 2)),
            "hbg_min_distance": int(cfg_get(self.args, "janus_hbg_min_distance", 14)),
            "hbg_avoid_radius": int(cfg_get(self.args, "janus_hbg_avoid_fg_radius", 18)),
            "sam_refine_points": int(cfg_get(self.args, "janus_sam_refine_points", 1)),
            "sam_mined_points": int(cfg_get(self.args, "janus_sam_mined_points", 1)),
            "sam_mined_min_distance": int(cfg_get(self.args, "janus_sam_mined_min_distance", 20)),
            "sam_mined_avoid_radius": int(cfg_get(self.args, "janus_sam_mined_avoid_radius", 28)),
        }

        if not bool(cfg_get(self.args, "janus_scale_aware_policy", False)):
            return default_policy

        mask = support_mask.detach().float()
        area_ratio = float((mask > 0.5).float().mean().item())
        large_thresh = float(cfg_get(self.args, "janus_large_area_thresh", 0.08))

        if area_ratio >= large_thresh:
            return {
                "policy_name": "large_conservative",
                "fg_points": int(cfg_get(self.args, "janus_large_fg_points", 1)),
                "base_bg_points": int(cfg_get(self.args, "janus_large_base_bg_points", 3)),
                "hard_background": bool(cfg_get(self.args, "janus_large_hard_background", False)),
                "hard_bg_ratio": float(cfg_get(self.args, "janus_large_hard_bg_ratio", 0.0)),
                "hard_bg_max_points": int(cfg_get(self.args, "janus_large_hard_bg_max_points", 0)),
                "hbg_min_distance": int(cfg_get(self.args, "janus_hbg_min_distance", 14)),
                "hbg_avoid_radius": int(cfg_get(self.args, "janus_hbg_avoid_fg_radius", 18)),
                "sam_refine_points": int(cfg_get(self.args, "janus_large_sam_refine_points", 1)),
                "sam_mined_points": int(cfg_get(self.args, "janus_large_sam_mined_points", 1)),
                "sam_mined_min_distance": int(cfg_get(self.args, "janus_large_sam_mined_min_distance", 28)),
                "sam_mined_avoid_radius": int(cfg_get(self.args, "janus_large_sam_mined_avoid_radius", 40)),
            }

        return {
            "policy_name": "small_sparse_hbg",
            "fg_points": int(cfg_get(self.args, "janus_small_fg_points", 1)),
            "base_bg_points": int(cfg_get(self.args, "janus_small_base_bg_points", 2)),
            "hard_background": bool(cfg_get(self.args, "janus_small_hard_background", True)),
            "hard_bg_ratio": float(cfg_get(self.args, "janus_small_hard_bg_ratio", 0.20)),
            "hard_bg_max_points": int(cfg_get(self.args, "janus_small_hard_bg_max_points", 2)),
            "hbg_min_distance": int(cfg_get(self.args, "janus_hbg_min_distance", 14)),
            "hbg_avoid_radius": int(cfg_get(self.args, "janus_hbg_avoid_fg_radius", 18)),
            "sam_refine_points": int(cfg_get(self.args, "janus_small_sam_refine_points", 1)),
            "sam_mined_points": int(cfg_get(self.args, "janus_small_sam_mined_points", 1)),
            "sam_mined_min_distance": int(cfg_get(self.args, "janus_small_sam_mined_min_distance", 20)),
            "sam_mined_avoid_radius": int(cfg_get(self.args, "janus_small_sam_mined_avoid_radius", 28)),
        }

"""
        anchor = "    def _build_prompt_bundle("
        if anchor not in s:
            raise RuntimeError("Cannot find _build_prompt_bundle in models/FoB.py")
        s = s.replace(anchor, method + anchor, 1)

    if "policy = self._resolve_janus_policy(support_mask, img_size)" not in s:
        pattern = r"(def _build_prompt_bundle\(.*?h, w = int\(img_size\[0\]\), int\(img_size\[1\]\))"
        s = re.sub(
            pattern,
            r"\1\n        policy = self._resolve_janus_policy(support_mask, img_size)",
            s,
            count=1,
            flags=re.S,
        )

    s = s.replace(
        'num_samples=int(cfg_get(self.args, "janus_fg_points", 6))',
        'num_samples=int(policy["fg_points"])',
    )
    s = s.replace(
        'num_points=int(cfg_get(self.args, "janus_fg_points", 6))',
        'num_points=int(policy["fg_points"])',
    )
    s = s.replace(
        'fallback=int(cfg_get(self.args, "janus_base_bg_points", 10))',
        'fallback=int(policy["base_bg_points"])',
    )

    hard_block_regex = (
        r'if self\.janus_enabled and self\.use_hard_background:\s*'
        r'hard_ratio = float\(cfg_get\(self\.args, "janus_hard_bg_ratio", 0\.40\)\)\s*'
        r'hard_budget = max\(1, int\(round\(total_bg \* hard_ratio\)\)\)\s*'
        r'hard_budget = min\(\s*hard_budget,\s*'
        r'int\(cfg_get\(self\.args, "janus_hard_bg_max_points", 8\)\),\s*'
        r'total_bg,\s*\)\s*'
        r'else:\s*hard_budget = 0'
    )
    hard_block_new = (
        'if self.janus_enabled and bool(policy["hard_background"]): '
        'hard_ratio = float(policy["hard_bg_ratio"]) '
        'hard_budget = max(1, int(round(total_bg * hard_ratio))) '
        'hard_budget = min(hard_budget, int(policy["hard_bg_max_points"]), total_bg) '
        'else: hard_budget = 0'
    )
    s, n = re.subn(hard_block_regex, hard_block_new, s, count=1)
    if n == 0 and 'policy["hard_background"]' not in s:
        print("warning: hard-budget block not replaced; inspect models/FoB.py manually")

    s = s.replace(
        'min_distance=int(cfg_get(self.args, "janus_hbg_min_distance", 14))',
        'min_distance=int(policy["hbg_min_distance"])',
    )
    s = s.replace(
        'avoid_radius=int(cfg_get(self.args, "janus_hbg_avoid_fg_radius", 18))',
        'avoid_radius=int(policy["hbg_avoid_radius"])',
    )
    s = s.replace(
        'int(cfg_get(self.args, "janus_sam_refine_points", 4))',
        'int(policy["sam_refine_points"])',
    )

    if '"sam_mined_min_distance": int(policy["sam_mined_min_distance"])' not in s:
        target = '"shape_complexity": float(compactness),'
        insert = (
            '"shape_complexity": float(compactness), '
            '"scale_policy": policy.get("policy_name", "default_sparse"), '
            '"sam_mined_points": int(policy["sam_mined_points"]), '
            '"sam_mined_min_distance": int(policy["sam_mined_min_distance"]), '
            '"sam_mined_avoid_radius": int(policy["sam_mined_avoid_radius"]),'
        )
        s = s.replace(target, insert, 1)

    if '"effective_policy": policy.get("policy_name", "default_sparse")' not in s:
        s = s.replace(
            '"curvature_allocation": bool(self.use_curvature_allocation),',
            '"curvature_allocation": bool(self.use_curvature_allocation), '
            '"effective_policy": policy.get("policy_name", "default_sparse"),',
            1,
        )

    p.write_text(s)
    print("patched models/FoB.py")


def patch_sam() -> None:
    p = Path("SAM.py")
    backup(p)
    s = p.read_text()

    s = s.replace(
        'num_points=int(cfg_get(config, "janus_sam_mined_points", 4))',
        'num_points=int(prompt_meta.get("sam_mined_points", cfg_get(config, "janus_sam_mined_points", 4)))',
    )
    s = s.replace(
        'min_distance=int(cfg_get(config, "janus_sam_mined_min_distance", 14))',
        'min_distance=int(prompt_meta.get("sam_mined_min_distance", cfg_get(config, "janus_sam_mined_min_distance", 14)))',
    )
    s = s.replace(
        'avoid_radius=int(cfg_get(config, "janus_sam_mined_avoid_radius", 18))',
        'avoid_radius=int(prompt_meta.get("sam_mined_avoid_radius", cfg_get(config, "janus_sam_mined_avoid_radius", 18)))',
    )

    if "janus_refine_score_margin" not in s and "return mask1" in s:
        gate = (
            'area0 = float(np.asarray(mask0).astype(np.float32).sum()) '
            'area1 = float(np.asarray(mask1).astype(np.float32).sum()) '
            'area_ratio = area1 / max(area0, 1.0) if area0 > 1.0 else 1.0 '
            'score_margin = float(cfg_get(config, "janus_refine_score_margin", 0.02)) '
            'min_area_ratio = float(cfg_get(config, "janus_refine_min_area_ratio", 0.35)) '
            'max_area_ratio = float(cfg_get(config, "janus_refine_max_area_ratio", 2.50)) '
            'accept_refine = (float(score1) >= float(score0) - score_margin and '
            'min_area_ratio <= area_ratio <= max_area_ratio) '
            'return mask1 if accept_refine else mask0'
        )
        s = s.replace("return mask1", gate, 1)

    p.write_text(s)
    print("patched SAM.py")


def patch_test() -> None:
    p = Path("test.py")
    backup(p)
    s = p.read_text()

    s = s.replace(
        "DataLoader(test_dataset, batch_size=_config['batch_size'], shuffle=True, num_workers=_config['num_workers'], pin_memory=True, drop_last=True)",
        "DataLoader(test_dataset, batch_size=_config['batch_size'], shuffle=False, num_workers=_config['num_workers'], pin_memory=True, drop_last=False)",
    )
    s = s.replace(
        "test_loader = DataLoader(test_dataset, batch_size=_config['batch_size'], shuffle=True, num_workers=_config['num_workers'], pin_memory=True, drop_last=True)",
        "test_loader = DataLoader(test_dataset, batch_size=_config['batch_size'], shuffle=False, num_workers=_config['num_workers'], pin_memory=True, drop_last=False)",
    )

    p.write_text(s)
    print("patched test.py")


def main() -> None:
    required = ["config.py", "models/FoB.py", "SAM.py", "test.py"]
    missing = [x for x in required if not Path(x).exists()]
    if missing:
        raise SystemExit(f"Run this from repository root. Missing: {missing}")

    patch_config()
    patch_fob()
    patch_sam()
    patch_test()

    print("\nDone. Now run:")
    print("  python -m compileall config.py models/FoB.py SAM.py test.py")
    print("  bash scripts/janus/check_retrain_env.sh")
    print("\nFor scale-aware testing, pass:")
    print("  janus_scale_aware_policy=True")


if __name__ == "__main__":
    main()
