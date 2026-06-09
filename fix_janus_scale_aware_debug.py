#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


def read(path: str) -> str:
    return Path(path).read_text()


def write(path: str, text: str) -> None:
    Path(path).write_text(text)


def backup(path: str) -> None:
    p = Path(path)
    b = p.with_suffix(p.suffix + ".bak_janus_debug_fix")
    if not b.exists():
        b.write_text(p.read_text())


def patch_config() -> None:
    path = "config.py"
    backup(path)
    s = read(path)

    if "janus_debug_policy" not in s:
        if "janus_scale_aware_policy" in s:
            s = s.replace(
                "janus_scale_aware_policy = False",
                "janus_scale_aware_policy = False\n    janus_debug_policy = False",
                1,
            )
        elif "janus_enabled" in s:
            s = s.replace(
                "janus_enabled = True",
                "janus_enabled = True\n    janus_debug_policy = False",
                1,
            )
        else:
            s += "\n    janus_debug_policy = False\n"

    if "janus_large_area_thresh" not in s:
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
        marker = "janus_hbg_train_top_quantile = 0.90"
        if marker in s:
            s = s.replace(marker, marker + insert, 1)
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

    write(path, s)
    print("OK config.py")


def fix_fob_syntax_and_debug() -> None:
    path = "models/FoB.py"
    backup(path)
    s = read(path)

    bad = (
        'if self.janus_enabled and bool(policy["hard_background"]): '
        'hard_ratio = float(policy["hard_bg_ratio"]) '
        'hard_budget = max(1, int(round(total_bg * hard_ratio))) '
        'hard_budget = min(hard_budget, int(policy["hard_bg_max_points"]), total_bg) '
        'else: hard_budget = 0'
    )
    good = '''if self.janus_enabled and bool(policy["hard_background"]):
            hard_ratio = float(policy["hard_bg_ratio"])
            hard_budget = max(1, int(round(total_bg * hard_ratio)))
            hard_budget = min(hard_budget, int(policy["hard_bg_max_points"]), total_bg)
        else:
            hard_budget = 0'''
    if bad in s:
        s = s.replace(bad, good, 1)

    if "_resolve_janus_policy" not in s:
        method = '''
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

        if bool(cfg_get(self.args, "janus_debug_policy", False)):
            policy_name = "large_conservative" if area_ratio >= large_thresh else "small_sparse_hbg"
            print(
                f"[JANUS policy] area_ratio={area_ratio:.6f}, "
                f"large_thresh={large_thresh:.6f}, policy={policy_name}"
            )

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

'''
        anchor = "    def _build_prompt_bundle("
        if anchor not in s:
            raise RuntimeError("Cannot find _build_prompt_bundle in models/FoB.py")
        s = s.replace(anchor, method + anchor, 1)

    if "janus_debug_policy" not in s and 'large_thresh = float(cfg_get(self.args, "janus_large_area_thresh", 0.08))' in s:
        s = s.replace(
            'large_thresh = float(cfg_get(self.args, "janus_large_area_thresh", 0.08))',
            '''large_thresh = float(cfg_get(self.args, "janus_large_area_thresh", 0.08))

        if bool(cfg_get(self.args, "janus_debug_policy", False)):
            policy_name = "large_conservative" if area_ratio >= large_thresh else "small_sparse_hbg"
            print(
                f"[JANUS policy] area_ratio={area_ratio:.6f}, "
                f"large_thresh={large_thresh:.6f}, policy={policy_name}"
            )''',
            1,
        )

    if "policy = self._resolve_janus_policy(support_mask, img_size)" not in s:
        s = re.sub(
            r"(h, w = int\(img_size\[0\]\), int\(img_size\[1\]\))",
            r"\1\n        policy = self._resolve_janus_policy(support_mask, img_size)",
            s,
            count=1,
        )

    write(path, s)
    print("OK models/FoB.py")


def fix_sam_syntax() -> None:
    path = "SAM.py"
    backup(path)
    s = read(path)

    bad = (
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
    good = '''area0 = float(np.asarray(mask0).astype(np.float32).sum())
        area1 = float(np.asarray(mask1).astype(np.float32).sum())
        area_ratio = area1 / max(area0, 1.0) if area0 > 1.0 else 1.0

        score_margin = float(cfg_get(config, "janus_refine_score_margin", 0.02))
        min_area_ratio = float(cfg_get(config, "janus_refine_min_area_ratio", 0.35))
        max_area_ratio = float(cfg_get(config, "janus_refine_max_area_ratio", 2.50))

        accept_refine = (
            float(score1) >= float(score0) - score_margin
            and min_area_ratio <= area_ratio <= max_area_ratio
        )

        return mask1 if accept_refine else mask0'''
    if bad in s:
        s = s.replace(bad, good, 1)

    s = s.replace(
        'num_points=int(cfg_get(config, "janus_sam_mined_points", 4))',
        'num_points=int(prompt_meta.get("sam_mined_points", cfg_get(config, "janus_sam_mined_points", 1)))',
    )
    s = s.replace(
        'min_distance=int(cfg_get(config, "janus_sam_mined_min_distance", 14))',
        'min_distance=int(prompt_meta.get("sam_mined_min_distance", cfg_get(config, "janus_sam_mined_min_distance", 20)))',
    )
    s = s.replace(
        'avoid_radius=int(cfg_get(config, "janus_sam_mined_avoid_radius", 18))',
        'avoid_radius=int(prompt_meta.get("sam_mined_avoid_radius", cfg_get(config, "janus_sam_mined_avoid_radius", 28)))',
    )

    write(path, s)
    print("OK SAM.py")


def fix_test_loader() -> None:
    path = "test.py"
    backup(path)
    s = read(path)
    s = s.replace("shuffle=True", "shuffle=False")
    s = s.replace("drop_last=True", "drop_last=False")
    write(path, s)
    print("OK test.py")


def main() -> None:
    for p in ["config.py", "models/FoB.py", "SAM.py", "test.py"]:
        if not Path(p).exists():
            raise SystemExit(f"Missing {p}; run from repository root.")

    patch_config()
    fix_fob_syntax_and_debug()
    fix_sam_syntax()
    fix_test_loader()

    print("\nDone.")
    print("Run:")
    print("  python -m compileall config.py models/FoB.py SAM.py test.py")
    print("  DATA_DIR=$PWD/data/CHAOST2 bash scripts/janus/check_retrain_env.sh")


if __name__ == "__main__":
    main()
