#!/usr/bin/env python3
"""Patch _qb_bank_family_table: accept a PER-CLIP bank (clip_order == declared
racket clip_names, one row per loaded clip) by returning None = clip_id identity
addressing — the same contract every family_table-None caller already implements
and validate_runtime_motion_contract(clip_families=None) proves. The
(forehand, backhand) 2-row convention stays required for true 2-family banks
shared by many-clip speed lists (spdmix)."""
import io

PATH = ("/workspace/codexschema/chingmu101_20260728/hope_training/whole_body_tracking/"
        "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_commands.py")

OLD = """            order = list(metadata.get("clip_order") or [])
            if order != ["forehand", "backhand"]:
"""
NEW = """            order = list(metadata.get("clip_order") or [])
            declared = [str(n) for n in (getattr(self.cfg, "clip_names_per_clip", ()) or ())]
            _m = getattr(motion, "motion", motion)
            _nseg = int(getattr(_m, "num_segments", 0) or 0)
            if declared and order == declared and (_nseg == 0 or _nseg == len(order)):
                # PER-CLIP bank (chingmu101 whole-library arm, 2026-07-28): the bank carries
                # ONE row per loaded clip (clip_order == racket clip_names, 1:1), so clip_id
                # IS the bank row. Return None = the identity addressing every
                # family_table-None caller implements, and the exact contract
                # validate_runtime_motion_contract(clip_families=None) verifies (per-clip
                # SHA/n_frames/anchor). The (forehand, backhand) 2-row convention below only
                # binds when a many-clip motion list shares a 2-family bank (spdmix 变速表).
                return None
            if order != ["forehand", "backhand"]:
"""

src = io.open(PATH, encoding="utf-8").read()
assert src.count(OLD) == 1, f"expected exactly 1 anchor, found {src.count(OLD)}"
io.open(PATH, "w", encoding="utf-8").write(src.replace(OLD, NEW))
print("patched _qb_bank_family_table")
import py_compile

py_compile.compile(PATH, doraise=True)
print("py_compile OK")
