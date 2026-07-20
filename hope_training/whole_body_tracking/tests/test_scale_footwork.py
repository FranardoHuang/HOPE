"""Unit tests for scripts/scale_footwork.py (Axis-D footwork amplitude β, 脚步幅度).

Pure CPU, numpy + stdlib only — the module under test is loaded by file path
(same pattern as test_retime_motion_clip.py). Synthetic side-shuffle clips with
hand-checkable numbers exercise:

  * hard refusals: β=0 (must reference a real stationary strike), β<0 (mirror,
    not a scale), off-grid β (buckets are preregistered), missing contact-phase
    annotation, non-finite inputs, unresolved body order;
  * parameterization extraction: separation vector (robust median), root signed
    lateral displacement, support anchors, footfall positions, contact table;
  * β action: root displacement and footfall lateral offsets scale, support
    anchors stay pinned, fore-aft and height NEVER scale;
  * frozen stance gate with the four known M0 0/4 failures as negative examples
    (left1 0.095425 / left2 0.200557 / right1 0.076532 / right2 0.024300 — the
    last one must die on the independent 0.005 m narrowing hard gate alone);
  * foot crossing / minimum stance width: per-frame on the source AND on the
    β-scaled footfall schedule (β=1.35 crosses where β=0.8 is legal);
  * manifest content: frozen β buckets written verbatim, input sha256, a
    self-consistent one-layer spec content sha, training_authorized=False.

Run:  python3 -m pytest hope_training/whole_body_tracking/tests/test_scale_footwork.py -q
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# --- load the module under test by path (scripts/ is not a package) ----------- #
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_spec = importlib.util.spec_from_file_location("scale_footwork", _SCRIPTS / "scale_footwork.py")
sf = importlib.util.module_from_spec(_spec)
sys.modules["scale_footwork"] = sf
_spec.loader.exec_module(sf)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CATALOG = _REPO_ROOT / "configs" / "motion_role_catalog.json"

T = 120
FPS = 50.0
BODY_NAMES = ["base_Link", sf.LEFT_FOOT_BODY, sf.RIGHT_FOOT_BODY]

# window layout shared by every synthetic clip:
#   [0,19] both support | [20,39] first foot swings | [40,59] both support
#   [60,79] second foot swings | [80,119] both support (recover stance)
W0_END, A0, A1, W1_END, B0, B1 = 19, 20, 39, 59, 60, 79


def _swing(y: np.ndarray, z: np.ndarray, s: int, e: int, y_from: float, y_to: float) -> None:
    """Foot leaves the ground at s, lands at e+1: linear y, parabolic z bump."""
    for t in range(s, e + 1):
        p = (t - (s - 1)) / (e + 2 - s)
        y[t] = y_from + p * (y_to - y_from)
        z[t] = 0.05 * 4.0 * p * (1.0 - p)


def make_clip(l0: float = 0.10, r0: float = -0.10, l_step: float = 0.15,
              r_step: float = 0.15, d: float = 0.15, first: str = "left") -> dict:
    """Side-shuffle toward +lateral (robot-left). `first` foot steps on [20,39],
    the other on [60,79]; root ramps 0 -> d over [20,80]. All float64."""
    left_y = np.full(T, l0, dtype=np.float64)
    right_y = np.full(T, r0, dtype=np.float64)
    left_z = np.zeros(T)
    right_z = np.zeros(T)
    if first == "left":
        _swing(left_y, left_z, A0, A1, l0, l0 + l_step)
        left_y[A1 + 1:] = l0 + l_step
        _swing(right_y, right_z, B0, B1, r0, r0 + r_step)
        right_y[B1 + 1:] = r0 + r_step
        first_contacts = [[0, W0_END], [A1 + 1, T - 1]]
        second_contacts = [[0, W1_END], [B1 + 1, T - 1]]
        contacts = {"left": first_contacts, "right": second_contacts}
    else:
        _swing(right_y, right_z, A0, A1, r0, r0 + r_step)
        right_y[A1 + 1:] = r0 + r_step
        _swing(left_y, left_z, B0, B1, l0, l0 + l_step)
        left_y[B1 + 1:] = l0 + l_step
        contacts = {"right": [[0, W0_END], [A1 + 1, T - 1]],
                    "left": [[0, W1_END], [B1 + 1, T - 1]]}

    root_y = np.zeros(T)
    ramp = np.linspace(0.0, d, (B1 + 1) - A0 + 1)
    root_y[A0:B1 + 2] = ramp
    root_y[B1 + 2:] = d

    body_pos = np.zeros((T, 3, 3), dtype=np.float64)
    body_pos[:, 0, 1] = root_y
    body_pos[:, 0, 2] = 0.85
    body_pos[:, 1, 1] = left_y
    body_pos[:, 1, 2] = left_z
    body_pos[:, 2, 1] = right_y
    body_pos[:, 2, 2] = right_z
    body_quat = np.zeros((T, 3, 4), dtype=np.float64)
    body_quat[..., 0] = 1.0  # identity wxyz -> yaw 0, movement frame == world
    return {
        "fps": np.float64(FPS),
        "joint_pos": np.zeros((T, sf.N_JOINTS), dtype=np.float64),
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "contacts": contacts,
    }


def write_clip(tmp_path: Path, clip: dict, *, embed_contacts: bool = True,
               embed_body_names: bool = True, name: str = "clip.npz") -> Path:
    arrays = {k: clip[k] for k in ("fps", "joint_pos", "body_pos_w", "body_quat_w")}
    if embed_body_names:
        arrays["body_names"] = np.asarray(BODY_NAMES)
    if embed_contacts:
        for foot in ("left", "right"):
            mask = np.zeros(T, dtype=np.int64)
            for s, e in clip["contacts"][foot]:
                mask[s:e + 1] = 1
            arrays[f"{foot}_foot_contact"] = mask
    path = tmp_path / name
    np.savez(path, **arrays)
    return path


def cli(npz: Path, out: Path, *, beta: str = "1.20", direction: str | None = "left",
        extra: tuple = ()) -> list:
    args = ["--footwork-npz", str(npz), "--beta", beta,
            "--strike-asset-id", "hope_forehand_v4rg_cal", "--strike-phase", "0.471",
            "--output", str(out)]
    if direction is not None:
        args += ["--direction", direction]
    return args + list(extra)


def run_spec(args: list) -> dict:
    return sf.build_spec(sf.build_parser().parse_args(args))


def refuse(args: list) -> sf.FootworkScaleError:
    with pytest.raises(sf.FootworkScaleError) as ei:
        run_spec(args)
    return ei.value


# ------------------------------------------------------------ beta refusals -- #
def test_beta_zero_rejected_points_to_stationary_strike(tmp_path):
    err = refuse(cli(tmp_path / "missing.npz", tmp_path / "o.json", beta="0.0"))
    assert err.code == "beta_zero_stationary"
    assert "stationary_strike" in str(err)  # d=0 must cite the real stationary asset


def test_beta_negative_rejected_as_mirror(tmp_path):
    err = refuse(cli(tmp_path / "missing.npz", tmp_path / "o.json", beta="-1.0"))
    assert err.code == "beta_negative_mirror"
    assert "mirror" in str(err).lower()


def test_beta_off_grid_rejected(tmp_path):
    for bad in ("1.05", "0.5", "2.0"):
        err = refuse(cli(tmp_path / "missing.npz", tmp_path / "o.json", beta=bad))
        assert err.code == "beta_off_grid"


def test_beta_bucket_mapping_is_frozen():
    assert sf.BETA_BUCKETS == {"train": (0.80, 1.00, 1.20),
                               "interpolation": (0.90, 1.10),
                               "ood": (0.65, 1.35)}
    for b, want in ((0.80, "train"), (1.00, "train"), (1.20, "train"),
                    (0.90, "interpolation"), (1.10, "interpolation"),
                    (0.65, "ood"), (1.35, "ood")):
        assert sf.validate_beta(b) == want


# --------------------------------------------------- contact-phase fail-closed #
def test_missing_contact_annotation_fail_closed(tmp_path):
    clip = make_clip()
    npz = write_clip(tmp_path, clip, embed_contacts=False)
    err = refuse(cli(npz, tmp_path / "o.json"))
    assert err.code == "contact_phase_missing"

    # both npz-embedded AND --contact-json -> conflict, refuse
    npz2 = write_clip(tmp_path, clip, embed_contacts=True, name="c2.npz")
    cj = tmp_path / "contacts.json"
    cj.write_text(json.dumps(clip["contacts"]))
    err = refuse(cli(npz2, tmp_path / "o.json", extra=("--contact-json", str(cj))))
    assert err.code == "contact_phase_conflict"

    # --contact-json alone is a valid annotation source
    spec = run_spec(cli(npz, tmp_path / "o.json", extra=("--contact-json", str(cj))))
    assert spec["inputs"]["contact_phase_source"] == str(cj)


# ------------------------------------------------------------ parameterization #
def test_parameterization_extraction_and_robust_median(tmp_path):
    clip = make_clip()
    # one noisy video frame inside the initial stance window: median must shrug it off
    clip["body_pos_w"][5, 1, 1] += 0.015  # below the 0.02 support-slip budget
    npz = write_clip(tmp_path, clip)
    spec = run_spec(cli(npz, tmp_path / "o.json", beta="1.00"))
    p = spec["parameterization"]
    assert abs(p["initial_separation_m"]["lateral"] - 0.20) < 1e-9   # median, not mean
    assert abs(p["initial_separation_m"]["fore_aft"]) < 1e-9
    assert abs(p["terminal_separation_m"]["lateral"] - 0.20) < 1e-9
    assert abs(p["root_signed_lateral_displacement_m"] - 0.15) < 1e-9
    # contact-phase table survives into the spec
    left_iv = [(a["start_frame"], a["end_frame"]) for a in p["support_anchors"]["left"]]
    assert left_iv == [(0, W0_END), (A1 + 1, T - 1)]
    # footfalls: one landing per foot, at the annotated touchdown frames
    assert p["footfalls"]["left"][0]["touchdown_frame"] == A1 + 1
    assert abs(p["footfalls"]["left"][0]["lateral_m"] - 0.25) < 1e-9
    assert p["footfalls"]["right"][0]["touchdown_frame"] == B1 + 1
    assert abs(p["footfalls"]["right"][0]["lateral_m"] - 0.05) < 1e-9


def test_beta_scales_displacement_not_geometry(tmp_path):
    npz = write_clip(tmp_path, make_clip())
    spec = run_spec(cli(npz, tmp_path / "o.json", beta="1.20"))
    s = spec["scaled"]
    assert abs(s["achieved_root_signed_lateral_displacement_m"] - 1.2 * 0.15) < 1e-9
    left = s["support_anchors_scaled"]["left"]
    # initial support anchor pinned (support foot fixed in its support phase)
    assert abs(left[0]["lateral_m"] - 0.10) < 1e-9
    # landing lateral offset scales: 0.10 + 1.2 * 0.15
    assert abs(left[1]["lateral_m"] - 0.28) < 1e-9
    # fore-aft and height NEVER scale
    src = spec["parameterization"]["support_anchors"]["left"]
    for a, b in zip(src, left):
        assert abs(a["fore_aft_m"] - b["fore_aft_m"]) < 1e-12
        assert abs(a["height_m"] - b["height_m"]) < 1e-12
    assert spec["forbidden_implementations"] == [
        "hip_roll_only_amplification", "uniform_whole_body_scaling", "z_axis_scaling"]


# ------------------------------------- frozen stance gate: M0 0/4 as negatives #
def _gate(init_lat: float, term_lat: float) -> dict:
    return sf.stance_gate({
        "initial_separation_m": {"fore_aft": 0.0, "lateral": init_lat},
        "terminal_separation_m": {"fore_aft": 0.0, "lateral": term_lat},
    })


def _m0_clip(narrow_delta: float) -> dict:
    """Terminal separation = 0.20 - narrow_delta via the second foot's landing."""
    return make_clip(r_step=0.15 + narrow_delta)


def test_stance_m0_left1_rejected(tmp_path):
    npz = write_clip(tmp_path, _m0_clip(0.095425))
    err = refuse(cli(npz, tmp_path / "o.json"))
    assert err.code == "stance_gate_failed"
    assert "lateral_band" in str(err) and "terminal_narrowing_hard_gate" in str(err)
    g = _gate(0.20, 0.20 - 0.095425)
    assert abs(g["checks"]["terminal_narrowing_hard_gate"]["value_m"] - 0.095425) < 1e-9


def test_stance_m0_left2_rejected_widening(tmp_path):
    npz = write_clip(tmp_path, _m0_clip(-0.200557))  # widens: band fails, narrowing passes
    err = refuse(cli(npz, tmp_path / "o.json"))
    assert err.code == "stance_gate_failed"
    g = _gate(0.20, 0.20 + 0.200557)
    assert not g["checks"]["lateral_band"]["passed"]
    assert g["checks"]["terminal_narrowing_hard_gate"]["passed"]


def test_stance_m0_right1_rejected(tmp_path):
    npz = write_clip(tmp_path, _m0_clip(0.076532))
    err = refuse(cli(npz, tmp_path / "o.json"))
    assert err.code == "stance_gate_failed"
    g = _gate(0.20, 0.20 - 0.076532)
    assert abs(g["checks"]["terminal_narrowing_hard_gate"]["value_m"] - 0.076532) < 1e-9


def test_stance_m0_right2_dies_on_narrowing_hard_gate_alone(tmp_path):
    # 0.0243 m sits INSIDE the 0.03 m component band — the independent 0.005 m
    # narrowing hard gate must still kill it (the exact way right2 died).
    npz = write_clip(tmp_path, _m0_clip(0.024300))
    err = refuse(cli(npz, tmp_path / "o.json"))
    assert err.code == "stance_gate_failed"
    assert "terminal_narrowing_hard_gate" in str(err)
    assert "lateral_band" not in str(err)
    g = _gate(0.20, 0.20 - 0.024300)
    assert g["checks"]["lateral_band"]["passed"]
    assert not g["checks"]["terminal_narrowing_hard_gate"]["passed"]


# ------------------------------------------------- crossing / min stance width #
def _cross_clip(mid_step: float) -> dict:
    """Right foot steps LEFT first by mid_step (transient narrowing), left foot
    follows by the same amount: terminal stance recovers, mid-motion may not."""
    return make_clip(l0=0.15, r0=-0.15, l_step=mid_step, r_step=mid_step,
                     d=0.24, first="right")


def test_crossing_detected_on_source(tmp_path):
    npz = write_clip(tmp_path, _cross_clip(0.28))  # mid separation 0.02 < 0.03
    err = refuse(cli(npz, tmp_path / "o.json", beta="1.00"))
    assert err.code == "foot_crossing"


def test_crossing_detected_on_scaled_schedule_beta_dependent(tmp_path):
    clip = _cross_clip(0.24)  # source mid separation 0.06: legal
    npz = write_clip(tmp_path, clip)
    ok = run_spec(cli(npz, tmp_path / "ok.json", beta="0.80"))
    assert ok["checks"]["foot_crossing_scaled"]["passed"]
    err = refuse(cli(npz, tmp_path / "bad.json", beta="1.35"))
    assert err.code == "foot_crossing_scaled"  # 0.30 - 1.35*0.24 = -0.024 crosses


# --------------------------------------------------------- other fail-closed -- #
def test_direction_mismatch_rejected(tmp_path):
    npz = write_clip(tmp_path, make_clip())  # moves left (d > 0)
    err = refuse(cli(npz, tmp_path / "o.json", direction="right"))
    assert err.code == "direction_mismatch"


def test_nonfinite_input_rejected(tmp_path):
    clip = make_clip()
    clip["body_pos_w"][3, 1, 0] = np.nan
    npz = write_clip(tmp_path, clip)
    err = refuse(cli(npz, tmp_path / "o.json"))
    assert err.code == "nonfinite_input"


def test_body_order_unresolved_fail_loud(tmp_path):
    clip = make_clip()
    npz = write_clip(tmp_path, clip, embed_body_names=False)
    err = refuse(cli(npz, tmp_path / "o.json"))
    assert err.code == "body_order_unresolved"
    # an explicit --body-order resolves the same clip
    spec = run_spec(cli(npz, tmp_path / "o.json",
                        extra=("--body-order", ",".join(BODY_NAMES))))
    assert spec["parameterization"]["root_signed_lateral_displacement_m"] > 0


def test_support_slip_annotation_contradiction(tmp_path):
    clip = make_clip()
    y = clip["body_pos_w"][:, 1, 1]
    y[A1 + 1:] = np.linspace(y[A1 + 1], y[A1 + 1] + 0.10, T - (A1 + 1))  # planted foot slides
    err = refuse(cli(write_clip(tmp_path, clip), tmp_path / "o.json"))
    assert err.code == "support_slip"


def test_root_displacement_too_small_rejected(tmp_path):
    npz = write_clip(tmp_path, make_clip(d=0.005))
    err = refuse(cli(npz, tmp_path / "o.json"))
    assert err.code == "root_displacement_too_small"


def test_strike_phase_invalid_rejected(tmp_path):
    npz = tmp_path / "missing.npz"
    for bad in ("1.5", "0.0", "nan"):
        args = ["--footwork-npz", str(npz), "--beta", "1.20", "--direction", "left",
                "--strike-asset-id", "hope_forehand_v4rg_cal", "--strike-phase", bad,
                "--output", str(tmp_path / "o.json")]
        err = refuse(args)
        assert err.code == "strike_metadata_invalid"


@pytest.mark.skipif(not _CATALOG.is_file(), reason="repo catalog not present")
def test_catalog_gate_rejected_footwork_asset_refused(tmp_path):
    npz = write_clip(tmp_path, make_clip())
    err = refuse(cli(npz, tmp_path / "o.json",
                     extra=("--catalog", str(_CATALOG),
                            "--footwork-asset-id", "lateral_step_left_1")))
    assert err.code == "footwork_source_gate_rejected"  # M0 stance 0/4 stays rejected
    # a stationary strike cannot be passed off as the footwork module either
    err = refuse(cli(npz, tmp_path / "o.json",
                     extra=("--catalog", str(_CATALOG),
                            "--footwork-asset-id", "hope_forehand_v4rg_cal")))
    assert err.code == "catalog_role_mismatch"


# --------------------------------------------------------------- manifest/CLI #
def test_manifest_buckets_sha_and_gate_chain(tmp_path):
    npz = write_clip(tmp_path, make_clip())
    out = tmp_path / "spec.json"
    rc = sf.main(cli(npz, out, beta="1.20"))
    assert rc == 0
    spec = json.loads(out.read_text())
    # frozen buckets written verbatim into every manifest
    assert spec["beta_buckets"] == {"train": [0.80, 1.00, 1.20],
                                    "interpolation": [0.90, 1.10],
                                    "ood": [0.65, 1.35]}
    assert spec["beta_bucket"] == "train"
    # one-layer content addressing: input sha + self-consistent spec sha
    assert spec["inputs"]["footwork_npz_sha256"] == hashlib.sha256(npz.read_bytes()).hexdigest()
    payload = {k: v for k, v in spec.items() if k != "spec_sha256"}
    assert sf.canonical_sha256(payload) == spec["spec_sha256"]
    # a spec is never a trainable asset
    assert spec["training_authorized"] is False
    assert len(spec["requires_full_gate_chain"]) >= 5
    assert spec["recovery_ready_budget"]["terminal_narrowing_hard_cap_m"] == 0.005


def test_main_exit_code_2_and_no_output_on_refusal(tmp_path):
    out = tmp_path / "spec.json"
    rc = sf.main(cli(tmp_path / "missing.npz", out, beta="0.0"))
    assert rc == 2
    assert not out.exists()  # fail-closed: nothing is written on refusal
