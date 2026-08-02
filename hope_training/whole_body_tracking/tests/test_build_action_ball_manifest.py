"""CPU tests for the action-ball manifest builder's formal holdout contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


WHOLE_BODY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WHOLE_BODY_ROOT.parents[1]
BUILDER_PATH = (
    WHOLE_BODY_ROOT / "scripts" / "build_action_ball_manifest.py"
)
BUILDER_SPEC = importlib.util.spec_from_file_location(
    "build_action_ball_manifest_under_test", BUILDER_PATH
)
assert BUILDER_SPEC is not None and BUILDER_SPEC.loader is not None
B = importlib.util.module_from_spec(BUILDER_SPEC)
sys.modules[BUILDER_SPEC.name] = B
BUILDER_SPEC.loader.exec_module(B)

MANIFEST_PATH = (
    WHOLE_BODY_ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "action_ball_manifest.py"
)
MANIFEST_SPEC = importlib.util.spec_from_file_location(
    "action_ball_manifest_for_builder_test", MANIFEST_PATH
)
assert MANIFEST_SPEC is not None and MANIFEST_SPEC.loader is not None
M = importlib.util.module_from_spec(MANIFEST_SPEC)
sys.modules[MANIFEST_SPEC.name] = M
MANIFEST_SPEC.loader.exec_module(M)


def _write_synthetic_batch(
    tmp_path: Path,
    *,
    canonical_ready_root_z_m: float = 0.0,
    uid: str = "synthetic_unit_fh",
    family: str = "FH",
    measured_mount_sign=None,
    measured_schema_version: int = 4,
    omit_measured_key=None,
) -> Path:
    """Create the smallest 50 Hz clip accepted by the production builder."""

    frame_count = 11
    body_count = 32
    body_pos = np.zeros((frame_count, body_count, 3), dtype=np.float32)
    body_pos[:, 0, 2] = canonical_ready_root_z_m
    body_pos[:, 31, 1] = np.arange(frame_count, dtype=np.float32) * 0.02
    body_quat = np.zeros((frame_count, body_count, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    body_names = np.asarray(
        ["pelvis_link"]
        + [f"body_{index}" for index in range(1, body_count - 1)]
        + ["right_wrist_yaw_Link"]
    )

    arrays = {
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_names": body_names,
        "fps": np.asarray(50, dtype=np.int64),
    }
    if measured_mount_sign is not None:
        measured_position = np.zeros((frame_count, 3), dtype=np.float32)
        measured_position[:, 0] = (
            np.arange(frame_count, dtype=np.float32) * 0.04
        )
        arrays.update(
            {
                "measured_racket_site_pos_w": measured_position,
                "measured_racket_normal_w": np.tile(
                    np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                    (frame_count, 1),
                ),
                "measured_racket_long_axis_w": np.tile(
                    np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                    (frame_count, 1),
                ),
                "measured_racket_schema_version": np.asarray(
                    [measured_schema_version], dtype=np.int64
                ),
                "measured_racket_position_semantics": np.asarray(
                    "physical_blade_center"
                ),
                "measured_racket_normal_semantics": np.asarray(
                    "signed_physical_hitting_face"
                ),
                "measured_racket_long_axis_semantics": np.asarray(
                    "measured_paddle_butt_to_blade"
                ),
                "measured_racket_robot_mount_normal_sign": np.asarray(
                    [measured_mount_sign], dtype=np.int8
                ),
                "measured_racket_robot_butt_to_blade_axis_local": np.asarray(
                    B.ROBOT_BUTT_TO_BLADE_AXIS_LOCAL, dtype=np.float64
                ),
                "measured_racket_robot_rigid_visual_mesh_sha256": np.asarray(
                    B.ROBOT_RIGID_VISUAL_MESH_SHA256
                ),
                "measured_racket_source_sha256": np.asarray("a" * 64),
                "measured_racket_retarget_admitted": np.asarray(
                    [1], dtype=np.int64
                ),
                "measured_racket_retarget_receipt_sha256": np.asarray(
                    "b" * 64
                ),
                "measured_racket_joint_order_contract_id": np.asarray(
                    "a3-gmr-dof-pos-to-runtime-articulation-v1"
                ),
                "measured_racket_joint_order_contract_sha256": np.asarray(
                    "c" * 64
                ),
            }
        )
        if omit_measured_key is not None:
            arrays.pop(omit_measured_key)

    clip_path = tmp_path / f"{uid}.npz"
    np.savez(clip_path, **arrays)
    clip_sha256 = hashlib.sha256(clip_path.read_bytes()).hexdigest()

    batch = {
        "schema": "synthetic_chingmu_manifest_v1",
        "units": [
            {
                "uid": uid,
                "family": family,
                "npz": clip_path.name,
                "npz_sha256": clip_sha256,
                "T": frame_count,
                "fps": 50,
                "hit_frame_50": 5,
                "strike_phase": 0.5,
                "yaw_before_deg": 0.0,
                "station_xy_hope_m": [0.0, 0.0],
                "ball_pos_hit_hope_m": [0.5, 0.0, 0.30],
                "v_in_fit_hope_ms": [-3.0, 0.0, 0.0],
                "v_out_fit_hope_ms": [3.0, 0.0, 1.0],
                "w_out_nominal_radps": [0.0, 0.0, 0.0],
            }
        ],
    }
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(json.dumps(batch), encoding="utf-8")
    return batch_path


def _build_argv(
    batch_path: Path, output_path: Path, manifest_id: str
) -> list:
    return [
        "build",
        "--batch-manifest",
        str(batch_path),
        "--batch-root",
        str(batch_path.parent),
        "--repo-root",
        str(REPO_ROOT),
        "--out",
        str(output_path),
        "--manifest-id",
        manifest_id,
        "--expect-units",
        "1",
    ]


def _write_measured_bank_receipt(
    batch_path: Path,
    *,
    measured_file: Path | None = None,
    measured_mount_sign: int = -1,
) -> tuple[Path, str]:
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    unit = batch["units"][0]
    if measured_file is None:
        measured_file = batch_path.parent / Path(unit["npz"]).name
    measured_sha256 = hashlib.sha256(measured_file.read_bytes()).hexdigest()
    batch_sha256 = hashlib.sha256(batch_path.read_bytes()).hexdigest()
    frames = int(unit["T"])
    receipt = {
        "schema_version": 1,
        "kind": "chingmu73_measured_racket_schema_v4_repo_import",
        "source_manifest": {
            "file": batch_path.name,
            "sha256": batch_sha256,
        },
        "authorities": {
            "source_manifest": {"sha256": batch_sha256},
        },
        "publication": {
            "all_npz_sha256_verified_before_publish": True,
            "historical_bank_overwritten": False,
            "versioned_sibling": True,
        },
        "denominators": {
            "catalog_actions": 1,
            "materialized_npz": 1,
            "schema_v4_npz": 1,
            "solver_admitted": 1,
            "solver_all_gates_true": 1,
            "fk_audit_admitted": 1,
            "fk_audit_all_gates_true": 1,
            "fk_audit_finite": 1,
            "total_materialized_frames": frames,
        },
        "actions": [
            {
                "clip_id": 0,
                "uid": unit["uid"],
                "file": measured_file.name,
                "sha256": measured_sha256,
                "frames": frames,
                "hit_frame_50": unit["hit_frame_50"],
                "robot_mount_normal_sign": measured_mount_sign,
            }
        ],
    }
    receipt_path = batch_path.parent / "BANK_IMPORT_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return receipt_path, hashlib.sha256(receipt_path.read_bytes()).hexdigest()


def _measured_receipt_argv(receipt_path: Path, receipt_sha256: str) -> list:
    return [
        "--racket-authority",
        "measured_channel",
        "--measured-bank-receipt",
        str(receipt_path),
        "--expected-measured-bank-receipt-sha256",
        receipt_sha256,
    ]


def test_contact_z_is_relative_to_nonzero_canonical_ready_root(tmp_path):
    ready_root_z = 0.62
    batch_path = _write_synthetic_batch(
        tmp_path, canonical_ready_root_z_m=ready_root_z
    )
    output_path = tmp_path / "action_ball_nonzero_root.json"
    assert B.main(
        [
            "build",
            "--batch-manifest",
            str(batch_path),
            "--batch-root",
            str(tmp_path),
            "--repo-root",
            str(REPO_ROOT),
            "--out",
            str(output_path),
            "--manifest-id",
            "action_ball_nonzero_root_v1",
            "--expect-units",
            "1",
        ]
    ) == 0

    document = json.loads(output_path.read_text(encoding="utf-8"))
    contact_z = document["actions"][0]["ball_profile"][
        "contact_offset_center_b_yaw_m"
    ][2]
    absolute_ball_contact_z = 0.30 + 0.76
    assert contact_z == pytest.approx(absolute_ball_contact_z - ready_root_z)
    assert ready_root_z + contact_z == pytest.approx(absolute_ball_contact_z)
    # Regression guard: the old producer emitted absolute contact Z as the
    # base-relative offset and runtime added ready_root_z a second time.
    assert contact_z != pytest.approx(absolute_ball_contact_z)


def test_canonical_ready_root_z_rejects_nonfinite_value(tmp_path):
    clip_path = tmp_path / "nonfinite_root.npz"
    body_pos = np.zeros((2, 1, 3), dtype=np.float32)
    body_pos[0, 0, 2] = np.nan
    np.savez(
        clip_path,
        body_pos_w=body_pos,
        body_names=np.asarray(["pelvis_link"]),
    )

    with pytest.raises(
        SystemExit, match="canonical-ready pelvis/root Z must be finite"
    ):
        B._canonical_ready_root_z(clip_path)


def test_builder_rejects_512_and_default_768_roundtrips(tmp_path):
    batch_path = _write_synthetic_batch(tmp_path)
    output_path = tmp_path / "action_ball_synthetic.json"
    common_argv = [
        "build",
        "--batch-manifest",
        str(batch_path),
        "--batch-root",
        str(tmp_path),
        "--repo-root",
        str(REPO_ROOT),
        "--out",
        str(output_path),
        "--manifest-id",
        "action_ball_synthetic_formal_v1",
        "--expect-units",
        "1",
    ]

    with pytest.raises(
        SystemExit,
        match=r"512 below formal per-action minimum 768.*cannot populate",
    ):
        B.main(common_argv + ["--holdout-samples", "512"])
    assert not output_path.exists()

    assert B.main(common_argv) == 0
    loaded = M.load_action_ball_manifest(output_path)
    assert loaded.manifest.holdout.samples_per_action == 768
    assert loaded.manifest.action_order == ("synthetic_unit_fh",)
    assert loaded.file_sha256 == hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()

    sha_sidecar = output_path.with_name(output_path.name + ".sha256")
    assert sha_sidecar.read_text(encoding="utf-8") == (
        f"{loaded.file_sha256}  {output_path.name}\n"
    )
    report_path = output_path.with_name(
        output_path.name.replace(".json", "") + ".buildreport.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["n_actions"] == 1
    assert report["file_sha256"] == loaded.file_sha256


def test_measured_channel_replaces_legacy_fk_sign_and_speed(tmp_path):
    """Synthetic Take_061 equivalent: legacy +Y FK says +1, admitted measured face says -1."""

    batch_path = _write_synthetic_batch(
        tmp_path,
        uid="Take_061_unit10_BH_equivalent",
        family="BH",
        measured_mount_sign=-1,
    )
    legacy_output = tmp_path / "legacy.json"
    measured_output = tmp_path / "measured.json"
    receipt_path, receipt_sha256 = _write_measured_bank_receipt(
        batch_path, measured_mount_sign=-1
    )

    assert B.main(
        _build_argv(batch_path, legacy_output, "synthetic_legacy_v1")
    ) == 0
    assert B.main(
        _build_argv(batch_path, measured_output, "synthetic_measured_v1")
        + _measured_receipt_argv(receipt_path, receipt_sha256)
    ) == 0

    legacy = json.loads(legacy_output.read_text(encoding="utf-8"))
    measured = json.loads(measured_output.read_text(encoding="utf-8"))
    legacy_action = legacy["actions"][0]
    measured_action = measured["actions"][0]
    assert legacy_action["mount_normal_sign"] == 1
    assert measured_action["mount_normal_sign"] == -1
    assert legacy_action["reference_racket_site_speed_mps"] == pytest.approx(1.0)
    assert measured_action["reference_racket_site_speed_mps"] == pytest.approx(2.0)

    legacy_report = json.loads(
        (tmp_path / "legacy.buildreport.json").read_text(encoding="utf-8")
    )
    measured_report = json.loads(
        (tmp_path / "measured.buildreport.json").read_text(encoding="utf-8")
    )
    assert "racket_authority" not in legacy_report
    assert measured_report["racket_authority"] == "measured_channel"
    assert measured_report["measured_bank_receipt_sha256"] == receipt_sha256
    measured_row = measured_report["per_action"][0]
    assert measured_row["measured_racket_schema_version"] == 4
    assert measured_row["measured_racket_source_sha256"] == "a" * 64
    assert measured_row["measured_racket_retarget_receipt_sha256"] == "b" * 64
    assert measured_row["measured_signed_face_velocity_cos"] == pytest.approx(1.0)


def test_measured_channel_rejects_missing_channel_without_fk_fallback(tmp_path):
    batch_path = _write_synthetic_batch(
        tmp_path,
        measured_mount_sign=-1,
        omit_measured_key="measured_racket_normal_w",
    )
    output_path = tmp_path / "partial.json"
    receipt_path, receipt_sha256 = _write_measured_bank_receipt(batch_path)

    with pytest.raises(
        SystemExit,
        match=r"requires the complete schema-v4.*measured_racket_normal_w",
    ):
        B.main(
            _build_argv(batch_path, output_path, "synthetic_partial_v1")
            + _measured_receipt_argv(receipt_path, receipt_sha256)
        )
    assert not output_path.exists()


def test_measured_channel_rejects_non_v4_schema(tmp_path):
    batch_path = _write_synthetic_batch(
        tmp_path,
        measured_mount_sign=-1,
        measured_schema_version=2,
    )
    output_path = tmp_path / "schema2.json"
    receipt_path, receipt_sha256 = _write_measured_bank_receipt(batch_path)

    with pytest.raises(
        SystemExit, match="measured_racket_schema_version must be exactly 4"
    ):
        B.main(
            _build_argv(batch_path, output_path, "synthetic_schema2_v1")
            + _measured_receipt_argv(receipt_path, receipt_sha256)
        )
    assert not output_path.exists()


def test_measured_channel_requires_content_bound_bank_receipt(tmp_path):
    batch_path = _write_synthetic_batch(tmp_path, measured_mount_sign=-1)
    output_path = tmp_path / "no-receipt.json"

    with pytest.raises(
        SystemExit,
        match=r"requires --measured-bank-receipt.*expected-measured-bank",
    ):
        B.main(
            _build_argv(batch_path, output_path, "synthetic_no_receipt_v1")
            + ["--racket-authority", "measured_channel"]
        )
    assert not output_path.exists()


def test_measured_channel_forbids_skip_npz_hash(tmp_path):
    batch_path = _write_synthetic_batch(tmp_path, measured_mount_sign=-1)
    output_path = tmp_path / "skip-hash.json"
    receipt_path, receipt_sha256 = _write_measured_bank_receipt(batch_path)

    with pytest.raises(SystemExit, match="forbids --skip-npz-hash"):
        B.main(
            _build_argv(batch_path, output_path, "synthetic_skip_hash_v1")
            + _measured_receipt_argv(receipt_path, receipt_sha256)
            + ["--skip-npz-hash"]
        )
    assert not output_path.exists()


def test_measured_receipt_replaces_legacy_source_path_and_sha(tmp_path):
    batch_path = _write_synthetic_batch(tmp_path, measured_mount_sign=-1)
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    measured_file = tmp_path / Path(batch["units"][0]["npz"]).name
    measured_sha256 = hashlib.sha256(measured_file.read_bytes()).hexdigest()
    batch["units"][0]["npz"] = "clips/legacy_source_only.npz"
    batch["units"][0]["npz_sha256"] = "d" * 64
    batch_path.write_text(json.dumps(batch), encoding="utf-8")
    receipt_path, receipt_sha256 = _write_measured_bank_receipt(
        batch_path,
        measured_file=measured_file,
        measured_mount_sign=-1,
    )
    output_path = tmp_path / "receipt-bound.json"

    argv = _build_argv(batch_path, output_path, "synthetic_receipt_bound_v1")
    argv += _measured_receipt_argv(receipt_path, receipt_sha256)
    assert "--skip-npz-hash" not in argv
    assert B.main(argv) == 0

    action = json.loads(output_path.read_text(encoding="utf-8"))["actions"][0]
    assert action["motion_path"] == measured_file.name
    assert action["motion_sha256"] == measured_sha256
    assert action["motion_sha256"] != "d" * 64


def test_ttc_lattice_quantizes_inward_and_keeps_both_sides_enabled():
    lattice = B._ttc_lattice(
        continuous_min_s=0.611,
        continuous_max_s=1.389,
        requested_center_s=1.003,
        requested_initial_width_s=0.005,
        policy_dt_s=0.02,
        label="fresh",
    )
    assert lattice == {
        "policy_dt_s": 0.02,
        "lower_tick": 31,
        "center_tick": 50,
        "upper_tick": 69,
        "min_s": 0.62,
        "center_s": 1.0,
        "max_s": 1.3800000000000001,
        "lower_initial_s": 0.02,
        "lower_max_s": 0.38,
        "upper_initial_s": 0.02,
        "upper_max_s": 0.38,
    }
    assert lattice["min_s"] >= 0.611
    assert lattice["max_s"] <= 1.389


def test_ttc_lattice_keeps_one_ulp_feasibility_at_decimal_tick():
    # The manifest loader recomputes this exact expression and intentionally has no tolerance.
    continuous_min = 0.96 / 0.6 + 0.1
    assert continuous_min == 1.7000000000000002
    lattice = B._ttc_lattice(
        continuous_min_s=continuous_min,
        continuous_max_s=2.1,
        requested_center_s=None,
        requested_initial_width_s=0.05,
        policy_dt_s=0.02,
        label="one_ulp_lower_bound",
    )
    assert lattice["lower_tick"] == 85
    assert lattice["min_s"] == continuous_min
    assert lattice["min_s"] >= continuous_min


def test_ttc_lattice_rejects_window_without_three_ticks():
    with pytest.raises(SystemExit, match="fewer than three policy ticks"):
        B._ttc_lattice(
            continuous_min_s=0.991,
            continuous_max_s=1.029,
            requested_center_s=None,
            requested_initial_width_s=0.05,
            policy_dt_s=0.02,
            label="too_narrow",
        )


def test_ttc_lattice_rejects_nonpositive_initial_width():
    with pytest.raises(SystemExit, match="initial width must be positive"):
        B._ttc_lattice(
            continuous_min_s=0.60,
            continuous_max_s=1.40,
            requested_center_s=1.0,
            requested_initial_width_s=0.0,
            policy_dt_s=0.02,
            label="zero_width",
        )


def _fresh_n5_args(**overrides):
    values = {
        "skip_npz_hash": False,
        "prototype_scope": "upper",
        "prototype_path": "configs/stroke_prototypes_fresh_n5_upper.json",
        "motion_path_prefix": (
            "vendor_assets/canonical_motion_v3append_v12/upper"
        ),
        "expected_geometry_source_sha256": "a" * 64,
        "solver_profile_sha256": "b" * 64,
        "physics_profile_sha256": "c" * 64,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _fresh_n5_units():
    families = ("BH", "FH", "BH", "BH", "FH")
    return [
        {"uid": action_id, "family": family}
        for action_id, family in zip(B.FRESH_N5_ACTION_ORDER, families)
    ]


def test_fresh_n5_build_request_accepts_only_exact_upper_order_and_pins():
    B._validate_fresh_n5_build_request(
        units=_fresh_n5_units(),
        args=_fresh_n5_args(),
        exact_geometry_sha256="a" * 64,
    )


@pytest.mark.parametrize(
    ("units", "args", "match"),
    [
        (
            list(reversed(_fresh_n5_units())),
            _fresh_n5_args(),
            "action order must be exactly",
        ),
        (
            [
                *_fresh_n5_units()[:4],
                {"uid": "fh_loop", "family": "FH"},
            ],
            _fresh_n5_args(),
            "action order must be exactly",
        ),
        (
            _fresh_n5_units(),
            _fresh_n5_args(skip_npz_hash=True),
            "forbids --skip-npz-hash",
        ),
        (
            _fresh_n5_units(),
            _fresh_n5_args(prototype_scope="full"),
            "scope must be exactly upper",
        ),
        (
            _fresh_n5_units(),
            _fresh_n5_args(motion_path_prefix="../escaped"),
            "must stay inside the repo root",
        ),
        (
            _fresh_n5_units(),
            _fresh_n5_args(prototype_path="../escaped.json"),
            "--prototype-path must stay inside the repo root",
        ),
        (
            _fresh_n5_units(),
            _fresh_n5_args(expected_geometry_source_sha256="d" * 64),
            "expected geometry SHA must equal",
        ),
        (
            _fresh_n5_units(),
            _fresh_n5_args(
                solver_profile_sha256=hashlib.sha256(b"solver").hexdigest()
            ),
            "forbids placeholder",
        ),
    ],
)
def test_fresh_n5_build_request_rejects_drift(units, args, match):
    with pytest.raises(SystemExit, match=match):
        B._validate_fresh_n5_build_request(
            units=units,
            args=args,
            exact_geometry_sha256="a" * 64,
        )
