"""CPU-only contract tests for the measured VendorV2 N1 bundle materializer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/materialize_measured_action_ball_n1_bundle.py"
SPEC = importlib.util.spec_from_file_location("materialize_measured_n1", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    return path


def test_final_action_binding_is_exact_and_revoked_candidate_is_absent():
    assert module.ACTION_ID == "take_061_unit04_bh"
    assert module.MEASURED_UID == "Take_061_unit04_BH"
    assert module.ACTION_FACTS == {
        "action_uid": 5527597793770800,
        "motion_path": (
            "assets/motions/chingmu73_measured_v4_20260803/"
            "hope_Take_061_unit04_BH.npz"
        ),
        "motion_sha256": (
            "aab1953b9a857d0a7663a92d85fe4de5bd1d991d22249aa3d4d22ce7ef9fdd8e"
        ),
        "reference_t_hit_s": 0.96,
        "reference_t_cycle_s": 1.12,
        "reference_racket_site_speed_mps": 1.8901338577270508,
        "strike_phase": 0.8571,
        "family": "backhand",
    }


def test_prepare_does_not_bypass_full_solver_admission_preflight():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "skip_full_solver_preflight_for_immutable_tape=True" not in source
    assert "full_solver_preflight_support_source" in source


def test_five_recipes_keep_explicit_identity_when_masks_match():
    assert module.RECIPES == {
        "current_lm": (True, True, True),
        "analytic_full": (True, True, True),
        "analytic_no_velocity": (True, False, True),
        "teacher_pos_face_no_velocity": (True, False, True),
        "outcome_dense_only": (False, False, False),
    }
    assert module.RECIPES["current_lm"] == module.RECIPES["analytic_full"]
    assert (
        module.RECIPES["analytic_no_velocity"]
        == module.RECIPES["teacher_pos_face_no_velocity"]
    )


def test_fixed_question_projection_zeroes_widths_but_keeps_legal_support():
    profile = {
        "contact_offset_center_b_yaw_m": [1.0, 2.0, 3.0],
        "contact_offset_min_b_yaw_m": [0.0, 0.0, 0.0],
        "contact_offset_max_b_yaw_m": [4.0, 4.0, 4.0],
        "contact_offset_std_lower_initial_m": [0.1, 0.2, 0.3],
        "contact_offset_std_lower_max_m": [0.4, 0.5, 0.6],
        "contact_offset_std_upper_initial_m": [0.1, 0.2, 0.3],
        "contact_offset_std_upper_max_m": [0.4, 0.5, 0.6],
        "time_to_contact_center_s": 1.0,
        "time_to_contact_min_s": 0.5,
        "time_to_contact_max_s": 1.5,
        "time_to_contact_std_lower_initial_s": 0.1,
        "time_to_contact_std_lower_max_s": 0.2,
        "time_to_contact_std_upper_initial_s": 0.1,
        "time_to_contact_std_upper_max_s": 0.2,
        "incoming_speed_center_mps": 3.0,
        "incoming_speed_min_mps": 2.0,
        "incoming_speed_max_mps": 4.0,
        "spin_magnitude_center_radps": 0.0,
        "spin_magnitude_min_radps": 0.0,
        "spin_magnitude_max_radps": 10.0,
        "base_spawn_center_w_xy_m": [0.1, 0.2],
        "base_spawn_min_w_xy_m": [-1.0, -1.0],
        "base_spawn_max_w_xy_m": [1.0, 1.0],
        "base_spawn_std_lower_initial_m": [0.1, 0.1],
        "base_spawn_std_lower_max_m": [0.2, 0.2],
        "base_spawn_std_upper_initial_m": [0.1, 0.1],
        "base_spawn_std_upper_max_m": [0.2, 0.2],
        "base_travel_center_b_yaw_xy_m": [0.0, 0.0],
        "base_travel_min_b_yaw_xy_m": [-1.0, -1.0],
        "base_travel_max_b_yaw_xy_m": [1.0, 1.0],
        "base_travel_std_lower_initial_m": [0.1, 0.1],
        "base_travel_std_lower_max_m": [0.2, 0.2],
        "base_travel_std_upper_initial_m": [0.1, 0.1],
        "base_travel_std_upper_max_m": [0.2, 0.2],
        "incoming_direction_tangent_u_neg_initial_deg": 3.0,
        "incoming_direction_tangent_u_neg_max_deg": 15.0,
        "incoming_direction_tangent_u_pos_initial_deg": 3.0,
        "incoming_direction_tangent_u_pos_max_deg": 15.0,
    }
    fixed = module._freeze_ball_profile(profile)
    assert fixed["contact_offset_min_b_yaw_m"] == [0.0, 0.0, 0.0]
    assert fixed["contact_offset_max_b_yaw_m"] == [4.0, 4.0, 4.0]
    assert fixed["time_to_contact_min_s"] == 0.5
    assert fixed["time_to_contact_max_s"] == 1.5
    assert fixed["incoming_speed_min_mps"] == 2.0
    assert fixed["incoming_speed_max_mps"] == 4.0
    assert fixed["base_spawn_min_w_xy_m"] == [-1.0, -1.0]
    assert fixed["base_spawn_max_w_xy_m"] == [1.0, 1.0]
    assert all(
        value == 0.0 or value == [0.0, 0.0, 0.0] or value == [0.0, 0.0]
        for key, value in fixed.items()
        if "std_lower_" in key or "std_upper_" in key
    )
    assert fixed["incoming_direction_tangent_u_neg_max_deg"] == 0.0


def test_tape_receipt_binds_recipe_mask_artifact_and_producers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    artifact = tmp_path / "tape.npz"
    artifact.write_bytes(b"immutable tape")
    producers = {name: str(index) * 64 for index, name in enumerate(
        ("incoming_ball", "teacher_contact", "desired_contact", "landing_spin_task"), start=1
    )}
    receipt = _write(
        tmp_path / "receipt.json",
        {
            "schema_version": 1,
            "kind": module.TAPE_RECEIPT_KIND,
            "action_id": module.ACTION_ID,
            "target_recipe": "analytic_no_velocity",
            "target_validity": {"order": list(module.TARGET_ORDER), "mask": [True, False, True]},
            "artifact": {"path": "tape.npz", "sha256": _sha(artifact)},
            "row_count": 1,
            "per_column_producer_sha256": producers,
            "physical_ball_semantics": module.PHYSICAL_BALL_SEMANTICS,
            "reset_inverse_solve": False,
            "diagnostic_unauthorized": True,
        },
    )
    class FakeTape:
        canonical_sha256 = "5" * 64
        question_sha256 = "6" * 64
        source_receipt = SimpleNamespace(
            action_uid=module.ACTION_FACTS["action_uid"],
            action_slot=0,
            profile_sha256="a" * 64,
            motion_sha256=module.ACTION_FACTS["motion_sha256"],
            manifest_sha256="b" * 64,
            sampler_sha256="c" * 64,
            physics_sha256="d" * 64,
            solver_sha256="e" * 64,
            mobility_mode="no_move",
        )

        def target_lineage(self, recipe):
            return {
                "base_question_sha256": self.question_sha256,
                "target_recipe": recipe,
                "target_producer_sha256": (
                    producers["desired_contact"] if recipe == "analytic_no_velocity" else "9" * 64
                ),
                "target_column_sha256": "8" * 64,
                "target_validity_mask": list(module.RECIPES[recipe]),
                "tape_canonical_sha256": self.canonical_sha256,
            }

    fake_module = SimpleNamespace(
        TARGET_RECIPES=tuple(module.RECIPES),
        TARGET_VALIDITY_BY_RECIPE=module.RECIPES,
        load_immutable_n1_tape=lambda *_args, **_kwargs: FakeTape(),
    )
    monkeypatch.setattr(module, "_load_module", lambda *_args, **_kwargs: fake_module)
    pin, summary = module._validate_tape_receipt(
        tmp_path,
        receipt,
        _sha(receipt),
        action_id=module.ACTION_ID,
        action_uid=module.ACTION_FACTS["action_uid"],
        motion_sha=module.ACTION_FACTS["motion_sha256"],
        recipe="analytic_no_velocity",
    )
    assert pin["sha256"] == _sha(receipt)
    assert summary["artifact"]["sha256"] == _sha(artifact)
    assert summary["target_validity"]["mask"] == [True, False, True]
    assert summary["row_count"] == 1
    assert summary["selected_target_lineage"]["target_recipe"] == "analytic_no_velocity"


def test_mechanical_unknown_requires_explicit_diagnostic_acceptance(tmp_path: Path):
    motion_sha = "a" * 64
    report = _write(
        tmp_path / "mechanical.json",
        {
            "schema_version": 1,
            "kind": "measured_racket_mechanical_admission_audit_v1",
            "diagnostic_unauthorized": True,
            "actions": [
                {
                    "uid": module.MEASURED_UID,
                    "sha256": motion_sha,
                    "kinematic_limit_verdict": "PASS",
                    "mechanical_verdict": "UNKNOWN",
                    "mechanical_admitted": False,
                }
            ],
        },
    )
    with pytest.raises(module.BundleError, match="allow-mechanical-unknown"):
        module._mechanical_selection(
            tmp_path,
            report,
            _sha(report),
            motion_sha=motion_sha,
            action_uid=module.MEASURED_UID,
            allow_unknown=False,
        )
    _pin, selected = module._mechanical_selection(
        tmp_path,
        report,
        _sha(report),
        motion_sha=motion_sha,
        action_uid=module.MEASURED_UID,
        allow_unknown=True,
    )
    assert selected["mechanical_verdict"] == "UNKNOWN"
    assert selected["unknown_explicitly_accepted_for_sim_diagnostic"] is True


def test_alignment_receipt_requires_all_11_gates_and_diagonal_long_axis(tmp_path: Path):
    gates = {name: True for name in module.RACKET_ALIGNMENT_GATES}
    report = _write(
        tmp_path / "alignment.json",
        {
            "schema_version": 3,
            "kind": "materialized_measured_racket_fk_audit_v3",
            "uid": module.MEASURED_UID,
            "motion_sha256": module.ACTION_FACTS["motion_sha256"],
            "frames": 57,
            "finite": True,
            "admitted": True,
            "robot_butt_to_blade_axis_local": [math.sqrt(0.5), 0.0, math.sqrt(0.5)],
            "robot_rigid_visual_mesh_sha256": module.RACKET_MESH_SHA256,
            "gates": gates,
            "hit": {"frame": 48, "position_error_m": 0.001},
            "position_error_m": {"p95": 0.01},
            "face_error_deg": {"p95": 1.0},
            "long_axis_error_deg": {"p95": 1.0},
            "so3_error_deg": {"p95": 1.0},
            "authorization": {
                "diagnostic_unauthorized": True,
                "training": False,
                "promotion": False,
                "deployment": False,
            },
        },
    )
    _pin, summary = module._validate_racket_alignment(
        tmp_path,
        report,
        _sha(report),
        motion_sha=module.ACTION_FACTS["motion_sha256"],
        action_uid=module.MEASURED_UID,
        frame_count=57,
        strike_frame=48,
    )
    assert summary["all_11_gates_pass"] is True

    failed = json.loads(report.read_text())
    failed["gates"]["hit_velocity_direction_le_15_deg"] = False
    failed_path = _write(tmp_path / "alignment_failed.json", failed)
    with pytest.raises(module.BundleError, match="failed gate"):
        module._validate_racket_alignment(
            tmp_path,
            failed_path,
            _sha(failed_path),
            motion_sha=module.ACTION_FACTS["motion_sha256"],
            action_uid=module.MEASURED_UID,
            frame_count=57,
            strike_frame=48,
        )

    missing = json.loads(report.read_text())
    del missing["gates"]["hit_velocity_direction_observable"]
    missing_path = _write(tmp_path / "alignment_missing_gate.json", missing)
    with pytest.raises(module.BundleError, match="failed gate"):
        module._validate_racket_alignment(
            tmp_path,
            missing_path,
            _sha(missing_path),
            motion_sha=module.ACTION_FACTS["motion_sha256"],
            action_uid=module.MEASURED_UID,
            frame_count=57,
            strike_frame=48,
        )


def test_local_v4_bank_and_build_report_bind_the_frozen_action():
    root = Path(__file__).resolve().parents[3]
    bank = root / "assets/motions/chingmu73_measured_v4_20260803/BANK_IMPORT_RECEIPT.json"
    report = root / "configs/action_ball_chingmu73_measured_v4_f10_20260803.buildreport.json"
    source = root / "configs/action_ball_chingmu73_measured_v4_f10_20260803.json"
    if not (bank.exists() and report.exists() and source.exists()):
        pytest.skip("local measured-racket authority assets are unavailable")
    bank_pin, report_pin, evidence = module._validate_measured_provenance(
        root,
        bank_receipt_path=bank,
        bank_receipt_sha=_sha(bank),
        build_report_path=report,
        build_report_sha=_sha(report),
        source_manifest_sha=_sha(source),
        action_id=module.ACTION_ID,
        measured_uid=module.MEASURED_UID,
        motion_sha=module.ACTION_FACTS["motion_sha256"],
        frame_count=57,
        strike_frame=48,
    )
    assert bank_pin["sha256"] == _sha(bank)
    assert report_pin["sha256"] == _sha(report)
    assert evidence["bank_action_row"]["hit_frame_50"] == 48
    assert evidence["build_report_action"]["racket_authority"] == "measured_channel"


# --- the offline live re-pinner, and whether it speaks solver profile v3 ------
#
# 人话:`_materialize_live_profile_pins` 是"把 pins 模板重新按活体源码封章"的那一步。
# v2 时代它往 solver payload 里塞七份源文件的整文件 SHA;v3 的 payload 里根本没有
# 这个键(封的是逐符号语义面 + counter-rally 两份未裁定源码的整文件 SHA)。这支
# 函数此前一条测试都没有,于是"多按一枚指纹"没人发现 —— 离线铸出的 pin 与 runtime
# 现算的那枚永远差一个键,manifest 一进 boot 就崩。下面五条把这件事钉住。

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MDP_REL = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp"
)
_LIVE_V3_PINS_REL = (
    "configs/action_ball_n1_measured_20260806/fresh_core_seed0_20260806_r2/"
    "action_ball_profile_pins.live.v2.5564d5b3c09d.json"
)
# The union of what the re-pinner reads: the seven solver sources it digests, the
# sixth adjudicated source that only the semantic surface pins, and the surface
# module itself.
_REPIN_INPUTS = (
    "hope_commands.py",
    "continuous_questions.py",
    "racket_contact_geometry.py",
    "stroke_adapt_torch.py",
    "strike_spec_torch.py",
    "virtual_ball.py",
    "counter_rally.py",
    "counter_rally_torch.py",
    "action_ball_solver_semantic_surface.py",
)


def _mirror_repin_inputs(root: Path) -> Path:
    mdp = root / _MDP_REL
    mdp.mkdir(parents=True, exist_ok=True)
    for name in _REPIN_INPUTS:
        (mdp / name).write_bytes((_REPO_ROOT / _MDP_REL / name).read_bytes())
    return mdp


def _repin(root: Path, template: Path, destination: Path) -> dict:
    path, digest = module._materialize_live_profile_pins(
        root=root,
        template_path=template,
        template_expected_sha=_sha(template),
        destination=destination,
        output_relative=destination.name,
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    assert _sha(path) == digest
    return document


def _reseal(document: dict) -> dict:
    document["solver_profile_sha256"] = module._canonical_payload_sha(
        document["solver_payload"]
    )
    return document


def _staged_v3_template(root: Path, mutate=None) -> Path:
    """Copy the live v3 pins document into a scratch root, optionally mutated."""

    document = json.loads(
        (_REPO_ROOT / _LIVE_V3_PINS_REL).read_text(encoding="utf-8")
    )
    if mutate is not None:
        mutate(document)
    target = root / "pins.json"
    target.write_text(
        json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def test_live_repin_of_a_v3_document_is_a_fixed_point_at_the_same_checkout(
    tmp_path: Path,
):
    """The pin the offline producer mints must be the pin the runtime computes.

    The live v3 pins document was minted by ``pin_action_ball_profile_contracts``
    from this same checkout, and its ``solver_profile_sha256`` is what the boot
    gate compares against.  Re-sealing it here must reproduce it exactly.  Before
    the v3 branch existed this returned a different digest, because the payload
    came back carrying a whole-file byte map the runtime never puts in.
    """

    template = _REPO_ROOT / _LIVE_V3_PINS_REL
    expected = json.loads(template.read_text(encoding="utf-8"))
    assert expected["solver_payload"]["schema_version"] == 3

    document = _repin(_REPO_ROOT, template, tmp_path / "out")

    assert "implementation_source_sha256" not in document["solver_payload"]
    assert document["solver_payload"] == expected["solver_payload"]
    assert (
        document["solver_profile_sha256"] == expected["solver_profile_sha256"]
    )
    assert (
        document["solver_payload"]["semantic_surface"]["sha256"]
        == document["solver_semantic_surface"]["sha256"]
    )


def test_live_repin_refuses_a_v3_payload_that_still_carries_the_v2_byte_map(
    tmp_path: Path,
):
    """A hybrid document is exactly the shape the old code used to emit."""

    _mirror_repin_inputs(tmp_path)

    def mutate(document: dict) -> None:
        document["solver_payload"]["implementation_source_sha256"] = dict(
            document["solver_implementation_source_sha256"]
        )
        _reseal(document)

    template = _staged_v3_template(tmp_path, mutate)
    with pytest.raises(module.BundleError, match="not a v3 pin"):
        _repin(tmp_path, template, tmp_path / "out")


def test_live_repin_refuses_when_a_covered_solver_symbol_moved(tmp_path: Path):
    """Refuse a re-seal that is really a re-draw, and name the migration script.

    The mutation moves one covered symbol's body and nothing else: the pinned
    source list, the covered symbol names and the covered symbol count are all
    unchanged.  A check one notch coarser -- comparing counts, names or file
    lists instead of the sealed per-symbol digest -- would wave this through.
    """

    mdp = _mirror_repin_inputs(tmp_path)
    ball = mdp / "virtual_ball.py"
    source = ball.read_text(encoding="utf-8")
    needle = "a[..., 2] -= prm.g"
    assert source.count(needle) == 1, "the mutation target moved; fix the test"
    ball.write_text(source.replace(needle, needle + " * 1.05"), encoding="utf-8")

    template = _staged_v3_template(tmp_path)
    with pytest.raises(module.BundleError) as excinfo:
        _repin(tmp_path, template, tmp_path / "out")
    message = str(excinfo.value)
    assert "migrate_action_ball_solver_pin_to_semantic_surface.py" in message
    assert "re-drawing the questions" in message


def test_live_repin_rereads_the_unadjudicated_counter_rally_bytes(
    tmp_path: Path,
):
    """counter-rally is still whole-file pinned, so live bytes must win.

    The runtime hashes these two files off the live checkout every boot.  If the
    producer copied the template's values instead of re-reading, a counter-rally
    edit would leave the manifest pinned to bytes that no longer exist and the
    boot gate would reject it.
    """

    clean_root = tmp_path / "clean"
    dirty_root = tmp_path / "dirty"
    for root in (clean_root, dirty_root):
        _mirror_repin_inputs(root)
    dirty_rally = dirty_root / _MDP_REL / "counter_rally.py"
    dirty_rally.write_bytes(
        dirty_rally.read_bytes() + b"\n# offline re-pin liveness probe\n"
    )

    clean = _repin(clean_root, _staged_v3_template(clean_root), clean_root / "o")
    dirty = _repin(dirty_root, _staged_v3_template(dirty_root), dirty_root / "o")

    live_dirty_sha = hashlib.sha256(dirty_rally.read_bytes()).hexdigest()
    assert (
        dirty["solver_payload"]["unadjudicated_whole_file_sha256"][
            "counter_rally.py"
        ]
        == live_dirty_sha
    )
    assert (
        clean["solver_payload"]["unadjudicated_whole_file_sha256"][
            "counter_rally.py"
        ]
        != live_dirty_sha
    )
    assert clean["solver_profile_sha256"] != dirty["solver_profile_sha256"]
    # The semantic half is untouched: counter-rally has never been adjudicated
    # symbol by symbol, which is exactly why it stays on the coarse pin.
    assert (
        clean["solver_payload"]["semantic_surface"]
        == dirty["solver_payload"]["semantic_surface"]
    )


def test_live_repin_refuses_a_solver_profile_schema_it_does_not_know(
    tmp_path: Path,
):
    _mirror_repin_inputs(tmp_path)

    def mutate(document: dict) -> None:
        document["solver_payload"]["schema_version"] = 4
        _reseal(document)

    template = _staged_v3_template(tmp_path, mutate)
    with pytest.raises(module.BundleError, match="solver profile schema 4"):
        _repin(tmp_path, template, tmp_path / "out")
