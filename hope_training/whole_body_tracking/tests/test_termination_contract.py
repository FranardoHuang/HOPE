"""CPU-only tests for frozen Isaac-to-MuJoCo termination parity."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "termination_contract.py"
ISAAC_EVALUATOR = SCRIPT.parent / "eval_deterministic.py"
SPEC = importlib.util.spec_from_file_location("termination_contract_tested", SCRIPT)
termination = importlib.util.module_from_spec(SPEC)
sys.modules["termination_contract_tested"] = termination
assert SPEC.loader is not None
SPEC.loader.exec_module(termination)

TRACKED = [
    "pelvis_link",
    "left_ankle_roll_Link",
    "right_ankle_roll_Link",
    "left_wrist_yaw_Link",
    "right_wrist_yaw_Link",
]


def _document(*, envelope_as_penalty: bool = False, upper_only: bool = False):
    terms = {
        "time_out": {
            "func": "isaaclab.envs.mdp.terminations:time_out",
            "time_out": True,
            "params": {},
        },
        "anchor_pos": {
            "func": "whole_body_tracking.tasks.tracking.mdp.terminations:bad_anchor_pos_z_only",
            "params": {"command_name": "motion", "threshold": 0.25},
            "time_out": False,
        },
        "anchor_ori": {
            "func": "whole_body_tracking.tasks.tracking.mdp.terminations:bad_anchor_ori",
            "params": {
                "command_name": "motion",
                "threshold": 0.8,
                "asset_cfg": {"name": "robot"},
            },
            "time_out": False,
        },
        "ee_body_pos": {
            "func": "whole_body_tracking.tasks.tracking.mdp.terminations:bad_motion_body_pos_z_only",
            "params": {
                "command_name": "motion",
                "threshold": 0.25,
                "body_names": [
                    "left_ankle_roll_Link",
                    "right_ankle_roll_Link",
                    "left_wrist_yaw_Link",
                    "right_wrist_yaw_Link",
                ],
            },
            "time_out": False,
        },
        "base_fell_tilt": {
            "func": "isaaclab.envs.mdp.terminations:bad_orientation",
            "params": {"limit_angle": 0.7},
            "time_out": False,
        },
        "base_too_low": {
            "func": "isaaclab.envs.mdp.terminations:root_height_below_minimum",
            "params": {"minimum_height": 0.5},
            "time_out": False,
        },
    }
    if envelope_as_penalty:
        terms["anchor_pos"] = None
        terms["ee_body_pos"] = None
    elif upper_only:
        terms["ee_body_pos"]["params"]["body_names"] = [
            "left_wrist_yaw_Link", "right_wrist_yaw_Link"
        ]
    return {
        "sim": {"dt": 0.005},
        "decimation": 4,
        "episode_length_s": 10.0,
        "commands": {"motion": {"anchor_body_name": "torso_Link"}},
        "terminations": terms,
    }


def _write(path: Path, document) -> Path:
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return path


def _runtime_cfg(document):
    terms = {}
    for name, raw in document["terminations"].items():
        if raw is None:
            terms[name] = None
            continue
        params = copy.deepcopy(raw["params"])
        if "asset_cfg" in params:
            params["asset_cfg"] = SimpleNamespace(**params["asset_cfg"])
        terms[name] = SimpleNamespace(
            func=raw["func"], params=params, time_out=raw["time_out"]
        )
    return SimpleNamespace(
        sim=SimpleNamespace(dt=document["sim"]["dt"]),
        decimation=document["decimation"],
        episode_length_s=document["episode_length_s"],
        commands=SimpleNamespace(
            motion=SimpleNamespace(**document["commands"]["motion"])
        ),
        terminations=SimpleNamespace(**terms),
    )


def test_full_contract_parses_active_terms_thresholds_and_body_indices(tmp_path: Path):
    env = _write(tmp_path / "env.yaml", _document())
    contract = termination.build_termination_contract(env)
    termination.verify_termination_contract(contract, env_yaml=env)
    assert contract["contract_id"].startswith("sha256:")
    assert contract["terms"]["anchor_pos"]["threshold"] == 0.25
    assert contract["terms"]["base_fell_tilt"]["limit_angle"] == 0.7
    assert contract["runtime"] == {
        "sim_dt": 0.005,
        "decimation": 4,
        "control_dt": 0.02,
        "episode_length_s": 10.0,
        "max_episode_steps": 500,
        "motion_command_name": "motion",
        "anchor_body_name": "torso_Link",
    }

    runtime = termination.runtime_termination_settings(contract, TRACKED)
    assert runtime["max_episode_steps"] == 500
    assert runtime["anchor_body_name"] == "torso_Link"
    assert runtime["ee_body_pos_z"]["indices"] == [1, 2, 3, 4]
    assert termination.tracking_reset_reasons(
        runtime,
        anchor_pos_z_error=0.3,
        anchor_ori_error=0.9,
        ee_body_pos_z_errors=[0.3, 0.0, 0.0, 0.0],
    ) == ["anchor_pos", "anchor_ori", "ee_body_pos"]
    assert termination.physical_reset_reasons(
        runtime, tilt_rad=0.71, root_height=0.49
    ) == ["fall_tilt", "fall_root_z"]


def test_envelope_penalty_contract_disables_only_training_removed_resets(tmp_path: Path):
    env = _write(tmp_path / "env.yaml", _document(envelope_as_penalty=True))
    contract = termination.build_termination_contract(env)
    runtime = termination.runtime_termination_settings(contract, TRACKED)
    assert runtime["anchor_pos_z"] is None
    assert runtime["ee_body_pos_z"] is None
    assert runtime["anchor_ori"] == 0.8
    assert "anchor_pos" not in runtime["active_terms"]
    assert "ee_body_pos" not in runtime["active_terms"]
    # Large reference-envelope errors cannot reset R8b/C2/C3/C4; the still-active orientation
    # and absolute physical terms remain effective.
    assert termination.tracking_reset_reasons(
        runtime,
        anchor_pos_z_error=99.0,
        anchor_ori_error=0.81,
        ee_body_pos_z_errors=[99.0],
    ) == ["anchor_ori"]
    assert termination.physical_reset_reasons(
        runtime, tilt_rad=0.8, root_height=0.4
    ) == ["fall_tilt", "fall_root_z"]


def test_upper_only_contract_keeps_wrist_guards_and_drops_ankles(tmp_path: Path):
    env = _write(tmp_path / "env.yaml", _document(upper_only=True))
    runtime = termination.runtime_termination_settings(
        termination.build_termination_contract(env), TRACKED
    )
    assert runtime["ee_body_pos_z"]["body_names"] == [
        "left_wrist_yaw_Link", "right_wrist_yaw_Link"
    ]
    assert runtime["ee_body_pos_z"]["indices"] == [3, 4]


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d.pop("terminations"), "terminations object"),
        (lambda d: d["terminations"].update({"mystery_reset": {"func": "x", "params": {}}}),
         "unsupported active termination"),
        (lambda d: d["terminations"]["anchor_pos"]["params"].pop("threshold"),
         "YAML number"),
        (lambda d: d["terminations"]["anchor_ori"].update({"func": "wrong_guard"}),
         "must be one of"),
        (lambda d: d["terminations"].update({"time_out": None}),
         "time_out termination"),
        (lambda d: d["terminations"]["anchor_pos"].update({"time_out": True}),
         "time_out must be false"),
        (lambda d: d["terminations"]["anchor_pos"]["params"].update({"threshold": True}),
         "YAML number"),
        (lambda d: d["terminations"]["ee_body_pos"]["params"].update({"body_names": [3]}),
         "non-empty strings"),
        (lambda d: d["terminations"]["anchor_ori"]["params"].update(
            {"asset_cfg": {"name": "other"}}
         ), "asset_cfg.name"),
        (lambda d: d["commands"]["motion"].update({"anchor_body_name": "pelvis_link"}),
         "anchor_body_name"),
    ],
)
def test_formal_parser_fails_closed_on_unrepresentable_config(
    tmp_path: Path, mutate, match: str
):
    document = _document()
    mutate(document)
    env = _write(tmp_path / "env.yaml", document)
    with pytest.raises(termination.TerminationContractError, match=match):
        termination.build_termination_contract(env)


def test_contract_id_and_source_verification_detect_changes(tmp_path: Path):
    env = _write(tmp_path / "env.yaml", _document())
    contract = termination.build_termination_contract(env)

    tampered = copy.deepcopy(contract)
    tampered["terms"]["anchor_pos"]["threshold"] = 0.3
    with pytest.raises(termination.TerminationContractError, match="contract_id mismatch"):
        termination.verify_termination_contract(tampered)

    document = _document()
    document["terminations"]["anchor_pos"]["params"]["threshold"] = 0.3
    _write(env, document)
    changed = termination.build_termination_contract(env)
    assert changed["contract_id"] != contract["contract_id"]
    with pytest.raises(termination.TerminationContractError, match="no longer matches"):
        termination.verify_termination_contract(contract, env_yaml=env)


def _rehash(contract):
    payload = dict(contract)
    payload.pop("contract_id", None)
    contract["contract_id"] = termination._content_id(payload)


def test_rehashed_malformed_contract_still_fails_closed(tmp_path: Path):
    env = _write(tmp_path / "env.yaml", _document())
    original = termination.build_termination_contract(env)

    inactive_timeout = copy.deepcopy(original)
    inactive_timeout["terms"]["time_out"] = {"active": False}
    _rehash(inactive_timeout)
    with pytest.raises(termination.TerminationContractError, match="time_out must be active"):
        termination.verify_termination_contract(inactive_timeout)

    extra_field = copy.deepcopy(original)
    extra_field["terms"]["anchor_pos"]["invented"] = 123
    _rehash(extra_field)
    with pytest.raises(termination.TerminationContractError, match="fields must be exactly"):
        termination.verify_termination_contract(extra_field)

    wrong_timeout_step = copy.deepcopy(original)
    wrong_timeout_step["runtime"]["max_episode_steps"] = 499
    _rehash(wrong_timeout_step)
    with pytest.raises(termination.TerminationContractError, match="computed 500"):
        termination.verify_termination_contract(wrong_timeout_step)


def test_function_identity_requires_the_verified_full_module_path(tmp_path: Path):
    document = _document()
    document["terminations"]["anchor_ori"]["func"] = "evil.module:bad_anchor_ori"
    env = _write(tmp_path / "env.yaml", document)
    with pytest.raises(termination.TerminationContractError, match="matching function-name suffix"):
        termination.build_termination_contract(env)


def test_runtime_timing_changes_are_frozen_into_the_contract(tmp_path: Path):
    first = _document()
    env = _write(tmp_path / "env.yaml", first)
    contract_500 = termination.build_termination_contract(env)
    first["episode_length_s"] = 8.0
    _write(env, first)
    contract_400 = termination.build_termination_contract(env)
    assert contract_500["runtime"]["max_episode_steps"] == 500
    assert contract_400["runtime"]["max_episode_steps"] == 400
    assert contract_500["contract_id"] != contract_400["contract_id"]


def test_lossless_runtime_pickle_fields_must_match_human_readable_yaml(tmp_path: Path):
    document = _document()
    env = _write(tmp_path / "env.yaml", document)
    contract = termination.build_termination_contract(env)
    runtime_cfg = _runtime_cfg(document)
    termination.verify_runtime_env_cfg(contract, runtime_cfg)

    runtime_cfg.terminations.anchor_pos.params["threshold"] = 0.3
    with pytest.raises(termination.TerminationContractError, match="do not match"):
        termination.verify_runtime_env_cfg(contract, runtime_cfg)


def test_runtime_pickle_episode_and_function_identity_mismatches_fail(tmp_path: Path):
    document = _document()
    env = _write(tmp_path / "env.yaml", document)
    contract = termination.build_termination_contract(env)

    wrong_time = _runtime_cfg(document)
    wrong_time.episode_length_s = 8.0
    with pytest.raises(termination.TerminationContractError, match="does not match"):
        termination.verify_runtime_env_cfg(contract, wrong_time)

    wrong_function = _runtime_cfg(document)
    wrong_function.terminations.anchor_ori.func = "evil.module:bad_anchor_ori"
    with pytest.raises(termination.TerminationContractError, match="matching function-name suffix"):
        termination.verify_runtime_env_cfg(contract, wrong_function)


@pytest.mark.skip(
    reason="schema-v3 Isaac BankExam adapter is not wired yet; termination_contract is library-only"
)
def test_isaac_formal_evaluator_checks_pickle_yaml_before_environment_construction():
    source = ISAAC_EVALUATOR.read_text()
    load = source.index("env_cfg = pickle.load(stream)")
    parity = source.index("verify_runtime_env_cfg(saved_termination_contract, env_cfg)")
    construct = source.index("env = gym.make(task_id")
    assert load < parity < construct


def test_duplicate_yaml_key_is_rejected(tmp_path: Path):
    text = yaml.safe_dump(_document(), sort_keys=False)
    text = text.replace("episode_length_s: 10.0", "episode_length_s: 10.0\nepisode_length_s: 8.0")
    env = tmp_path / "env.yaml"
    env.write_text(text)
    with pytest.raises(termination.TerminationContractError, match="duplicate key"):
        termination.build_termination_contract(env)


def test_real_isaac_container_tags_are_read_as_data_without_object_construction(
    tmp_path: Path,
):
    text = yaml.safe_dump(_document(), sort_keys=False)
    text = (
        "camera_eye: !!python/tuple [3.0, 3.0, 3.0]\n"
        "joint_ids: !!python/object/apply:builtins.slice [null, null, null]\n"
        + text
    )
    env = tmp_path / "env.yaml"
    env.write_text(text)
    contract = termination.build_termination_contract(env)
    assert contract["terms"]["time_out"]["active"] is True


def test_unknown_python_yaml_tag_is_rejected_without_execution(tmp_path: Path):
    text = "unsafe: !!python/object/apply:os.system ['echo never-run']\n" + yaml.safe_dump(
        _document(), sort_keys=False
    )
    env = tmp_path / "env.yaml"
    env.write_text(text)
    with pytest.raises(termination.TerminationContractError, match="cannot safely parse"):
        termination.build_termination_contract(env)
