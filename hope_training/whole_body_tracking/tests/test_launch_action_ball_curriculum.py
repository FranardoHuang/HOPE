"""Host-only tests for the ActionBall fresh-N5 launcher and exact V4 runtime."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import pytest


WBT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WBT_ROOT.parents[1]
LAUNCHER_PATH = WBT_ROOT / "scripts/launch_action_ball_curriculum.py"
KIT_LAUNCHER_PATH = WBT_ROOT / "scripts/launch_kit_training_locked.sh"
NOSITE_BOOTSTRAP_PATH = (
    WBT_ROOT / "scripts/action_ball_python_nosite_bootstrap.py"
)


def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        "launch_action_ball_curriculum_under_test", LAUNCHER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


M = _load_launcher()


def _load_nosite_bootstrap():
    spec = importlib.util.spec_from_file_location(
        "action_ball_nosite_bootstrap_for_launcher_tests",
        NOSITE_BOOTSTRAP_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


NOSITE = _load_nosite_bootstrap()


def _json_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bytes(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return hashlib.sha256(value).hexdigest()


def _write_text(path: Path, value: str) -> str:
    return _write_bytes(path, value.encode("utf-8"))


def _initial_heartbeat(payload, evaluator_pid: int) -> dict:
    content = {
        "owner_id": payload["frozen_evaluation_runtime"]["owner_id"],
        "run_id": payload["frozen_evaluation_runtime"]["run_id"],
        "pid": evaluator_pid,
        "sidecar_code_sha256": payload["sidecar_launch_receipt"][
            "sidecar_code_sha256"
        ],
        "launch_sha256": payload["sidecar_launch_receipt"]["content_sha256"],
        "backend_contract_sha256": payload["sidecar_launch_receipt"][
            "backend_contract_sha256"
        ],
        "heartbeat_seq": 1,
        "phase": "ready",
        "request_seq": None,
        "request_sha256": "",
        "attempts_completed": 0,
        "attempts_total": 0,
        "request_started_unix_ns": 0,
        "request_started_monotonic_ns": 0,
        "request_deadline_unix_ns": 0,
        "request_deadline_monotonic_ns": 0,
        "heartbeat_unix_ns": time.time_ns(),
        "heartbeat_monotonic_ns": time.monotonic_ns(),
        "error_type": "",
    }
    return {
        "schema_version": 1,
        "kind": "whole_body_tracking.action_ball.formal_sidecar_heartbeat",
        "content": content,
        "content_sha256": M.canonical_sha256(content),
    }


def _install_live_gpu_admission(plan: dict) -> str:
    payload = plan["canonical_payload"]
    tool = {
        "name": "nvidia-smi",
        "requested_path": "/usr/bin/nvidia-smi",
        "path": "/usr/bin/nvidia-smi",
        "sha256": "a" * 64,
    }
    return M._write_live_gpu_admission(
        plan,
        {
            role: {
                "nvidia_smi": tool,
                "gpu_index": payload["gpus"][role]["index"],
                "gpu_uuid": payload["gpus"][role]["uuid"],
                "compute_process_count": 0,
            }
            for role in ("trainer", "evaluator")
        },
    )


class LaunchFixture:
    """A tiny exact-commit repo plus external operator/evaluator evidence."""

    def __init__(self, tmp_path: Path):
        cryptography = pytest.importorskip("cryptography")
        del cryptography
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )

        self.repo = (tmp_path / "repo").resolve()
        self.control = (tmp_path / "control").resolve()
        self.runs = (tmp_path / "runs").resolve()
        self.repo.mkdir()
        self.control.mkdir()
        self.runs.mkdir()
        self.private_key = Ed25519PrivateKey.generate()
        self.public_key_hex = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ).hex()
        self.order = list(M.ACTION_ORDER)
        self.reward_sha = "e" * 64
        self.ppo_recipe = {"schema_version": 1, "fixture": "ppo"}
        self.ppo_sha = M.canonical_sha256(self.ppo_recipe)
        self.ground_sha = M.GROUND_PLANT_ABSENT_SHA256

        _write_text(
            self.repo / ".gitignore",
            "/hope_training/whole_body_tracking/logs/\n/ignored/\n",
        )
        self.bindings = []
        for index, action_id in enumerate(self.order):
            family = (
                "forehand"
                if action_id in {"v12_forehand_block", "fh_loop_high"}
                else "backhand"
            )
            relative = f"motions/{index:02d}_{action_id}.npz"
            digest = _write_text(
                self.repo / relative, f"fixture motion {index} {action_id}\n"
            )
            self.bindings.append(
                {
                    "motion_id": action_id,
                    "action_uid": 1000 + index,
                    "family": family,
                    "motion_path": relative,
                    "motion_sha256": digest,
                }
            )

        scopes = {
            "upper": [
                {
                    "motion_id": binding["motion_id"],
                    "scope": "upper",
                    "clip_index": index,
                    "family": binding["family"],
                    "npz_sha256": binding["motion_sha256"],
                }
                for index, binding in enumerate(self.bindings)
            ]
        }
        self.prototype_relative = "configs/prototype.json"
        self.prototype_sha = _write_json(
            self.repo / self.prototype_relative,
            {
                "schema_version": 2,
                "prototype_set_id": "fixture-upper-n5",
                "scopes": scopes,
                "derived_sha256": M.canonical_sha256(scopes),
            },
        )
        self.solver_sha = "1" * 64
        self.physics_sha = "2" * 64
        self.policy_sha = "3" * 64
        self.manifest_relative = "configs/manifest.json"
        self.manifest_sha = _write_json(
            self.repo / self.manifest_relative,
            {
                "schema_version": 3,
                "manifest_id": "fixture-fresh-upper-nomove-n5",
                "mobility_mode": "no_move",
                "action_order": self.order,
                "prototype": {
                    "path": self.prototype_relative,
                    "sha256": self.prototype_sha,
                    "scope": "upper",
                },
                "solver_profile_sha256": self.solver_sha,
                "physics_profile_sha256": self.physics_sha,
                "landing_aim": {},
                "actions": [
                    {
                        "action_id": binding["motion_id"],
                        "action_uid": binding["action_uid"],
                        "family": binding["family"],
                        "motion_path": binding["motion_path"],
                        "motion_sha256": binding["motion_sha256"],
                    }
                    for binding in self.bindings
                ],
                "curriculum": {
                    "min_proposals": 320,
                    "min_safe_closed": 256,
                    "target_failure_rate": 0.10,
                    "failure_band_half_width": 0.03,
                    "min_solver_admit_rate": 0.50,
                    "min_install_rate": 1.0,
                    "min_start_rate": 1.0,
                    "min_close_rate": 1.0,
                    "max_other_unsafe_rate": 0.0,
                    "confidence_z": 1.96,
                    "max_center_failures": 0,
                },
                "holdout": {
                    "seed": 20260729,
                    "samples_per_action": 960,
                    "split_id": "fixture-heldout-v1",
                },
                "notes": "strict launcher fixture",
            },
        )
        self.contract_profile = M.LAUNCH_PROFILE
        self.order_uid_digest = M._order_uid_digest(
            self.order,
            [binding["action_uid"] for binding in self.bindings],
        )
        self.namespace_identity = (
            f"n{len(self.order)}-{self.order_uid_digest[:12]}"
        )
        self.action_set_contract_row = {
            "profile_id": self.contract_profile,
            "expected_n": len(self.order),
            "scope": "upper",
            "mobility_mode": "no_move",
            "ordered_action_ids": self.order,
            "ordered_action_uids": [
                binding["action_uid"] for binding in self.bindings
            ],
            "order_uid_digest_sha256": self.order_uid_digest,
            "manifest_path": self.manifest_relative,
            "manifest_sha256": self.manifest_sha,
            "experiment_name": M.ACTION_BALL_EXPERIMENT_NAME,
        }
        self.action_set_contract_identity = {
            "schema_version": 1,
            "kind": "whole_body_tracking.action_ball.action_set_contract",
            **self.action_set_contract_row,
            "actor_obs_contract": (
                "action_ball_table_pose_twist_heading_task_teacher_start_n"
                f"{len(self.order)}"
            ),
            "actor_obs_width": 194 + len(self.order),
            "namespace_identity": self.namespace_identity,
        }
        self.action_set_contract_identity["contract_sha256"] = (
            M.canonical_sha256(self.action_set_contract_identity)
        )

        self.ready_sha = "4" * 64
        self.ready_fk_sha = "5" * 64
        self.alignment_sha = "6" * 64
        self.registry_relative = "configs/registry.json"
        self.registry_sha = _write_json(
            self.repo / self.registry_relative,
            {
                "schema_version": 2,
                "bank_id": "fixture-upper-n5",
                "scope": "upper",
                "canonical_ready_sha256": self.ready_sha,
                "canonical_ready_fk_sha256": self.ready_fk_sha,
                "entries": [
                    {
                        "motion_id": binding["motion_id"],
                        "scope": "upper",
                        "npz_path": binding["motion_path"],
                        "npz_sha256": binding["motion_sha256"],
                        "training_authorized": True,
                    }
                    for binding in self.bindings
                ],
            },
        )
        self.promotion_relative = "configs/promotion.json"
        self.promotion_sha = _write_json(
            self.repo / self.promotion_relative,
            {
                "schema_version": 1,
                "kind": "fixture-promotion-certificate",
                "authorization_purpose": "training",
            },
        )
        admission_unsigned = {
            "schema_version": 1,
            "kind": "action_ball_static_motion_admission_launch",
            "authorization_purpose": "training",
            "scope": "upper",
            "mobility_mode": "no_move",
            "ordered_action_ids": self.order,
            "registry_sha256": self.registry_sha,
            "promotion_certificate_sha256": self.promotion_sha,
            "motion_rows": [
                {
                    "motion_id": binding["motion_id"],
                    "action_uid": binding["action_uid"],
                    "motion_path": binding["motion_path"],
                    "motion_sha256": binding["motion_sha256"],
                }
                for binding in self.bindings
            ],
        }
        self.admission_relative = "configs/admission.json"
        self.admission_sha = _write_json(
            self.repo / self.admission_relative,
            {
                **admission_unsigned,
                "canonical_sha256": M.canonical_sha256(admission_unsigned),
            },
        )

        self.sidecar_code_sha = _write_text(
            self.repo / M.SIDECAR_CODE_SOURCE,
            "#!/usr/bin/env python3\n# fixture formal sidecar\n",
        )
        sidecar_content = {
            "protocol_contract_sha256": "6" * 64,
            "sidecar_code_sha256": self.sidecar_code_sha,
            "backend_contract_sha256": "7" * 64,
            "policy_evaluation_contract_sha256": "8" * 64,
            "resolved_recipe_contract_sha256": "9" * 64,
            "runtime_identity_contract_sha256": "a" * 64,
            "window_contract": M.SIDECAR_WINDOW_CONTRACT,
            "heartbeat_contract": M.SIDECAR_HEARTBEAT_CONTRACT,
        }
        sidecar = {
            "schema_version": 1,
            "kind": "action_ball_frozen_eval_sidecar_launch",
            "content": sidecar_content,
            "content_sha256": M.canonical_sha256(sidecar_content),
        }
        self.sidecar_content_sha = sidecar["content_sha256"]
        self.sidecar_relative = "configs/sidecar-launch.json"
        self.sidecar_sha = _write_json(
            self.repo / self.sidecar_relative, sidecar
        )
        self.attempt_relative = M.EVALUATION_INBOX_SOURCE
        self.attempt_contract_sha = "d" * 64
        self.attempt_sha = _write_text(
            self.repo / self.attempt_relative,
            (
                f"{M.SIDECAR_CODE_TRUST_NAME} = "
                f"frozenset({{{self.sidecar_code_sha!r}}})\n"
                f"{M.SIDECAR_LAUNCH_TRUST_NAME} = "
                f"frozenset({{{self.sidecar_content_sha!r}}})\n"
            ),
        )
        evaluator = {
            "schema_version": 4,
            "kind": "action_ball_frozen_evaluator_v4_launch",
            "authority_contract_sha256": "8" * 64,
            "curriculum_contract_sha256": "9" * 64,
            "profile_order": [
                {
                    "action_uid": binding["action_uid"],
                    "profile_sha256": f"{index + 10:064x}",
                    "mobility": "no_move",
                }
                for index, binding in enumerate(self.bindings)
            ],
            "arm_catalog_sha256": "a" * 64,
            "scheduler_contract_sha256": "b" * 64,
            "sampler_sha256": "c" * 64,
            "solver_sha256": self.solver_sha,
            "policy_contract_sha256": self.policy_sha,
            "attempt_source_contract_sha256": self.attempt_contract_sha,
            "attempt_source_path": self.attempt_relative,
            "attempt_source_sha256": self.attempt_sha,
            "window_contract": M.V4_EVALUATOR_WINDOW_CONTRACT,
        }
        self.evaluator_canonical_sha = M.canonical_sha256(evaluator)
        self.evaluator_relative = "configs/evaluator.json"
        self.evaluator_sha = _write_json(
            self.repo / self.evaluator_relative, evaluator
        )
        self.hope_commands_sha = _write_text(
            self.repo / M.HOPE_COMMANDS_SOURCE,
            "# fixture formal drain/reset runtime coordinator\n",
        )
        drain = {
            "schema_version": 1,
            "kind": "action_ball_drain_reset_launch",
            "authority_contract_sha256": "f" * 64,
            "curriculum_contract_sha256": evaluator[
                "curriculum_contract_sha256"
            ],
            "profile_order": evaluator["profile_order"],
            "arm_catalog_sha256": evaluator["arm_catalog_sha256"],
            "scheduler_contract_sha256": evaluator[
                "scheduler_contract_sha256"
            ],
            "sampler_sha256": evaluator["sampler_sha256"],
            "solver_sha256": evaluator["solver_sha256"],
            "policy_contract_sha256": evaluator[
                "policy_contract_sha256"
            ],
            "runtime_source_contract_sha256": "1" * 64,
            "runtime_source_path": M.HOPE_COMMANDS_SOURCE,
            "runtime_source_sha256": self.hope_commands_sha,
            "broker_contract_sha256": "2" * 64,
            "attempt_pool_contract_sha256": "3" * 64,
            "task_receipt_pool_contract_sha256": "4" * 64,
            "env_reset_contract_sha256": "5" * 64,
        }
        self.drain_canonical_sha = M.canonical_sha256(drain)
        self.drain_relative = "configs/drain-reset-launch.json"
        self.drain_sha = _write_json(
            self.repo / self.drain_relative, drain
        )
        self.profile_pins_relative = "configs/profile-pins.json"
        self.profile_pins_sha = _write_json(
            self.repo / self.profile_pins_relative, {"fixture": "profile-pins"}
        )
        self.launch_trust_spec_relative = "configs/launch-trust-spec.json"
        self.launch_trust_spec_sha = _write_json(
            self.repo / self.launch_trust_spec_relative,
            {"fixture": "launch-trust-spec"},
        )
        self.launch_trust_root_relative = "configs/launch-trust-root.json"
        self.launch_trust_root_sha = _write_json(
            self.repo / self.launch_trust_root_relative,
            {"fixture": "launch-trust-root"},
        )

        _write_text(
            self.repo / M.PROMOTION_TRUST_SOURCE,
            f"{M.PROMOTION_TRUST_NAME} = frozenset({{{self.promotion_sha!r}}})\n",
        )
        _write_text(
            self.repo / M.EVALUATOR_TRUST_SOURCE,
            f"{M.EVALUATOR_TRUST_NAME} = "
            f"frozenset({{{self.evaluator_canonical_sha!r}}})\n",
        )
        _write_text(
            self.repo / M.CURRICULUM_TRUST_SOURCE,
            f"{M.DRAIN_RESET_TRUST_NAME} = "
            f"frozenset({{{self.drain_canonical_sha!r}}})\n",
        )
        _write_text(
            self.repo / M.FRESH_ORDER_SOURCE,
            f"{M.FRESH_ORDER_NAME} = {tuple(M.ACTION_ORDER)!r}\n",
        )
        _write_text(
            self.repo / M.ACTION_SET_CONTRACT_SOURCE,
            (
                f"{M.ACTION_SET_PROFILE_POLICIES_NAME} = "
                f"{ {self.contract_profile: {'expected_n': len(self.order), 'scope': 'upper', 'mobility_mode': 'no_move', 'required_action_ids': self.order, 'retired_action_ids': ['fh_loop', 'fh_block_syn']}}!r}\n"
                f"{M.ACTION_SET_CONTRACTS_NAME} = "
                f"{ {self.contract_profile: self.action_set_contract_row}!r}\n"
            ),
        )
        launcher_sha = _write_bytes(
            self.repo / M.LAUNCHER_SOURCE, LAUNCHER_PATH.read_bytes()
        )
        _write_text(self.repo / M.TRAIN_SOURCE, "# fixture trainer\n")
        _write_text(
            self.repo
            / (
                "hope_training/whole_body_tracking/source/whole_body_tracking/"
                "whole_body_tracking/__init__.py"
            ),
            "# fixture package\n",
        )
        _write_text(self.repo / M.SETUP_SOURCE, "# fixture setup\n")
        _write_text(self.repo / M.PROCESS_GROUP_SOURCE, "# fixture process helper\n")
        _write_text(
            self.repo / M.STAGE_SUPERVISOR_SOURCE,
            "# fixture dual-GPU supervisor\n",
        )
        _write_text(
            self.repo / M.EXACT_RESUME_VERIFIER_SOURCE,
            "# fixture exact-resume verifier\n",
        )
        _write_text(
            self.repo / M.RUNTIME_INVENTORY_SOURCE,
            """#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="command", required=True)
verify = sub.add_parser("verify")
verify.add_argument("--receipt", required=True)
args = parser.parse_args()
path = Path(args.receipt)
raw = path.read_bytes()
receipt = json.loads(raw)
content_raw = json.dumps(
    receipt["content"],
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
).encode("utf-8")
if hashlib.sha256(content_raw).hexdigest() != receipt["content_sha256"]:
    raise SystemExit(2)
result = {
    "ok": True,
    "kind": receipt["kind"],
    "content_sha256": receipt["content_sha256"],
    "receipt_path": str(path),
    "receipt_sha256": hashlib.sha256(raw).hexdigest(),
}
print(json.dumps(result, sort_keys=True, separators=(",", ":")))
""",
        )
        _write_text(
            self.repo / M.PROPOSAL_SAMPLER_SOURCE,
            """from __future__ import annotations
import hashlib
import json

def frozen_evaluation_proposal_sampler_contract():
    with open(__file__, "rb") as stream:
        source_sha256 = hashlib.sha256(stream.read()).hexdigest()
    payload = {
        "schema_version": 1,
        "kind": "action_ball_frozen_evaluation_proposal_sampler",
        "random_access": "fixture random access",
        "training_state_isolation": "fixture isolated",
        "sampling_core": "fixture sampling core",
        "mixture": "fixture mixture",
        "frontier": "fixture frontier",
        "proposal_accounting": "fixture one proposal no redraw",
        "sampling_schema_version": 3,
        "arm_catalog_sha256": "f" * 64,
        "draws_per_birth": 3,
        "draws_per_sample": 18,
        "implementation_source_sha256": source_sha256,
    }
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return {"payload": payload, "sha256": hashlib.sha256(raw).hexdigest()}
""",
        )
        _write_text(
            self.repo / M.RUNTIME_BOOTSTRAP_SOURCE,
            "# fixture runtime bootstrap protocol\n",
        )
        _write_bytes(
            self.repo / M.NOSITE_BOOTSTRAP_SOURCE,
            NOSITE_BOOTSTRAP_PATH.read_bytes(),
        )
        _write_text(
            self.repo / M.PPO_RUNNER_SOURCE,
            "# fixture PPO runner\n",
        )
        kit = self.repo / M.KIT_LAUNCHER_SOURCE
        _write_text(kit, "#!/bin/sh\nexit 0\n")
        kit.chmod(0o755)

        authority_unsigned = {
            "schema_version": 1,
            "kind": "action_ball_frozen_stage_evaluator_authority",
            "evaluator_id": "fixture-stage-evaluator",
            "public_key_ed25519_hex": self.public_key_hex,
            "evaluator_source_path": M.LAUNCHER_SOURCE,
            "evaluator_source_sha256": launcher_sha,
        }
        self.authority_relative = "configs/stage-evaluator-authority.json"
        self.authority_sha = _write_json(
            self.repo / self.authority_relative,
            {
                **authority_unsigned,
                "canonical_sha256": M.canonical_sha256(authority_unsigned),
            },
        )

        self._git("init", "-b", "main")
        self._git("config", "user.name", "ActionBall Fixture")
        self._git("config", "user.email", "fixture@example.invalid")
        self._git("add", "-A")
        self._git("commit", "-m", "fixture source")
        self.commit = self._git("rev-parse", "HEAD")

        self.fitted_gate_path = self.control / "fitted-gate.json"
        self.fitted_gate_sha = _write_json(
            self.fitted_gate_path, {"status": "passed", "fixture": "fitted"}
        )
        self.table_smoke_path = self.control / "table-smoke.json"
        self.table_smoke_sha = _write_json(
            self.table_smoke_path, {"status": "passed", "fixture": "table"}
        )
        safety_payload = {
            "schema_version": 1,
            "kind": "action_ball_prelaunch_safety_attestation",
            "status": "passed",
            "source_commit_sha": self.commit,
            "launch_profile": M.LAUNCH_PROFILE,
            "action_set_contract_sha256": self.action_set_contract_identity[
                "contract_sha256"
            ],
            "ordered_action_ids": self.order,
            "manifest_sha256": self.manifest_sha,
            "profile_pins_sha256": self.profile_pins_sha,
            "fitted_ball_launch_trust_spec_sha256": self.launch_trust_spec_sha,
            "fitted_ball_launch_trust_root_sha256": self.launch_trust_root_sha,
            "fitted_ball_gate_receipt_sha256": self.fitted_gate_sha,
            "isaac_table_smoke_receipt_sha256": self.table_smoke_sha,
            "stage_evaluator_authority_sha256": self.authority_sha,
            "per_action": [
                {
                    "action_id": binding["motion_id"],
                    "action_uid": binding["action_uid"],
                    "motion_sha256": binding["motion_sha256"],
                    "t_hit_s": 0.4,
                    "t_cycle_s": 2.0,
                    "physical_racket_site_speed_mps": 3.0,
                    "all_body_table_pair_count": 160,
                    "table_contact_count": 0,
                    "fall_count": 0,
                    "hard_limit_count": 0,
                    "unsafe_count": 0,
                    "t_hit_pass": True,
                    "t_cycle_pass": True,
                    "physical_racket_site_speed_pass": True,
                    "shared_ready_recovery_pass": True,
                    "recorded_incoming_ball_returned_to_table": True,
                    "no_table_contact": True,
                    "grounded_safety_pass": True,
                    "hard_limit_pass": True,
                    "isaac_filtered_contact_pass": True,
                }
                for binding in self.bindings
            ],
        }
        self.safety_path = self.control / "prelaunch-safety.json"
        self.safety_sha = self._write_signed(
            self.safety_path,
            "action_ball_signed_prelaunch_safety_attestation",
            safety_payload,
        )

        runtime_path = Path(sys.executable).resolve()
        runtime_identity = M._probe_python_runtime(runtime_path)
        runtime_import_root = self.control / "runtime-import-root"
        runtime_import_root.mkdir()
        self.runtime_import_roots = NOSITE.bind_import_roots(
            [runtime_import_root]
        )
        self.runtime_spec = {
            "path": str(runtime_path),
            "sha256": M.sha256_file(runtime_path),
            "version": runtime_identity["version"],
            "cache_tag": runtime_identity["cache_tag"],
            "import_roots": runtime_identity["import_roots"],
        }
        runtime_inventory_content = {
            "python": {
                "requested_path": str(runtime_path),
                "probe": {
                    "version": runtime_identity["version"],
                    "cache_tag": runtime_identity["cache_tag"],
                    "no_site_execution": {
                        "outer": {
                            "import_roots": self.runtime_import_roots
                        }
                    },
                },
            },
            "isaaclab_checkout": {"fixture": True},
        }
        self.runtime_inventory_path = self.control / "runtime-inventory.json"
        self.runtime_inventory_sha = _write_json(
            self.runtime_inventory_path,
            {
                "schema_version": 2,
                "kind": "action_ball_runtime_inventory_v2",
                "content": runtime_inventory_content,
                "content_sha256": M.canonical_sha256(
                    runtime_inventory_content
                ),
            },
        )
        self.namespaces = {
            stage: self.runs
            / f"fixture-{stage}-{self.namespace_identity}-attempt-001"
            for stage in M.STAGE_ORDER
        }
        self.spec_path = self.control / "launch-spec-v3.json"
        self.spec = {
            "schema_version": M.SCHEMA_VERSION,
            "kind": M.SPEC_KIND,
            "launch_profile": M.LAUNCH_PROFILE,
            "source": {"checkout": str(self.repo), "commit_sha": self.commit},
            "action_set": {
                "contract_profile": self.contract_profile,
            },
            "inputs": {
                "manifest": self._repo_pin(
                    self.manifest_relative, self.manifest_sha
                ),
                "prototype": self._repo_pin(
                    self.prototype_relative, self.prototype_sha
                ),
                "motion_admission_receipt": self._repo_pin(
                    self.admission_relative, self.admission_sha
                ),
                "evaluator_launch_receipt": self._repo_pin(
                    self.evaluator_relative, self.evaluator_sha
                ),
                "sidecar_launch_receipt": self._repo_pin(
                    self.sidecar_relative, self.sidecar_sha
                ),
                "drain_reset_launch_receipt": self._repo_pin(
                    self.drain_relative, self.drain_sha
                ),
                "canonical_registry": {
                    "path": self.registry_relative,
                    "sha256": self.registry_sha,
                    "alignment_sha256": self.alignment_sha,
                    "canonical_ready_sha256": self.ready_sha,
                    "canonical_ready_fk_sha256": self.ready_fk_sha,
                },
                "promotion_certificate": self._repo_pin(
                    self.promotion_relative, self.promotion_sha
                ),
                "fitted_ball_profile_pins": self._repo_pin(
                    self.profile_pins_relative, self.profile_pins_sha
                ),
                "fitted_ball_launch_trust_spec": self._repo_pin(
                    self.launch_trust_spec_relative, self.launch_trust_spec_sha
                ),
                "fitted_ball_launch_trust_root": self._repo_pin(
                    self.launch_trust_root_relative, self.launch_trust_root_sha
                ),
                "fitted_ball_gate_receipt": self._external_pin(
                    self.fitted_gate_path
                ),
                "isaac_table_smoke_receipt": self._external_pin(
                    self.table_smoke_path
                ),
                "stage_evaluator_authority": self._repo_pin(
                    self.authority_relative, self.authority_sha
                ),
                "prelaunch_safety_attestation": self._external_pin(
                    self.safety_path
                ),
            },
            "policy_contract_sha256": self.policy_sha,
            "train": {
                "isaac_python": self.runtime_spec,
                "runtime_inventory": self._external_pin(
                    self.runtime_inventory_path
                ),
                "seed": 20260729,
                "extra_overrides": ["logger=tensorboard"],
                "ground_plant_contract_sha256": self.ground_sha,
                "effective_reward_recipe_sha256": self.reward_sha,
                "ppo_recipe_sha256": self.ppo_sha,
            },
            "gpus": {
                "trainer": {
                    "index": 0,
                    "uuid": "GPU-fixture-trainer",
                    "owner": "Franco",
                    "lock_path": "/tmp/hope_lean_queue_gpu0.lock",
                    "boot_lock_path": "/workspace/.kit_boot.lock",
                    "require_empty": True,
                },
                "evaluator": {
                    "index": 1,
                    "uuid": "GPU-fixture-evaluator",
                    "owner": "Franco",
                    "lock_path": "/tmp/hope_lean_queue_gpu1.lock",
                    "boot_lock_path": "/workspace/.kit_boot.lock",
                    "require_empty": True,
                },
            },
            "stages": {
                "smoke": self._stage_spec("smoke", 1, 2, 1),
                "canary": self._stage_spec("canary", 8, 256, 32),
                "long": self._stage_spec(
                    "long",
                    M.LONG_MIN_NUM_ENVS,
                    M.LONG_MIN_ITERATIONS,
                    M.LONG_MAX_SAVE_INTERVAL,
                ),
            },
        }
        self.refresh_owner_receipts()

    @staticmethod
    def _repo_pin(path: str, digest: str) -> dict[str, str]:
        return {"path": path, "sha256": digest}

    @staticmethod
    def _external_pin(path: Path) -> dict[str, str]:
        return {"path": str(path), "sha256": M.sha256_file(path)}

    def _stage_spec(
        self, stage: str, envs: int, iterations: int, save_interval: int
    ) -> dict:
        evaluation_interval = {
            "smoke": 2,
            "canary": 32,
            "long": 100,
        }[stage]
        return {
            "namespace": str(self.namespaces[stage]),
            "num_envs": envs,
            "max_iterations": iterations,
            "save_interval": save_interval,
            "evaluation_inbox_root": str(
                self.namespaces[stage] / "frozen_eval_inbox"
            ),
            "evaluation_owner_id": "Franco",
            "evaluation_run_id": self.namespaces[stage].name,
            "frozen_eval_interval_updates": evaluation_interval,
            "trainer_gpu_owner_receipt": None,
            "evaluator_gpu_owner_receipt": None,
            "predecessor_receipt": None,
        }

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _write_signed(self, path: Path, envelope_kind: str, payload: dict) -> str:
        signature = self.private_key.sign(M._canonical_bytes(payload)).hex()
        return _write_json(
            path,
            {
                "schema_version": 1,
                "kind": envelope_kind,
                "payload": payload,
                "signature_ed25519_hex": signature,
            },
        )

    def write_spec(self) -> None:
        _write_json(self.spec_path, self.spec)

    def refresh_owner_receipts(self) -> None:
        self.spec["source"]["commit_sha"] = self.commit
        for stage in M.STAGE_ORDER:
            for role in ("trainer", "evaluator"):
                gpu = self.spec["gpus"][role]
                path = self.control / f"gpu-owner-{role}-{stage}.json"
                _write_json(
                    path,
                    {
                        "schema_version": 1,
                        "kind": "action_ball_gpu_owner",
                        "owner": "Franco",
                        "gpu_index": gpu["index"],
                        "gpu_uuid": gpu["uuid"],
                        "lock_path": gpu["lock_path"],
                        "stage": stage,
                        "namespace": str(self.namespaces[stage]),
                        "source_commit_sha": self.commit,
                    },
                )
                self.spec["stages"][stage][
                    f"{role}_gpu_owner_receipt"
                ] = self._external_pin(path)
        self.write_spec()

    def commit_repo_change(self, message: str = "fixture mutation") -> None:
        self._git("add", "-A")
        self._git("commit", "-m", message)
        self.commit = self._git("rev-parse", "HEAD")
        self.refresh_owner_receipts()

    def install_completed_stage(self, stage: str) -> Path:
        plan = M.prepare_launch_plan(self.spec_path, stage)
        namespace = M._claim_namespace(plan)
        index = M.STAGE_ORDER.index(stage)
        stamp = f"2026-07-29_0{index + 1}-00-00"
        rsl_dir = (
            self.repo
            / "hope_training/whole_body_tracking/logs/rsl_rl"
            / M.ACTION_BALL_EXPERIMENT_NAME
            / f"{stamp}_{namespace.name}"
        )
        (rsl_dir / "params").mkdir(parents=True)
        reward = {
            "schema_version": 1,
            "terms": [],
            "sha256": self.reward_sha,
        }
        reward_path = rsl_dir / "params/effective_reward_recipe.json"
        _write_json(reward_path, reward)
        contract = {
            "schema_version": 3,
            "effective_reward_recipe": reward,
            "action_ball_training": {
                "effective_reward_recipe_sha256": self.reward_sha
            },
            "action_ball_ppo_runner_recipe": self.ppo_recipe,
        }
        contract_path = rsl_dir / "params/training_contract.json"
        _write_json(contract_path, contract)
        budget = self.spec["stages"][stage]
        checkpoint_path = rsl_dir / f"model_{budget['max_iterations']}.pt"
        _write_text(checkpoint_path, f"finite checkpoint for {stage}\n")
        metrics_path = namespace / "frozen-evaluator-metrics.json"
        _write_json(metrics_path, {"stage": stage, "status": "passed"})
        train_log = namespace / "train.log"
        _write_text(
            train_log,
            "[INFO] Task: HOPE-PingPong-ActionBall-AgibotA3-v0 | "
            f"experiment: {M.ACTION_BALL_EXPERIMENT_NAME} | log: {rsl_dir}\n",
        )
        recipe = plan["canonical_payload"]["training_recipe"]
        payload = {
            "schema_version": 1,
            "kind": "action_ball_stage_result",
            "status": "passed",
            "completed_stage": stage,
            "launch_profile": M.LAUNCH_PROFILE,
            "action_set_contract_sha256": plan["canonical_payload"][
                "action_set_contract"
            ]["contract_sha256"],
            "source_commit_sha": self.commit,
            "ordered_action_ids": self.order,
            "manifest_sha256": self.manifest_sha,
            "prototype_sha256": self.prototype_sha,
            "motion_admission_receipt_sha256": self.admission_sha,
            "evaluator_launch_receipt_sha256": self.evaluator_sha,
            "sidecar_launch_receipt_sha256": self.sidecar_sha,
            "drain_reset_launch_receipt_sha256": self.drain_sha,
            "policy_contract_sha256": self.policy_sha,
            "fitted_ball_profile_pins_sha256": self.profile_pins_sha,
            "fitted_ball_launch_trust_spec_sha256": self.launch_trust_spec_sha,
            "fitted_ball_launch_trust_root_sha256": self.launch_trust_root_sha,
            "fitted_ball_gate_receipt_sha256": self.fitted_gate_sha,
            "isaac_table_smoke_receipt_sha256": self.table_smoke_sha,
            "prelaunch_safety_attestation_sha256": self.safety_sha,
            "stage_evaluator_authority_sha256": self.authority_sha,
            "namespace": str(namespace),
            "launch_claim_sha256": plan["launch_claim_sha256"],
            "stage_budget": {
                "num_envs": budget["num_envs"],
                "max_iterations": budget["max_iterations"],
                "save_interval": budget["save_interval"],
            },
            "training_recipe_sha256": M.canonical_sha256(recipe),
            "isaac_python_runtime_sha256": M.canonical_sha256(
                plan["canonical_payload"]["isaac_python_runtime"]
            ),
            "frozen_evaluation_runtime_sha256": M.canonical_sha256(
                plan["canonical_payload"]["frozen_evaluation_runtime"]
            ),
            "gpu_roles_sha256": M.canonical_sha256(
                plan["canonical_payload"]["gpus"]
            ),
            "isolated_training_entrypoint_sha256": M.canonical_sha256(
                plan["canonical_payload"]["isolated_training_entrypoint"]
            ),
            "trainer_output": {
                "rsl_log_dir": str(rsl_dir),
                "timestamp_prefix": stamp,
                "run_name": namespace.name,
                "launcher_log_sha256": M.sha256_file(train_log),
                "training_contract": {
                    "path": "params/training_contract.json",
                    "sha256": M.sha256_file(contract_path),
                },
                "effective_reward_recipe": {
                    "path": "params/effective_reward_recipe.json",
                    "sha256": M.sha256_file(reward_path),
                },
            },
            "metrics_evidence": {
                "path": metrics_path.name,
                "sha256": M.sha256_file(metrics_path),
            },
            "checkpoint": {
                "path": checkpoint_path.name,
                "sha256": M.sha256_file(checkpoint_path),
                "finite": True,
                "exact_resume_passed": True,
            },
            "metrics": {
                "proposed_count": 10,
                "solver_admitted_count": 8,
                "solver_rejected_count": 2,
                "solver_rejection_reason_counts": {"invalid_ball": 2},
                "attempt_count": 8,
                "return_success_count": 7,
                "policy_return_failure_count": 1,
                "return_success_lcb": 0.7,
                "policy_return_failure_rate": 0.125,
                "unsafe_count": 0,
                "table_hit_count": 0,
                "fall_count": 0,
                "hard_limit_count": 0,
                "nan_count": 0,
                "counter_violation_count": 0,
                "domain_epoch_stale_count": 0,
                "curriculum_counter_invariants_passed": True,
            },
        }
        receipt_path = self.control / f"{stage}-passed-v2.json"
        self._write_signed(
            receipt_path, "action_ball_signed_stage_result", payload
        )
        if stage != "long":
            next_stage = M.STAGE_ORDER[index + 1]
            self.spec["stages"][next_stage]["predecessor_receipt"] = (
                self._external_pin(receipt_path)
            )
            self.write_spec()
        return receipt_path


@pytest.fixture
def launch_fixture(tmp_path: Path) -> LaunchFixture:
    return LaunchFixture(tmp_path)


def test_v3_plan_binds_exact_commit_runtime_recipe_and_claim(
    launch_fixture: LaunchFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        M, "_verify_live_gpu_empty", lambda *_: pytest.fail("plan queried GPU")
    )
    plan = M.prepare_launch_plan(launch_fixture.spec_path, "smoke")
    payload = plan["canonical_payload"]
    assert plan["schema_version"] == M.SCHEMA_VERSION
    assert plan["kind"] == M.CLAIM_KIND
    assert payload["ordered_action_ids"] == list(M.ACTION_ORDER)
    assert payload["training_recipe"]["extra_overrides"] == ["logger=tensorboard"]
    assert payload["training_recipe"]["ground_plant_contract_sha256"] == (
        M.GROUND_PLANT_ABSENT_SHA256
    )
    assert payload["isaac_python_runtime"]["sha256"] == (
        launch_fixture.runtime_spec["sha256"]
    )
    runtime_inventory = payload["isaac_python_runtime"][
        "runtime_inventory"
    ]
    assert {
        key: runtime_inventory[key]
        for key in (
            "path",
            "file_sha256",
            "content_sha256",
            "kind",
        )
    } == {
        "path": str(launch_fixture.runtime_inventory_path),
        "file_sha256": launch_fixture.runtime_inventory_sha,
        "content_sha256": json.loads(
            launch_fixture.runtime_inventory_path.read_text(encoding="utf-8")
        )["content_sha256"],
        "kind": "action_ball_runtime_inventory_v2",
    }
    assert runtime_inventory["import_roots"] == (
        launch_fixture.runtime_import_roots
    )
    assert len(runtime_inventory["nosite_verification_contract_sha256"]) == 64
    assert payload["evaluator_launch_receipt"]["canonical_sha256"] == (
        launch_fixture.evaluator_canonical_sha
    )
    assert payload["sidecar_launch_receipt"]["content_sha256"] == (
        launch_fixture.sidecar_content_sha
    )
    assert payload["sidecar_launch_receipt"]["heartbeat_contract"] == (
        M.SIDECAR_HEARTBEAT_CONTRACT
    )
    assert payload["drain_reset_launch_receipt"]["canonical_sha256"] == (
        launch_fixture.drain_canonical_sha
    )
    assert payload["frozen_evaluation_runtime"] == {
        "inbox_root": str(
            launch_fixture.namespaces["smoke"] / "frozen_eval_inbox"
        ),
        "owner_id": "Franco",
        "run_id": launch_fixture.namespaces["smoke"].name,
        "interval_updates": 2,
        "heartbeat_path": str(
            launch_fixture.namespaces["smoke"]
            / "frozen_eval_inbox"
            / "sidecar_status"
            / "Franco"
            / launch_fixture.namespaces["smoke"].name
            / "heartbeat.json"
        ),
        "heartbeat_contract": M.SIDECAR_HEARTBEAT_CONTRACT,
        "evaluator_v4_identity": payload[
            "frozen_evaluation_runtime"
        ]["evaluator_v4_identity"],
    }
    assert payload["gpus"]["trainer"]["index"] == 0
    assert payload["gpus"]["evaluator"]["index"] == 1
    assert (
        payload["gpus"]["trainer"]["uuid"]
        != payload["gpus"]["evaluator"]["uuid"]
    )
    assert payload["proposal_sampler"]["source_path"] == (
        M.PROPOSAL_SAMPLER_SOURCE
    )
    assert payload["proposal_sampler"]["source_sha256"] == payload[
        "runtime_code_sha256"
    ][M.PROPOSAL_SAMPLER_SOURCE]
    assert payload["proposal_sampler_contract_sha256"] == payload[
        "proposal_sampler"
    ]["contract_sha256"]
    assert payload["runtime_bootstrap"]["source_sha256"] == payload[
        "runtime_code_sha256"
    ][M.RUNTIME_BOOTSTRAP_SOURCE]
    assert payload["ppo_runner"]["source_sha256"] == payload[
        "runtime_code_sha256"
    ][M.PPO_RUNNER_SOURCE]
    assert payload["runtime_code_sha256"][
        M.EXACT_RESUME_VERIFIER_SOURCE
    ] == M.sha256_file(
        launch_fixture.repo / M.EXACT_RESUME_VERIFIER_SOURCE
    )
    sidecar_command = NOSITE.validate_exact_nosite_argv(
        payload["sidecar_argv"]
    )
    sidecar_argv = sidecar_command.contract["entrypoint_argv"]
    assert sidecar_argv[-4:] == [
        "--backend",
        "formal",
        "--device",
        "cuda:0",
    ]
    heartbeat_flag = sidecar_argv.index(
        "--heartbeat-interval-s"
    )
    assert sidecar_argv[heartbeat_flag + 1] == "5.0"
    deadline_flag = sidecar_argv.index("--request-deadline-s")
    assert sidecar_argv[deadline_flag + 1] == "7200.0"
    trainer_command = NOSITE.validate_exact_nosite_argv(plan["argv"])
    assert trainer_command.contract["entrypoint"]["path"].endswith(
        M.LAUNCHER_SOURCE
    )
    training_argv = trainer_command.contract["entrypoint_argv"]
    assert training_argv[0] == "train-entrypoint"
    assert payload["isolated_training_entrypoint"]["import_root"].endswith(
        "source/whole_body_tracking"
    )
    assert (
        f"task.experiment_name={M.ACTION_BALL_EXPERIMENT_NAME}"
        in training_argv
    )
    assert "task.rewards.full_body_mimic=false" in training_argv
    assert "task.rewards.full_body_mimic" in M._OWNED_OVERRIDE_KEYS
    assert "algo.policy.init_noise_std=0.02" in training_argv
    assert "action_ball_shared_ready_bootstrap=true" in training_argv
    assert "+task.racket.reference_guard_mode=metrics_only" in training_argv
    assert "algo.policy.init_noise_std" in M._OWNED_OVERRIDE_KEYS
    assert "action_ball_shared_ready_bootstrap" in M._OWNED_OVERRIDE_KEYS
    assert "task.racket.reference_guard_mode" in M._OWNED_OVERRIDE_KEYS
    assert (
        "expected_effective_reward_recipe_sha256="
        + launch_fixture.reward_sha
        in training_argv
    )
    for override in (
        "task.racket.action_ball_sidecar_launch_receipt_path="
        + launch_fixture.sidecar_relative,
        "task.racket.action_ball_sidecar_launch_receipt_file_sha256="
        + launch_fixture.sidecar_sha,
        "task.racket.action_ball_drain_reset_launch_receipt_path="
        + launch_fixture.drain_relative,
        "task.racket.action_ball_drain_reset_launch_receipt_file_sha256="
        + launch_fixture.drain_sha,
        "task.racket.action_ball_evaluation_inbox_root="
        + str(launch_fixture.namespaces["smoke"] / "frozen_eval_inbox"),
        "task.racket.action_ball_evaluation_owner_id=Franco",
        "task.racket.action_ball_evaluation_run_id="
        + launch_fixture.namespaces["smoke"].name,
        "task.racket.action_ball_frozen_eval_interval_updates=2",
    ):
        assert override in training_argv
    assert plan["confirmation_claim_sha256"] == plan["launch_claim_sha256"]
    assert training_argv[-1] == (
        "++training_launch_claim_sha256=" + plan["launch_claim_sha256"]
    )
    assert (
        "++training_launch_claim_path="
        + str(
            launch_fixture.namespaces["smoke"] / "launch_claim.json"
        )
        in training_argv[:-1]
    )


def test_plan_ignores_caller_git_path_and_repository_environment(
    launch_fixture: LaunchFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "attacker.git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "attacker-tree"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "attacker-index"))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(tmp_path / "objects"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.fsmonitor")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "evil")
    plan = M.prepare_launch_plan(launch_fixture.spec_path, "smoke")
    identity = plan["canonical_payload"]["runtime_tool_identity"]["git"]
    assert identity["path"] != str(fake_git)
    assert identity == M._trusted_system_executable("git")


def test_gpu_probe_uses_resolved_system_nvidia_smi_and_fixed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class Completed:
        returncode = 0
        stderr = ""

        def __init__(self, stdout):
            self.stdout = stdout

    monkeypatch.setattr(
        M,
        "_trusted_system_executable",
        lambda name: {
            "name": name,
            "requested_path": "/usr/bin/nvidia-smi",
            "path": "/usr/bin/nvidia-smi",
            "sha256": "a" * 64,
        },
    )

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if "--query-gpu=index,uuid" in argv:
            return Completed("0, GPU-test\n")
        return Completed("")

    monkeypatch.setattr(M.subprocess, "run", fake_run)
    result = M._verify_live_gpu_empty(0, "GPU-test")
    assert result["nvidia_smi"]["sha256"] == "a" * 64
    assert len(calls) == 2
    for argv, kwargs in calls:
        assert argv[0] == "/usr/bin/nvidia-smi"
        assert kwargs["env"] == {
            "PATH": os.defpath,
            "LANG": "C",
            "LC_ALL": "C",
        }


@pytest.mark.parametrize(
    "scope,field",
    [
        ("inputs", "sidecar_launch_receipt"),
        ("inputs", "drain_reset_launch_receipt"),
        ("stage", "evaluation_inbox_root"),
        ("stage", "evaluation_owner_id"),
        ("stage", "evaluation_run_id"),
        ("stage", "frozen_eval_interval_updates"),
        ("stage", "trainer_gpu_owner_receipt"),
        ("stage", "evaluator_gpu_owner_receipt"),
    ],
)
def test_formal_v4_runtime_field_is_mandatory(
    launch_fixture: LaunchFixture, scope: str, field: str
) -> None:
    if scope == "inputs":
        del launch_fixture.spec["inputs"][field]
    else:
        del launch_fixture.spec["stages"]["smoke"][field]
    launch_fixture.write_spec()
    with pytest.raises(M.LaunchRefused, match="missing"):
        M.prepare_launch_plan(launch_fixture.spec_path, "smoke")


def test_runtime_inventory_is_mandatory(
    launch_fixture: LaunchFixture,
) -> None:
    del launch_fixture.spec["train"]["runtime_inventory"]
    launch_fixture.write_spec()
    with pytest.raises(M.LaunchRefused, match="runtime_inventory"):
        M.prepare_launch_plan(launch_fixture.spec_path, "smoke")


def test_runtime_inventory_receipt_drift_is_rejected(
    launch_fixture: LaunchFixture,
) -> None:
    receipt = json.loads(
        launch_fixture.runtime_inventory_path.read_text(encoding="utf-8")
    )
    receipt["content"]["isaaclab_checkout"]["fixture"] = False
    _write_json(launch_fixture.runtime_inventory_path, receipt)
    launch_fixture.spec["train"]["runtime_inventory"] = (
        launch_fixture._external_pin(launch_fixture.runtime_inventory_path)
    )
    launch_fixture.write_spec()
    with pytest.raises(M.LaunchRefused, match="content_sha256"):
        M.prepare_launch_plan(launch_fixture.spec_path, "smoke")


def test_sidecar_heartbeat_contract_is_exact(
    launch_fixture: LaunchFixture,
) -> None:
    document = json.loads(
        (launch_fixture.repo / launch_fixture.sidecar_relative).read_text(
            encoding="utf-8"
        )
    )
    document["content"]["heartbeat_contract"][
        "heartbeat_stale_after_seconds"
    ] = 121.0
    document["content_sha256"] = M.canonical_sha256(document["content"])
    with pytest.raises(M.LaunchRefused, match="heartbeat_contract"):
        M._validate_sidecar_launch_receipt(
            document,
            checkout=launch_fixture.repo,
            source_commit=launch_fixture.commit,
        )


def test_sidecar_heartbeat_contract_rejects_equal_integer_tamper(
    launch_fixture: LaunchFixture,
) -> None:
    document = json.loads(
        (launch_fixture.repo / launch_fixture.sidecar_relative).read_text(
            encoding="utf-8"
        )
    )
    document["content"]["heartbeat_contract"][
        "heartbeat_interval_seconds"
    ] = 5
    document["content_sha256"] = M.canonical_sha256(document["content"])
    with pytest.raises(M.LaunchRefused, match="heartbeat_contract"):
        M._validate_sidecar_launch_receipt(
            document,
            checkout=launch_fixture.repo,
            source_commit=launch_fixture.commit,
        )


def test_trainer_and_evaluator_cannot_share_gpu_identity(
    launch_fixture: LaunchFixture,
) -> None:
    launch_fixture.spec["gpus"]["evaluator"]["uuid"] = (
        launch_fixture.spec["gpus"]["trainer"]["uuid"]
    )
    launch_fixture.write_spec()
    with pytest.raises(M.LaunchRefused, match="distinct UUIDs"):
        M.prepare_launch_plan(launch_fixture.spec_path, "smoke")


def test_shared_kit_boot_lock_is_fixed_non_truncating_and_nofollow() -> None:
    source = KIT_LAUNCHER_PATH.read_text(encoding="utf-8")
    assert "lock_file=/workspace/.kit_boot.lock" in source
    assert "os.O_APPEND" in source
    assert "os.O_NOFOLLOW" in source
    assert "os.O_CREAT | os.O_APPEND" not in source
    assert 'exec 9>"$lock_file"' not in source


def test_legacy_evaluator_receipt_is_never_accepted(
    launch_fixture: LaunchFixture,
) -> None:
    legacy = json.loads(
        (
            launch_fixture.repo / launch_fixture.evaluator_relative
        ).read_text(encoding="utf-8")
    )
    legacy["schema_version"] = 1
    legacy["kind"] = "action_ball_frozen_evaluator_launch"
    with pytest.raises(M.LaunchRefused, match="schema/kind"):
        M._validate_evaluator_receipt(
            legacy,
            checkout=launch_fixture.repo,
                source_commit=launch_fixture.commit,
                bindings=launch_fixture.bindings,
                mobility_mode="no_move",
                solver_sha256=launch_fixture.solver_sha,
            policy_sha256=launch_fixture.policy_sha,
        )


def test_ignored_uncommitted_input_cannot_impersonate_commit_blob(
    launch_fixture: LaunchFixture,
) -> None:
    ignored = launch_fixture.repo / "ignored/manifest.json"
    ignored.parent.mkdir()
    shutil.copy2(
        launch_fixture.repo / launch_fixture.manifest_relative, ignored
    )
    launch_fixture.spec["inputs"]["manifest"] = {
        "path": "ignored/manifest.json",
        "sha256": launch_fixture.manifest_sha,
    }
    launch_fixture.write_spec()
    assert launch_fixture._git("status", "--porcelain", "--ignored").splitlines()
    with pytest.raises(M.LaunchRefused, match="must exist exactly once in commit"):
        M.prepare_launch_plan(launch_fixture.spec_path, "smoke")


def test_worktree_bytes_must_equal_commit_tree(
    launch_fixture: LaunchFixture,
) -> None:
    path = launch_fixture.repo / launch_fixture.manifest_relative
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(M.LaunchRefused, match="checkout is dirty"):
        M.prepare_launch_plan(launch_fixture.spec_path, "smoke")


@pytest.mark.parametrize("field", ["sha256", "version", "cache_tag", "import_roots"])
def test_isaac_python_runtime_identity_is_strict(
    launch_fixture: LaunchFixture, field: str
) -> None:
    runtime = launch_fixture.spec["train"]["isaac_python"]
    if field == "sha256":
        runtime[field] = "0" * 64
    elif field == "import_roots":
        runtime[field] = ["/tmp/not-the-runtime"]
    else:
        runtime[field] = "drifted"
    launch_fixture.write_spec()
    with pytest.raises(M.LaunchRefused, match="[Ii]saac Python|isaac_python"):
        M.prepare_launch_plan(launch_fixture.spec_path, "smoke")


def test_python_symlink_chain_is_bound_without_losing_requested_venv_path(
    launch_fixture: LaunchFixture,
) -> None:
    target = Path(sys.executable).resolve()
    link = launch_fixture.control / "isaac-python"
    link.symlink_to(target)
    observed = M._probe_python_runtime(link)
    launch_fixture.spec["train"]["isaac_python"] = {
        "path": str(link),
        "sha256": M.sha256_file(target),
        "version": observed["version"],
        "cache_tag": observed["cache_tag"],
        "import_roots": observed["import_roots"],
    }
    inventory = json.loads(
        launch_fixture.runtime_inventory_path.read_text(encoding="utf-8")
    )
    inventory["content"]["python"]["requested_path"] = str(link)
    inventory["content_sha256"] = M.canonical_sha256(inventory["content"])
    _write_json(launch_fixture.runtime_inventory_path, inventory)
    launch_fixture.spec["train"]["runtime_inventory"] = (
        launch_fixture._external_pin(launch_fixture.runtime_inventory_path)
    )
    launch_fixture.write_spec()
    plan = M.prepare_launch_plan(launch_fixture.spec_path, "smoke")
    runtime = plan["canonical_payload"]["isaac_python_runtime"]
    assert runtime["path"] == str(link)
    assert runtime["resolved_path"] == str(target)
    assert runtime["resolution"]["resolution_chain"][0]["kind"] == "symlink"
    assert runtime["resolution"]["resolution_chain"][-1]["kind"] == "regular"
    assert plan["argv"][0] == str(link)


def test_long_budget_cannot_be_disguised_canary(
    launch_fixture: LaunchFixture,
) -> None:
    launch_fixture.spec["stages"]["long"]["num_envs"] = 16
    launch_fixture.spec["stages"]["long"]["max_iterations"] = 1000
    launch_fixture.write_spec()
    with pytest.raises(M.LaunchRefused, match="full preregistered run"):
        M.prepare_launch_plan(launch_fixture.spec_path, "smoke")


def test_frozen_evaluator_interval_cannot_exceed_stage_budget(
    launch_fixture: LaunchFixture,
) -> None:
    launch_fixture.spec["stages"]["canary"][
        "frozen_eval_interval_updates"
    ] = 257
    launch_fixture.write_spec()
    with pytest.raises(M.LaunchRefused, match="exceeds max_iterations"):
        M.prepare_launch_plan(launch_fixture.spec_path, "smoke")


def test_canary_budget_must_schedule_every_contracted_action(
    launch_fixture: LaunchFixture,
) -> None:
    launch_fixture.spec["stages"]["canary"][
        "frozen_eval_interval_updates"
    ] = 64
    launch_fixture.write_spec()
    with pytest.raises(M.LaunchRefused, match="per contracted action"):
        M.prepare_launch_plan(launch_fixture.spec_path, "smoke")


def test_stage_authority_needs_no_invented_runtime_trust_name(
    launch_fixture: LaunchFixture,
) -> None:
    source = (
        launch_fixture.repo / M.EVALUATOR_TRUST_SOURCE
    ).read_text(encoding="utf-8")
    assert "TRUSTED_FROZEN_STAGE_EVALUATOR" not in source
    assert M.prepare_launch_plan(launch_fixture.spec_path, "smoke")


def test_valid_signed_smoke_receipt_binds_real_timestamped_output(
    launch_fixture: LaunchFixture,
) -> None:
    launch_fixture.install_completed_stage("smoke")
    canary = M.prepare_launch_plan(launch_fixture.spec_path, "canary")
    assert canary["canonical_payload"]["predecessor_receipt_sha256"]


@pytest.mark.parametrize(
    "field",
    [
        "sidecar_launch_receipt_sha256",
        "drain_reset_launch_receipt_sha256",
        "frozen_evaluation_runtime_sha256",
        "gpu_roles_sha256",
        "isolated_training_entrypoint_sha256",
    ],
)
def test_predecessor_replay_binds_complete_v4_runtime(
    launch_fixture: LaunchFixture, field: str
) -> None:
    receipt_path = launch_fixture.install_completed_stage("smoke")
    envelope = json.loads(receipt_path.read_text(encoding="utf-8"))
    envelope["payload"][field] = "0" * 64
    launch_fixture._write_signed(
        receipt_path,
        "action_ball_signed_stage_result",
        envelope["payload"],
    )
    launch_fixture.spec["stages"]["canary"]["predecessor_receipt"] = (
        launch_fixture._external_pin(receipt_path)
    )
    launch_fixture.write_spec()
    with pytest.raises(
        M.LaunchRefused,
        match=(
            "lineage|frozen-evaluation runtime|dual-GPU role binding|"
            "isolated training entrypoint"
        ),
    ):
        M.prepare_launch_plan(launch_fixture.spec_path, "canary")


@pytest.mark.parametrize(
    "field,value",
    [
        ("extra_overrides", ["logger=wandb"]),
        ("ground_plant_contract_sha256", "1" * 64),
        ("effective_reward_recipe_sha256", "2" * 64),
        ("ppo_recipe_sha256", "3" * 64),
    ],
)
def test_predecessor_compares_complete_training_recipe(
    launch_fixture: LaunchFixture, field: str, value
) -> None:
    launch_fixture.install_completed_stage("smoke")
    launch_fixture.spec["train"][field] = value
    launch_fixture.write_spec()
    with pytest.raises(
        M.LaunchRefused,
        match=(
            "recipe differs|training_recipe_sha256|effective Reward|"
            "formal fresh launch"
        ),
    ):
        M.prepare_launch_plan(launch_fixture.spec_path, "canary")


def test_stage_receipt_rejects_forged_nontrainer_output_even_when_resigned(
    launch_fixture: LaunchFixture,
) -> None:
    receipt_path = launch_fixture.install_completed_stage("smoke")
    envelope = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload = envelope["payload"]
    fake_dir = launch_fixture.control / (
        "2026-07-29_01-00-00_" + launch_fixture.namespaces["smoke"].name
    )
    fake_dir.mkdir()
    payload["trainer_output"]["rsl_log_dir"] = str(fake_dir)
    launch_fixture._write_signed(
        receipt_path, "action_ball_signed_stage_result", payload
    )
    launch_fixture.spec["stages"]["canary"]["predecessor_receipt"] = (
        launch_fixture._external_pin(receipt_path)
    )
    launch_fixture.write_spec()
    with pytest.raises(M.LaunchRefused, match="outside|escapes"):
        M.prepare_launch_plan(launch_fixture.spec_path, "canary")


def test_existing_rsl_suffix_is_spent_before_namespace_claim(
    launch_fixture: LaunchFixture,
) -> None:
    spent = (
        launch_fixture.repo
        / "hope_training/whole_body_tracking/logs/rsl_rl"
        / M.ACTION_BALL_EXPERIMENT_NAME
        / f"2026-07-29_01-00-00_{launch_fixture.namespaces['smoke'].name}"
    )
    spent.mkdir(parents=True)
    with pytest.raises(M.LaunchRefused, match="permanently spent"):
        M.prepare_launch_plan(launch_fixture.spec_path, "smoke")
    assert not launch_fixture.namespaces["smoke"].exists()


def test_claim_specific_confirmation_rejects_static_or_wrong_token(
    launch_fixture: LaunchFixture,
) -> None:
    plan = M.prepare_launch_plan(launch_fixture.spec_path, "smoke")
    with pytest.raises(M.LaunchRefused, match="confirmation claim"):
        M.launch_from_spec(
            launch_fixture.spec_path,
            "smoke",
            "0" * 64,
        )
    assert not launch_fixture.namespaces["smoke"].exists()
    assert plan["launch_claim_sha256"] != "0" * 64


def test_exact_claim_confirmation_rechecked_under_lock(
    launch_fixture: LaunchFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = M.prepare_launch_plan(launch_fixture.spec_path, "smoke")
    lock = tmp_path / "lock"
    monkeypatch.setattr(
        M,
        "acquire_gpu_lock",
        lambda _path: os.open(lock, os.O_RDWR | os.O_CREAT, 0o600),
    )
    monkeypatch.setattr(
        M,
        "_verify_live_gpu_empty",
        lambda index, uuid: {
            "nvidia_smi": {
                "name": "nvidia-smi",
                "requested_path": "/usr/bin/nvidia-smi",
                "path": "/usr/bin/nvidia-smi",
                "sha256": "a" * 64,
            },
            "gpu_index": index,
            "gpu_uuid": uuid,
            "compute_process_count": 0,
        },
    )
    monkeypatch.setattr(
        M,
        "_start_stage_supervisor",
        lambda *_args, **_kwargs: {
            "schema_version": 1,
            "kind": "action_ball_launch_accepted",
            "launch_claim_sha256": plan["launch_claim_sha256"],
        },
    )
    accepted = M.launch_from_spec(
        launch_fixture.spec_path,
        "smoke",
        plan["launch_claim_sha256"],
    )
    assert accepted["launch_claim_sha256"] == plan["launch_claim_sha256"]
    assert (launch_fixture.namespaces["smoke"] / "launch_claim.json").is_file()


def test_gpu_lock_must_preexist_and_is_never_created_or_truncated(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.lock"
    with pytest.raises(M.LaunchRefused, match="already exist"):
        M.acquire_gpu_lock(missing)
    assert not missing.exists()

    lock = tmp_path / "gpu.lock"
    lock.write_bytes(b"operator-owned-lock\n")
    before = lock.stat()
    descriptor = M.acquire_gpu_lock(lock)
    try:
        opened = os.fstat(descriptor)
        assert (opened.st_dev, opened.st_ino) == (
            before.st_dev,
            before.st_ino,
        )
        assert lock.read_bytes() == b"operator-owned-lock\n"
        with pytest.raises(M.LaunchRefused, match="already owned"):
            M.acquire_gpu_lock(lock)
    finally:
        os.close(descriptor)
    after = lock.stat()
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )


def test_gpu_lock_refuses_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.lock"
    target.write_bytes(b"do-not-follow\n")
    link = tmp_path / "gpu.lock"
    link.symlink_to(target)
    with pytest.raises(M.LaunchRefused, match="regular non-symlink"):
        M.acquire_gpu_lock(link)


def test_gpu_lock_replacement_before_post_flock_binding_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = tmp_path / "gpu.lock"
    lock.write_bytes(b"original\n")
    real_lstat = Path.lstat
    calls = {"count": 0}

    def replacing_lstat(path):
        if path == lock:
            calls["count"] += 1
            if calls["count"] == 3:
                path.unlink()
                path.write_bytes(b"replacement\n")
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", replacing_lstat)
    with pytest.raises(M.LaunchRefused, match="changed before ownership"):
        M.acquire_gpu_lock(lock)
    assert lock.read_bytes() == b"replacement\n"


def test_acceptance_ack_chain_binds_ready_and_both_processes() -> None:
    claim_sha = "1" * 64
    ready_sha = "2" * 64
    intent_sha = "3" * 64
    ack_sha = "4" * 64
    accepted_sha = "5" * 64
    live_gpu_sha = "8" * 64
    processes = {
        "evaluator": {
            "pid": 101,
            "pgid": 101,
            "starttime_ticks": 1001,
            "argv_sha256": "6" * 64,
        },
        "trainer": {
            "pid": 102,
            "pgid": 102,
            "starttime_ticks": 1002,
            "argv_sha256": "7" * 64,
        },
    }
    intent_ack = {
        "schema_version": 1,
        "kind": "action_ball_stage_supervisor_accept_ack",
        "launch_claim_sha256": claim_sha,
        "supervisor_ready_sha256": ready_sha,
        "accept_intent_sha256": intent_sha,
        "live_gpu_admission_sha256": live_gpu_sha,
    }
    assert (
        M._validate_supervisor_accept_ack(
            intent_ack,
            launch_claim_sha256=claim_sha,
            supervisor_ready_sha256=ready_sha,
            accept_intent_sha256=intent_sha,
            live_gpu_admission_sha256=live_gpu_sha,
        )
        == intent_ack
    )
    commit_ack = {
        "schema_version": 1,
        "kind": "action_ball_stage_supervisor_launch_commit_ack",
        "launch_claim_sha256": claim_sha,
        "supervisor_ready_sha256": ready_sha,
        "accept_intent_sha256": intent_sha,
        "supervisor_accept_ack_sha256": ack_sha,
        "launch_accepted_sha256": accepted_sha,
        "live_gpu_admission_sha256": live_gpu_sha,
        "processes": processes,
    }
    assert (
        M._validate_supervisor_launch_commit_ack(
            commit_ack,
            launch_claim_sha256=claim_sha,
            supervisor_ready_sha256=ready_sha,
            accept_intent_sha256=intent_sha,
            supervisor_accept_ack_sha256=ack_sha,
            launch_accepted_sha256=accepted_sha,
            live_gpu_admission_sha256=live_gpu_sha,
            ready_processes=processes,
        )
        == commit_ack
    )
    drifted = json.loads(json.dumps(commit_ack))
    drifted["processes"]["trainer"]["starttime_ticks"] += 1
    with pytest.raises(M.LaunchRefused, match="process identities"):
        M._validate_supervisor_launch_commit_ack(
            drifted,
            launch_claim_sha256=claim_sha,
            supervisor_ready_sha256=ready_sha,
            accept_intent_sha256=intent_sha,
            supervisor_accept_ack_sha256=ack_sha,
            launch_accepted_sha256=accepted_sha,
            live_gpu_admission_sha256=live_gpu_sha,
            ready_processes=processes,
        )


def test_supervisor_ready_binds_exact_dual_process_argv_and_leaders(
    launch_fixture: LaunchFixture,
) -> None:
    plan = M.prepare_launch_plan(launch_fixture.spec_path, "smoke")
    payload = plan["canonical_payload"]
    namespace = Path(payload["namespace"])
    namespace.mkdir()
    processes = {}
    for offset, role in enumerate(("evaluator", "trainer"), start=1):
        pid = 5000 + offset
        leader_path = namespace / f"{role}_leader_identity.json"
        leader_sha = _write_json(
            leader_path,
            {
                "schema_version": 1,
                "kind": "leader_identity",
                "leader": {
                    "pid": pid,
                    "pgid": pid,
                    "starttime_ticks": 9000 + offset,
                },
            },
        )
        processes[role] = {
            "pid": pid,
            "pgid": pid,
            "starttime_ticks": 9000 + offset,
            "argv_sha256": M.canonical_sha256(
                (
                    payload["sidecar_argv"]
                    if role == "evaluator"
                    else plan["argv"]
                )
            ),
            "returncode": None,
            "leader_receipt": str(leader_path),
            "leader_receipt_sha256": leader_sha,
            "term_receipt": "",
            "term_receipt_sha256": "",
            "kill_receipt": "",
            "kill_receipt_sha256": "",
        }
    ready = {
        "schema_version": 1,
        "kind": "action_ball_stage_supervisor_ready",
        "ready_utc": "2026-07-29T12:00:00Z",
        "claim_sha256": plan["launch_claim_sha256"],
        "source_commit_sha": payload["source_commit_sha"],
        "stage": payload["stage"],
        "namespace": str(namespace),
        "gpu_roles": payload["gpus"],
        "sidecar_ready": {
            "schema_version": 1,
            "kind": "whole_body_tracking.action_ball.formal_sidecar_ready",
            "owner_id": payload["frozen_evaluation_runtime"]["owner_id"],
            "run_id": payload["frozen_evaluation_runtime"]["run_id"],
            "backend": "formal",
            "device": "cuda:0",
            "launch_receipt_canonical_sha256": payload[
                "sidecar_launch_receipt"
            ]["content_sha256"],
        },
        "sidecar_heartbeat_initial": _initial_heartbeat(
            payload, processes["evaluator"]["pid"]
        ),
        "trainer_learning_line": "Learning iteration 1/2",
        "processes": processes,
        "logs": {
            "evaluator": str(namespace / "evaluator.log"),
            "trainer": str(namespace / "train.log"),
        },
    }
    assert (
        M._validate_supervisor_ready_receipt(
            ready, plan=plan, namespace=namespace
        )
        == ready
    )
    drifted = json.loads(json.dumps(ready))
    drifted["processes"]["trainer"]["argv_sha256"] = "0" * 64
    with pytest.raises(M.LaunchRefused, match="process identity"):
        M._validate_supervisor_ready_receipt(
            drifted, plan=plan, namespace=namespace
        )


def test_launcher_runs_control_intent_ack_commit_ack_protocol(
    launch_fixture: LaunchFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = M.prepare_launch_plan(launch_fixture.spec_path, "smoke")
    namespace = M._claim_namespace(plan)
    live_gpu_admission_sha = _install_live_gpu_admission(plan)
    payload = plan["canonical_payload"]
    thread_errors = []
    observed = {}

    class FakeExact:
        @staticmethod
        def bind_leader(_proc_root, pid, pgid, output):
            assert pid == pgid
            value = {
                "schema_version": 1,
                "kind": "leader_identity",
                "leader": {
                    "pid": pid,
                    "pgid": pgid,
                    "starttime_ticks": 424242,
                },
            }
            _write_json(output, value)
            return value

    class FakeProcess:
        pid = 4242

        def __init__(self, argv, **kwargs):
            del kwargs
            observed["argv"] = list(argv)
            gate_fd = os.dup(int(argv[6]))
            supervisor_command = NOSITE.validate_exact_nosite_argv(argv[7:])
            supervisor_argv = list(
                supervisor_command.contract["entrypoint_argv"]
            )
            observed["supervisor_argv"] = supervisor_argv
            control_index = supervisor_argv.index("--launcher-control-fd")
            control_fd = os.dup(
                int(supervisor_argv[control_index + 1])
            )

            def supervisor_protocol():
                try:
                    assert os.read(gate_fd, 1) == b"G"
                    os.close(gate_fd)
                    processes = {}
                    for offset, role in enumerate(
                        ("evaluator", "trainer"), start=1
                    ):
                        pid = 6000 + offset
                        leader_path = (
                            namespace / f"{role}_leader_identity.json"
                        )
                        leader_sha = _write_json(
                            leader_path,
                            {
                                "schema_version": 1,
                                "kind": "leader_identity",
                                "leader": {
                                    "pid": pid,
                                    "pgid": pid,
                                    "starttime_ticks": 7000 + offset,
                                },
                            },
                        )
                        processes[role] = {
                            "pid": pid,
                            "pgid": pid,
                            "starttime_ticks": 7000 + offset,
                            "argv_sha256": M.canonical_sha256(
                                (
                                    payload["sidecar_argv"]
                                    if role == "evaluator"
                                    else plan["argv"]
                                )
                            ),
                            "returncode": None,
                            "leader_receipt": str(leader_path),
                            "leader_receipt_sha256": leader_sha,
                            "term_receipt": "",
                            "term_receipt_sha256": "",
                            "kill_receipt": "",
                            "kill_receipt_sha256": "",
                        }
                    ready = {
                        "schema_version": 1,
                        "kind": "action_ball_stage_supervisor_ready",
                        "ready_utc": "2026-07-29T12:00:00Z",
                        "claim_sha256": plan["launch_claim_sha256"],
                        "source_commit_sha": payload["source_commit_sha"],
                        "stage": payload["stage"],
                        "namespace": str(namespace),
                        "gpu_roles": payload["gpus"],
                        "sidecar_ready": {
                            "schema_version": 1,
                            "kind": (
                                "whole_body_tracking.action_ball."
                                "formal_sidecar_ready"
                            ),
                            "owner_id": payload[
                                "frozen_evaluation_runtime"
                            ]["owner_id"],
                            "run_id": payload[
                                "frozen_evaluation_runtime"
                            ]["run_id"],
                            "backend": "formal",
                            "device": "cuda:0",
                            "launch_receipt_canonical_sha256": payload[
                                "sidecar_launch_receipt"
                            ]["content_sha256"],
                        },
                        "sidecar_heartbeat_initial": _initial_heartbeat(
                            payload, processes["evaluator"]["pid"]
                        ),
                        "trainer_learning_line": "Learning iteration 1/2",
                        "processes": processes,
                        "logs": {
                            "evaluator": str(namespace / "evaluator.log"),
                            "trainer": str(namespace / "train.log"),
                        },
                    }
                    ready_path = namespace / "supervisor_ready.json"
                    ready_sha = _write_json(ready_path, ready)
                    intent_path = namespace / "launch_accept_intent.json"
                    while not intent_path.is_file():
                        time.sleep(0.001)
                    assert os.read(control_fd, 1) == b"A"
                    intent_sha = M.sha256_file(intent_path)
                    ack_path = namespace / "launch_accept_ack.json"
                    ack_sha = _write_json(
                        ack_path,
                        {
                            "schema_version": 1,
                            "kind": (
                                "action_ball_stage_supervisor_accept_ack"
                            ),
                            "launch_claim_sha256": plan[
                                "launch_claim_sha256"
                            ],
                            "supervisor_ready_sha256": ready_sha,
                            "accept_intent_sha256": intent_sha,
                            "live_gpu_admission_sha256": (
                                live_gpu_admission_sha
                            ),
                        },
                    )
                    accepted_path = namespace / "launch_accepted.json"
                    while not accepted_path.is_file():
                        time.sleep(0.001)
                    _write_json(
                        namespace / "launch_commit_ack.json",
                        {
                            "schema_version": 1,
                            "kind": (
                                "action_ball_stage_supervisor_"
                                "launch_commit_ack"
                            ),
                            "launch_claim_sha256": plan[
                                "launch_claim_sha256"
                            ],
                            "supervisor_ready_sha256": ready_sha,
                            "accept_intent_sha256": intent_sha,
                            "supervisor_accept_ack_sha256": ack_sha,
                            "launch_accepted_sha256": M.sha256_file(
                                accepted_path
                            ),
                            "live_gpu_admission_sha256": (
                                live_gpu_admission_sha
                            ),
                            "processes": processes,
                        },
                    )
                    os.close(control_fd)
                except BaseException as exc:
                    thread_errors.append(exc)

            self.thread = threading.Thread(
                target=supervisor_protocol, daemon=True
            )
            self.thread.start()

        @staticmethod
        def poll():
            return None

    monkeypatch.setattr(
        M,
        "_verify_repo_blob",
        lambda *_args, **_kwargs: (
            LAUNCHER_PATH,
            M.LAUNCHER_SOURCE,
            M.sha256_file(LAUNCHER_PATH),
            "100644",
        ),
    )
    monkeypatch.setattr(M, "_load_exact_process_group", lambda _path: FakeExact())
    monkeypatch.setattr(
        M,
        "_supervisor_proc_identity",
        lambda **_kwargs: {
            "executable_path": payload["isaac_python_runtime"][
                "resolved_path"
            ],
            "executable_sha256": payload["isaac_python_runtime"]["sha256"],
            "cgroup_sha256": "9" * 64,
        },
    )
    monkeypatch.setattr(M.subprocess, "Popen", FakeProcess)
    lock_path = tmp_path / "locks"
    lock_path.mkdir()
    trainer_lock = os.open(lock_path / "trainer", os.O_RDWR | os.O_CREAT, 0o600)
    evaluator_lock = os.open(
        lock_path / "evaluator", os.O_RDWR | os.O_CREAT, 0o600
    )
    try:
        accepted = M._start_stage_supervisor(
            plan,
            trainer_lock_fd=trainer_lock,
            evaluator_lock_fd=evaluator_lock,
            live_gpu_admission_sha256=live_gpu_admission_sha,
        )
    finally:
        os.close(trainer_lock)
        os.close(evaluator_lock)
    assert accepted["launch_claim_sha256"] == plan["launch_claim_sha256"]
    assert (namespace / "launch_commit_ack.json").is_file()
    assert "--launcher-control-fd" in observed["supervisor_argv"]
    assert not thread_errors


def test_invalid_ready_receipt_cancels_only_created_supervisor(
    launch_fixture: LaunchFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = M.prepare_launch_plan(launch_fixture.spec_path, "smoke")
    namespace = M._claim_namespace(plan)
    live_gpu_admission_sha = _install_live_gpu_admission(plan)
    payload = plan["canonical_payload"]
    created = []
    cancelled = []

    class FakeExact:
        @staticmethod
        def bind_leader(_proc_root, pid, pgid, output):
            value = {
                "schema_version": 1,
                "kind": "leader_identity",
                "leader": {
                    "pid": pid,
                    "pgid": pgid,
                    "starttime_ticks": 333,
                },
            }
            _write_json(output, value)
            return value

    class FakeProcess:
        pid = 333

        def __init__(self, argv, **kwargs):
            del kwargs
            gate_fd = os.dup(int(argv[6]))
            created.append(self)

            def publish_bad_ready():
                if os.read(gate_fd, 1) == b"G":
                    _write_json(namespace / "supervisor_ready.json", {})
                os.close(gate_fd)

            self.thread = threading.Thread(
                target=publish_bad_ready, daemon=True
            )
            self.thread.start()

        @staticmethod
        def poll():
            return None

    monkeypatch.setattr(
        M,
        "_verify_repo_blob",
        lambda *_args, **_kwargs: (
            LAUNCHER_PATH,
            M.LAUNCHER_SOURCE,
            M.sha256_file(LAUNCHER_PATH),
            "100644",
        ),
    )
    monkeypatch.setattr(M, "_load_exact_process_group", lambda _path: FakeExact())
    monkeypatch.setattr(
        M,
        "_supervisor_proc_identity",
        lambda **_kwargs: {
            "executable_path": payload["isaac_python_runtime"][
                "resolved_path"
            ],
            "executable_sha256": payload["isaac_python_runtime"]["sha256"],
            "cgroup_sha256": "8" * 64,
        },
    )
    monkeypatch.setattr(M.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        M,
        "_cancel_and_reap_supervisor",
        lambda **kwargs: cancelled.append(kwargs["process"]),
    )
    lock_root = tmp_path / "locks"
    lock_root.mkdir()
    trainer_lock = os.open(
        lock_root / "trainer", os.O_RDWR | os.O_CREAT, 0o600
    )
    evaluator_lock = os.open(
        lock_root / "evaluator", os.O_RDWR | os.O_CREAT, 0o600
    )
    try:
        with pytest.raises(M.LaunchRefused, match="keys differ"):
            M._start_stage_supervisor(
                plan,
                trainer_lock_fd=trainer_lock,
                evaluator_lock_fd=evaluator_lock,
                live_gpu_admission_sha256=live_gpu_admission_sha,
            )
    finally:
        os.close(trainer_lock)
        os.close(evaluator_lock)
    assert len(created) == 1
    assert cancelled == created
    assert not (namespace / "launch_accepted.json").exists()


def test_canary_then_full_long_preserves_exact_lineage(
    launch_fixture: LaunchFixture,
) -> None:
    launch_fixture.install_completed_stage("smoke")
    launch_fixture.install_completed_stage("canary")
    long_plan = M.prepare_launch_plan(launch_fixture.spec_path, "long")
    assert long_plan["canonical_payload"]["stage_budget"]["num_envs"] >= 4096
    assert long_plan["canonical_payload"]["stage_budget"]["max_iterations"] >= 20001


def test_bank_resume_and_launcher_owned_injection_fail_closed(
    launch_fixture: LaunchFixture,
) -> None:
    for override in (
        "resume=true",
        "question_bank=/tmp/x.json",
        "expected_effective_reward_recipe_sha256=" + "0" * 64,
        "task.experiment_name=other",
        "task.table_obstacle=false",
        "task.actions.qdes_clamp=false",
        "task.rewards.death_penalty_weight=0",
        "task.env.episode_length_s=1",
        "algo.learning_rate=0",
        "hydra.searchpath=[file:///tmp/evil]",
        "training_launch_claim_path=/tmp/evil.json",
        "logger=wandb",
    ):
        launch_fixture.spec["train"]["extra_overrides"] = [override]
        launch_fixture.write_spec()
        with pytest.raises(M.LaunchRefused):
            M.prepare_launch_plan(launch_fixture.spec_path, "smoke")


def _validate_fixture_manifest(
    launch_fixture: LaunchFixture, document: dict
) -> None:
    M._validate_manifest(
        document,
        checkout=launch_fixture.repo,
        source_commit=launch_fixture.commit,
        order=tuple(launch_fixture.order),
        ordered_action_uids=tuple(
            binding["action_uid"] for binding in launch_fixture.bindings
        ),
        scope="upper",
        mobility_mode="no_move",
        prototype_relative=launch_fixture.prototype_relative,
        prototype_sha256=launch_fixture.prototype_sha,
    )


def test_launcher_rejects_nonformal_manifest_holdout(
    launch_fixture: LaunchFixture,
) -> None:
    document = json.loads(
        (launch_fixture.repo / launch_fixture.manifest_relative).read_text()
    )
    document["holdout"]["samples_per_action"] = 512
    with pytest.raises(M.LaunchRefused, match="formal per-action window"):
        _validate_fixture_manifest(launch_fixture, document)


def test_launcher_rejects_unknown_manifest_key_and_non_n1_counter_objective(
    launch_fixture: LaunchFixture,
) -> None:
    document = json.loads(
        (launch_fixture.repo / launch_fixture.manifest_relative).read_text()
    )
    document["fake_authorization"] = True
    with pytest.raises(M.LaunchRefused, match="keys differ"):
        _validate_fixture_manifest(launch_fixture, document)
    document.pop("fake_authorization")
    document["counter_rally_objective"] = {}
    with pytest.raises(M.LaunchRefused, match="restricted to exact N=1"):
        _validate_fixture_manifest(launch_fixture, document)
