import importlib.util
import inspect
import json
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


WBT = Path(__file__).resolve().parents[1]
SIDECAR_PATH = WBT / "scripts" / "action_ball_frozen_eval_sidecar.py"


def _load_sidecar():
    name = "action_ball_formal_sidecar_unit"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, SIDECAR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


S = _load_sidecar()


def _minimal_launch_claim():
    payload = {
        "namespace": "/fixture",
        "argv_without_launch_claim": [
            "train.py",
            "++training_launch_claim_path=/fixture/launch_claim.json",
        ]
    }
    claim_sha256 = S.inbox_protocol.canonical_sha256(payload)
    return {
        "schema_version": 3,
        "kind": "action_ball_no_clobber_launch_claim_v3",
        "launch_claim_sha256": claim_sha256,
        "canonical_payload": payload,
        "argv": [
            *payload["argv_without_launch_claim"],
            "++training_launch_claim_sha256={}".format(claim_sha256),
        ],
        "confirmation_claim_sha256": claim_sha256,
    }


def test_exact_resume_rejects_live_module_origin_drift_before_isaac_import(
    monkeypatch,
):
    events = []

    def reject_drift(*, claim_payload):
        events.append("inventory")
        raise RuntimeError("module origin drift")

    class MustNotImportIsaac:
        def __init__(self, **_kwargs):
            events.append("isaac")
            raise AssertionError(
                "Isaac evaluator was constructed before inventory passed"
            )

    monkeypatch.setattr(
        S, "_live_verify_runtime_inventory", reject_drift
    )
    monkeypatch.setattr(
        S, "FormalIsaacEvaluator", MustNotImportIsaac
    )
    with pytest.raises(RuntimeError, match="module origin drift"):
        S.build_exact_resume_runtime_from_claim(
            claim_document=_minimal_launch_claim(),
            final_checkpoint_path="/fixture/model_0.pt",
            device="cuda:0",
        )
    assert events == ["inventory"]


def _inventory_claim_fixture(tmp_path):
    script = tmp_path / S.RUNTIME_INVENTORY_SOURCE
    script.parent.mkdir(parents=True)
    script.write_bytes(b"# pinned inventory verifier\n")
    requested = str(Path(sys.executable).absolute())
    content = {
        "python": {"requested_path": requested},
        "unicode_probe_path": str(tmp_path / "乒乓"),
    }
    inventory = {
        "schema_version": 1,
        "kind": "action_ball_runtime_inventory_v1",
        "content": content,
        "content_sha256": S._runtime_inventory_content_sha256(
            content
        ),
    }
    inventory_path = tmp_path / "runtime_inventory.json"
    inventory_path.write_text(
        json.dumps(
            inventory,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    inventory_path.chmod(0o600)
    inventory_receipt = S.inbox_protocol.artifact_receipt(
        inventory_path
    )
    claim_payload = {
        "source_checkout": str(tmp_path),
        "runtime_code_sha256": {
            S.RUNTIME_INVENTORY_SOURCE: (
                S.inbox_protocol.artifact_receipt(script)["sha256"]
            ),
        },
        "isaac_python_runtime": {
            "path": requested,
            "runtime_inventory": {
                "path": str(inventory_path),
                "file_sha256": inventory_receipt["sha256"],
                "content_sha256": inventory["content_sha256"],
                "kind": inventory["kind"],
            },
        },
    }
    return claim_payload, script, inventory, inventory_receipt


def test_live_runtime_inventory_proof_binds_exact_verifier_result(
    tmp_path, monkeypatch
):
    payload, script, inventory, inventory_receipt = (
        _inventory_claim_fixture(tmp_path)
    )
    expected_result = {
        "ok": True,
        "kind": inventory["kind"],
        "content_sha256": inventory["content_sha256"],
        "receipt_path": inventory_receipt["path"],
        "receipt_sha256": inventory_receipt["sha256"],
    }

    def run(argv, **kwargs):
        assert argv[0] == str(Path(sys.executable).absolute())
        assert argv[1:4] == ["-I", "-B", "-c"]
        assert argv[4] == S._INVENTORY_STDIN_WRAPPER
        assert argv[5] == str(script)
        assert argv[6:] == [
            "verify",
            "--receipt",
            inventory_receipt["path"],
        ]
        assert kwargs["cwd"] == "/"
        assert kwargs["input"] == script.read_bytes()
        return SimpleNamespace(
            returncode=0,
            stdout=(
                json.dumps(
                    expected_result,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                + b"\n"
            ),
            stderr=b"",
        )

    monkeypatch.setattr(S.subprocess, "run", run)
    proof = S._live_verify_runtime_inventory(
        claim_payload=payload
    )
    assert proof["content"]["verification_result"] == expected_result
    assert proof["content"]["verifier_source"]["path"] == str(script)
    assert proof["content"]["inventory_artifact"] == inventory_receipt
    assert proof["content_sha256"] == (
        S.inbox_protocol.canonical_sha256(proof["content"])
    )
    wrong_runtime = copy.deepcopy(payload)
    wrong_runtime["isaac_python_runtime"]["path"] = "/wrong/python"
    with pytest.raises(RuntimeError, match="differs"):
        S._validate_preimport_live_inventory_verification(
            proof,
            claim_payload=wrong_runtime,
        )
    wrong_kind = copy.deepcopy(payload)
    wrong_kind["isaac_python_runtime"]["runtime_inventory"][
        "kind"
    ] = "wrong_kind"
    with pytest.raises(RuntimeError, match="differs"):
        S._validate_preimport_live_inventory_verification(
            proof,
            claim_payload=wrong_kind,
        )


def test_live_runtime_inventory_executes_pinned_bytes_and_rejects_path_swap(
    tmp_path, monkeypatch
):
    payload, script, inventory, inventory_receipt = (
        _inventory_claim_fixture(tmp_path)
    )
    admitted_script = script.read_bytes()
    expected_result = {
        "ok": True,
        "kind": inventory["kind"],
        "content_sha256": inventory["content_sha256"],
        "receipt_path": inventory_receipt["path"],
        "receipt_sha256": inventory_receipt["sha256"],
    }

    def swap_before_execution(_argv, **kwargs):
        assert kwargs["input"] == admitted_script
        script.write_bytes(b"# swapped after admission\n")
        return SimpleNamespace(
            returncode=0,
            stdout=(
                json.dumps(
                    expected_result,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                + b"\n"
            ),
            stderr=b"",
        )

    monkeypatch.setattr(
        S.subprocess, "run", swap_before_execution
    )
    with pytest.raises(RuntimeError, match="changed during execution"):
        S._live_verify_runtime_inventory(claim_payload=payload)


def test_live_runtime_inventory_rejects_requested_interpreter_drift(
    tmp_path, monkeypatch
):
    payload, _script, inventory, _receipt = (
        _inventory_claim_fixture(tmp_path)
    )
    inventory["content"]["python"]["requested_path"] = "/wrong/python"
    inventory["content_sha256"] = (
        S._runtime_inventory_content_sha256(inventory["content"])
    )
    inventory_path = Path(
        payload["isaac_python_runtime"]["runtime_inventory"]["path"]
    )
    inventory_path.write_text(
        json.dumps(
            inventory,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    receipt = S.inbox_protocol.artifact_receipt(inventory_path)
    identity = payload["isaac_python_runtime"]["runtime_inventory"]
    identity["file_sha256"] = receipt["sha256"]
    identity["content_sha256"] = inventory["content_sha256"]
    monkeypatch.setattr(
        S.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("drift must reject before verifier process")
        ),
    )
    with pytest.raises(RuntimeError, match="interpreter differs"):
        S._live_verify_runtime_inventory(claim_payload=payload)


def test_live_runtime_inventory_propagates_module_origin_rejection(
    tmp_path, monkeypatch
):
    payload, _script, _inventory, _receipt = (
        _inventory_claim_fixture(tmp_path)
    )
    monkeypatch.setattr(
        S.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=2,
            stdout=b"",
            stderr=b"module origin differs from frozen receipt",
        ),
    )
    with pytest.raises(
        RuntimeError, match="module origin differs"
    ):
        S._live_verify_runtime_inventory(claim_payload=payload)


def test_live_runtime_inventory_second_restore_reuses_one_full_scan(
    tmp_path, monkeypatch
):
    payload, _script, inventory, inventory_receipt = (
        _inventory_claim_fixture(tmp_path)
    )
    expected_result = {
        "ok": True,
        "kind": inventory["kind"],
        "content_sha256": inventory["content_sha256"],
        "receipt_path": inventory_receipt["path"],
        "receipt_sha256": inventory_receipt["sha256"],
    }
    calls = []

    def run(*_args, **_kwargs):
        assert _kwargs["timeout"] == 600
        calls.append("full_recursive_inventory")
        return SimpleNamespace(
            returncode=0,
            stdout=(
                json.dumps(
                    expected_result,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                + b"\n"
            ),
            stderr=b"",
        )

    monkeypatch.setattr(S.subprocess, "run", run)
    with S._LIVE_RUNTIME_INVENTORY_CACHE_LOCK:
        S._LIVE_RUNTIME_INVENTORY_CACHE.clear()
    first = S._live_verify_runtime_inventory_cached(
        claim_payload=payload
    )
    second = S._live_verify_runtime_inventory_cached(
        claim_payload=payload
    )
    assert first == second
    assert calls == ["full_recursive_inventory"]


def test_public_factory_accepts_only_deep_bound_preimport_inventory_proof(
    tmp_path, monkeypatch
):
    payload, _script, inventory, inventory_receipt = (
        _inventory_claim_fixture(tmp_path)
    )
    expected_result = {
        "ok": True,
        "kind": inventory["kind"],
        "content_sha256": inventory["content_sha256"],
        "receipt_path": inventory_receipt["path"],
        "receipt_sha256": inventory_receipt["sha256"],
    }
    monkeypatch.setattr(
        S.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                json.dumps(
                    expected_result,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                + b"\n"
            ),
            stderr=b"",
        ),
    )
    proof = S._live_verify_runtime_inventory(claim_payload=payload)
    namespace = tmp_path / "namespace"
    payload["namespace"] = str(namespace)
    payload["argv_without_launch_claim"] = [
        "train.py",
        "++training_launch_claim_path={}".format(
            namespace / "launch_claim.json"
        ),
    ]
    claim_sha = S.inbox_protocol.canonical_sha256(payload)
    claim = {
        "schema_version": 3,
        "kind": "action_ball_no_clobber_launch_claim_v3",
        "launch_claim_sha256": claim_sha,
        "canonical_payload": payload,
        "argv": [
            *payload["argv_without_launch_claim"],
            "++training_launch_claim_sha256={}".format(claim_sha),
        ],
        "confirmation_claim_sha256": claim_sha,
    }
    calls = []

    class Evaluator:
        def __init__(self, *, device):
            calls.append(("construct", device))

        def build_exact_resume_runtime_from_claim(
            self,
            *,
            claim_document,
            final_checkpoint_path,
            _live_inventory_verification,
        ):
            calls.append(
                (
                    "build",
                    claim_document,
                    final_checkpoint_path,
                    _live_inventory_verification,
                )
            )
            return "runtime"

        def _close_environment(self):
            calls.append(("close",))

    with S._LIVE_RUNTIME_INVENTORY_CACHE_LOCK:
        S._LIVE_RUNTIME_INVENTORY_CACHE.clear()
    monkeypatch.setattr(S, "FormalIsaacEvaluator", Evaluator)
    assert S.build_exact_resume_runtime_from_claim(
        claim_document=claim,
        final_checkpoint_path="/fixture/model_0.pt",
        device="cuda:0",
        _preimport_live_inventory_verification=proof,
    ) == "runtime"
    assert calls[0] == ("construct", "cuda:0")
    assert calls[1][0] == "build"
    assert calls[1][3] == proof

    forged = copy.deepcopy(proof)
    forged["content"]["verification_result"]["ok"] = False
    forged["content_sha256"] = S.inbox_protocol.canonical_sha256(
        forged["content"]
    )
    calls.clear()
    with S._LIVE_RUNTIME_INVENTORY_CACHE_LOCK:
        S._LIVE_RUNTIME_INVENTORY_CACHE.clear()
    with pytest.raises(RuntimeError, match="differs"):
        S.build_exact_resume_runtime_from_claim(
            claim_document=claim,
            final_checkpoint_path="/fixture/model_0.pt",
            device="cuda:0",
            _preimport_live_inventory_verification=forged,
        )
    assert calls == []


@pytest.mark.parametrize(
    "replacement",
    [
        "training_launch_claim_path=/tmp/evil.json",
        "+training_launch_claim_path=/tmp/evil.json",
        "+++training_launch_claim_path=/tmp/evil.json",
        "~training_launch_claim_path",
        "training_launch_claim_sha256=" + "a" * 64,
        "+training_launch_claim_sha256=" + "a" * 64,
        "+++training_launch_claim_sha256=" + "a" * 64,
        "~training_launch_claim_sha256",
        17,
    ],
)
def test_preimport_claim_rejects_noncanonical_hydra_bindings(
    replacement,
):
    claim = _minimal_launch_claim()
    payload = claim["canonical_payload"]
    payload["argv_without_launch_claim"].append(replacement)
    claim_sha = S.inbox_protocol.canonical_sha256(payload)
    claim["launch_claim_sha256"] = claim_sha
    claim["confirmation_claim_sha256"] = claim_sha
    claim["argv"] = [
        *payload["argv_without_launch_claim"],
        "++training_launch_claim_sha256={}".format(claim_sha),
    ]
    with pytest.raises(RuntimeError, match="canonical binding"):
        S._preimport_claim_payload(claim)


def test_preimport_claim_requires_one_exact_namespace_path_binding():
    claim = _minimal_launch_claim()
    payload = claim["canonical_payload"]
    payload["argv_without_launch_claim"].append(
        payload["argv_without_launch_claim"][-1]
    )
    claim_sha = S.inbox_protocol.canonical_sha256(payload)
    claim["launch_claim_sha256"] = claim_sha
    claim["confirmation_claim_sha256"] = claim_sha
    claim["argv"] = [
        *payload["argv_without_launch_claim"],
        "++training_launch_claim_sha256={}".format(claim_sha),
    ]
    with pytest.raises(RuntimeError, match="canonical binding"):
        S._preimport_claim_payload(claim)

    claim = _minimal_launch_claim()
    payload = claim["canonical_payload"]
    payload["argv_without_launch_claim"] = ["train.py"]
    claim_sha = S.inbox_protocol.canonical_sha256(payload)
    claim["launch_claim_sha256"] = claim_sha
    claim["confirmation_claim_sha256"] = claim_sha
    claim["argv"] = [
        "train.py",
        "++training_launch_claim_sha256={}".format(claim_sha),
    ]
    with pytest.raises(RuntimeError, match="canonical binding"):
        S._preimport_claim_payload(claim)


class _Normalizer:
    def state_dict(self):
        return {"mean": [1.0], "variance": [2.0]}


def test_normalizer_binding_rejects_absent_runner_attribute():
    with pytest.raises(RuntimeError, match="absent"):
        S.FormalIsaacEvaluator._normalizer_payload(
            SimpleNamespace(), "privileged_obs_normalizer"
        )


def test_normalizer_binding_distinguishes_disabled_and_real_state():
    runner = SimpleNamespace(
        privileged_obs_normalizer=_Normalizer(),
        obs_normalizer=None,
    )
    assert S.FormalIsaacEvaluator._normalizer_payload(
        runner, "obs_normalizer"
    ) == {"enabled": False}
    assert S.FormalIsaacEvaluator._normalizer_payload(
        runner, "privileged_obs_normalizer"
    ) == {
        "enabled": True,
        "state": {"mean": [1.0], "variance": [2.0]},
    }


def test_critic_normalizer_resolution_is_version_aware_and_unambiguous():
    privileged = _Normalizer()
    assert S.FormalIsaacEvaluator._critic_normalizer_name(
        SimpleNamespace(privileged_obs_normalizer=privileged)
    ) == "privileged_obs_normalizer"
    assert S.FormalIsaacEvaluator._critic_normalizer_name(
        SimpleNamespace(critic_obs_normalizer=privileged)
    ) == "critic_obs_normalizer"
    assert S.FormalIsaacEvaluator._critic_normalizer_name(
        SimpleNamespace(
            privileged_obs_normalizer=privileged,
            critic_obs_normalizer=privileged,
        )
    ) == "privileged_obs_normalizer"
    with pytest.raises(RuntimeError, match="two different"):
        S.FormalIsaacEvaluator._critic_normalizer_name(
            SimpleNamespace(
                privileged_obs_normalizer=privileged,
                critic_obs_normalizer=_Normalizer(),
            )
        )
    with pytest.raises(RuntimeError, match="no critic"):
        S.FormalIsaacEvaluator._critic_normalizer_name(
            SimpleNamespace()
        )


def _request():
    return {
        "bindings": {
            "action_order": [11, 29],
            "actions": [
                {
                    "action_uid": 11,
                    "motion": {"sha256": "1" * 64},
                },
                {
                    "action_uid": 29,
                    "motion": {"sha256": "2" * 64},
                },
            ],
            "manifest_sha256": "3" * 64,
            "sampler_sha256": "4" * 64,
            "proposal_sampler_contract_sha256": "d" * 64,
            "solver_sha256": "5" * 64,
            "physics_sha256": "6" * 64,
            "curriculum_sha256": "7" * 64,
            "policy_contract_sha256": "8" * 64,
        },
        "target": {
            "action_uid": 29,
            "profile_sha256": "9" * 64,
            "mobility_mode": "no_move",
        },
        "windows": [
            {"role": "scheduler", "proposal_count": 2},
        ],
    }


def _hard_contract():
    return {
        "kind": (
            "whole_body_tracking.RacketTargetCommand."
            "action_ball_hard_contract"
        ),
        "action_uids": [11, 29],
        "mobility_mode": "no_move",
        "manifest": {"file_sha256": "3" * 64},
        "profiles": {
            "sampler_contract_sha256": "4" * 64,
            "frozen_evaluation_proposal_sampler_contract_sha256": (
                "d" * 64
            ),
            "adapter_contract_sha256": "7" * 64,
            "profile_sha256": ["a" * 64, "9" * 64],
        },
        "solver": {"sha256": "5" * 64},
        "physics": {"sha256": "6" * 64},
        "curriculum": {"policy_contract_sha256": "8" * 64},
        "bindings": [
            {
                "action_uid": 11,
                "action_slot": 0,
                "motion_sha256": "1" * 64,
                "profile_sha256": "a" * 64,
            },
            {
                "action_uid": 29,
                "action_slot": 1,
                "motion_sha256": "2" * 64,
                "profile_sha256": "9" * 64,
            },
        ],
    }


class _Term:
    def __init__(self, hard):
        self._hard = hard

    def action_ball_hard_contract(self):
        return self._hard


def test_live_term_identity_accepts_arbitrary_ordered_action_catalog():
    S.FormalIsaacEvaluator._validate_action_ball_term_identity(
        _Term(_hard_contract()), _request()
    )


def test_live_term_identity_rejects_motion_or_target_profile_drift():
    request = _request()
    request["bindings"]["actions"][1]["motion"]["sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="binding"):
        S.FormalIsaacEvaluator._validate_action_ball_term_identity(
            _Term(_hard_contract()), request
        )


def test_formal_attempt_rows_bind_unique_frozen_proposal_receipts():
    request = _request()
    attempts = {
        "scheduler": [
            {
                "proposal_sampler_contract_sha256": "d" * 64,
                "proposal_receipt_sha256": "1" * 64,
            },
            {
                "proposal_sampler_contract_sha256": "d" * 64,
                "proposal_receipt_sha256": "2" * 64,
            },
        ]
    }
    S.FormalIsaacEvaluator._validate_proposal_receipt_bindings(
        request=request,
        attempts_by_role=attempts,
    )
    attempts["scheduler"][1]["proposal_receipt_sha256"] = "1" * 64
    with pytest.raises(RuntimeError, match="repeated"):
        S.FormalIsaacEvaluator._validate_proposal_receipt_bindings(
            request=request,
            attempts_by_role=attempts,
        )
    attempts["scheduler"][1]["proposal_receipt_sha256"] = "2" * 64
    attempts["scheduler"][1][
        "proposal_sampler_contract_sha256"
    ] = "e" * 64
    with pytest.raises(RuntimeError, match="not bound"):
        S.FormalIsaacEvaluator._validate_proposal_receipt_bindings(
            request=request,
            attempts_by_role=attempts,
        )

    request = _request()
    request["target"]["profile_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="target profile"):
        S.FormalIsaacEvaluator._validate_action_ball_term_identity(
            _Term(_hard_contract()), request
        )


def test_training_contract_must_bind_ppo_and_effective_reward():
    bindings = {
        "ppo_recipe_sha256": "a" * 64,
        "reward_sha256": "b" * 64,
    }
    contract = {
        "action_ball_ppo_runner_recipe": {"sha256": "a" * 64},
        "effective_reward_recipe": {"sha256": "b" * 64},
    }
    S.FormalIsaacEvaluator._validate_training_contract_bindings(
        contract, bindings
    )
    contract["effective_reward_recipe"]["sha256"] = "c" * 64
    with pytest.raises(RuntimeError, match="PPO/Reward"):
        S.FormalIsaacEvaluator._validate_training_contract_bindings(
            contract, bindings
        )


def test_formal_sidecar_does_not_resize_exact_resume_environment():
    source = SIDECAR_PATH.read_text(encoding="utf-8")
    assert "scene_cfg.num_envs = proposal_count" not in source
    assert "Do not resize the environment before loading" in source


def test_inbox_protocol_loader_compiles_bound_source_not_pycache():
    source = inspect.getsource(S._load_inbox_protocol)
    assert "spec_from_file_location" not in source
    assert "compile(" in source
    assert "O_NOFOLLOW" in source


class _SnapshotRunner:
    def __init__(self):
        self.alg = SimpleNamespace(policy=torch.nn.Linear(2, 1))
        self.obs_normalizer = torch.nn.BatchNorm1d(2)
        self.privileged_obs_normalizer = torch.nn.BatchNorm1d(3)
        self.current_learning_iteration = 0
        self.preflight_calls = []

    def _preflight_required_exact_resume_checkpoint(
        self, loaded, *, path, load_optimizer
    ):
        self.preflight_calls.append(
            (loaded["iter"], path, load_optimizer)
        )
        return {"validated": True}


def _write_pickle_execution_marker(path):
    Path(path).write_text("executed\n", encoding="utf-8")
    return {}


class _MaliciousCheckpointValue:
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return (_write_pickle_execution_marker, (self.marker,))


def _checkpoint(tmp_path, *, generation=7, nonfinite=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = _SnapshotRunner()
    with torch.no_grad():
        source.alg.policy.weight.fill_(
            float("nan") if nonfinite else 3.0
        )
        source.alg.policy.bias.fill_(4.0)
        source.obs_normalizer.running_mean.fill_(5.0)
        source.privileged_obs_normalizer.running_mean.fill_(6.0)
    document = {
        "iter": generation,
        "model_state_dict": source.alg.policy.state_dict(),
        "obs_norm_state_dict": source.obs_normalizer.state_dict(),
        "privileged_obs_norm_state_dict": (
            source.privileged_obs_normalizer.state_dict()
        ),
    }
    path = tmp_path / "snapshot.pt"
    torch.save(document, path)
    return path, document


def _bare_evaluator():
    evaluator = object.__new__(S.FormalIsaacEvaluator)
    evaluator.device = "cpu"
    return evaluator


def test_actor_only_snapshot_load_validates_full_envelope_and_two_normalizers(
    tmp_path,
):
    path, expected = _checkpoint(tmp_path)
    runner = _SnapshotRunner()
    evaluator = _bare_evaluator()
    evaluator._load_frozen_snapshot(
        runner=runner,
        checkpoint_receipt=S.inbox_protocol.artifact_receipt(path),
        checkpoint_path=path,
        expected_generation=7,
    )
    assert runner.preflight_calls == [(7, str(path), True)]
    assert runner.current_learning_iteration == 8
    assert runner.alg.policy.training is False
    assert runner.obs_normalizer.training is False
    assert runner.privileged_obs_normalizer.training is False
    for name, value in runner.alg.policy.state_dict().items():
        assert torch.equal(value, expected["model_state_dict"][name])
    for name, value in runner.obs_normalizer.state_dict().items():
        assert torch.equal(value, expected["obs_norm_state_dict"][name])
    for (
        name,
        value,
    ) in runner.privileged_obs_normalizer.state_dict().items():
        assert torch.equal(
            value,
            expected["privileged_obs_norm_state_dict"][name],
        )


def test_actor_only_snapshot_load_accepts_first_policy_generation_zero(
    tmp_path,
):
    path, _ = _checkpoint(tmp_path, generation=0)
    runner = _SnapshotRunner()
    _bare_evaluator()._load_frozen_snapshot(
        runner=runner,
        checkpoint_receipt=S.inbox_protocol.artifact_receipt(path),
        checkpoint_path=path,
        expected_generation=0,
    )
    assert runner.preflight_calls == [(0, str(path), True)]
    assert runner.current_learning_iteration == 1


def test_actor_only_snapshot_load_rejects_generation_and_nonfinite_state(
    tmp_path,
):
    path, _ = _checkpoint(tmp_path)
    with pytest.raises(RuntimeError, match="iteration"):
        _bare_evaluator()._load_frozen_snapshot(
            runner=_SnapshotRunner(),
            checkpoint_receipt=S.inbox_protocol.artifact_receipt(path),
            checkpoint_path=path,
            expected_generation=8,
        )

    bad_path, _ = _checkpoint(
        tmp_path / "bad", nonfinite=True
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        _bare_evaluator()._load_frozen_snapshot(
            runner=_SnapshotRunner(),
            checkpoint_receipt=S.inbox_protocol.artifact_receipt(
                bad_path
            ),
            checkpoint_path=bad_path,
            expected_generation=7,
        )


def test_actor_only_snapshot_safe_load_does_not_execute_reduce(
    tmp_path,
):
    marker = tmp_path / "pickle-executed"
    path = tmp_path / "malicious.pt"
    torch.save(
        {
            "iter": 0,
            "exploit": _MaliciousCheckpointValue(marker),
        },
        path,
    )
    with pytest.raises(RuntimeError, match="could not decode"):
        _bare_evaluator()._load_frozen_snapshot(
            runner=_SnapshotRunner(),
            checkpoint_receipt=S.inbox_protocol.artifact_receipt(path),
            checkpoint_path=path,
            expected_generation=0,
        )
    assert not marker.exists()


def _heartbeat_document(path):
    document = json.loads(path.read_text(encoding="ascii"))
    assert document["schema_version"] == 1
    assert document["kind"] == S.HEARTBEAT_KIND
    assert document["content_sha256"] == (
        S.inbox_protocol.canonical_sha256(document["content"])
    )
    return document


def test_heartbeat_publishes_atomic_request_progress_and_clears_deadline(
    tmp_path,
):
    progress = S.SidecarProgressPublisher(
        inbox_root=tmp_path,
        owner_id="owner",
        run_id="run",
        launch_sha256="a" * 64,
        interval_s=60.0,
    )
    progress.start()
    try:
        progress.publish("ready")
        progress.begin_request(
            request_seq=3,
            request_sha256="b" * 64,
            attempts_total=5,
            deadline_s=60.0,
        )
        progress.publish("evaluating", attempts_completed=2)
        active = _heartbeat_document(progress.path)["content"]
        assert active["request_seq"] == 3
        assert active["request_sha256"] == "b" * 64
        assert active["attempts_completed"] == 2
        assert active["attempts_total"] == 5
        assert active["request_deadline_monotonic_ns"] > 0
        active_seq = active["heartbeat_seq"]

        progress.waiting_for_request_or_ack()
        waiting = _heartbeat_document(progress.path)["content"]
        assert waiting["heartbeat_seq"] > active_seq
        assert waiting["phase"] == "waiting_for_request_or_ack"
        assert waiting["request_seq"] is None
        assert waiting["request_deadline_monotonic_ns"] == 0
        assert waiting["attempts_total"] == 0
    finally:
        progress.stop()


def test_heartbeat_deadline_and_launch_binding_fail_closed(
    tmp_path, monkeypatch
):
    clock = [10_000_000_000]
    monkeypatch.setattr(S.time, "monotonic_ns", lambda: clock[0])
    progress = S.SidecarProgressPublisher(
        inbox_root=tmp_path,
        owner_id="owner",
        run_id="deadline",
        launch_sha256="a" * 64,
        interval_s=5.0,
    )
    progress.begin_request(
        request_seq=0,
        request_sha256="b" * 64,
        attempts_total=1,
        deadline_s=1.0,
    )
    clock[0] += 1_000_000_001
    with pytest.raises(RuntimeError, match="deadline"):
        progress.assert_before_deadline()

    launch_content = {
        "heartbeat_contract": {
            "schema_version": 1,
            "heartbeat_interval_seconds": 5.0,
            "heartbeat_stale_after_seconds": 120.0,
            "request_deadline_seconds": 7200.0,
        }
    }
    assert S._launch_heartbeat_settings(
        launch_content,
        cli_heartbeat_interval_s=5.0,
        cli_request_deadline_s=7200.0,
    )["request_deadline_s"] == 7200.0
    with pytest.raises(RuntimeError, match="differs"):
        S._launch_heartbeat_settings(
            launch_content,
            cli_heartbeat_interval_s=10.0,
            cli_request_deadline_s=7200.0,
        )


def test_shared_exact_resume_factory_is_strict_no_step_and_close_idempotent():
    source = inspect.getsource(
        S.FormalIsaacEvaluator
        .build_exact_resume_runtime_from_claim
    )
    assert "load_formal_action_ball_checkpoint_bytes" in source
    assert "checkpoint_bytes," in source
    assert 'runner.load(str(checkpoint_path)' not in source
    assert "exact_resume_live_state_receipt" in source
    assert '"common_step_counter_delta"] != 0' in source
    assert '"exact_resume_live_state": exact_resume_live_state' in source
    assert '"environment_restore": True' not in source
    assert '"rng_restore": True' not in source
    assert '"simulator_step_count": 0' not in source
    assert "runner.learn(" not in source

    class Owner:
        def __init__(self):
            self.closes = 0

        def _close_environment(self):
            self.closes += 1

    owner = Owner()
    runtime = S.ExactResumeRuntime(
        wrapped_env=object(),
        runner=object(),
        construction_receipt={},
        _owner=owner,
    )
    runtime.close()
    runtime.close()
    assert owner.closes == 1
