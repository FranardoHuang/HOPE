from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFIER_PATH = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/scripts/"
    "action_ball_exact_resume_verifier.py"
)
SPEC = importlib.util.spec_from_file_location(
    "action_ball_exact_resume_verifier_under_test", VERIFIER_PATH
)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERIFIER
SPEC.loader.exec_module(VERIFIER)


def _canonical_bytes(value):
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _numpy_rng_state():
    return {
        "schema_version": 1,
        "bit_generator": "MT19937",
        "state_uint32": list(range(624)),
        "position": 17,
        "has_gauss": 0,
        "cached_gaussian": 0.0,
    }


def _checkpoint(*, claim_sha, bootstrap_sha, lineage_sha, iteration=2):
    bootstrap_receipt = {
        "path": "/fixture/bootstrap.json",
        "sha256": "d" * 64,
        "size_bytes": 123,
    }
    exact_state = {
        "schema_version": 3,
        "next_learning_iteration": iteration + 1,
        "python_random_state": [1, 2, 3],
        "numpy_random_state": _numpy_rng_state(),
        "torch_random_state": [7, 8, 9],
        "environment_resume_state": {
            "schema_version": 3,
            "common_step_counter": 48,
        },
        "runtime_bootstrap_receipt_sha256": bootstrap_sha,
        "runtime_bootstrap_lineage_payload_sha256": lineage_sha,
        "runtime_bootstrap_receipt": bootstrap_receipt,
    }
    return {
        "iter": iteration,
        "model_state_dict": {"weight": [1.0, 2.0]},
        "optimizer_state_dict": {"state": {"step": 19}},
        "obs_normalizer_state_dict": {"mean": [0.1, 0.2]},
        "infos": {
            "training_contract_schema_version": 3,
            "training_contract_lineage_exact": 1,
            "training_contract_sha256": "c" * 64,
            "training_launch_claim_sha256": claim_sha,
            "hope_exact_resume_state": exact_state,
            "runtime_bootstrap_receipt_sha256": bootstrap_sha,
            "runtime_bootstrap_lineage_payload_sha256": lineage_sha,
            "runtime_bootstrap_receipt": bootstrap_receipt,
        },
    }


def _live_inventory_verification(
    *,
    checkout,
    inventory_identity,
    inventory_source_sha,
    interpreter,
    verifier_size=101,
    inventory_size=202,
):
    live_content = {
        "schema_version": 1,
        "kind": "action_ball_runtime_inventory_live_verification",
        "verifier_source": {
            "path": str(checkout / VERIFIER.RUNTIME_INVENTORY_SOURCE),
            "sha256": inventory_source_sha,
            "size_bytes": verifier_size,
        },
        "inventory_artifact": {
            "path": inventory_identity["path"],
            "sha256": inventory_identity["file_sha256"],
            "size_bytes": inventory_size,
        },
        "inventory_content_sha256": inventory_identity[
            "content_sha256"
        ],
        "current_interpreter": interpreter,
        "verification_result": {
            "ok": True,
            "kind": "action_ball_runtime_inventory_v1",
            "content_sha256": inventory_identity["content_sha256"],
            "receipt_path": inventory_identity["path"],
            "receipt_sha256": inventory_identity["file_sha256"],
        },
    }
    return {
        "schema_version": 1,
        "kind": "action_ball_runtime_inventory_live_verification",
        "content": live_content,
        "content_sha256": VERIFIER.canonical_sha256(live_content),
    }


def _construction_receipt(
    *,
    checkpoint_path,
    checkpoint_sha,
    claim_sha,
    bootstrap_sha,
    lineage_sha,
    checkout,
    inventory_identity,
    inventory_source_sha,
    interpreter,
    live_verification=None,
    iteration=2,
):
    if live_verification is None:
        live_verification = _live_inventory_verification(
            checkout=checkout,
            inventory_identity=inventory_identity,
            inventory_source_sha=inventory_source_sha,
            interpreter=interpreter,
        )
    resume_content = {
        "schema_version": 1,
        "kind": "action_ball_exact_resume_live_state",
        "source_embedded_iteration": iteration,
        "current_learning_iteration": iteration + 1,
        "roundtrip_pending": True,
        "resume_reset_pending": True,
        "model_state_sha256": "1" * 64,
        "optimizer_state_sha256": "2" * 64,
        "actor_normalizer_state_sha256": "3" * 64,
        "critic_normalizer_state_sha256": "4" * 64,
        "exact_resume_state_sha256": "5" * 64,
        "environment_resume_state_sha256": "6" * 64,
        "rng_state_sha256": "7" * 64,
        "runtime_bootstrap_binding_sha256": "8" * 64,
        "common_step_counter": 48,
        "common_step_counter_delta": 0,
        "live_core_sha256": "9" * 64,
    }
    exact_resume_live_state = {
        "schema_version": 1,
        "kind": "action_ball_exact_resume_live_state",
        "content": resume_content,
        "content_sha256": VERIFIER.canonical_sha256(resume_content),
    }
    content = {
        "schema_version": 1,
        "kind": "action_ball_exact_resume_runtime_construction",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_iteration": iteration,
        "load_optimizer": True,
        "bootstrap_content_sha256": bootstrap_sha,
        "bootstrap_artifact_sha256": "d" * 64,
        "bootstrap_artifact_size_bytes": 123,
        "bootstrap_lineage_payload_sha256": lineage_sha,
        "runtime_inventory_live_verification": live_verification,
        "exact_resume_live_state": exact_resume_live_state,
        "training_contract_sha256": "c" * 64,
        "training_launch_claim_sha256": claim_sha,
        "environment_count": 4,
        "runner_current_learning_iteration": iteration + 1,
    }
    return {
        "schema_version": 1,
        "kind": "action_ball_exact_resume_runtime_construction",
        "content": content,
        "content_sha256": VERIFIER.canonical_sha256(content),
    }


def _fixture(tmp_path, monkeypatch, *, mutate_roundtrip=None):
    namespace = tmp_path / "namespace"
    namespace.mkdir()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    rsl = tmp_path / "rsl"
    rsl.mkdir()
    claim_path = namespace / "launch_claim.json"
    claim_path.write_text("{}\n", encoding="utf-8")
    claim_sha = "a" * 64
    bootstrap_sha = "b" * 64
    lineage_sha = "e" * 64
    source_checkpoint = rsl / "model_2.pt"
    source_value = _checkpoint(
        claim_sha=claim_sha,
        bootstrap_sha=bootstrap_sha,
        lineage_sha=lineage_sha,
    )
    source_checkpoint.write_bytes(_canonical_bytes(source_value))
    claim = {
        "schema_version": 3,
        "kind": "action_ball_no_clobber_launch_claim_v3",
        "launch_claim_sha256": claim_sha,
        "canonical_payload": {},
        "argv": [],
        "confirmation_claim_sha256": claim_sha,
    }
    payload = {
        "namespace": str(namespace),
        "stage": "smoke",
        "stage_budget": {"max_iterations": 2},
        "action_set_contract": {
            "experiment_name": VERIFIER.EXPERIMENT_NAME,
        },
    }
    inventory_identity = {
        "path": str(tmp_path / "runtime_inventory.json"),
        "file_sha256": "8" * 64,
        "content_sha256": "7" * 64,
        "kind": "action_ball_runtime_inventory_v1",
    }
    inventory_source_sha = "9" * 64
    interpreter = "/fixture/isaac/python"
    payload["isaac_python_runtime"] = {
        "path": interpreter,
        "runtime_inventory": inventory_identity,
    }
    payload["runtime_code_sha256"] = {
        VERIFIER.RUNTIME_INVENTORY_SOURCE: inventory_source_sha,
    }
    live_inventory_verification = _live_inventory_verification(
        checkout=checkout,
        inventory_identity=inventory_identity,
        inventory_source_sha=inventory_source_sha,
        interpreter=interpreter,
    )
    monkeypatch.setattr(
        VERIFIER,
        "_validate_claim",
        lambda path: (
            claim,
            payload,
            {"sha256": "f" * 64},
            checkout,
            "1" * 40,
        ),
    )
    monkeypatch.setattr(
        VERIFIER,
        "_rsl_log_dir",
        lambda **kwargs: rsl,
    )
    monkeypatch.setattr(
        VERIFIER,
        "_committed_source",
        lambda checkout, commit, relative, label: {
            "path": VERIFIER_PATH,
            "sha256": hashlib.sha256(
                VERIFIER_PATH.read_bytes()
            ).hexdigest(),
            "size_bytes": VERIFIER_PATH.stat().st_size,
            "raw": VERIFIER_PATH.read_bytes(),
        },
    )
    monkeypatch.setattr(
        VERIFIER,
        "_preimport_runtime_inventory_verification",
        lambda **kwargs: copy.deepcopy(live_inventory_verification),
    )

    calls = {
        "factory": 0,
        "save": 0,
        "close": 0,
        "app_close": 0,
        "learn": 0,
        "step": 0,
    }

    def factory(
        *,
        claim_document,
        final_checkpoint_path,
        device,
        _preimport_live_inventory_verification,
    ):
        calls["factory"] += 1
        assert (
            _preimport_live_inventory_verification
            == live_inventory_verification
        )
        checkpoint_path = Path(final_checkpoint_path)
        checkpoint_sha = hashlib.sha256(
            checkpoint_path.read_bytes()
        ).hexdigest()

        class Runner:
            _exact_resume_roundtrip_pending = True

            def learn(self, *args, **kwargs):
                calls["learn"] += 1
                raise AssertionError("verifier must not call learn")

            def step(self, *args, **kwargs):
                calls["step"] += 1
                raise AssertionError("verifier must not call step")

            def save_exact_resume_roundtrip(self, target):
                calls["save"] += 1
                output_value = copy.deepcopy(source_value)
                if mutate_roundtrip is not None:
                    mutate_roundtrip(output_value)
                target_path = Path(target)
                target_path.write_bytes(_canonical_bytes(output_value))
                self._exact_resume_roundtrip_pending = False
                raw = target_path.read_bytes()
                return {
                    "checkpoint": {
                        "path": str(target_path),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "size_bytes": len(raw),
                    },
                    "source_embedded_iteration": 2,
                    "before_current_learning_iteration": 3,
                    "after_current_learning_iteration": 3,
                    "output_embedded_iteration": 2,
                    "output_next_learning_iteration": 3,
                    "runtime_bootstrap_receipt_sha256": bootstrap_sha,
                    "runtime_bootstrap_lineage_payload_sha256": lineage_sha,
                    "runtime_bootstrap_receipt": source_value["infos"][
                        "runtime_bootstrap_receipt"
                    ],
                }

        def close():
            calls["close"] += 1

        return SimpleNamespace(
            wrapped_env=object(),
            runner=Runner(),
            construction_receipt=_construction_receipt(
                checkpoint_path=checkpoint_path,
                checkpoint_sha=checkpoint_sha,
                claim_sha=claim_sha,
                bootstrap_sha=bootstrap_sha,
                lineage_sha=lineage_sha,
                checkout=checkout,
                inventory_identity=inventory_identity,
                inventory_source_sha=inventory_source_sha,
                interpreter=interpreter,
                live_verification=live_inventory_verification,
            ),
            close=close,
        )

    monkeypatch.setattr(
        VERIFIER,
        "_load_checkpoint",
        lambda snapshot: json.loads(snapshot["raw"].decode("ascii")),
    )
    monkeypatch.setattr(
        VERIFIER,
        "_load_sidecar_factory",
        lambda sidecar_source, inbox_source: factory,
    )
    monkeypatch.setattr(
        VERIFIER,
        "_production_torch_module",
        lambda: SimpleNamespace(is_tensor=lambda value: False),
    )

    class FakeApp:
        def close(self):
            calls["app_close"] += 1

    monkeypatch.setattr(
        VERIFIER, "_launch_isaac_app", lambda: FakeApp()
    )
    return {
        "claim_path": claim_path,
        "checkpoint_path": source_checkpoint,
        "output_path": namespace / "exact_resume_verification.json",
        "calls": calls,
    }


def test_real_restore_zero_step_save_reload_receipt(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    receipt = VERIFIER.verify_exact_resume(
        claim_path=str(fixture["claim_path"]),
        checkpoint_path=str(fixture["checkpoint_path"]),
        output_path=str(fixture["output_path"]),
    )

    assert fixture["calls"] == {
        "factory": 2,
        "save": 1,
        "close": 2,
        "app_close": 1,
        "learn": 0,
        "step": 0,
    }
    assert receipt["restore"]["factory_call_count"] == 2
    assert receipt["restore"]["closed_runtime_count"] == 2
    assert receipt["restore"]["common_step_counter_delta"] == 0
    assert receipt["restore"]["fresh_strict_load_token_consumed"] is True
    assert "learn_or_update_called" not in receipt["restore"]
    assert receipt["state"]["source_core_sha256"] == receipt["state"][
        "roundtrip_core_sha256"
    ]
    assert receipt["natural_exit"] is True
    published = json.loads(
        fixture["output_path"].read_text(encoding="ascii")
    )
    assert published == receipt


def test_roundtrip_optimizer_drift_is_rejected(tmp_path, monkeypatch):
    def mutate(checkpoint):
        checkpoint["optimizer_state_dict"]["state"]["step"] += 1

    fixture = _fixture(
        tmp_path, monkeypatch, mutate_roundtrip=mutate
    )
    with pytest.raises(
        VERIFIER.ExactResumeVerificationError,
        match="core drifted",
    ):
        VERIFIER.verify_exact_resume(
            claim_path=str(fixture["claim_path"]),
            checkpoint_path=str(fixture["checkpoint_path"]),
            output_path=str(fixture["output_path"]),
        )
    assert not fixture["output_path"].exists()
    assert fixture["calls"]["learn"] == 0
    assert fixture["calls"]["step"] == 0


def test_roundtrip_requires_consumed_strict_load_token(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    original_factory = VERIFIER._load_sidecar_factory(
        {"path": VERIFIER_PATH}, {"path": VERIFIER_PATH}
    )

    def factory(**kwargs):
        runtime = original_factory(**kwargs)
        original_save = runtime.runner.save_exact_resume_roundtrip

        def save_without_consuming(target):
            receipt = original_save(target)
            runtime.runner._exact_resume_roundtrip_pending = True
            return receipt

        runtime.runner.save_exact_resume_roundtrip = save_without_consuming
        return runtime

    monkeypatch.setattr(
        VERIFIER,
        "_load_sidecar_factory",
        lambda sidecar_source, inbox_source: factory,
    )

    with pytest.raises(
        VERIFIER.ExactResumeVerificationError,
        match="strict-load token",
    ):
        VERIFIER.verify_exact_resume(
            claim_path=str(fixture["claim_path"]),
            checkpoint_path=str(fixture["checkpoint_path"]),
            output_path=str(fixture["output_path"]),
        )
    assert not fixture["output_path"].exists()


def test_public_publisher_has_no_injectable_factory_or_loader(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    with pytest.raises(TypeError, match="unexpected keyword"):
        VERIFIER.verify_exact_resume(
            claim_path=str(fixture["claim_path"]),
            checkpoint_path=str(fixture["checkpoint_path"]),
            output_path=str(fixture["output_path"]),
            runtime_factory=lambda **kwargs: None,
        )
    assert not fixture["output_path"].exists()


def test_checkpoint_loader_never_falls_back_to_unsafe_pickle(
    tmp_path, monkeypatch
):
    marker = tmp_path / "pickle_executed"

    def fake_load(stream, *, map_location, weights_only):
        if weights_only is not True:
            marker.write_text("executed\n", encoding="utf-8")
        raise RuntimeError("safe loader rejected fixture")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(load=fake_load),
    )

    with pytest.raises(
        VERIFIER.ExactResumeVerificationError,
        match="safe weights-only",
    ):
        VERIFIER._load_checkpoint({"raw": b"malicious-pickle-fixture"})
    assert not marker.exists()


@pytest.mark.parametrize(
    "bad_state",
    (
        ("MT19937", list(range(624)), 17, 0, 0.0),
        {
            "schema_version": 1,
            "bit_generator": "MT19937",
            "state_uint32": list(range(623)),
            "position": 17,
            "has_gauss": 0,
            "cached_gaussian": 0.0,
        },
        {
            "schema_version": 1,
            "bit_generator": "MT19937",
            "state_uint32": list(range(624)),
            "position": 17,
            "has_gauss": 0,
            "cached_gaussian": float("nan"),
        },
    ),
)
def test_checkpoint_core_refuses_nonportable_numpy_rng_state(bad_state):
    checkpoint = _checkpoint(
        claim_sha="a" * 64,
        bootstrap_sha="b" * 64,
        lineage_sha="c" * 64,
    )
    checkpoint["infos"]["hope_exact_resume_state"][
        "numpy_random_state"
    ] = bad_state

    with pytest.raises(
        VERIFIER.ExactResumeVerificationError,
        match="NumPy RNG state",
    ):
        VERIFIER._checkpoint_core(
            checkpoint,
            expected_iteration=2,
            claim_sha256="a" * 64,
            torch_module=SimpleNamespace(is_tensor=lambda _value: False),
        )


def test_claim_argv_rejects_a_second_claim_path(tmp_path):
    claim_path = tmp_path / "launch_claim.json"
    good_path = "++training_launch_claim_path={}".format(claim_path)
    evil_path = "++training_launch_claim_path=/tmp/evil.json"

    with pytest.raises(
        VERIFIER.ExactResumeVerificationError,
        match="duplicate Hydra override",
    ):
        VERIFIER._hydra_overrides(
            ["train.py", "--", good_path, evil_path]
        )


def test_inventory_refusal_precedes_checkpoint_and_runtime_imports(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    calls = []

    def refuse_inventory(**kwargs):
        calls.append("inventory")
        raise VERIFIER.ExactResumeVerificationError("inventory refused")

    def forbidden(name):
        def call(*args, **kwargs):
            calls.append(name)
            raise AssertionError("{} ran before inventory passed".format(name))

        return call

    monkeypatch.setattr(
        VERIFIER,
        "_preimport_runtime_inventory_verification",
        refuse_inventory,
    )
    monkeypatch.setattr(VERIFIER, "_load_sidecar_factory", forbidden("sidecar"))
    monkeypatch.setattr(VERIFIER, "_load_checkpoint", forbidden("checkpoint"))
    monkeypatch.setattr(
        VERIFIER, "_production_torch_module", forbidden("torch")
    )
    monkeypatch.setattr(VERIFIER, "_launch_isaac_app", forbidden("isaac"))

    with pytest.raises(
        VERIFIER.ExactResumeVerificationError, match="inventory refused"
    ):
        VERIFIER.verify_exact_resume(
            claim_path=str(fixture["claim_path"]),
            checkpoint_path=str(fixture["checkpoint_path"]),
            output_path=str(fixture["output_path"]),
        )
    assert calls == ["inventory"]
    assert not fixture["output_path"].exists()
    assert not any(
        path.name.startswith("exact_resume_roundtrip_")
        for path in fixture["checkpoint_path"].parent.iterdir()
    )


def test_preimport_inventory_accepts_utf8_canonical_receipt(
    tmp_path, monkeypatch
):
    checkout = tmp_path / "含乒乓的仓库"
    source_path = checkout / VERIFIER.RUNTIME_INVENTORY_SOURCE
    source_path.parent.mkdir(parents=True)
    source_raw = b"# exact committed inventory source\n"
    source_path.write_bytes(source_raw)
    bootstrap_path = checkout / VERIFIER.NOSITE_BOOTSTRAP_SOURCE
    bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_raw = (
        REPO_ROOT / VERIFIER.NOSITE_BOOTSTRAP_SOURCE
    ).read_bytes()
    bootstrap_path.write_bytes(bootstrap_raw)
    bootstrap_sha = hashlib.sha256(bootstrap_raw).hexdigest()
    receipt_dir = tmp_path / "收据"
    receipt_dir.mkdir()
    receipt_path = receipt_dir / "运行时.json"
    current_interpreter = str(Path(sys.executable).absolute())
    nosite = VERIFIER._load_nosite_bootstrap_module(bootstrap_path)
    import_root = tmp_path / "显式导入根"
    import_root.mkdir()
    import_roots = [nosite.bind_import_root(import_root)]
    verification_command = nosite.build_exact_nosite_argv(
        python=Path(current_interpreter),
        bootstrap=bootstrap_path,
        bootstrap_sha256=bootstrap_sha,
        entrypoint=source_path,
        entrypoint_sha256=hashlib.sha256(source_raw).hexdigest(),
        import_roots=import_roots,
        entrypoint_argv=[
            "verify",
            "--receipt",
            str(receipt_path),
        ],
    )
    content = {
        "python": {
            "requested_path": current_interpreter,
            "probe": {
                "no_site_execution": {
                    "outer": {"import_roots": import_roots}
                }
            },
        },
        "unicode_witness": "乒乓",
    }
    content_sha = hashlib.sha256(
        VERIFIER._canonical_utf8_bytes(content)
    ).hexdigest()
    document = {
        "schema_version": 2,
        "kind": "action_ball_runtime_inventory_v2",
        "content": content,
        "content_sha256": content_sha,
    }
    raw_receipt = VERIFIER._canonical_utf8_bytes(document) + b"\n"
    receipt_path.write_bytes(raw_receipt)
    inventory_identity = {
        "path": str(receipt_path),
        "file_sha256": hashlib.sha256(raw_receipt).hexdigest(),
        "content_sha256": content_sha,
        "kind": "action_ball_runtime_inventory_v2",
        "import_roots": import_roots,
        "nosite_verification_contract_sha256": (
            verification_command.contract_sha256
        ),
    }
    source_sha = hashlib.sha256(source_raw).hexdigest()
    payload = {
        "runtime_code_sha256": {
            VERIFIER.RUNTIME_INVENTORY_SOURCE: source_sha,
            VERIFIER.NOSITE_BOOTSTRAP_SOURCE: bootstrap_sha,
        },
        "isaac_python_runtime": {
            "path": current_interpreter,
            "runtime_inventory": inventory_identity,
        },
    }
    expected_result = {
        "ok": True,
        "kind": "action_ball_runtime_inventory_v2",
        "content_sha256": content_sha,
        "receipt_path": str(receipt_path),
        "receipt_sha256": inventory_identity["file_sha256"],
    }

    def fake_run(argv, **kwargs):
        assert tuple(argv) == verification_command.argv
        assert kwargs["stdin"] is VERIFIER.subprocess.DEVNULL
        return SimpleNamespace(
            returncode=0,
            stdout=VERIFIER._canonical_utf8_bytes(expected_result) + b"\n",
            stderr=b"",
        )

    monkeypatch.setattr(VERIFIER.subprocess, "run", fake_run)
    monkeypatch.setattr(
        VERIFIER,
        "sys",
        SimpleNamespace(executable=current_interpreter, modules={}),
    )
    proof = VERIFIER._preimport_runtime_inventory_verification(
        payload=payload,
        checkout=checkout,
        inventory_source={
            "path": source_path,
            "sha256": source_sha,
            "size_bytes": len(source_raw),
            "raw": source_raw,
        },
        nosite_bootstrap_source={
            "path": bootstrap_path,
            "sha256": bootstrap_sha,
            "size_bytes": len(bootstrap_raw),
            "raw": bootstrap_raw,
        },
    )
    assert proof["content"]["verification_result"] == expected_result
    assert proof["content_sha256"] == VERIFIER.canonical_sha256(
        proof["content"]
    )


def test_factory_executes_committed_raw_bytes_not_live_path_or_pyc(tmp_path):
    marker = tmp_path / "malicious_executed"
    inbox_path = tmp_path / "action_ball_evaluation_inbox.py"
    sidecar_path = tmp_path / "action_ball_frozen_eval_sidecar.py"
    inbox_raw = b"TRUSTED = True\n"
    sidecar_raw = (
        b"import sys\n"
        b"inbox_protocol=sys.modules['action_ball_evaluation_inbox']\n"
        b"def build_exact_resume_runtime_from_claim(**kwargs):\n"
        b"    return inbox_protocol.TRUSTED\n"
    )
    malicious = (
        "from pathlib import Path\n"
        "Path({!r}).write_text('executed')\n".format(str(marker))
    ).encode("utf-8")
    inbox_path.write_bytes(malicious)
    sidecar_path.write_bytes(malicious)
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    (pycache / "action_ball_frozen_eval_sidecar.cpython-38.pyc").write_bytes(
        b"malicious cached bytecode must be ignored"
    )
    inbox_source = {
        "path": inbox_path,
        "sha256": hashlib.sha256(inbox_raw).hexdigest(),
        "size_bytes": len(inbox_raw),
        "raw": inbox_raw,
    }
    sidecar_source = {
        "path": sidecar_path,
        "sha256": hashlib.sha256(sidecar_raw).hexdigest(),
        "size_bytes": len(sidecar_raw),
        "raw": sidecar_raw,
    }
    existing_inbox = sys.modules.pop("action_ball_evaluation_inbox", None)
    module_name = ""
    try:
        factory = VERIFIER._load_sidecar_factory(
            sidecar_source, inbox_source
        )
        module_name = factory.__module__
        assert factory() is True
        assert not marker.exists()
    finally:
        if module_name:
            sys.modules.pop(module_name, None)
        sys.modules.pop("action_ball_evaluation_inbox", None)
        if existing_inbox is not None:
            sys.modules["action_ball_evaluation_inbox"] = existing_inbox
