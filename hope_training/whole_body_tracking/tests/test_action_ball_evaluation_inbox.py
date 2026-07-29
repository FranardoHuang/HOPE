from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


WBT = Path(__file__).resolve().parents[1]
MDP = (
    WBT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)
SCRIPTS = WBT / "scripts"


def _load(name, path):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


I = _load(
    "action_ball_evaluation_inbox",
    MDP / "action_ball_evaluation_inbox.py",
)
S = _load(
    "action_ball_frozen_eval_sidecar",
    SCRIPTS / "action_ball_frozen_eval_sidecar.py",
)
R = _load("action_ball_runtime", MDP / "action_ball_runtime.py")
C = _load("action_ball_curriculum", MDP / "action_ball_curriculum.py")
E = _load("action_ball_evaluation", MDP / "action_ball_evaluation.py")


def _load_identity_module():
    package_name = "action_ball_identity_test_package"
    module_name = (
        package_name + ".action_ball_frozen_eval_identity"
    )
    if module_name in sys.modules:
        return sys.modules[module_name]
    package = ModuleType(package_name)
    package.__path__ = [str(MDP)]
    sys.modules[package_name] = package
    sys.modules[
        package_name + ".action_ball_evaluation_inbox"
    ] = I
    spec = importlib.util.spec_from_file_location(
        module_name,
        MDP / "action_ball_frozen_eval_identity.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _bindings(tmp_path):
    checkpoint = tmp_path / "model.pt"
    motion_a = tmp_path / "backhand_a.npz"
    motion_b = tmp_path / "backhand_b.npz"
    training_contract = tmp_path / "training_contract.json"
    environment_config = tmp_path / "env.pkl"
    agent_config = tmp_path / "agent.pkl"
    runtime_identity = tmp_path / "runtime_identity.json"
    runtime_bootstrap = tmp_path / "runtime_bootstrap.json"
    checkpoint.write_bytes(b"frozen checkpoint bytes")
    motion_a.write_bytes(b"motion-a-bytes")
    motion_b.write_bytes(b"motion-b-bytes")
    training_contract.write_bytes(b"training-contract")
    environment_config.write_bytes(b"environment-config")
    agent_config.write_bytes(b"agent-config")
    runtime_identity.write_bytes(b"runtime-identity")
    bootstrap_lineage_sha256 = "c" * 64
    bootstrap_content = {
        "lineage_payload_sha256": bootstrap_lineage_sha256
    }
    bootstrap_document = {
        "schema_version": 1,
        "kind": "action_ball_runtime_bootstrap_receipt_v1",
        "content": bootstrap_content,
        "content_sha256": I.canonical_sha256(bootstrap_content),
    }
    runtime_bootstrap.write_bytes(_canonical_bytes(bootstrap_document))
    return {
        "checkpoint": I.artifact_receipt(checkpoint),
        "training_contract": I.artifact_receipt(training_contract),
        "environment_config_pickle": I.artifact_receipt(
            environment_config
        ),
        "agent_config_pickle": I.artifact_receipt(agent_config),
        "runtime_identity": I.artifact_receipt(runtime_identity),
        "runtime_bootstrap_receipt_sha256": bootstrap_document[
            "content_sha256"
        ],
        "runtime_bootstrap_lineage_payload_sha256": (
            bootstrap_lineage_sha256
        ),
        "runtime_bootstrap_receipt": I.artifact_receipt(
            runtime_bootstrap
        ),
        "training_launch_claim_sha256": "0" * 64,
        "policy_generation": 17,
        "policy_state": I.state_binding(
            sha256="a" * 64, size_bytes=4096
        ),
        "actor_obs_normalizer": I.state_binding(
            sha256="1" * 64, size_bytes=712
        ),
        "critic_obs_normalizer": I.state_binding(
            sha256="2" * 64, size_bytes=944
        ),
        "ppo_recipe_sha256": "3" * 64,
        "policy_contract_sha256": "4" * 64,
        "action_order": [11, 29],
        "actions": [
            {
                "action_uid": 11,
                "motion": I.artifact_receipt(motion_a),
            },
            {
                "action_uid": 29,
                "motion": I.artifact_receipt(motion_b),
            },
        ],
        "manifest_sha256": "5" * 64,
        "sampler_sha256": "6" * 64,
        "proposal_sampler_contract_sha256": "b" * 64,
        "solver_sha256": "7" * 64,
        "physics_sha256": "8" * 64,
        "reward_sha256": "9" * 64,
        "curriculum_sha256": "a" * 64,
    }


def _target():
    return {
        "action_uid": 11,
        "profile_sha256": "b" * 64,
        "mobility_mode": "no_move",
        "domain_epoch": 4,
        "stratum": "marginal",
        "selected_arm_key": "incoming_speed.lower",
        "selection_round": 8,
        "arm_levels": [0.0, 0.1, 0.2, 0.3],
        "rho": 0.1,
    }


def _launch(monkeypatch):
    code_sha256 = S.sidecar_code_sha256()
    launch = I.build_sidecar_launch_document(
        sidecar_code_sha256=code_sha256,
        backend_contract_sha256=S.CPU_FAKE_BACKEND_CONTRACT_SHA256,
    )
    monkeypatch.setattr(
        I,
        "TRUSTED_ACTION_BALL_EVALUATION_SIDECAR_CODE_SHA256",
        frozenset((code_sha256,)),
    )
    monkeypatch.setattr(
        I,
        "TRUSTED_ACTION_BALL_EVALUATION_SIDECAR_LAUNCH_SHA256",
        frozenset((launch["content_sha256"],)),
    )
    return launch


def _request(
    tmp_path,
    launch,
    *,
    seq=0,
    seed_start=0,
    sample_start=10000,
    birth_start=20000,
    bindings=None,
):
    return I.build_request_document(
        owner_id="trainer-owner",
        run_id="n5-curriculum",
        request_seq=seq,
        sidecar_launch_sha256=launch["content_sha256"],
        bindings=_bindings(tmp_path) if bindings is None else bindings,
        target=_target(),
        seed_start=seed_start,
        sample_start=sample_start,
        birth_start=birth_start,
    )


def _rehash(document):
    document["content_sha256"] = I.canonical_sha256(
        document["content"]
    )
    return document


def _canonical_bytes(document):
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def test_request_binds_policy_and_both_normalizers_per_generation(
    tmp_path,
):
    bindings = _bindings(tmp_path)
    launch_sha256 = "a" * 64
    static = I._static_run_contract(bindings, launch_sha256)
    baseline = I.build_request_document(
        owner_id="trainer-owner",
        run_id="n5-curriculum",
        request_seq=0,
        sidecar_launch_sha256=launch_sha256,
        bindings=bindings,
        target=_target(),
        seed_start=0,
        sample_start=0,
        birth_start=0,
    )

    actor_changed = deepcopy(bindings)
    actor_changed["actor_obs_normalizer"] = I.state_binding(
        sha256="e" * 64,
        size_bytes=bindings["actor_obs_normalizer"]["size_bytes"],
    )
    critic_changed = deepcopy(bindings)
    critic_changed["critic_obs_normalizer"] = I.state_binding(
        sha256="f" * 64,
        size_bytes=bindings["critic_obs_normalizer"]["size_bytes"],
    )
    policy_changed = deepcopy(bindings)
    policy_changed["policy_generation"] += 1
    policy_changed["policy_state"] = I.state_binding(
        sha256="d" * 64,
        size_bytes=bindings["policy_state"]["size_bytes"],
    )

    for changed in (
        actor_changed,
        critic_changed,
        policy_changed,
    ):
        request = I.build_request_document(
            owner_id="trainer-owner",
            run_id="n5-curriculum",
            request_seq=0,
            sidecar_launch_sha256=launch_sha256,
            bindings=changed,
            target=_target(),
            seed_start=0,
            sample_start=0,
            birth_start=0,
        )
        assert request["content_sha256"] != baseline["content_sha256"]
        # These fields are deliberately generation-scoped rather than part
        # of the owner/run invariant, so one append-only namespace can serve
        # successive frozen policy generations.
        assert I._static_run_contract(changed, launch_sha256) == static


def _accepted_pair(tmp_path, monkeypatch):
    launch = _launch(monkeypatch)
    request = _request(tmp_path, launch)
    queue = I.EvaluationInbox(tmp_path / "inbox")
    queue.publish_request(request)
    evidence_path = S.process_one(
        inbox_root=queue.root,
        owner_id="trainer-owner",
        run_id="n5-curriculum",
        launch_document=launch,
        backend=S.DeterministicCpuFakeEvaluator(),
    )
    assert evidence_path is not None
    evidence = queue.load_evidence(
        "trainer-owner", "n5-curriculum", 0
    )
    resume_checkpoint = tmp_path / "consumer-resume-000.pt"
    resume_checkpoint.write_bytes(b"persisted exact consumer state 000")
    ack = I.build_ack_document(
        request,
        evidence,
        consumer_code_sha256="c" * 64,
        consumer_state_sha256="d" * 64,
        consumer_checkpoint=I.artifact_receipt(resume_checkpoint),
    )
    queue.publish_ack(ack)
    return launch, request, evidence, ack, queue


def _formal_target(
    *,
    action_uid=11,
    profile_sha256="b" * 64,
    domain_epoch=4,
    level=0.0,
    selected_arm_key=None,
):
    target = _target()
    target.update(
        {
            "action_uid": action_uid,
            "profile_sha256": profile_sha256,
            "domain_epoch": domain_epoch,
            "selected_arm_key": (
                R.ARM_KEYS[0]
                if selected_arm_key is None
                else selected_arm_key
            ),
            "arm_levels": [level] * len(R.ARM_KEYS),
            "rho": level,
        }
    )
    return target


def _runtime_request(
    request_document,
    *,
    role="frozen_canary",
    offset=1,
    generation=None,
    window_sha256="d" * 64,
    target_override=None,
):
    content = request_document["content"]
    if generation is None:
        generation = content["bindings"]["policy_generation"]
    target = (
        content["target"]
        if target_override is None
        else target_override
    )
    allocation = next(
        item for item in content["windows"] if item["role"] == role
    )
    levels = R.ActionDomainLevels(
        **dict(zip(R.ARM_KEYS, target["arm_levels"]))
    )
    return R.FrozenEvaluationProposalRequest.create(
        policy_checkpoint_sha256=content["bindings"]["checkpoint"][
            "sha256"
        ],
        policy_generation=generation,
        window_sha256=window_sha256,
        evidence_role=role,
        proposal_offset=offset,
        seed=allocation["seed_start"] + offset,
        sample_index=allocation["sample_start"] + offset,
        birth_index=allocation["birth_start"] + offset,
        action_uid=target["action_uid"],
        profile_sha256=target["profile_sha256"],
        mobility_mode=target["mobility_mode"],
        domain_epoch=target["domain_epoch"],
        domain_levels=levels,
        selected_arm_key=target["selected_arm_key"],
    )


def _publish_sidecar_evidence(queue, launch):
    path = S.process_one(
        inbox_root=queue.root,
        owner_id="trainer-owner",
        run_id="n5-curriculum",
        launch_document=launch,
        backend=S.DeterministicCpuFakeEvaluator(),
    )
    assert path is not None
    return path


def _v4_inbox_system(tmp_path, monkeypatch, *, suffix):
    sidecar_launch = _launch(monkeypatch)
    queue = I.EvaluationInbox(tmp_path / f"{suffix}-inbox")
    queue.initialize()
    source = I.FrozenSidecarInboxAttemptSource(
        inbox=queue,
        owner_id="trainer-owner",
        run_id="n5-curriculum",
        runtime_module=R,
    )
    target = _formal_target()
    key = C.ActionProfileKey(
        target["action_uid"],
        target["profile_sha256"],
        target["mobility_mode"],
    )
    evaluator_launch = E.launch_receipt_document_v4(
        curriculum_contract_sha256="a" * 64,
        profile_order=(key,),
        arm_catalog_sha256=C.ARM_CATALOG_SHA256,
        scheduler_contract_sha256=(
            C.ArmSchedulerConfig().contract_sha256
        ),
        sampler_sha256="6" * 64,
        solver_sha256="7" * 64,
        policy_contract_sha256="4" * 64,
        attempt_source_contract_sha256=source.source_contract_sha256,
        attempt_source_path=source.source_path,
        attempt_source_sha256=source.source_code_sha256,
    )
    launch_sha = E._canonical_sha256(evaluator_launch)
    monkeypatch.setattr(
        E,
        "TRUSTED_FROZEN_EVALUATOR_V4_LAUNCH_RECEIPT_SHA256",
        frozenset(
            set(E.TRUSTED_FROZEN_EVALUATOR_V4_LAUNCH_RECEIPT_SHA256)
            | {launch_sha}
        ),
    )
    authority = E.FrozenEvaluatorV4Authority.from_trusted_launch_receipt(
        evaluator_launch,
        attempt_source=source,
    )
    bindings = _bindings(tmp_path)
    coordinator = I.FrozenEvaluationInboxCoordinator(
        inbox=queue,
        owner_id="trainer-owner",
        run_id="n5-curriculum",
        sidecar_launch_sha256=sidecar_launch["content_sha256"],
        consumer_code_sha256="c" * 64,
        evaluator_authority=authority,
    )
    return {
        "sidecar_launch": sidecar_launch,
        "queue": queue,
        "source": source,
        "key": key,
        "evaluator_launch": evaluator_launch,
        "authority": authority,
        "bindings": bindings,
        "coordinator": coordinator,
    }


def _open_sidecar_session(
    authority,
    snapshot,
    key,
    *,
    role,
    domain_epoch,
    stratum,
    selected_arm_key,
    selection_round,
    arm_levels,
    rho,
):
    return authority.open_window(
        snapshot=snapshot,
        key=key,
        evidence_role=role,
        domain_epoch=domain_epoch,
        stratum=stratum,
        selected_arm_key=selected_arm_key,
        selection_round=selection_round,
        arm_levels=tuple(arm_levels),
        rho=rho,
    )


def test_cpu_fake_evaluator_end_to_end_is_fixed_and_append_only(
    tmp_path, monkeypatch
):
    launch, request, evidence, ack, queue = _accepted_pair(
        tmp_path, monkeypatch
    )
    del launch, request, ack
    windows = evidence["content"]["windows"]
    assert [item["allocation"]["proposal_count"] for item in windows] == [
        320,
        960,
    ]
    assert windows[0]["allocation"]["seed_end_exclusive"] == windows[1][
        "allocation"
    ]["seed_start"]
    for window in windows:
        ledger = window["ledger"]
        assert ledger["physics_invalid"] > 0
        assert ledger["solver_rejected"] > 0
        assert ledger["physics_invalid_reasons"] == {
            "incoming_ball_physics_invalid": ledger["physics_invalid"]
        }
        assert ledger["solver_reject_reasons"] == {
            "ball_to_task_geometry_unreachable": ledger[
                "solver_rejected"
            ]
        }
        assert (
            ledger["proposed"]
            == ledger["physics_invalid"]
            + ledger["solver_rejected"]
            + ledger["solver_admitted"]
        )
        assert all(
            "terminal_outcome" not in attempt
            for attempt in window["attempts"]
        )
    assert queue.next_pending_request(
        "trainer-owner", "n5-curriculum"
    ) is None
    evidence_bytes = queue.evidence_path(
        "trainer-owner", "n5-curriculum", 0
    ).read_bytes()
    with pytest.raises(
        (FileExistsError, I.EvaluationInboxError),
        match="replace|already|acknowledgement",
    ):
        queue.publish_evidence(evidence)
    assert (
        queue.evidence_path(
            "trainer-owner", "n5-curriculum", 0
        ).read_bytes()
        == evidence_bytes
    )


@pytest.mark.parametrize(
    "raw,match",
    [
        (b'{"x":1,"x":2}\n', "duplicate"),
        (b'{"x":NaN}\n', "non-finite"),
        (b'{"x":Infinity}\n', "non-finite"),
        (b'{"x":1e999}\n', "non-finite"),
        (b'{"x":', "incomplete"),
        (b"\xff\xfe", "UTF-8"),
    ],
)
def test_strict_json_rejects_duplicates_nonfinite_and_partial(raw, match):
    with pytest.raises(I.EvaluationInboxError, match=match):
        I.strict_json_loads(
            raw, label="malformed", require_canonical=False
        )


def test_partial_evidence_is_never_observed_as_complete(
    tmp_path, monkeypatch
):
    launch = _launch(monkeypatch)
    request = _request(tmp_path, launch)
    queue = I.EvaluationInbox(tmp_path / "inbox")
    queue.publish_request(request)
    evidence_path = queue.evidence_path(
        "trainer-owner", "n5-curriculum", 0
    )
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_bytes(b'{"schema_version":1')
    with pytest.raises(I.EvaluationInboxError, match="incomplete"):
        queue.next_pending_request(
            "trainer-owner", "n5-curriculum"
        )


def test_sidecar_code_and_launch_are_independently_fail_closed(
    tmp_path, monkeypatch
):
    code_sha256 = S.sidecar_code_sha256()
    launch = I.build_sidecar_launch_document(
        sidecar_code_sha256=code_sha256,
        backend_contract_sha256=S.CPU_FAKE_BACKEND_CONTRACT_SHA256,
    )
    with pytest.raises(I.EvaluationInboxError, match="code SHA"):
        I.validate_sidecar_launch_document(
            launch,
            actual_sidecar_code_sha256=code_sha256,
            backend_contract_sha256=(
                S.CPU_FAKE_BACKEND_CONTRACT_SHA256
            ),
        )
    monkeypatch.setattr(
        I,
        "TRUSTED_ACTION_BALL_EVALUATION_SIDECAR_CODE_SHA256",
        frozenset((code_sha256,)),
    )
    with pytest.raises(I.EvaluationInboxError, match="launch SHA"):
        I.validate_sidecar_launch_document(
            launch,
            actual_sidecar_code_sha256=code_sha256,
            backend_contract_sha256=(
                S.CPU_FAKE_BACKEND_CONTRACT_SHA256
            ),
        )
    monkeypatch.setattr(
        I,
        "TRUSTED_ACTION_BALL_EVALUATION_SIDECAR_LAUNCH_SHA256",
        frozenset((launch["content_sha256"],)),
    )
    tampered = deepcopy(launch)
    tampered["content"]["backend_contract_sha256"] = "d" * 64
    _rehash(tampered)
    with pytest.raises(I.EvaluationInboxError, match="contract|code"):
        I.validate_sidecar_launch_document(
            tampered,
            actual_sidecar_code_sha256=code_sha256,
            backend_contract_sha256=(
                S.CPU_FAKE_BACKEND_CONTRACT_SHA256
            ),
        )


def test_sidecar_launch_binds_exact_heartbeat_contract():
    launch = I.build_sidecar_launch_document(
        sidecar_code_sha256="1" * 64,
        backend_contract_sha256="2" * 64,
    )
    assert (
        launch["content"]["heartbeat_contract"]
        == I.SIDECAR_HEARTBEAT_CONTRACT
    )
    tampered = deepcopy(launch)
    tampered["content"]["heartbeat_contract"][
        "heartbeat_interval_seconds"
    ] = 5
    _rehash(tampered)
    with pytest.raises(
        I.EvaluationInboxError, match="heartbeat contract"
    ):
        I.validate_sidecar_launch_document(
            tampered,
            actual_sidecar_code_sha256="1" * 64,
            backend_contract_sha256="2" * 64,
            require_trust=False,
        )


def test_wrong_owner_and_replayed_evidence_fail_closed(
    tmp_path, monkeypatch
):
    launch, request0, evidence0, _ack0, queue = _accepted_pair(
        tmp_path, monkeypatch
    )
    wrong_path = queue.request_path(
        "other-owner", "n5-curriculum", 0
    )
    wrong_path.parent.mkdir(parents=True, exist_ok=True)
    wrong_path.write_bytes(_canonical_bytes(request0))
    with pytest.raises(I.EvaluationInboxError, match="another owner"):
        queue.load_request("other-owner", "n5-curriculum", 0)

    bindings = deepcopy(request0["content"]["bindings"])
    checkpoint = tmp_path / "model-generation-2.pt"
    checkpoint.write_bytes(b"new frozen checkpoint")
    bindings["checkpoint"] = I.artifact_receipt(checkpoint)
    request1 = _request(
        tmp_path,
        launch,
        seq=1,
        seed_start=1280,
        sample_start=11280,
        birth_start=21280,
        bindings=bindings,
    )
    queue.publish_request(request1)
    replay_path = queue.evidence_path(
        "trainer-owner", "n5-curriculum", 1
    )
    replay_path.write_bytes(_canonical_bytes(evidence0))
    with pytest.raises(
        I.EvaluationInboxError, match="request_seq|exact request"
    ):
        queue.load_evidence("trainer-owner", "n5-curriculum", 1)


def test_fixed_count_and_seed_overlap_are_rejected(
    tmp_path, monkeypatch
):
    launch, request0, _evidence, _ack, queue = _accepted_pair(
        tmp_path, monkeypatch
    )
    bad_count = deepcopy(request0)
    bad_count["content"]["windows"][0]["proposal_count"] = 319
    _rehash(bad_count)
    with pytest.raises(I.EvaluationInboxError, match="proposal_count"):
        I.validate_request_document(bad_count)

    bindings = deepcopy(request0["content"]["bindings"])
    checkpoint = tmp_path / "model-generation-2.pt"
    checkpoint.write_bytes(b"new frozen checkpoint")
    bindings["checkpoint"] = I.artifact_receipt(checkpoint)
    overlapping = _request(
        tmp_path,
        launch,
        seq=1,
        seed_start=1200,
        sample_start=50000,
        birth_start=60000,
        bindings=bindings,
    )
    with pytest.raises(I.EvaluationInboxError, match="seed ranges overlap"):
        queue.publish_request(overlapping)

    internal_overlap = deepcopy(overlapping)
    canary = internal_overlap["content"]["windows"][0]
    heldout = internal_overlap["content"]["windows"][1]
    heldout["seed_start"] = canary["seed_start"]
    heldout["seed_end_exclusive"] = (
        heldout["seed_start"] + I.HELDOUT_PROPOSALS
    )
    _rehash(internal_overlap)
    with pytest.raises(I.EvaluationInboxError, match="heldout seed"):
        I.validate_request_document(internal_overlap)


def test_checkpoint_or_motion_byte_drift_stops_sidecar(
    tmp_path, monkeypatch
):
    launch = _launch(monkeypatch)
    request = _request(tmp_path, launch)
    queue = I.EvaluationInbox(tmp_path / "inbox")
    queue.publish_request(request)
    checkpoint_path = Path(
        request["content"]["bindings"]["checkpoint"]["path"]
    )
    checkpoint_path.write_bytes(b"tampered after request publication")
    with pytest.raises(I.EvaluationInboxError, match="differ"):
        S.process_one(
            inbox_root=queue.root,
            owner_id="trainer-owner",
            run_id="n5-curriculum",
            launch_document=launch,
            backend=S.DeterministicCpuFakeEvaluator(),
        )


def test_runtime_bootstrap_byte_drift_stops_sidecar(
    tmp_path, monkeypatch
):
    launch = _launch(monkeypatch)
    request = _request(tmp_path, launch)
    queue = I.EvaluationInbox(tmp_path / "inbox")
    queue.publish_request(request)
    receipt_path = Path(
        request["content"]["bindings"][
            "runtime_bootstrap_receipt"
        ]["path"]
    )
    receipt_path.write_bytes(b"replaced bootstrap receipt")
    with pytest.raises(I.EvaluationInboxError, match="differ"):
        S.process_one(
            inbox_root=queue.root,
            owner_id="trainer-owner",
            run_id="n5-curriculum",
            launch_document=launch,
            backend=S.DeterministicCpuFakeEvaluator(),
        )


def test_evidence_count_tamper_and_ack_tamper_are_rejected(
    tmp_path, monkeypatch
):
    _launch_doc, request, evidence, ack, _queue = _accepted_pair(
        tmp_path, monkeypatch
    )
    count_tamper = deepcopy(evidence)
    count_tamper["content"]["windows"][0]["attempts"].pop()
    count_tamper["content"]["windows"][0]["ledger"]["proposed"] -= 1
    _rehash(count_tamper)
    with pytest.raises(I.EvaluationInboxError, match="attempt count"):
        I.validate_evidence_document(
            count_tamper, request_document=request
        )
    ack_tamper = deepcopy(ack)
    ack_tamper["content"]["evidence_sha256"] = "0" * 64
    _rehash(ack_tamper)
    with pytest.raises(I.EvaluationInboxError, match="evidence_sha256"):
        I.validate_ack_document(
            ack_tamper,
            request_document=request,
            evidence_document=evidence,
        )


def test_raw_terminal_precedence_is_code_classified():
    signals = {
        "infrastructure_invalid": False,
        "joint_actual_limit": True,
        "joint_qdes_limit": True,
        "fall": True,
        "table_hit": True,
        "collision": True,
        "legal_return": True,
    }
    assert I.classify_terminal_signals(signals) == "joint_actual_limit"
    signals["infrastructure_invalid"] = True
    assert I.classify_terminal_signals(signals) is None


def test_raw_safety_ledger_preserves_overlapping_sticky_flags():
    signals = {
        "infrastructure_invalid": False,
        "joint_actual_limit": True,
        "joint_qdes_limit": False,
        "fall": False,
        "table_hit": True,
        "collision": False,
        "legal_return": False,
    }
    ledger = I._derive_ledger(
        [
            {
                "solver_disposition": "admitted",
                "reject_reason": "",
                "installed": True,
                "started": True,
                "closed": True,
                "terminal_signals": signals,
            }
        ]
    )
    assert ledger["closed"] == 1
    assert ledger["joint_actual_limit"] == 1
    assert ledger["table_hit"] == 1
    assert ledger["legal_return"] == 0
    assert ledger["safe_nonreturn"] == 0


def test_runtime_identity_binds_git_object_format_and_oid(tmp_path):
    identity = _load_identity_module()
    assert (
        identity._git_object_id(
            "a" * 40,
            object_format="sha1",
            name="test HEAD",
        )
        == "a" * 40
    )
    assert (
        identity._git_object_id(
            "b" * 64,
            object_format="sha256",
            name="test HEAD",
        )
        == "b" * 64
    )
    for value, object_format in (
        ("c" * 64, "sha1"),
        ("D" * 40, "sha1"),
        ("e" * 40, "md5"),
    ):
        with pytest.raises(
            identity.FrozenEvaluationRuntimeIdentityError,
            match="Git object|unsupported",
        ):
            identity._git_object_id(
                value,
                object_format=object_format,
                name="test HEAD",
            )

    training_contract = tmp_path / "training_contract.json"
    env_pickle = tmp_path / "env.pkl"
    agent_pickle = tmp_path / "agent.pkl"
    training_contract.write_bytes(b"{}")
    env_pickle.write_bytes(b"env")
    agent_pickle.write_bytes(b"agent")
    document = identity.build_runtime_identity_document(
        repo_root=WBT.parents[1],
        task_id=identity.TASK_ID,
        training_launch_claim_sha256="f" * 64,
        training_contract_path=training_contract,
        environment_config_pickle_path=env_pickle,
        agent_config_pickle_path=agent_pickle,
    )
    source = document["content"]["source"]
    assert source["object_format"] in ("sha1", "sha256")
    assert source["head_commit_oid"] == identity._git_object_id(
        source["head_commit_oid"],
        object_format=source["object_format"],
        name="observed HEAD",
    )
    assert "head_commit_sha" not in source


def test_binding_tamper_and_noncanonical_file_are_rejected(
    tmp_path, monkeypatch
):
    launch = _launch(monkeypatch)
    request = _request(tmp_path, launch)
    tampered = deepcopy(request)
    tampered["content"]["bindings"]["reward_sha256"] = "f" * 64
    _rehash(tampered)
    with pytest.raises(I.EvaluationInboxError, match="run contract"):
        I.validate_request_document(tampered)

    queue = I.EvaluationInbox(tmp_path / "inbox")
    path = queue.request_path(
        "trainer-owner", "n5-curriculum", 0
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(request, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(I.EvaluationInboxError, match="canonical"):
        queue.load_request("trainer-owner", "n5-curriculum", 0)


def test_attempt_source_rebuilds_runtime_events_and_rejects_replay(
    tmp_path, monkeypatch
):
    launch = _launch(monkeypatch)
    bindings = _bindings(tmp_path)
    target = _target()
    target["selected_arm_key"] = R.ARM_KEYS[0]
    target["arm_levels"] = [0.0] * len(R.ARM_KEYS)
    request = I.build_request_document(
        owner_id="trainer-owner",
        run_id="n5-curriculum",
        request_seq=0,
        sidecar_launch_sha256=launch["content_sha256"],
        bindings=bindings,
        target=target,
        seed_start=0,
        sample_start=0,
        birth_start=0,
    )
    queue = I.EvaluationInbox(tmp_path / "inbox")
    queue.publish_request(request)
    assert (
        S.process_one(
            inbox_root=queue.root,
            owner_id="trainer-owner",
            run_id="n5-curriculum",
            launch_document=launch,
            backend=S.DeterministicCpuFakeEvaluator(),
        )
        is not None
    )
    source = I.FrozenSidecarInboxAttemptSource(
        inbox=queue,
        owner_id="trainer-owner",
        run_id="n5-curriculum",
        request_seq=0,
        runtime_module=R,
    )
    levels = R.ActionDomainLevels(
        **dict(zip(R.ARM_KEYS, target["arm_levels"]))
    )

    def proposal_request(offset):
        return R.FrozenEvaluationProposalRequest.create(
            policy_checkpoint_sha256=bindings["checkpoint"]["sha256"],
            policy_generation=bindings["policy_generation"],
            window_sha256="d" * 64,
            evidence_role="frozen_canary",
            proposal_offset=offset,
            seed=offset,
            sample_index=offset,
            birth_index=offset,
            action_uid=target["action_uid"],
            profile_sha256=target["profile_sha256"],
            mobility_mode=target["mobility_mode"],
            domain_epoch=target["domain_epoch"],
            domain_levels=levels,
            selected_arm_key=target["selected_arm_key"],
        )

    physics_request = proposal_request(0)
    physics_proposal = source.issue_proposal(physics_request)
    source.assert_exact_proposal(physics_request, physics_proposal)
    physics_solver = source.solver_event(
        physics_request, physics_proposal
    )
    assert physics_solver.disposition == "rejected"
    assert physics_solver.reject_reason.startswith("physics_invalid/")
    source.assert_solver_event(
        physics_request, physics_proposal, physics_solver
    )

    admitted_request = proposal_request(1)
    admitted_proposal = source.issue_proposal(admitted_request)
    admitted_solver = source.solver_event(
        admitted_request, admitted_proposal
    )
    assert admitted_solver.disposition == "admitted"
    installed = source.lifecycle_event(
        admitted_request,
        admitted_proposal,
        admitted_solver,
        "installed",
    )
    source.assert_lifecycle_event(
        admitted_request,
        admitted_proposal,
        admitted_solver,
        installed,
    )
    started = source.lifecycle_event(
        admitted_request,
        admitted_proposal,
        admitted_solver,
        "started",
    )
    source.assert_lifecycle_event(
        admitted_request,
        admitted_proposal,
        admitted_solver,
        started,
    )
    terminal = source.terminal_event(
        admitted_request, admitted_proposal, admitted_solver
    )
    source.assert_terminal_event(
        admitted_request,
        admitted_proposal,
        admitted_solver,
        terminal,
    )
    assert terminal.terminal_outcome == "legal_return"

    restored = I.FrozenSidecarInboxAttemptSource(
        inbox=queue,
        owner_id="trainer-owner",
        run_id="n5-curriculum",
        request_seq=0,
        runtime_module=R,
    )
    restored.load_state_dict(source.state_dict())
    assert restored.state_dict() == source.state_dict()
    with pytest.raises(I.EvaluationInboxError, match="replayed"):
        restored.issue_proposal(admitted_request)


def test_dynamic_source_constructs_empty_then_consumes_late_evidence(
    tmp_path, monkeypatch
):
    launch = _launch(monkeypatch)
    queue = I.EvaluationInbox(tmp_path / "dynamic-inbox")
    queue.initialize()
    source = I.FrozenSidecarInboxAttemptSource(
        inbox=queue,
        owner_id="trainer-owner",
        run_id="n5-curriculum",
        runtime_module=R,
    )
    owner_before = source.state_owner_sha256
    assert source.state_dict()["records"] == {}

    bindings = _bindings(tmp_path)
    request = I.build_request_document(
        owner_id="trainer-owner",
        run_id="n5-curriculum",
        request_seq=0,
        sidecar_launch_sha256=launch["content_sha256"],
        bindings=bindings,
        target=_formal_target(),
        seed_start=0,
        sample_start=0,
        birth_start=0,
    )
    queue.publish_request(request)
    _publish_sidecar_evidence(queue, launch)

    runtime_request = _runtime_request(request, offset=1)
    proposal = source.issue_proposal(runtime_request)
    source.assert_exact_proposal(runtime_request, proposal)
    assert source.state_owner_sha256 == owner_before
    state = source.state_dict()
    record = state["records"][runtime_request.reservation_sha256]
    assert record["request_seq"] == 0
    assert record["attempt_sha256"]


def test_v4_authority_binds_empty_source_before_late_evidence(
    tmp_path, monkeypatch
):
    sidecar_launch = _launch(monkeypatch)
    queue = I.EvaluationInbox(tmp_path / "authority-first-inbox")
    queue.initialize()
    source = I.FrozenSidecarInboxAttemptSource(
        inbox=queue,
        owner_id="trainer-owner",
        run_id="n5-curriculum",
        runtime_module=R,
    )
    target = _formal_target()
    key = C.ActionProfileKey(
        target["action_uid"],
        target["profile_sha256"],
        target["mobility_mode"],
    )
    evaluator_launch = E.launch_receipt_document_v4(
        curriculum_contract_sha256="a" * 64,
        profile_order=(key,),
        arm_catalog_sha256=C.ARM_CATALOG_SHA256,
        scheduler_contract_sha256=(
            C.ArmSchedulerConfig().contract_sha256
        ),
        sampler_sha256="6" * 64,
        solver_sha256="7" * 64,
        policy_contract_sha256="4" * 64,
        attempt_source_contract_sha256=(
            source.source_contract_sha256
        ),
        attempt_source_path=source.source_path,
        attempt_source_sha256=source.source_code_sha256,
    )
    evaluator_launch_sha256 = E._canonical_sha256(
        evaluator_launch
    )
    monkeypatch.setattr(
        E,
        "TRUSTED_FROZEN_EVALUATOR_V4_LAUNCH_RECEIPT_SHA256",
        frozenset((evaluator_launch_sha256,)),
    )
    authority = E.FrozenEvaluatorV4Authority.from_trusted_launch_receipt(
        evaluator_launch, attempt_source=source
    )
    owner_before = authority.binding_document()[
        "source_state_owner_sha256"
    ]

    bindings = _bindings(tmp_path)
    checkpoint_path = Path(bindings["checkpoint"]["path"])
    snapshot = authority.freeze_checkpoint(
        checkpoint_path.read_bytes(),
        policy_generation=bindings["policy_generation"],
    )
    request = I.build_request_document(
        owner_id="trainer-owner",
        run_id="n5-curriculum",
        request_seq=0,
        sidecar_launch_sha256=sidecar_launch["content_sha256"],
        bindings=bindings,
        target=target,
        seed_start=0,
        sample_start=0,
        birth_start=0,
    )
    queue.publish_request(request)
    _publish_sidecar_evidence(queue, sidecar_launch)
    assert source.state_owner_sha256 == owner_before

    session = authority.open_window(
        snapshot=snapshot,
        key=key,
        evidence_role="frozen_canary",
        domain_epoch=target["domain_epoch"],
        stratum=target["stratum"],
        selected_arm_key=target["selected_arm_key"],
        selection_round=target["selection_round"],
        arm_levels=tuple(target["arm_levels"]),
        rho=target["rho"],
    )
    handle = authority.issue_next(session)
    assert authority.capture_solver(handle) == "rejected"
    assert authority.attempt_rows(session)[0][
        "reject_reason"
    ].startswith("physics_invalid/")


def test_dynamic_source_supports_multiple_requests_without_cross_target(
    tmp_path, monkeypatch
):
    launch = _launch(monkeypatch)
    queue = I.EvaluationInbox(tmp_path / "multi-inbox")
    queue.initialize()
    source = I.FrozenSidecarInboxAttemptSource(
        inbox=queue,
        owner_id="trainer-owner",
        run_id="n5-curriculum",
        runtime_module=R,
    )
    stable_owner = source.state_owner_sha256

    bindings0 = _bindings(tmp_path)
    target0 = _formal_target(action_uid=11, level=0.0)
    request0 = I.build_request_document(
        owner_id="trainer-owner",
        run_id="n5-curriculum",
        request_seq=0,
        sidecar_launch_sha256=launch["content_sha256"],
        bindings=bindings0,
        target=target0,
        seed_start=0,
        sample_start=0,
        birth_start=0,
    )
    queue.publish_request(request0)
    _publish_sidecar_evidence(queue, launch)
    runtime0 = _runtime_request(request0, offset=1)
    proposal0 = source.issue_proposal(runtime0)
    solver0 = source.solver_event(runtime0, proposal0)
    assert solver0.disposition == "admitted"

    evidence0 = queue.load_evidence(
        "trainer-owner", "n5-curriculum", 0
    )
    resume_checkpoint = tmp_path / "consumer-resume-multi-000.pt"
    resume_checkpoint.write_bytes(b"persisted multi request state 000")
    queue.publish_ack(
        I.build_ack_document(
            request0,
            evidence0,
            consumer_code_sha256="c" * 64,
            consumer_state_sha256="d" * 64,
            consumer_checkpoint=I.artifact_receipt(
                resume_checkpoint
            ),
        )
    )

    bindings1 = deepcopy(bindings0)
    checkpoint1 = tmp_path / "model-generation-2.pt"
    checkpoint1.write_bytes(b"second frozen checkpoint")
    bindings1["checkpoint"] = I.artifact_receipt(checkpoint1)
    bindings1["policy_generation"] += 1
    bindings1["policy_state"] = I.state_binding(
        sha256="d" * 64, size_bytes=8192
    )
    bindings1["actor_obs_normalizer"] = I.state_binding(
        sha256="e" * 64, size_bytes=800
    )
    bindings1["critic_obs_normalizer"] = I.state_binding(
        sha256="f" * 64, size_bytes=1000
    )
    target1 = _formal_target(
        action_uid=29,
        profile_sha256="e" * 64,
        domain_epoch=5,
        level=0.2,
        selected_arm_key=R.ARM_KEYS[1],
    )
    request1 = I.build_request_document(
        owner_id="trainer-owner",
        run_id="n5-curriculum",
        request_seq=1,
        sidecar_launch_sha256=launch["content_sha256"],
        bindings=bindings1,
        target=target1,
        seed_start=1280,
        sample_start=1280,
        birth_start=1280,
    )
    queue.publish_request(request1)
    _publish_sidecar_evidence(queue, launch)
    assert source.state_owner_sha256 == stable_owner
    source.assert_exact_proposal(runtime0, proposal0)
    source.assert_solver_event(runtime0, proposal0, solver0)

    replay0 = _runtime_request(
        request0,
        offset=1,
        generation=2,
        window_sha256="a" * 64,
    )
    with pytest.raises(
        I.EvaluationInboxError,
        match="no accepted sidecar evidence row matches",
    ):
        source.issue_proposal(replay0)

    wrong_target = deepcopy(target0)
    wrong_target["arm_levels"] = target1["arm_levels"]
    wrong_target["selected_arm_key"] = target1[
        "selected_arm_key"
    ]
    wrong = _runtime_request(
        request1,
        offset=1,
        generation=bindings1["policy_generation"],
        window_sha256="1" * 64,
        target_override=wrong_target,
    )
    with pytest.raises(I.EvaluationInboxError, match="no accepted"):
        source.issue_proposal(wrong)

    runtime1 = _runtime_request(
        request1,
        offset=1,
        generation=bindings1["policy_generation"],
        window_sha256="2" * 64,
    )
    proposal1 = source.issue_proposal(runtime1)
    source.assert_exact_proposal(runtime1, proposal1)
    records = source.state_dict()["records"]
    assert {
        record["request_seq"] for record in records.values()
    } == {0, 1}


def test_dynamic_source_exact_restore_continues_same_row(
    tmp_path, monkeypatch
):
    launch = _launch(monkeypatch)
    queue = I.EvaluationInbox(tmp_path / "restore-inbox")
    bindings = _bindings(tmp_path)
    request = I.build_request_document(
        owner_id="trainer-owner",
        run_id="n5-curriculum",
        request_seq=0,
        sidecar_launch_sha256=launch["content_sha256"],
        bindings=bindings,
        target=_formal_target(),
        seed_start=0,
        sample_start=0,
        birth_start=0,
    )
    queue.publish_request(request)
    _publish_sidecar_evidence(queue, launch)
    source = I.FrozenSidecarInboxAttemptSource(
        inbox=queue,
        owner_id="trainer-owner",
        run_id="n5-curriculum",
        runtime_module=R,
    )
    runtime_request = _runtime_request(request, offset=1)
    proposal = source.issue_proposal(runtime_request)
    solver = source.solver_event(runtime_request, proposal)
    source.lifecycle_event(
        runtime_request, proposal, solver, "installed"
    )
    source.lifecycle_event(
        runtime_request, proposal, solver, "started"
    )
    frozen_state = deepcopy(source.state_dict())

    restored = I.FrozenSidecarInboxAttemptSource(
        inbox=queue,
        owner_id="trainer-owner",
        run_id="n5-curriculum",
        runtime_module=R,
    )
    assert restored.state_owner_sha256 == source.state_owner_sha256
    restored.load_state_dict(frozen_state)
    assert restored.state_dict() == frozen_state
    terminal = restored.terminal_event(
        runtime_request, proposal, solver
    )
    restored.assert_terminal_event(
        runtime_request, proposal, solver, terminal
    )
    assert terminal.terminal_outcome == "legal_return"

    tampered = deepcopy(frozen_state)
    tampered["records"][runtime_request.reservation_sha256][
        "request_seq"
    ] = 99
    with pytest.raises(I.EvaluationInboxError, match="missing|state"):
        restored.load_state_dict(tampered)


def test_scheduler_coordinator_replays_early_burn_and_recovers_ack(
    tmp_path, monkeypatch
):
    system = _v4_inbox_system(
        tmp_path,
        monkeypatch,
        suffix="scheduler-coordinator",
    )
    authority = system["authority"]
    bindings = system["bindings"]
    checkpoint = Path(bindings["checkpoint"]["path"])
    snapshot = authority.freeze_checkpoint(
        checkpoint.read_bytes(),
        policy_generation=bindings["policy_generation"],
    )
    selected_arm = R.ARM_KEYS[0]
    session = _open_sidecar_session(
        authority,
        snapshot,
        system["key"],
        role="scheduler",
        domain_epoch=4,
        stratum=f"marginal:{selected_arm}",
        selected_arm_key=selected_arm,
        selection_round=8,
        arm_levels=(0.0,) * len(R.ARM_KEYS),
        rho=0.0,
    )
    coordinator = system["coordinator"]
    assert coordinator.publish_sessions(
        sessions=(session,),
        bindings=bindings,
    ) == 0
    request = system["queue"].load_request(
        "trainer-owner",
        "n5-curriculum",
        0,
    )
    assert [row["role"] for row in request["content"]["windows"]] == [
        "scheduler"
    ]

    class EarlyInfrastructureBurn(
        S.DeterministicCpuFakeEvaluator
    ):
        def evaluate(self, request_document):
            rows = super().evaluate(request_document)
            attempt = rows["scheduler"][1]
            attempt.update(
                {
                    "installed": False,
                    "started": False,
                    "closed": False,
                    "terminal_signals": {
                        **attempt["terminal_signals"],
                        "infrastructure_invalid": True,
                        "legal_return": False,
                    },
                }
            )
            return rows

    assert (
        S.process_one(
            inbox_root=system["queue"].root,
            owner_id="trainer-owner",
            run_id="n5-curriculum",
            launch_document=system["sidecar_launch"],
            backend=EarlyInfrastructureBurn(),
        )
        is not None
    )
    evidence = system["queue"].load_evidence(
        "trainer-owner",
        "n5-curriculum",
        0,
    )
    attempts = evidence["content"]["windows"][0]["attempts"]
    assert {
        stratum: sum(
            row["sampling_stratum"] == stratum for row in attempts
        )
        for stratum in ("center", "interior", "frontier")
    } == {"center": 20, "interior": 60, "frontier": 20}
    assert all(
        row["frontier_arm"] == selected_arm
        for row in attempts
        if row["sampling_stratum"] == "frontier"
    )

    mixture_tamper = deepcopy(evidence)
    tamper_window = mixture_tamper["content"]["windows"][0]
    tamper_window["attempts"][0]["sampling_stratum"] = "interior"
    tamper_window["attempt_receipt_root_sha256"] = (
        I._attempt_receipt_root(tamper_window["attempts"])
    )
    _rehash(mixture_tamper)
    with pytest.raises(I.EvaluationInboxError, match="20/60/20"):
        I.validate_evidence_document(
            mixture_tamper,
            request_document=request,
        )

    capability = coordinator.consume_evidence(0)
    exact = authority.assert_scheduler_capabilities_many(
        {system["key"]: capability}
    )
    ledger, runtime_rows = (
        exact[system["key"]][0].ledger,
        exact[system["key"]][1],
    )
    assert ledger.P == I.SCHEDULER_PROPOSALS
    assert ledger.X == 1
    assert runtime_rows[1]["infrastructure_invalid"] is True
    assert runtime_rows[1]["closed"] is False
    assert runtime_rows[1]["terminal_outcome"] is None
    authority.consume_scheduler_capabilities_many(
        {system["key"]: capability}
    )
    coordinator.mark_curriculum_consumed(0)
    coordinator.prepare_ack(0)
    authority_state = deepcopy(authority.state_dict())
    coordinator_state = deepcopy(coordinator.state_dict())
    consumer_state_sha = I.canonical_sha256(
        {
            "authority": authority_state,
            "coordinator": coordinator_state,
        }
    )
    resume = tmp_path / "scheduler-consumer-resume.pt"
    resume.write_bytes(
        _canonical_bytes(
            {
                "authority": authority_state,
                "coordinator": coordinator_state,
            }
        )
    )
    coordinator.publish_ack(
        0,
        consumer_state_sha256=consumer_state_sha,
        consumer_checkpoint=I.artifact_receipt(resume),
    )

    restored = _v4_inbox_system(
        tmp_path,
        monkeypatch,
        suffix="scheduler-coordinator",
    )
    restored["authority"].load_state_dict(authority_state)
    restored["coordinator"].load_state_dict(coordinator_state)
    restored["coordinator"].reconcile_ack(0)
    assert restored["coordinator"].state_dict()["records"][0][
        "stage"
    ] == "acked"

    checkpoint_2 = tmp_path / "model-generation-2.pt"
    checkpoint_2.write_bytes(b"second exact frozen policy checkpoint")
    bindings_2 = deepcopy(restored["bindings"])
    bindings_2["checkpoint"] = I.artifact_receipt(checkpoint_2)
    bindings_2["policy_generation"] = 117
    bindings_2["policy_state"] = I.state_binding(
        sha256="d" * 64,
        size_bytes=8192,
    )
    snapshot_2 = restored["authority"].freeze_checkpoint(
        checkpoint_2.read_bytes(),
        policy_generation=bindings_2["policy_generation"],
    )
    session_2 = _open_sidecar_session(
        restored["authority"],
        snapshot_2,
        restored["key"],
        role="scheduler",
        domain_epoch=5,
        stratum=f"marginal:{selected_arm}",
        selected_arm_key=selected_arm,
        selection_round=9,
        arm_levels=(0.0,) * len(R.ARM_KEYS),
        rho=0.0,
    )
    assert restored["coordinator"].publish_sessions(
        sessions=(session_2,),
        bindings=bindings_2,
    ) == 1


def test_formal_coordinator_consumes_release_through_curriculum(
    tmp_path, monkeypatch
):
    system = _v4_inbox_system(
        tmp_path,
        monkeypatch,
        suffix="formal-coordinator",
    )
    authority = system["authority"]
    key = system["key"]
    curriculum = C.ActionBallCurriculum(
        contract_sha256="a" * 64,
        profile_order=(key,),
        sampler_sha256="6" * 64,
        solver_sha256="7" * 64,
        policy_contract_sha256="4" * 64,
        config=C.BallCurriculumConfig(),
        scheduler_config=C.ArmSchedulerConfig(),
        evaluator_authority=authority,
    )
    domain = curriculum.selected_formal_domain(key)
    assert domain is not None
    checkpoint = Path(system["bindings"]["checkpoint"]["path"])
    snapshot = authority.freeze_checkpoint(
        checkpoint.read_bytes(),
        policy_generation=system["bindings"]["policy_generation"],
    )
    sessions = tuple(
        _open_sidecar_session(
            authority,
            snapshot,
            key,
            role=role,
            domain_epoch=domain.domain_epoch,
            stratum=domain.stratum,
            selected_arm_key=domain.selected_arm_key,
            selection_round=domain.selection_round,
            arm_levels=domain.arm_levels,
            rho=domain.rho,
        )
        for role in ("frozen_canary", "frozen_heldout")
    )
    coordinator = system["coordinator"]
    assert coordinator.publish_sessions(
        sessions=sessions,
        bindings=system["bindings"],
    ) == 0
    _publish_sidecar_evidence(
        system["queue"],
        system["sidecar_launch"],
    )
    release = coordinator.consume_evidence(0)
    canary = release.canary_evidence.ledger
    heldout = release.heldout_evidence.ledger
    assert canary.P == 320 and canary.safe_closed >= 256
    assert heldout.P == 960 and heldout.safe_closed >= 768
    assert canary.U_table > 0
    assert canary.U_joint_actual > 0
    decisions = curriculum.stage_selected({key: release})
    assert decisions[0].kind == "canary_blocked"
    assert "table_hit_zero_tolerance" in decisions[0].blockers
    coordinator.mark_curriculum_consumed(0)
    coordinator.prepare_ack(0)
    consumer_state = {
        "authority": authority.state_dict(),
        "coordinator": coordinator.state_dict(),
        "curriculum": curriculum.state_dict(),
    }
    resume = tmp_path / "formal-consumer-resume.pt"
    resume.write_bytes(_canonical_bytes(consumer_state))
    coordinator.publish_ack(
        0,
        consumer_state_sha256=I.canonical_sha256(consumer_state),
        consumer_checkpoint=I.artifact_receipt(resume),
    )
    assert system["queue"].load_ack(
        "trainer-owner",
        "n5-curriculum",
        0,
    )["content"]["consumer_checkpoint"]["sha256"] == (
        I.artifact_receipt(resume)["sha256"]
    )
