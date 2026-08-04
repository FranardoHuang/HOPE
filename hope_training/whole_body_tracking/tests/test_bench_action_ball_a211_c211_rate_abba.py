from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "bench_action_ball_a211_c211_rate_abba.py"
)
SPEC = importlib.util.spec_from_file_location("bench_action_ball_a211_c211_rate_abba", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
B = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = B
SPEC.loader.exec_module(B)


def _write_canonical(path: Path, value: object) -> str:
    path.write_bytes(B.canonical_bytes(value) + b"\n")
    return B.sha256_file(path)


def _launcher_source(kit_relative: str) -> str:
    return f'''\
KIT_LAUNCHER_SOURCE = {kit_relative!r}

import hashlib
import json
from pathlib import Path


class _AdmissionConstants:
    GPU_NAMESPACE_RECEIPT_ENV = "HOPE_FIXTURE_GPU_NAMESPACE_RECEIPT"
    GPU_NAMESPACE_RECEIPT_SHA_ENV = "HOPE_FIXTURE_GPU_NAMESPACE_RECEIPT_SHA256"


_A = _AdmissionConstants()


class _RuntimeBridge:
    @staticmethod
    def _runtime_asset_exec_environment(_runtime_assets):
        return {{}}


_B = _RuntimeBridge()


def _runtime_namespace_receipt(spec, claim_sha):
    path = Path(spec["namespace"]) / "fixture_namespace_receipt.json"
    raw = json.dumps({{"claim": claim_sha}}, sort_keys=True, separators=(",", ":")).encode() + b"\\n"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _prelong_semantics_exec_environment(stage, reward_sha):
    if stage != "scale4096" or len(reward_sha) != 64:
        raise ValueError("invalid fixture prelong authority")
    return {{"HOPE_FIXTURE_PRELONG": reward_sha}}


def _revalidate_claim_payload(payload, claimed=False):
    if payload.get("fixture_validated") is not True:
        raise ValueError("fixture claim is not validated")
    if claimed is not False:
        raise ValueError("existing plan must use the non-claiming validator path")
    return {{}}, {{}}, {{}}
'''


def _deferred_rate_contract() -> dict[str, object]:
    return {
        "num_envs": 4096,
        "steps_per_env": 24,
        "warmup_updates": 10,
        "minimum_measured_updates": 50,
        "abba_order": ["current_A", "current_C", "current_C", "current_A"],
        "main_timing_mode": "profiler_off",
        "isolation": "exclusive_single_process_same_gpu",
    }


def _base_training_argv(
    *, family: str, python: Path, training_source: Path, run_name: str
) -> list[str]:
    target = "online_solver" if family == "A211" else "direct_ball"
    reuse = "true" if family == "A211" else "false"
    return [
        str(python),
        str(training_source),
        "seed=7",
        "num_envs=4096",
        "max_iterations=5",
        "algo.runner.save_interval=1",
        f"run_name={run_name}",
        "algo.runner.empirical_normalization=true",
        "algo.policy.actor_hidden_dims=[512,256,128]",
        "algo.policy.critic_hidden_dims=[512,256,128]",
        "algo.policy.init_noise_std=0.02",
        "algo.policy.noise_std_type=scalar",
        "algo.algorithm.entropy_coef=0.01",
        "algo.algorithm.schedule=fixed",
        "algo.algorithm.learning_rate=0.0001",
        "algo.algorithm.desired_kl=0.01",
        "algo.algorithm.clip_param=0.2",
        "algo.algorithm.num_learning_epochs=5",
        "algo.algorithm.num_mini_batches=4",
        "action_ball_dynamic_ready_artifact_sha256=" + "1" * 64,
        "action_ball_dynamic_ready_nominal_receipt_sha256=" + "2" * 64,
        "motion_file=/fixture/motion.npz",
        "task.racket.clip_names=[fixture_clip]",
        "task.racket.action_ball_manifest_sha256=" + "3" * 64,
        "task.racket.action_ball_seed=20260804",
        "task.actions.control_step_action_delay_min=0",
        "task.actions.control_step_action_delay_max=0",
        "task.push.enable=false",
        "task.physical_ball=true",
        "task.racket.virtual_ball=false",
        "task.racket.action_ball_target_observation_noise=false",
        f"task.racket.action_ball_target_source={target}",
        "task.racket.action_ball_reuse_exact_question_until_semantics_change=" + reuse,
    ]


def _plan_envelope(payload: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "fixture_validated_launcher_plan",
        "launch_claim_sha256": B.canonical_sha256(payload),
        "canonical_payload": payload,
    }


@pytest.fixture()
def validated_pair(tmp_path: Path) -> dict[str, object]:
    root = tmp_path.resolve()
    checkout = root / "checkout"
    scripts = checkout / "hope_training" / "whole_body_tracking" / "scripts"
    scripts.mkdir(parents=True)
    experiment_root = root / "experiments"
    (experiment_root / "a").mkdir(parents=True)
    (experiment_root / "c").mkdir(parents=True)
    outputs = root / "outputs"
    outputs.mkdir()

    training_source = scripts / "train.py"
    training_source.write_text("raise SystemExit('fixture only')\n", encoding="utf-8")
    kit = scripts / "launch_kit_training_locked.sh"
    kit.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    kit.chmod(0o755)
    kit_relative = kit.relative_to(checkout).as_posix()
    a_launcher = scripts / "launch_fixture_a211.py"
    c_launcher = scripts / "launch_fixture_c211.py"
    a_launcher.write_text(_launcher_source(kit_relative), encoding="utf-8")
    c_launcher.write_text(_launcher_source(kit_relative), encoding="utf-8")

    source = {
        "checkout": str(checkout),
        "commit_sha": "a" * 40,
        "isaac_python": str(Path(sys.executable).resolve()),
    }
    gpu = {
        "index": 0,
        "uuid": "GPU-fixture-0000",
        "lock_path": str(root / "gpu0.lock"),
        "require_empty": True,
    }
    common_recipe = {
        "ppo": {"learning_rate": 0.0001, "schedule": "fixed"},
        "soft_weights": {"balance": 1.0, "mimic": 1.0},
        "reference_guard_mode": "fixture_exact",
    }
    four_grid = {"path": "/fixture/grid.json", "sha256": "4" * 64}
    launcher_shas = {
        "A211": B.sha256_file(a_launcher),
        "C211": B.sha256_file(c_launcher),
    }
    launcher_paths = {"A211": a_launcher, "C211": c_launcher}
    kit_sha = B.sha256_file(kit)
    plans: dict[str, dict[str, str]] = {}
    payloads: dict[str, dict[str, object]] = {}
    for family in ("A211", "C211"):
        family_lower = family.lower()
        launcher = launcher_paths[family]
        recipe_key = "arm" if family == "A211" else "recipe"
        payload: dict[str, object] = {
            "fixture_validated": True,
            "diagnostic_unauthorized": True,
            "formal_evidence_prohibited": True,
            "promotion_prohibited": True,
            "resume_prohibited": True,
            "export_prohibited": True,
            "deployment_prohibited": True,
            "hardware_prohibited": True,
            "fresh_only": True,
            "spec": {
                "source": copy.deepcopy(source),
                "gpu": copy.deepcopy(gpu),
                "stage": "scale4096",
                "num_envs": 4096,
                "max_iterations": 5,
                "save_interval": 1,
                "namespace": str(experiment_root / family_lower / "validated-input"),
                "colocation_allowed": False,
            },
            "output_contract": {
                "update_profile": {
                    "forwarded_value": None,
                    "mode": "not_requested",
                },
                "deferred_matched_speed_measurement": _deferred_rate_contract(),
            },
            "training_argv": _base_training_argv(
                family=family,
                python=Path(sys.executable).resolve(),
                training_source=training_source,
                run_name="validated-input-" + family_lower,
            ),
            "runtime_sources": {
                "launcher": {
                    "path": launcher.relative_to(checkout).as_posix(),
                    "sha256": launcher_shas[family],
                },
                "kit_launcher": {
                    "path": kit_relative,
                    "sha256": kit_sha,
                },
            },
            "runtime_assets": {"fixture": True},
            "materialization_inputs": {
                (
                    "arm_materialization"
                    if family == "A211"
                    else "reward_materialization"
                ): {"runtime_effective_reward_sha256": "5" * 64}
            },
            "bundle": {
                recipe_key: copy.deepcopy(common_recipe),
                "isaac_four_grid_manifest": copy.deepcopy(four_grid),
            },
        }
        plan_path = outputs / (family_lower + "-validated-plan.json")
        plan_sha = _write_canonical(plan_path, _plan_envelope(payload))
        plans[family] = {"path": str(plan_path), "sha256": plan_sha}
        payloads[family] = payload

    return {
        "root": root,
        "checkout": checkout,
        "outputs": outputs,
        "plans": plans,
        "payloads": payloads,
        "launchers": {
            "A211": {"path": str(a_launcher), "sha256": launcher_shas["A211"]},
            "C211": {"path": str(c_launcher), "sha256": launcher_shas["C211"]},
        },
        "receipt": outputs / "rate-receipt.json",
    }


def _build(pair: dict[str, object]) -> dict[str, object]:
    plans = pair["plans"]
    launchers = pair["launchers"]
    assert isinstance(plans, dict) and isinstance(launchers, dict)
    return B.build_benchmark_plan(
        a_plan_path=Path(plans["A211"]["path"]),
        a_plan_sha256=plans["A211"]["sha256"],
        c_plan_path=Path(plans["C211"]["path"]),
        c_plan_sha256=plans["C211"]["sha256"],
        a_launcher_path=Path(launchers["A211"]["path"]),
        a_launcher_sha256=launchers["A211"]["sha256"],
        c_launcher_path=Path(launchers["C211"]["path"]),
        c_launcher_sha256=launchers["C211"]["sha256"],
        run_prefix="fixture-rate",
        receipt_path=pair["receipt"],
    )


def _reseal_base_plan(pair: dict[str, object], family: str) -> None:
    plans = pair["plans"]
    payloads = pair["payloads"]
    plan_path = Path(plans[family]["path"])
    plans[family]["sha256"] = _write_canonical(
        plan_path, _plan_envelope(payloads[family])
    )


def _rate_log(family: str, *, seconds_offset: float = 0.0, first_update: int = 0) -> str:
    rows: list[str] = []
    for index in range(B.MAX_ITERATIONS):
        seconds = 1.0 + seconds_offset + index * 0.001
        rows.append(f"Iteration time: {seconds:.3f}s")
        measured_index = index - B.WARMUP_UPDATES
        reset_present = 0 <= measured_index < B.MEASURED_UPDATES and measured_index % 2 == 1
        counters = {
            "terminal_reset_count": 2 if reset_present else 0,
            "timeout_reset_count": 1 if reset_present else 0,
            "termination_reason_fallen": 2 if reset_present else 0,
        }
        behavior = {"ppo_update": first_update + index, "counters": counters}
        rows.append(B.BEHAVIOR_PREFIX + json.dumps(behavior, sort_keys=True))
        cache = None
        if family == "A211":
            cache = {
                "policy": "reuse_exact_question_until_semantics_change",
                "novel_producer_count": 1,
                "consumer_hit_count": index + 1,
            }
        ledger = {
            "event": B.TRAINING_LEDGER_EVENT,
            "step": first_update + index,
            "solver_rejections": {"0": {"fixture": 0}},
            "exact_question_answer_cache": cache,
        }
        rows.append(json.dumps(ledger, sort_keys=True))
    return "\n".join(rows) + "\n"


def test_build_revalidates_pins_and_only_derives_the_rate_allowlist(
    validated_pair: dict[str, object],
) -> None:
    plan = _build(validated_pair)
    assert plan["order"] == ["A211", "C211", "C211", "A211"]
    assert plan["timing_contract"] == {
        "mode": "profiler_off_rsl_iteration_wall",
        "profile_environment": {B.PROFILE_ENV: "0"},
        "profile_rows_required": 0,
        "warmup_updates": 10,
        "measured_updates": 50,
        "tail_updates": 1,
        "total_updates": 61,
        "ci_method": "contiguous_5_update_batch_means_student_t",
        "ci_confidence": 0.95,
        "ci_block_length_updates": 5,
        "ci_block_count": 10,
        "isolation": "exclusive_single_process_same_gpu_full_abba_lock",
    }
    for index, block in enumerate(plan["blocks"]):
        assignments = B._assignment_rows(block["training_argv"], name="test argv")
        assert block["order_index"] == index
        assert assignments["max_iterations"] == "61"
        assert assignments["algo.runner.save_interval"] == "1000"
        assert assignments["run_name"] == block["run_name"]
        assert [row["field"] for row in block["argv_mutations"]] == [
            "max_iterations",
            "algo.runner.save_interval",
            "run_name",
        ]
        assert block["profiler_environment"] == {B.PROFILE_ENV: "0"}

    plan_path = validated_pair["outputs"] / "abba-plan.json"
    plan_sha = _write_canonical(plan_path, plan)
    assert B._load_and_revalidate_benchmark_plan(plan_path, plan_sha) == plan


def test_launcher_repin_detects_drift_after_plan_creation(
    validated_pair: dict[str, object],
) -> None:
    plan = _build(validated_pair)
    plan_path = validated_pair["outputs"] / "abba-plan.json"
    plan_sha = _write_canonical(plan_path, plan)
    launcher = Path(validated_pair["launchers"]["A211"]["path"])
    launcher.write_text(launcher.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(B.RateBenchRefused, match="launcher SHA-256 differs"):
        B._load_and_revalidate_benchmark_plan(plan_path, plan_sha)


def test_runtime_environment_comes_from_the_pinned_launcher_authority(
    validated_pair: dict[str, object],
) -> None:
    plan = _build(validated_pair)
    block = plan["blocks"][0]
    namespace = Path(block["namespace"])
    namespace.mkdir(parents=True)
    launcher_pin = validated_pair["launchers"]["A211"]
    launcher = B._load_launcher(
        Path(launcher_pin["path"]), launcher_pin["sha256"], family="A211"
    )
    environment = B._launcher_runtime_exec_environment(
        launcher=launcher,
        payload=validated_pair["payloads"]["A211"],
        block=block,
    )
    receipt = Path(environment["HOPE_FIXTURE_GPU_NAMESPACE_RECEIPT"])
    assert receipt.parent == namespace
    assert B.sha256_file(receipt) == environment[
        "HOPE_FIXTURE_GPU_NAMESPACE_RECEIPT_SHA256"
    ]
    assert environment["HOPE_FIXTURE_PRELONG"] == "5" * 64


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["output_contract"]["update_profile"].update(
                {"forwarded_value": "1", "mode": "enabled"}
            ),
            "profiler-instrumented",
        ),
        (
            lambda payload: payload["spec"].update({"colocation_allowed": True}),
            "disable colocation",
        ),
    ],
)
def test_base_plan_rejects_nonexclusive_or_profiled_inputs(
    validated_pair: dict[str, object], mutation, message: str
) -> None:
    mutation(validated_pair["payloads"]["A211"])
    _reseal_base_plan(validated_pair, "A211")
    with pytest.raises(B.RateBenchRefused, match=message):
        _build(validated_pair)


def test_cross_family_optimizer_drift_is_rejected(
    validated_pair: dict[str, object],
) -> None:
    argv = validated_pair["payloads"]["C211"]["training_argv"]
    argv[argv.index("algo.algorithm.learning_rate=0.0001")] = (
        "algo.algorithm.learning_rate=0.0002"
    )
    _reseal_base_plan(validated_pair, "C211")
    with pytest.raises(B.RateBenchRefused, match="learning_rate differs"):
        _build(validated_pair)


def test_derive_training_argv_refuses_missing_or_duplicate_budget_fields() -> None:
    base = [
        "/python",
        "/train.py",
        "max_iterations=5",
        "algo.runner.save_interval=1",
        "run_name=old",
    ]
    derived, mutations = B._derive_training_argv(base, run_name="new")
    assert derived[2:] == [
        "max_iterations=61",
        "algo.runner.save_interval=1000",
        "run_name=new",
    ]
    assert len(mutations) == 3
    with pytest.raises(B.RateBenchRefused, match="exactly one max_iterations"):
        B._derive_training_argv(base + ["+max_iterations=5"], run_name="new")
    with pytest.raises(B.RateBenchRefused, match="exactly one run_name"):
        B._derive_training_argv(base[:-1], run_name="new")


@pytest.mark.parametrize(("family", "offset"), [("A211", 0.0), ("C211", 0.2)])
def test_log_analysis_excludes_warmup_and_tail_and_reports_cache_and_strata(
    family: str, offset: float
) -> None:
    analysis = B.analyze_run_log(
        _rate_log(family, seconds_offset=offset, first_update=11), family=family
    )
    assert analysis["update_ids"]["warmup"] == list(range(11, 21))
    assert analysis["update_ids"]["measured"] == list(range(21, 71))
    assert analysis["update_ids"]["tail"] == [71]
    assert len(analysis["timing"]["measured_update_rows"]) == 50
    assert len(analysis["timing"]["block_confidence_interval"]["block_means_s"]) == 10
    assert analysis["reset_strata"]["reset_free"]["update_count"] == 25
    assert analysis["reset_strata"]["reset_present"]["update_count"] == 25
    inverse = analysis["solver_inverse_deltas"]
    assert inverse["measured"]["online_inverse_solve_count_delta"] == 0
    assert inverse["tail"]["online_inverse_solve_count_delta"] == 0
    assert inverse["warmup"]["online_inverse_solve_count_delta"] == (
        1 if family == "A211" else 0
    )


def test_log_analysis_rejects_profiler_leak_and_noncontiguous_updates() -> None:
    text = _rate_log("A211")
    with pytest.raises(B.RateBenchRefused, match="profiler instrumentation leaked"):
        B.analyze_run_log(B.PROFILE_PREFIX + "{}\n" + text, family="A211")

    lines = text.splitlines()
    behavior_index = 3 * 20 + 1
    row = json.loads(lines[behavior_index][len(B.BEHAVIOR_PREFIX) :])
    row["ppo_update"] += 1
    lines[behavior_index] = B.BEHAVIOR_PREFIX + json.dumps(row, sort_keys=True)
    with pytest.raises(B.RateBenchRefused, match="common contiguous window"):
        B.analyze_run_log("\n".join(lines) + "\n", family="A211")


def test_execute_writes_rate_only_receipt_with_symmetric_abba_contrast(
    validated_pair: dict[str, object],
) -> None:
    plan = _build(validated_pair)
    plan_path = validated_pair["outputs"] / "abba-plan.json"
    plan_sha = _write_canonical(plan_path, plan)

    def fake_probe(gpu_index: int, gpu_uuid: str) -> dict[str, object]:
        return {
            "gpu_index": gpu_index,
            "gpu_uuid": gpu_uuid,
            "compute_processes": [],
            "empty": True,
        }

    def fake_runner(
        _plan_path: Path, _plan_sha: str, block: dict[str, object]
    ) -> dict[str, object]:
        offset = 0.1 if block["family"] == "A211" else 0.0
        return {
            "family": block["family"],
            "order_index": block["order_index"],
            "analysis": B.analyze_run_log(
                _rate_log(block["family"], seconds_offset=offset),
                family=block["family"],
            ),
        }

    result = B.execute_benchmark(
        plan_path,
        plan_sha,
        block_runner=fake_runner,
        gpu_probe=fake_probe,
    )
    assert result["status"] == "COMPLETE_DIAGNOSTIC_RATE_ONLY"
    receipt_path = Path(result["receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["order"] == ["A211", "C211", "C211", "A211"]
    assert receipt["diagnostic_unauthorized"] is True
    assert receipt["rate_only"] is True
    assert receipt["interpretation"] == {
        "consumer_rate_only": True,
        "novel_question_producer_rate_excluded": True,
        "profile_attribution_excluded": True,
        "scale4096_or_long_readiness_claimed": False,
    }
    assert len(receipt["exclusive_gpu_probes"]) == 5
    assert receipt["abba_contrast"]["a_mean_iteration_time_s"] > receipt[
        "abba_contrast"
    ]["c_mean_iteration_time_s"]
    with pytest.raises(B.RateBenchRefused, match="already exists"):
        B.execute_benchmark(
            plan_path,
            plan_sha,
            block_runner=fake_runner,
            gpu_probe=fake_probe,
        )


def test_exclusive_json_writer_never_clobbers(tmp_path: Path) -> None:
    output = tmp_path.resolve() / "result.json"
    B._write_exclusive_json(output, {"first": True})
    with pytest.raises(B.RateBenchRefused, match="already exists"):
        B._write_exclusive_json(output, {"second": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"first": True}
