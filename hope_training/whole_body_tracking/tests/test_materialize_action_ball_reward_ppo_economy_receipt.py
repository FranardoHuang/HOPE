"""Host-light tests for the r6 ActionBall reward/PPO economy receipt."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/scripts/"
    "materialize_action_ball_reward_ppo_economy_receipt.py"
)
SPEC = importlib.util.spec_from_file_location("_reward_ppo_economy_receipt", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


# Independent, literal excerpts from the SHA-pinned rsl_rl 2.3.1 sources used
# on the exact Pod.  Do not construct these fixtures from
# ``M._RSL_SOURCE_MARKERS``: doing so would let a producer-marker regression
# rewrite both the assertion and its alleged evidence in one edit.
_PINNED_RSL_SOURCE_FIXTURES = {
    "ppo": b"\n".join(
        (
            b"normalize_advantage=not self.normalize_advantage_per_mini_batch",
            b"self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()",
            b"            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)",
        )
    )
    + b"\n",
    "rollout_storage": (
        b"self.advantages = (self.advantages - self.advantages.mean()) / "
        b"(self.advantages.std() + 1e-8)\n"
    ),
    "actor_critic": b"\n".join(
        (
            b'elif self.noise_std_type == "log":',
            b"self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))",
            b"std = torch.exp(self.log_std).expand_as(mean)",
            b"return self.distribution.entropy().sum(dim=-1)",
        )
    )
    + b"\n",
}


def _canonical(document):
    return json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _copy(repo: Path, relative: str) -> None:
    target = repo / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / relative, target)


def _write_json(path: Path, document) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical(document) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _r6_contract_from_r5(action_id: str, *, log_std: bool = True):
    source = (
        REPO_ROOT
        / "configs/a3_vendor_runtime_authority_20260801_r5/"
        f"{action_id}.shared_ready.training_contract.json"
    )
    contract = json.loads(source.read_text())
    policy = contract["action_ball_ppo_runner_recipe"]["recipe"]["policy"]
    policy["noise_std_type"] = "log" if log_std else "scalar"
    bootstrap = contract["action_ball_training"]["policy_bootstrap"]
    bootstrap["initialization"]["noise_std_type"] = (
        "log" if log_std else "scalar"
    )
    bootstrap["initialization"]["required_realized_init_noise_std"] = 0.02
    contract["action_ball_ppo_runner_recipe"]["recipe"][
        "policy_initialization"
    ] = deepcopy(bootstrap)
    reward = contract["effective_reward_recipe"]
    adopted = {
        "death_penalty": -300.0,
        "virtual_landing": 500.0,
        "qdes_limit_barrier": -5.0,
        "joint_limit": -5.0,
    }
    for term in reward["terms"]:
        if term["name"] in adopted:
            term["weight"] = adopted[term["name"]]
    reward_payload = {
        "schema_version": reward["schema_version"],
        "terms": reward["terms"],
    }
    reward["sha256"] = hashlib.sha256(_canonical(reward_payload)).hexdigest()
    contract["action_ball_training"][
        "effective_reward_recipe_sha256"
    ] = reward["sha256"]
    recipe = contract["action_ball_ppo_runner_recipe"]["recipe"]
    contract["action_ball_ppo_runner_recipe"]["sha256"] = hashlib.sha256(
        _canonical(recipe)
    ).hexdigest()
    return contract


def _fake_registry_pins(repo: Path, *, log_std: bool = True):
    pins = []
    for action_id in M.ACTION_IDS:
        relative = (
            "configs/a3_vendor_runtime_authority_20260802_r9/"
            f"{action_id}.shared_ready.training_contract.json"
        )
        digest = _write_json(
            repo / relative,
            _r6_contract_from_r5(action_id, log_std=log_std),
        )
        pins.append((action_id, relative, digest))
    identities = []
    for action_id in M.ACTION_IDS:
        identity = {
            "schema_version": 1,
            "action_id": action_id,
            "scope": "upper",
            "planned_paths": {
                "reward_economy_receipt": (
                    "configs/n1_reward_economy_20260802_r9/"
                    "reward_economy.v1.json"
                )
            },
        }
        identities.append(
            {
                "action_id": action_id,
                "identity": identity,
                "sha256": hashlib.sha256(_canonical(identity)).hexdigest(),
            }
        )
    return (
        pins,
        {
            "path": "configs/n1_reward_economy_20260802_r9/reward_economy.v1.json",
            "sha256": None,
        },
        identities,
    )


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _commit_fixture(repo: Path) -> None:
    _git(repo, "init")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=Receipt Test",
        "-c",
        "user.email=receipt-test@example.invalid",
        "commit",
        "-m",
        "fixture",
    )


def _fake_rsl_root(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "site-packages/rsl_rl"
    pins = []
    for role, relative, _ in M.RSL_RUNTIME_SOURCE_PINS:
        payload = _PINNED_RSL_SOURCE_FIXTURES[role]
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        pins.append((role, relative, hashlib.sha256(payload).hexdigest()))
    monkeypatch.setattr(M, "RSL_RUNTIME_SOURCE_PINS", tuple(pins))
    return root


def _fixture(tmp_path: Path, monkeypatch, *, log_std: bool = True):
    repo = tmp_path / "repo"
    for relative in (
        M.PRODUCER_SOURCE_PATH,
        M.ACTION_REGISTRY_PATH,
        M.TASK_PROFILE_PATH,
        M.PPO_CONFIG_PATH,
        *M.REPOSITORY_RUNTIME_SOURCE_PATHS,
    ):
        _copy(repo, relative)
    pins, output, identities = _fake_registry_pins(repo, log_std=log_std)
    monkeypatch.setattr(
        M,
        "_registry_pins",
        lambda _repo: (pins, output, identities),
    )
    rsl_root = _fake_rsl_root(tmp_path, monkeypatch)
    _commit_fixture(repo)
    return repo, rsl_root, pins, output, identities


def test_receipt_binds_exact_reward_scale_ppo_and_runtime_sources(
    tmp_path, monkeypatch
):
    repo, rsl_root, _, output, identities = _fixture(tmp_path, monkeypatch)
    receipt = M.build_receipt(
        repo_root=repo,
        rsl_rl_root=rsl_root,
        distribution_version="2.3.1",
    )

    M.validate_receipt_document(receipt)
    assert len(receipt["content_sha256"]) == 64
    reward = receipt["reward_economy"]
    assert reward["reward_global_scalar"] == 1.0
    assert reward["policy_step_dt_s"] == 0.02
    assert reward["effective_reward_recipe_sha256"] == M.EXPECTED_EFFECTIVE_REWARD_SHA256
    names = [term["name"] for term in reward["ordered_nonzero_terms"]]
    assert names == sorted(names)
    assert len(names) == len(set(names)) == 30
    assert reward["theoretical_bound_evidence_class"] == (
        "reviewed_analytic_assumption_bound_to_exact_sources"
    )
    progress_bound = next(
        row
        for row in reward["theoretical_weighted_dt_bounds"]
        if row["name"] == "racket_progress_per_step"
    )
    assert progress_bound["evidence_class"] == "reviewed_analytic_assumption"
    assert progress_bound["source_loci"] == [
        "mdp/hope_commands.py:RacketTargetCommand._update_footwork_signals",
        "mdp/hope_rewards.py:racket_progress",
    ]
    bounds = {
        row["name"]: (row["weighted_dt_min"], row["weighted_dt_max"])
        for row in reward["theoretical_weighted_dt_bounds"]
    }
    assert bounds == {
        "virtual_landing_one_shot": (0.0, 10.0),
        "hard_death_one_shot": (-6.0, 0.0),
        "qdes_limit_barrier_per_step": (-3.1, 0.0),
        "actual_joint_limit_barrier_per_step": (-3.1, 0.0),
        "action_rate_clamped_per_step": (-0.036, 0.0),
        "racket_position_fine_per_step": (0.0, 0.08),
        "racket_position_coarse_per_step": (0.0, 0.02),
        "racket_velocity_per_step": (0.0, 0.01),
        "racket_normal_per_step": (0.0, 0.01),
        "racket_progress_per_step": (-0.93, 0.93),
        "four_body_imitation_sum_per_step": (0.0, 0.08),
    }

    ppo = receipt["ppo_economy"]
    assert ppo["advantage_normalization"] == {
        "mode": "whole_rollout",
        "normalize_advantage_per_mini_batch": False,
        "storage_compute_returns_normalize_advantage": True,
        "rollout_samples_at_4096_env": 98304,
        "source_role": "rollout_storage",
    }
    assert ppo["required_final_policy"]["noise_std_type"] == "log"
    assert ppo["required_final_policy"]["required_realized_init_noise_std"] == 0.02
    assert ppo["std_parameterization_provenance"] == {
        "base_policy_init_noise_std": 1.0,
        "base_policy_noise_std_type": "scalar",
        "n1_runtime_override_init_noise_std": 0.02,
        "n1_runtime_materialized_noise_std_type": "log",
    }
    assert receipt["sources"]["registry_output"] == {"path": output["path"]}
    assert receipt["sources"]["producer"]["path"] == M.PRODUCER_SOURCE_PATH
    assert receipt["sources"]["registry_action_source_identities"] == identities
    rsl = receipt["sources"]["rsl_rl"]
    assert rsl["version"] == "2.3.1"
    assert [row["role"] for row in rsl["source_files"]] == [
        "ppo",
        "rollout_storage",
        "actor_critic",
    ]
    repository_runtime_paths = {
        row["path"] for row in receipt["sources"]["repository_runtime_sources"]
    }
    assert any(path.endswith("/mdp/hope_commands.py") for path in repository_runtime_paths)


def test_scalar_r6_runtime_contract_is_not_allowed_to_carry_new_policy_abi(
    tmp_path, monkeypatch
):
    repo, rsl_root, _, _, _ = _fixture(tmp_path, monkeypatch, log_std=False)
    with pytest.raises(M.ReceiptRefused, match="not exact log/.02"):
        M.build_receipt(
            repo_root=repo,
            rsl_rl_root=rsl_root,
            distribution_version="2.3.1",
        )


def test_runtime_contract_pin_must_exist_and_match_bytes(tmp_path, monkeypatch):
    repo, rsl_root, pins, _, _ = _fixture(tmp_path, monkeypatch)
    action_id, relative, digest = pins[0]
    path = repo / relative
    path.write_bytes(path.read_bytes() + b" ")
    # Keep the source-cleanliness gate intact while isolating the independent
    # registry-pin check: commit the drifted contract but deliberately retain
    # the old registry digest captured in ``pins``.  An uncommitted mutation is
    # correctly rejected earlier as dirty science input and is covered below.
    _git(repo, "add", relative)
    _git(
        repo,
        "-c",
        "user.name=Receipt Test",
        "-c",
        "user.email=receipt-test@example.invalid",
        "commit",
        "-m",
        "drift runtime contract behind stale registry pin",
    )

    with pytest.raises(M.ReceiptRefused, match="SHA-256 drift"):
        M.build_receipt(
            repo_root=repo,
            rsl_rl_root=rsl_root,
            distribution_version="2.3.1",
        )
    assert action_id == "bh_loop_c" and len(digest) == 64


def test_rsl_version_and_each_exact_source_are_fail_closed(tmp_path, monkeypatch):
    repo, rsl_root, _, _, _ = _fixture(tmp_path, monkeypatch)
    with pytest.raises(M.ReceiptRefused, match="version drift"):
        M.build_receipt(
            repo_root=repo,
            rsl_rl_root=rsl_root,
            distribution_version="2.3.2",
        )

    (rsl_root / "algorithms/ppo.py").write_bytes(b"drift\n")
    with pytest.raises(M.ReceiptRefused, match="ppo source drift"):
        M.build_receipt(
            repo_root=repo,
            rsl_rl_root=rsl_root,
            distribution_version="2.3.1",
        )


def test_ppo_clip_marker_is_exact_pinned_literal_not_producer_self_fixture(
    tmp_path, monkeypatch
):
    repo, rsl_root, _, _, _ = _fixture(tmp_path, monkeypatch)
    pinned_line = (
        b"            nn.utils.clip_grad_norm_(self.policy.parameters(), "
        b"self.max_grad_norm)\n"
    )
    old_wrong_line = (
        b"            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), "
        b"self.max_grad_norm)\n"
    )
    fixture = _PINNED_RSL_SOURCE_FIXTURES["ppo"]
    assert pinned_line in fixture
    assert old_wrong_line not in fixture
    assert pinned_line in M._RSL_SOURCE_MARKERS["ppo"]

    ppo_path = rsl_root / "algorithms/ppo.py"
    mutated = ppo_path.read_bytes().replace(pinned_line, old_wrong_line)
    assert mutated != ppo_path.read_bytes()
    ppo_path.write_bytes(mutated)
    mutation_sha = hashlib.sha256(mutated).hexdigest()
    monkeypatch.setattr(
        M,
        "RSL_RUNTIME_SOURCE_PINS",
        tuple(
            (role, relative, mutation_sha if role == "ppo" else digest)
            for role, relative, digest in M.RSL_RUNTIME_SOURCE_PINS
        ),
    )

    with pytest.raises(M.ReceiptRefused, match="ppo semantic marker is missing"):
        M.build_receipt(
            repo_root=repo,
            rsl_rl_root=rsl_root,
            distribution_version="2.3.1",
        )


def test_no_clobber_and_verify_require_file_and_semantic_identity(
    tmp_path, monkeypatch
):
    repo, rsl_root, _, output_pin, identities = _fixture(tmp_path, monkeypatch)
    receipt = M.build_receipt(
        repo_root=repo,
        rsl_rl_root=rsl_root,
        distribution_version="2.3.1",
    )
    output = M._registry_output_path(repo, output_pin)
    M._ensure_registry_output_parent(repo, output)
    M._write_no_clobber(output, receipt)
    raw = output.read_bytes()
    file_sha = hashlib.sha256(raw).hexdigest()
    output_pin["sha256"] = file_sha

    assert M._verify_file(
        output,
        repo_root=repo,
        rsl_rl_root=rsl_root,
        distribution_version="2.3.1",
        expected_file_sha256=file_sha,
    ) == receipt
    with pytest.raises(M.ReceiptRefused, match="already spent"):
        M._write_no_clobber(output, receipt)
    with pytest.raises(M.ReceiptRefused, match="registry file SHA"):
        M._verify_file(
            output,
            repo_root=repo,
            rsl_rl_root=rsl_root,
            distribution_version="2.3.1",
            expected_file_sha256="0" * 64,
        )
    identities[0]["identity"]["scope"] = "full"
    identities[0]["sha256"] = hashlib.sha256(
        _canonical(identities[0]["identity"])
    ).hexdigest()
    with pytest.raises(M.ReceiptRefused, match="live pinned inputs"):
        M._verify_file(
            output,
            repo_root=repo,
            rsl_rl_root=rsl_root,
            distribution_version="2.3.1",
            expected_file_sha256=file_sha,
        )


def test_self_hash_and_telemetry_consumer_are_explicit(tmp_path, monkeypatch):
    repo, rsl_root, _, _, _ = _fixture(tmp_path, monkeypatch)
    receipt = M.build_receipt(
        repo_root=repo,
        rsl_rl_root=rsl_root,
        distribution_version="2.3.1",
    )
    telemetry = receipt["runtime_4096x5_telemetry_consumer"]
    assert telemetry["status"] == "wired_probe_gate_runtime_evidence_required"
    assert telemetry["gate"] == {
        "num_envs": 4096,
        "ppo_updates": 5,
        "steps_per_env_per_update": 24,
        "rollout_samples_per_update": 98304,
    }
    assert "clip_fraction" in telemetry["required_fields"]["ppo"]
    assert telemetry["required_fields"]["gradient"] == [
        "pre_clip_actor_mean_parameter_grad_norm",
        "pre_clip_critic_parameter_grad_norm",
        "pre_clip_std_parameter_grad_norm",
        "pre_clip_total_grad_norm",
        "post_clip_total_grad_norm",
        "max_grad_norm",
        "pre_clip_actor_mean_parameter_grad_norm_distribution",
        "pre_clip_critic_parameter_grad_norm_distribution",
        "pre_clip_std_parameter_grad_norm_distribution",
        "pre_clip_total_grad_norm_distribution",
        "post_clip_total_grad_norm_distribution",
        "clip_factor_distribution",
        "optimizer_minibatch_count",
    ]
    assert "per_term_eligible_denominator" in telemetry["required_fields"]["reward"]
    assert "return_std" in telemetry["required_fields"]["reward"]
    assert "explained_variance" in telemetry["required_fields"]["reward"]

    tampered = deepcopy(receipt)
    tampered["reward_economy"]["reward_global_scalar"] = 0.1
    with pytest.raises(M.ReceiptRefused, match="self-hash"):
        M.validate_receipt_document(tampered)


def test_refuses_dirty_or_index_staged_repository_science_inputs(
    tmp_path, monkeypatch
):
    repo, rsl_root, _, _, _ = _fixture(tmp_path, monkeypatch)
    task = repo / M.TASK_PROFILE_PATH
    task.write_bytes(task.read_bytes() + b"\n# dirty science\n")
    with pytest.raises(M.ReceiptRefused, match="working tree"):
        M.build_receipt(
            repo_root=repo,
            rsl_rl_root=rsl_root,
            distribution_version="2.3.1",
        )

    staged_root = tmp_path / "staged"
    staged_root.mkdir()
    staged_monkeypatch = pytest.MonkeyPatch()
    try:
        staged_repo, staged_rsl, _, _, _ = _fixture(
            staged_root,
            staged_monkeypatch,
        )
        producer = staged_repo / M.PRODUCER_SOURCE_PATH
        producer.write_bytes(producer.read_bytes() + b"\n# staged producer drift\n")
        _git(staged_repo, "add", M.PRODUCER_SOURCE_PATH)
        with pytest.raises(M.ReceiptRefused, match="index"):
            M.build_receipt(
                repo_root=staged_repo,
                rsl_rl_root=staged_rsl,
                distribution_version="2.3.1",
            )
    finally:
        staged_monkeypatch.undo()


def test_registry_requires_two_materialized_r6_contracts_and_one_shared_output(
    tmp_path, monkeypatch
):
    output = SimpleNamespace(
        path="configs/n1_reward_economy_20260802_r9/reward_economy.v1.json",
        sha256=None,
    )
    configs = {
        action_id: SimpleNamespace(
            action_id=action_id,
            runtime_contract=SimpleNamespace(
                path=(
                    "configs/a3_vendor_runtime_authority_20260802_r9/"
                    f"{action_id}.shared_ready.training_contract.json"
                ),
                sha256=None,
            ),
            reward_economy_receipt=output,
        )
        for action_id in M.ACTION_IDS
    }
    monkeypatch.setattr(
        M,
        "_load_registry",
        lambda _repo: SimpleNamespace(
            ACTION_CONFIGS=configs,
            action_source_identity=lambda config: {
                "schema_version": 1,
                "action_id": config.action_id,
            },
            action_source_identity_sha256=lambda config: hashlib.sha256(
                _canonical(
                    {
                        "schema_version": 1,
                        "action_id": config.action_id,
                    }
                )
            ).hexdigest(),
        ),
    )
    with pytest.raises(M.ReceiptRefused, match="not materialized"):
        M._registry_pins(tmp_path)

    for config in configs.values():
        config.runtime_contract.sha256 = "a" * 64
    configs["bh_block"].reward_economy_receipt = SimpleNamespace(
        path=output.path,
        sha256="b" * 64,
    )
    with pytest.raises(M.ReceiptRefused, match="output pins disagree"):
        M._registry_pins(tmp_path)
