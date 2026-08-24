"""Independent offline verifier for the portable MuJoCo FullMDP long run."""
from __future__ import annotations
import argparse, hashlib, importlib.util, inspect, io, json, math, os, re, stat, sys
from pathlib import Path

def _ppo_recipe_module():
    source = (Path(__file__).resolve().parents[1] / "source" /
              "whole_body_tracking" / "action_ball_full_mdp_ppo_recipe.py")
    name = "_hope_mujoco_action_ball_full_mdp_ppo_recipe"
    cached = sys.modules.get(name)
    if cached is not None:
        if Path(cached.__file__).resolve() != source:
            raise RuntimeError("cached FullMDP PPO recipe origin differs")
        return cached
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None: raise RuntimeError("cannot load FullMDP PPO recipe")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    try: spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None); raise
    return module

def _epa48_runtime_module():
    source = Path(__file__).with_name("mujoco_full_mdp_epa48_runtime.py").resolve()
    name = "_hope_mujoco_full_mdp_epa48_runtime"
    cached = sys.modules.get(name)
    if cached is not None:
        if Path(getattr(cached, "__file__", "")).resolve() != source:
            raise RuntimeError("cached Full-A EPA48 runtime binder origin differs")
        return cached
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None: raise RuntimeError("cannot load Full-A EPA48 runtime identity")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    try: spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None); raise
    return module

def _plant_contract_module():
    source = Path(__file__).with_name("mujoco_full_mdp_plant_contract.py").resolve()
    name = "_hope_mujoco_full_mdp_plant_contract"
    cached = sys.modules.get(name)
    if cached is not None:
        if Path(getattr(cached, "__file__", "")).resolve() != source:
            raise RuntimeError("cached MuJoCo plant contract origin differs")
        return cached
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None: raise RuntimeError("cannot load MuJoCo plant contract")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    try: spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None); raise
    return module

def _reward_contract_module():
    source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "mdp"
        / "action_ball_full_mdp_reward_contract.py"
    ).resolve()
    name = "_hope_mujoco_consumer_action_ball_full_mdp_reward_contract"
    cached = sys.modules.get(name)
    if cached is not None:
        if Path(getattr(cached, "__file__", "")).resolve() != source:
            raise RuntimeError("cached FullMDP reward contract origin differs")
        return cached
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load FullMDP reward contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module

def _exact_reward_contract():
    contract = _reward_contract_module()
    names = getattr(contract, "MANAGER_NAMES", None)
    count = getattr(contract, "REWARD_TERM_COUNT", None)
    if (
        type(names) is not tuple
        or not names
        or any(type(name) is not str or not name for name in names)
        or len(set(names)) != len(names)
        or type(count) is not int
        or count != len(names)
    ):
        raise RuntimeError("FullMDP reward contract differs")
    return names, count

def _canonical_mujoco_identity_module():
    source = (Path(__file__).resolve().parents[1] / "scripts" /
              "canonical_mujoco_identity.py")
    name = "_hope_canonical_mujoco_identity"
    cached = sys.modules.get(name)
    if cached is not None:
        if Path(getattr(cached, "__file__", "")).resolve() != source:
            raise RuntimeError("cached canonical MuJoCo identity origin differs")
        return cached
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None: raise RuntimeError("cannot load canonical MuJoCo identity")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    try: spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None); raise
    return module

def _table_termination_module():
    root = Path(__file__).resolve().parent.parent
    expected = root / "mujoco_native/table_termination.py"
    inserted = str(root) not in sys.path
    if inserted: sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("mujoco_native.table_termination")
    finally:
        if inserted and sys.path and sys.path[0] == str(root): sys.path.pop(0)
    if (
        Path(getattr(module, "__file__", "")).resolve() != expected
        or Path(getattr(module, "REPO_ROOT", "")).resolve()
        != Path(__file__).resolve().parents[3]
    ):
        _fail("table termination authority origin")
    return module

def _mujoco_module():
    return importlib.import_module("mujoco")

FULL_MDP_PPO_RECIPE = _ppo_recipe_module().ACTION_BALL_FULL_MDP_PPO_RECIPE
FULL_MDP_PPO_RECIPE_SHA256 = FULL_MDP_PPO_RECIPE.recipe_sha256()
REWARD_TERM_NAMES, REWARD_TERM_COUNT = _exact_reward_contract()
PADDLE_PRIOR_TERM_NAMES = tuple(
    spec.manager_name
    for spec in _reward_contract_module().PADDLE_MOTION_PRIOR_SPECS
)
if REWARD_TERM_NAMES[-len(PADDLE_PRIOR_TERM_NAMES):] != PADDLE_PRIOR_TERM_NAMES:
    raise RuntimeError("FullMDP paddle prior contract differs")
PADDLE_PRIOR_TERM_COUNT = len(PADDLE_PRIOR_TERM_NAMES)
EVIDENCE_SCHEMA_VERSION, COMPLETION_SCHEMA_VERSION, SUMMARY_SCHEMA_VERSION = 9, 5, 6
COMPLETE_UPDATES, NUM_ENVS, STEPS_PER_UPDATE, SAVE_INTERVAL, ACTION_UID = (
    FULL_MDP_PPO_RECIPE.max_iterations, 4096,
    FULL_MDP_PPO_RECIPE.num_steps_per_env, FULL_MDP_PPO_RECIPE.save_interval,
    6907688916670928)
TRANSITIONS_PER_UPDATE = NUM_ENVS * STEPS_PER_UPDATE
def _names(raw): return frozenset(raw.split())
EVENT_KEYS = _names("""scheduled_due_rows due_terminal_overlap_rows reveal_rows reveal_due_rows reveal_deferred_rows launch_rows missed_launch_rows flight_terminal_rows
 shot_retired_rows completed_action_epoch_rows selected_reset_rows racket_contact_rows selected_contact_rows
 opposite_contact_rows edge_contact_rows between_contact_rows invalid_contact_rows actual_hard_edge_rows
 qdes_guard_intervention_rows r03_present_rows
 r03_physically_valid_rows landing_crossing_rows r06_present_rows r06_eligible_rows r06_common_rows
 r07_present_rows r07_eligible_rows recovery_success_rows recovery_failure_rows recovery_timeout_rows
 recovery_completion_fault_rows""")
TERMINAL_KEYS = _names("time_out base_fell_tilt base_too_low joint_qdes_forbidden robot_hit_table")
LIFECYCLE_KEYS = _names("""gym_reset_rows unknown_terminal_rows invalid_done_rows done_explanation_fault_rows
 time_out_rows timeout_fault_rows selected_reset_fault_rows reset_generation_rows reset_generation_fault_rows
 resolved_table_rows landing_on_opponent_rows landing_opponent_bound_rows classification_unknown_rows""")
FAULT_KEYS = _names("""unknown_terminal_rows invalid_done_rows done_explanation_fault_rows timeout_fault_rows
 selected_reset_fault_rows reset_generation_fault_rows classification_unknown_rows""")
FACT_INTEGRITY_KEYS = _names("""fact_integrity_r03_nonfinite_rows
 fact_integrity_r06_source_invalid_rows
 fact_integrity_r07_sequence_rows fact_integrity_r07_nonfinite_rows
 fact_integrity_unknown_bits_rows""")
TOP_KEYS = _names("""schema_version record_type diagnostic_unauthorized update_index
 run_identity num_envs num_steps_per_env transitions_delta transitions_cumulative
 environment_steps_delta environment_steps_cumulative storage_finite storage_domains extras_counts
 terminal_bit_counts classification_status_counts outcome_code_counts phase_counts episodes
 rollout_policy_mean_std selected_reset_rows gym_reset_rows lifecycle_counts fact_integrity_counts reward_graph
 action_identity prepared_update_sha256 snapshot optimizer_metrics learning_rate timings""")
ACK_KEYS = _names("prepared_update_sha256 snapshot optimizer_metrics learning_rate timings")
REWARD_GRAPH_KEYS = _names("""term_names term_count term_sums actual_reward_sum
 reward_terms_finite_rows reward_terms_nonfinite_rows
 actual_reward_finite_rows actual_reward_nonfinite_rows
 conservation_fault_rows playback_paddle_prior""")
PLAYBACK_PADDLE_PRIOR_KEYS = _names("""term_names row_count finite_rows
 kernel_sum kernel_sumsq domain_violation_rows""")
IDENTITY_KEYS = _names("""action_slot action_uid mount_normal_sign family family_source
 observed_rows slot0_rows uid_rows mount_sign_rows identity_rows family_counts""")
RECEIPT_KEYS = _names("name bytes sha256")
COMPLETION_KEYS = _names("""schema_version record_type diagnostic_unauthorized run_identity
 num_envs num_steps_per_env completed_updates environment_steps transitions evidence_jsonl
 snapshot_receipts final_observation_finite rollout_storage_finite optimizer_state_present
 optimizer_state_finite checkpoint_authority resume_authority action_contract
 action_ball_full_mdp_ppo_recipe_sha256""")
ACTION_CONTRACT = {
    "action_joint_order_contract_id": "a3-gmr-dof-pos-to-runtime-articulation-v1",
    "action_joint_order_contract_sha256": "b09987ff7a1bfa624b566cc8884d16672ba73c1acc3f92efb8a4faa99d314815",
    "action_offset_source": "runtime_plant.default_joint_pos_rad",
    "action_offset_sha256": "1b638d7b2e1ac7e552aace2ac8c2b00980dd9daf691f930b5fe775cebc84af78",
    "full_a_reset_joint_source": "runtime_plant.default_joint_pos_rad",
    "full_a_reset_root_source": "AGIBOT_A3_CFG.init_state.pos/rot",
    "full_a_policy_bootstrap": "a3_default_stand_zero_head_v1",
    "raw_action_clip": None,
    "executable_qdes_guard": "action_ball_shared_soft_hard_state_guard_v1",
    "transfer_authority": False, "matched_cross_backend_authority": False,
}
MODEL_SHAPES = (
    ("log_std", (31,)), ("actor.0.weight", (512, 203)), ("actor.0.bias", (512,)),
    ("actor.2.weight", (256, 512)), ("actor.2.bias", (256,)), ("actor.4.weight", (128, 256)), ("actor.4.bias", (128,)),
    ("actor.6.weight", (31, 128)), ("actor.6.bias", (31,)),
    ("critic.0.weight", (512, 219)), ("critic.0.bias", (512,)),
    ("critic.2.weight", (256, 512)), ("critic.2.bias", (256,)), ("critic.4.weight", (128, 256)), ("critic.4.bias", (128,)),
    ("critic.6.weight", (1, 128)), ("critic.6.bias", (1,)),
)
def _fail(message): raise ValueError("MuJoCo FullMDP evidence differs: " + message)
def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected): _fail(label + " keys")
def _same(value, expected):
    if type(value) is not type(expected): return False
    if type(value) is dict: return set(value) == set(expected) and all(_same(value[k], expected[k]) for k in expected)
    return (len(value) == len(expected) and all(_same(a, b) for a, b in zip(value, expected))) if type(value) is list else value == expected
def _int(value, label, limit=None):
    if type(value) is not int or value < 0 or (limit is not None and value > limit): _fail(label)
    return value
def _num(value, label):
    if type(value) not in (int, float) or not math.isfinite(value): _fail(label)
    return float(value)
def _hex(value, size, label):
    if type(value) is not str or re.fullmatch(f"[0-9a-f]{{{size}}}", value) is None: _fail(label)
    return value
def _verified_plant_model(path, final_augmented_mjb):
    contract, canonical = _plant_contract_module(), _canonical_mujoco_identity_module()
    try:
        verified = canonical.verify_exact_mujoco_identity(
            mjcf_path=Path(path),
            expected_manifest_path=contract.expected_manifest_path(),
            trusted_expected_manifest_sha256=contract.TRUSTED_EXPECTED_MANIFEST_SHA256,
        )
    except Exception as exc:
        _fail("expected plant exact verification: " + str(exc))
    expected = contract.expected_plant_model_identity()
    if (
        verified.portable_identity_sha256
        != expected["source_plant"]["portable_identity_sha256"]
    ):
        _fail("expected plant portable identity")
    try:
        mujoco = _mujoco_module()
        owner = _table_termination_module().consume_verified_owner_frame_contract(
            mujoco, verified
        )["content_sha256"]
    except Exception as exc:
        _fail("expected plant owner-frame verification: " + str(exc))
    return contract.verified_plant_model_identity(
        verification_receipt_sha256=verified.verification_receipt_sha256,
        owner_local_frame_sha256=owner,
        final_augmented_mjb=final_augmented_mjb,
    )
def _run_identity(
    commit, namespace, runtime_stack, plant_xml, final_augmented_mjb
):
    if (type(commit) is not str or re.fullmatch(r"[0-9a-f]{40}", commit) is None
            or type(namespace) is not str
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{15,159}", namespace) is None):
        _fail("expected run identity")
    _keys(
        runtime_stack,
        {"schema_version", "mujoco_warp", "rsl_rl", "mjlab"},
        "verified runtime stack",
    )
    if (
        type(runtime_stack["schema_version"]) is not int
        or runtime_stack["schema_version"] != 1
        or any(type(runtime_stack[key]) is not dict for key in (
            "mujoco_warp", "rsl_rl", "mjlab"
        ))
    ):
        _fail("verified runtime stack")
    return {"source_commit": commit, "run_namespace": namespace,
        "runtime_stack": {
            "schema_version": runtime_stack["schema_version"],
            "mujoco_warp": dict(runtime_stack["mujoco_warp"]),
            "rsl_rl": dict(runtime_stack["rsl_rl"]),
            "mjlab": dict(runtime_stack["mjlab"]),
        },
        "plant_model": _verified_plant_model(
            plant_xml, final_augmented_mjb)}
def _count_map(row, name, keys):
    _keys(row[name], keys, name)
    return {key: _int(value, f"{name}.{key}", TRANSITIONS_PER_UPDATE) for key, value in row[name].items()}
def _duplicates(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            _fail("duplicate JSON key " + str(key))
        out[key] = value
    return out
def _prepared_hash(row):
    base = {key: value for key, value in row.items() if key not in ACK_KEYS}
    return hashlib.sha256(json.dumps(base, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
def _receipt(value, label, name=None):
    _keys(value, RECEIPT_KEYS, label)
    if (type(value["name"]) is not str
            or re.fullmatch(r"model_(0|[1-9][0-9]*)\.pt", value["name"]) is None
            or (name is not None and value["name"] != name)):
        _fail(label + ".name")
    size = _int(value["bytes"], label + ".bytes")
    if size == 0: _fail(label + ".bytes")
    return {"name": value["name"], "bytes": size,
            "sha256": _hex(value["sha256"], 64, label + ".sha256")}
def _validate_record(row, index, run_identity):
    _keys(row, TOP_KEYS, "top-level")
    fixed = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "record_type": "mujoco_full_mdp_update_ack",
        "diagnostic_unauthorized": True, "update_index": index, "num_envs": NUM_ENVS,
        "num_steps_per_env": STEPS_PER_UPDATE,
        "transitions_delta": TRANSITIONS_PER_UPDATE,
        "transitions_cumulative": TRANSITIONS_PER_UPDATE * (index + 1),
        "environment_steps_delta": STEPS_PER_UPDATE,
        "environment_steps_cumulative": STEPS_PER_UPDATE * (index + 1),
    }
    if any(type(row[key]) is not type(value) or row[key] != value for key, value in fixed.items()):
        _fail(f"fixed fields at update {index}")
    _keys(row["run_identity"], {
        "source_commit", "run_namespace", "runtime_stack", "plant_model"
    }, "run identity")
    if not _same(row["run_identity"], run_identity):
        _fail(f"run identity at update {index}")
    if (_hex(row["prepared_update_sha256"], 64, "prepared update hash")
            != _prepared_hash(row)):
        _fail(f"prepared update hash at update {index}")
    _keys(row["storage_finite"], {
        "observations_policy", "observations_critic", "actions", "values",
        "actions_log_prob", "mu", "sigma", "rewards", "returns",
        "advantages",
    }, "storage")
    if any(value is not True for value in row["storage_finite"].values()):
        _fail("nonfinite rollout storage")
    _keys(row["storage_domains"], {
        "dones_binary", "sigma_positive",
    }, "storage domains")
    invalid_domains = [name for name, value
                       in row["storage_domains"].items() if value is not True]
    if invalid_domains:
        _fail("rollout storage domain: " + ",".join(sorted(invalid_domains)))
    events = _count_map(row, "extras_counts", EVENT_KEYS)
    named_faults = [name for name in (
        "missed_launch_rows", "recovery_completion_fault_rows",
    ) if events[name] != 0]
    if named_faults:
        _fail("named event fault counter: " + ",".join(named_faults))
    terms = _count_map(row, "terminal_bit_counts", TERMINAL_KEYS)
    life = _count_map(row, "lifecycle_counts", LIFECYCLE_KEYS)
    fact_integrity = _count_map(
        row, "fact_integrity_counts", FACT_INTEGRITY_KEYS
    )
    classes = _count_map(row, "classification_status_counts", map(str, range(6)))
    outcomes = _count_map(row, "outcome_code_counts", map(str, range(7)))
    phases = _count_map(row, "phase_counts", ("0", "2", "5", "6", "8"))
    if any(life[key] for key in FAULT_KEYS):
        _fail("lifecycle fault counter")
    if any(fact_integrity.values()):
        _fail("fact integrity fault counter")
    selected, gym = row["selected_reset_rows"], row["gym_reset_rows"]
    _int(selected, "selected_reset_rows", TRANSITIONS_PER_UPDATE)
    _int(gym, "gym_reset_rows", TRANSITIONS_PER_UPDATE)
    if (selected != events["selected_reset_rows"] or gym != life["gym_reset_rows"]
            or selected != gym or life["reset_generation_rows"] != gym
            or life["time_out_rows"] != terms["time_out"]
            or life["resolved_table_rows"] > terms["robot_hit_table"]
            or gym < max(terms.values())
            or gym > sum(terms.values())):
        _fail("lifecycle cross-check")
    episodes = row["episodes"]
    _keys(episodes, {"completed_count", "return_sum", "length_sum"}, "episodes")
    completed = _int(episodes["completed_count"], "episode count", TRANSITIONS_PER_UPDATE)
    length = _int(episodes["length_sum"], "episode length sum")
    if completed != gym or length < completed or (completed == 0 and length != 0):
        _fail("episode lengths/count")
    if (completed == 0 and _num(episodes["return_sum"], "episode return") != 0
            or _num(row["rollout_policy_mean_std"], "policy mean std") <= 0):
        _fail("episode return without completion or policy std")
    reward = row["reward_graph"]
    _keys(reward, REWARD_GRAPH_KEYS, "reward graph")
    if (
        type(reward["term_names"]) is not list
        or not _same(reward["term_names"], list(REWARD_TERM_NAMES))
        or type(reward["term_count"]) is not int
        or reward["term_count"] != REWARD_TERM_COUNT
        or type(reward["term_sums"]) is not list
        or len(reward["term_sums"]) != REWARD_TERM_COUNT
    ):
        _fail("reward graph term contract")
    total = math.fsum(
        _num(value, "reward graph term") for value in reward["term_sums"]
    )
    for key in REWARD_GRAPH_KEYS - {
        "term_names", "term_count", "term_sums", "actual_reward_sum",
        "playback_paddle_prior",
    }:
        _int(reward[key], "reward_graph." + key, TRANSITIONS_PER_UPDATE)
    paddle = reward["playback_paddle_prior"]
    _keys(paddle, PLAYBACK_PADDLE_PRIOR_KEYS, "playback paddle prior")
    if (
        type(paddle["term_names"]) is not list
        or not _same(paddle["term_names"], list(PADDLE_PRIOR_TERM_NAMES))
        or type(paddle["finite_rows"]) is not list
        or len(paddle["finite_rows"]) != PADDLE_PRIOR_TERM_COUNT
        or type(paddle["domain_violation_rows"]) is not list
        or len(paddle["domain_violation_rows"]) != PADDLE_PRIOR_TERM_COUNT
    ):
        _fail("playback paddle prior term contract")
    paddle_rows = _int(
        paddle["row_count"],
        "playback paddle prior row_count",
        TRANSITIONS_PER_UPDATE,
    )
    paddle_finite = [
        _int(value, "playback paddle prior finite_rows", paddle_rows)
        for value in paddle["finite_rows"]
    ]
    paddle_domain = [
        _int(
            value,
            "playback paddle prior domain_violation_rows",
            finite,
        )
        for value, finite in zip(
            paddle["domain_violation_rows"], paddle_finite
        )
    ]
    paddle_moments = {}
    for key in ("kernel_sum", "kernel_sumsq"):
        values = paddle[key]
        if type(values) is not list or len(values) != PADDLE_PRIOR_TERM_COUNT:
            _fail("playback paddle prior " + key)
        paddle_moments[key] = [
            _num(value, "playback paddle prior " + key) for value in values
        ]
    for term_index, finite in enumerate(paddle_finite):
        kernel_sum = paddle_moments["kernel_sum"][term_index]
        kernel_sumsq = paddle_moments["kernel_sumsq"][term_index]
        moment_floor = kernel_sum * kernel_sum
        moment_ceiling = finite * kernel_sumsq
        tolerance = 1.0e-9 * max(
            1.0, abs(moment_floor), abs(moment_ceiling)
        )
        if (
            kernel_sumsq < 0.0
            or moment_floor > moment_ceiling + tolerance
            or (
                finite == 0
                and (kernel_sum != 0.0 or kernel_sumsq != 0.0)
            )
        ):
            _fail("playback paddle prior moments")
    if (reward["reward_terms_finite_rows"] != TRANSITIONS_PER_UPDATE
            or reward["actual_reward_finite_rows"] != TRANSITIONS_PER_UPDATE
            or any(reward[key] for key in ("reward_terms_nonfinite_rows",
                "actual_reward_nonfinite_rows", "conservation_fault_rows"))
            or not math.isclose(total, _num(reward["actual_reward_sum"], "actual reward"),
                                rel_tol=1e-5, abs_tol=1.0)):
        _fail("reward graph fault or finite count")
    action = row["action_identity"]
    _keys(action, IDENTITY_KEYS, "action identity")
    expected_action = {
        "action_slot": 0, "action_uid": ACTION_UID, "mount_normal_sign": 1,
        "family": "forehand", "family_source": "runner_pinned_identity",
        "observed_rows": TRANSITIONS_PER_UPDATE, "slot0_rows": TRANSITIONS_PER_UPDATE,
        "uid_rows": TRANSITIONS_PER_UPDATE, "mount_sign_rows": TRANSITIONS_PER_UPDATE,
        "identity_rows": TRANSITIONS_PER_UPDATE,
        "family_counts": {"forehand": TRANSITIONS_PER_UPDATE, "backhand": 0},
    }
    if not _same(action, expected_action):
        _fail("action identity")
    snapshot = None if row["snapshot"] is None else _receipt(
        row["snapshot"], "snapshot receipt", f"model_{index}.pt")
    _keys(row["optimizer_metrics"], {"value_function", "surrogate", "entropy"},
          "optimizer metrics")
    if any(not math.isfinite(_num(value, "optimizer metric"))
           for value in row["optimizer_metrics"].values()):
        _fail("optimizer metrics")
    if _num(row["learning_rate"], "learning rate") <= 0:
        _fail("learning rate")
    timing = row["timings"]
    timing_keys = {"collection_seconds", "learning_seconds", "pre_ack_iteration_seconds",
                   "run_elapsed_pre_ack_seconds"}
    _keys(timing, timing_keys, "timings")
    timing = {key: _num(value, "timing " + key) for key, value in timing.items()}
    if (any(value <= 0 for value in timing.values())
            or timing["pre_ack_iteration_seconds"]
            < timing["collection_seconds"] + timing["learning_seconds"]):
        _fail("timings")
    return events, terms, life, outcomes, snapshot, timing["run_elapsed_pre_ack_seconds"]
def _stable_regular(path, label, *, retain_bytes):
    if not path.is_absolute():
        _fail(label + " path is not absolute")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        _fail(label + " open: " + str(exc))
    try:
        before = os.fstat(fd)
        digest = hashlib.sha256()
        chunks = [] if retain_bytes else None
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk: break
            digest.update(chunk)
            if chunks is not None: chunks.append(chunk)
        after, current = os.fstat(fd), os.stat(path, follow_symlinks=False)
        state = lambda item: (
            item.st_dev, item.st_ino, item.st_mode, item.st_nlink,
            item.st_size, item.st_mtime_ns, item.st_ctime_ns,
        )
        stable = state(before) == state(after) == state(current)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not stable
                or (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino)
                or path.resolve(strict=True) != path):
            _fail(label + " is not one stable regular file")
        return (
            b"".join(chunks) if chunks is not None else None,
            {"bytes": before.st_size, "sha256": digest.hexdigest()},
            state(before),
        )
    finally:
        os.close(fd)
def _read_regular(path, label):
    raw, inventory, _state = _stable_regular(
        path, label, retain_bytes=True,
    )
    return raw, inventory
def _verified_runtime_mjb(evidence_jsonl):
    """Independently hash and load the one run-owned augmented model."""
    contract = _plant_contract_module()
    expected = contract.expected_plant_model_identity()["runtime_attach"][
        "final_augmented_mjb"
    ]
    locator = expected["relative_locator"]
    if (
        type(locator) is not str
        or not locator
        or Path(locator).is_absolute()
        or Path(locator).name != locator
        or locator in (".", "..")
    ):
        _fail("runtime MJB relative locator")
    root = Path(evidence_jsonl).parent
    path = root / locator
    _raw, inventory, state = _stable_regular(
        path, "runtime MJB", retain_bytes=False,
    )
    observed = {
        "relative_locator": locator,
        "sha256": inventory["sha256"],
        "size_bytes": inventory["bytes"],
    }
    if not _same(observed, expected):
        _fail("runtime MJB receipt")
    try:
        model = _mujoco_module().MjModel.from_binary_path(str(path))
        current = os.stat(path, follow_symlinks=False)
    except Exception as exc:
        _fail("runtime MJB load: " + str(exc))
    current_state = (
        current.st_dev, current.st_ino, current.st_mode, current.st_nlink,
        current.st_size, current.st_mtime_ns, current.st_ctime_ns,
    )
    if model is None or current_state != state:
        _fail("runtime MJB changed during load")
    return observed
def _read_rows(path, count, identity):
    raw, inventory = _read_regular(path, "JSONL")
    if not raw.endswith(b"\n") or b"\n\n" in raw: _fail("JSONL framing")
    lines = raw.splitlines()
    if len(lines) != count: _fail(f"ACK count {len(lines)} != {count}")
    totals = {key: 0 for key in EVENT_KEYS}; life = {key: 0 for key in LIFECYCLE_KEYS}
    terms = {key: 0 for key in TERMINAL_KEYS}; outcomes = {str(i): 0 for i in range(7)}
    rows, elapsed = [], 0.0
    for index, line in enumerate(lines):
        try: row = json.loads(line, object_pairs_hook=_duplicates)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc: _fail(f"JSON at update {index}: {exc}")
        event, term, lifecycle, outcome, snapshot, now = _validate_record(row, index, identity)
        if now <= elapsed: _fail("run elapsed is not strictly increasing")
        elapsed = now
        for target, source in ((totals, event), (life, lifecycle), (terms, term), (outcomes, outcome)):
            for key in target: target[key] += source[key]
        rows.append((row, snapshot))
    return rows, totals, life, terms, outcomes, inventory
def _finite(value, torch):
    if isinstance(value, torch.Tensor):
        return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all())
    if type(value) is float: return math.isfinite(value)
    if type(value) in (str, int, bool, type(None)): return True
    if isinstance(value, dict): return all(_finite(k, torch) and _finite(v, torch) for k, v in value.items())
    if isinstance(value, (list, tuple)): return all(_finite(v, torch) for v in value)
    return False
def _model_optimizer(payload, torch, name):
    model, optimizer = payload["model_state_dict"], payload["optimizer_state_dict"]
    if not isinstance(model, dict) or set(model) != {key for key, _ in MODEL_SHAPES}:
        _fail("snapshot model ABI " + name)
    for key, shape in MODEL_SHAPES:
        value = model[key]
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or tuple(value.shape) != shape or not bool(torch.isfinite(value).all())):
            _fail("snapshot model ABI " + name + ":" + key)
    _keys(optimizer, {"state", "param_groups"}, "snapshot optimizer")
    state, groups = optimizer["state"], optimizer["param_groups"]
    if (not isinstance(state, dict) or type(groups) is not list or len(groups) != 1
            or not isinstance(groups[0], dict) or type(groups[0].get("params")) is not list):
        _fail("snapshot optimizer ABI " + name)
    ids = groups[0]["params"]
    if (len(ids) != len(MODEL_SHAPES) or any(type(i) is not int for i in ids)
            or len(set(ids)) != len(ids) or set(state) != set(ids)):
        _fail("snapshot optimizer parameter map " + name)
    for param, (key, shape) in zip(ids, MODEL_SHAPES):
        item = state[param]; _keys(item, {"step", "exp_avg", "exp_avg_sq"}, "snapshot optimizer state")
        step = item["step"]
        if (not isinstance(step, torch.Tensor) or tuple(step.shape) != ()
                or not (step.is_floating_point() or step.dtype in (torch.int32, torch.int64))
                or not _finite(step, torch) or float(step.item()) <= 0):
            _fail("snapshot optimizer step " + name + ":" + key)
        for state_name in ("exp_avg", "exp_avg_sq"):
            value = item[state_name]
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                    or tuple(value.shape) != shape or not bool(torch.isfinite(value).all())):
                _fail("snapshot optimizer state shape " + name + ":" + key)
    if (not _finite(optimizer, torch) or type(groups[0].get("lr")) not in (int, float)
            or not math.isfinite(groups[0]["lr"]) or groups[0]["lr"] <= 0):
        _fail("snapshot optimizer finite " + name)
def _snapshot_indices(complete):
    if not complete: return [0]
    indices = list(range(0, COMPLETE_UPDATES, SAVE_INTERVAL))
    if COMPLETE_UPDATES - 1 not in indices: indices.append(COMPLETE_UPDATES - 1)
    return indices
def _snapshots(root, rows, identity, complete):
    import torch
    try: root_stat = root.lstat()
    except OSError as exc: _fail("snapshot directory: " + str(exc))
    if (not root.is_absolute() or not stat.S_ISDIR(root_stat.st_mode)
            or root.resolve(strict=True) != root): _fail("snapshot directory")
    indices = _snapshot_indices(complete); names = [f"model_{i}.pt" for i in indices]
    if sorted(p.name for p in root.iterdir()) != sorted(names): _fail("snapshot names")
    if any(i >= len(rows) for i in indices): _fail("snapshot index beyond ACK frontier")
    inventory = []; dfd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for index, name in zip(indices, names):
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dfd)
            try:
                before = os.fstat(fd); chunks = []
                while True:
                    chunk = os.read(fd, 1024 * 1024)
                    if not chunk: break
                    chunks.append(chunk)
                after, current = os.fstat(fd), os.stat(name, dir_fd=dfd, follow_symlinks=False)
                stable = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                          before.st_ctime_ns) == (after.st_dev, after.st_ino, after.st_size,
                          after.st_mtime_ns, after.st_ctime_ns)
                if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size <= 0
                        or not stable or (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino)):
                    _fail("snapshot file " + name)
            finally: os.close(fd)
            raw = b"".join(chunks); actual = {"name": name, "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest()}
            if rows[index][1] != actual: _fail("snapshot ACK receipt binding " + name)
            if "weights_only" not in inspect.signature(torch.load).parameters:
                _fail("safe snapshot decoder is unavailable")
            try: payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            except Exception as exc: _fail("snapshot decode " + name + ": " + str(exc))
            _keys(payload, {"model_state_dict", "optimizer_state_dict", "iter", "infos"}, "snapshot payload")
            infos = {"diagnostic_unauthorized": True, "checkpoint_authority": False,
                "resume_authority": False, "update_index": index, "completed_updates": index + 1,
                "run_identity": dict(identity),
                "action_ball_full_mdp_ppo_recipe_sha256": FULL_MDP_PPO_RECIPE_SHA256,
                "prepared_update_sha256": rows[index][0]["prepared_update_sha256"]}
            if type(payload["iter"]) is not int or payload["iter"] != index or not _same(payload["infos"], infos):
                _fail("snapshot infos binding " + name)
            _model_optimizer(payload, torch, name); inventory.append(actual)
    finally: os.close(dfd)
    scheduled = set(indices)
    if any((i in scheduled) != (snapshot is not None) for i, (_row, snapshot) in enumerate(rows)):
        _fail("snapshot ACK schedule")
    return inventory
def _completion(path, identity, count, evidence, snapshots):
    raw, _ = _read_regular(path, "completion receipt")
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1: _fail("completion receipt framing")
    try: record = json.loads(raw[:-1], object_pairs_hook=_duplicates)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc: _fail("completion receipt JSON: " + str(exc))
    _keys(record, COMPLETION_KEYS, "completion receipt")
    expected = {"schema_version": COMPLETION_SCHEMA_VERSION,
        "record_type": "mujoco_full_mdp_completion",
        "diagnostic_unauthorized": True, "checkpoint_authority": False,
        "resume_authority": False, "run_identity": identity, "action_contract": ACTION_CONTRACT,
        "action_ball_full_mdp_ppo_recipe_sha256": FULL_MDP_PPO_RECIPE_SHA256,
        "num_envs": NUM_ENVS, "num_steps_per_env": STEPS_PER_UPDATE,
        "completed_updates": count, "environment_steps": STEPS_PER_UPDATE * count,
        "transitions": TRANSITIONS_PER_UPDATE * count, "evidence_jsonl": evidence,
        "snapshot_receipts": snapshots, "final_observation_finite": True,
        "rollout_storage_finite": True, "optimizer_state_present": True,
        "optimizer_state_finite": True}
    if not _same(record, expected): _fail("completion seal binding")
def _rate(numerator, denominator):
    return numerator / denominator if denominator else None
def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None
def consume(evidence_jsonl: Path, *, expected_updates: int, expected_source_commit: str,
            expected_run_namespace: str, expected_plant_xml: Path, snapshot_dir: Path,
            completion_json=None) -> dict:
    if type(expected_updates) is not int or expected_updates not in (1, 5, COMPLETE_UPDATES):
        _fail("expected update count")
    runtime_module = _epa48_runtime_module()
    runtime_verification = runtime_module.verify_runtime_stack_preimport()
    runtime_stack = runtime_module.verified_runtime_stack_identity(
        runtime_verification
    )
    runtime_mjb = _verified_runtime_mjb(evidence_jsonl)
    identity = _run_identity(
        expected_source_commit, expected_run_namespace, runtime_stack,
        expected_plant_xml, runtime_mjb)
    complete = expected_updates == COMPLETE_UPDATES
    if snapshot_dir is None or complete != (completion_json is not None): _fail("artifact mode")
    rows, events, life, terms, outcomes, evidence = _read_rows(
        evidence_jsonl, expected_updates, identity)
    snapshots = _snapshots(snapshot_dir, rows, identity, complete)
    if complete: _completion(completion_json, identity, expected_updates, evidence, snapshots)
    recovery = sum(events[key] for key in (
        "recovery_success_rows", "recovery_failure_rows", "recovery_timeout_rows"))
    required = ("reveal_due_rows", "reveal_rows", "launch_rows", "racket_contact_rows",
                "selected_contact_rows", "r03_present_rows", "r03_physically_valid_rows",
                "landing_crossing_rows", "flight_terminal_rows",
                "r06_present_rows", "r06_eligible_rows", "r06_common_rows",
                "r07_present_rows", "shot_retired_rows",
                "completed_action_epoch_rows")
    missing = [key for key in required if events[key] == 0]
    transitions = TRANSITIONS_PER_UPDATE * expected_updates
    milestones = dict(sorted(events.items()))
    milestones.update({key: life[key] for key in (
        "landing_on_opponent_rows", "landing_opponent_bound_rows", "gym_reset_rows")})
    return {"schema_version": SUMMARY_SCHEMA_VERSION,
        "diagnostic_unauthorized": True,
        "evidence_level": "sealed_engineering_longrun" if complete else "advisory_prefix",
        "run_identity": identity, "engineering_run_complete": complete,
        "producer_attested_milestone_coverage_complete": not missing,
        "producer_attested_milestone_coverage_missing": missing,
        "same_epoch_chain_replay_status": "not_produced",
        "full_a_complete": False, "update_ack_count": expected_updates,
        "last_update_index": expected_updates - 1,
        "environment_steps": STEPS_PER_UPDATE * expected_updates, "transitions": transitions,
        "milestones": milestones, "outcome_code_totals": outcomes, "terminal_bit_totals": terms,
        "table_terminal": {
            "robot_hit_table_rows": terms["robot_hit_table"],
            "resolved_rows": life["resolved_table_rows"],
            "keepout_only_rows": (
                terms["robot_hit_table"] - life["resolved_table_rows"]
            ),
        },
        "action_coverage": {
            "slot0": {"status": "observed", "observed_rows": transitions, "denominator": transitions},
            "forehand": {"status": "observed", "observed_rows": transitions, "denominator": transitions},
            "backhand": {"status": "未测", "observed_rows": 0, "denominator": 0}},
        "opportunity_d05": {"status": "not_produced", "denominator": None},
        "portable_reveal_opportunity": {
            "scheduled_rows": events["scheduled_due_rows"],
            "terminal_overlap_rows": events["due_terminal_overlap_rows"],
            "due_rows": events["reveal_due_rows"], "accepted_rows": events["reveal_rows"],
            "deferred_rows": events["reveal_deferred_rows"],
            "accept_rate": _rate(events["reveal_rows"], events["reveal_due_rows"]),
            "defer_rate": _rate(events["reveal_deferred_rows"], events["reveal_due_rows"])},
        # Contact and exact-R03 publication are distinct clocked marginals, so
        # neither is a rowwise subset of the other.  Preserve the requested
        # R03 denominators without turning a legal early-contact prefix into a
        # false conservation failure or disguising the quotient as a rate.
        "hit_opportunity_r03": {
            "present_rows": events["r03_present_rows"],
            "physically_valid_rows": events["r03_physically_valid_rows"],
            "selected_contact_rows": events["selected_contact_rows"],
            "selected_contact_to_physically_valid_ratio": _ratio(
                events["selected_contact_rows"],
                events["r03_physically_valid_rows"],
            ),
        },
        "rates": {
            "selected_contact_per_launch": _rate(
                events["selected_contact_rows"], events["launch_rows"]),
            "r03_physically_valid_per_present": _rate(
                events["r03_physically_valid_rows"], events["r03_present_rows"]),
            "r06_common_per_eligible": _rate(
                events["r06_common_rows"], events["r06_eligible_rows"]),
            "opponent_landing_per_crossing": _rate(
                life["landing_on_opponent_rows"], events["landing_crossing_rows"]),
            "recovery_success_per_terminal": _rate(events["recovery_success_rows"], recovery)},
        "last_learning_rate": rows[-1][0]["learning_rate"], "evidence_jsonl": evidence,
        "snapshot_count": len(snapshots), "snapshot_inventory": snapshots,
        "model_abi_verified": True, "optimizer_state_verified": True,
        "runtime_mjb_verified": True,
        "completion_seal_verified": complete,
        "action_ball_full_mdp_ppo_recipe_sha256": FULL_MDP_PPO_RECIPE_SHA256,
        "action_contract": dict(ACTION_CONTRACT) if complete else None}
def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--evidence-jsonl", type=Path, required=True); parser.add_argument("--expected-updates", type=int, default=COMPLETE_UPDATES)
    parser.add_argument("--expected-source-commit", required=True); parser.add_argument("--expected-run-namespace", required=True)
    parser.add_argument("--expected-plant-xml", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True); parser.add_argument("--completion-json", type=Path); args = parser.parse_args()
    print(json.dumps(consume(args.evidence_jsonl, expected_updates=args.expected_updates,
        expected_source_commit=args.expected_source_commit,
        expected_run_namespace=args.expected_run_namespace,
        expected_plant_xml=args.expected_plant_xml, snapshot_dir=args.snapshot_dir,
        completion_json=args.completion_json), sort_keys=True, separators=(",", ":"), allow_nan=False)); return 0
if __name__ == "__main__": raise SystemExit(main())
