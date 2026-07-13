#!/usr/bin/env python3
"""Export a training checkpoint to deploy ONNX WITHOUT booting Isaac (pure CPU torch).

Purpose: unblock exports when Isaac/Kit cannot boot (e.g. the 2026-07-04 runpod host whose Kit
hangs at extension load). Replicates utils/exporter.py faithfully:

- actor MLP rebuilt from the checkpoint's own weight shapes (no config guessing);
- EmpiricalNormalization loaded from the checkpoint's obs_norm_state_dict (exact rsl_rl math);
- the 6 motion-reference buffers read straight from the same .npz clips training used
  (MotionLoader semantics: concatenate clip arrays along time);
- graph/io identical to _OnnxMotionPolicyExporter (opset 11, same input/output names);
- metadata copied from a DONOR onnx exported earlier under the SAME task config (all metadata
  fields are config-derived — joint names/gains/obs layout/strike phases — so a same-config donor
  is exact; run_path is overridden).

Usage:
  python scripts/standalone_onnx_export.py --ckpt <model_x.pt> --fh fh.npz --bh bh.npz \
      --donor <older exported/policy.onnx> --out <run_dir>/exported --run-path <label>
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np


def _load_light_module(name: str, filename: str):
    """Load one Isaac-free source file without executing whole_body_tracking/__init__.py."""
    utils_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "source", "whole_body_tracking",
        "whole_body_tracking", "utils",
    ))
    path = os.path.join(utils_dir, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load dependency-light contract module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TC = _load_light_module("standalone_training_contract", "training_contract.py")
TRAINING_CONTRACT_SCHEMA_VERSION = _TC.TRAINING_CONTRACT_SCHEMA_VERSION
checkpoint_claims_contract = _TC.checkpoint_claims_contract
checkpoint_contract_lineage_exact = _TC.checkpoint_contract_lineage_exact
require_checkpoint_contract_binding = _TC.require_checkpoint_contract_binding
bind_actor_leg_ref_mask_metadata = _TC.bind_actor_leg_ref_mask_metadata
require_actor_leg_ref_mask_provenance = _TC.require_actor_leg_ref_mask_provenance
resolve_motion_body_lin_vel_points = _TC.resolve_motion_body_lin_vel_points
validate_schema3_contract = _TC.validate_schema3_contract
validate_schema3_contract_structure = _TC.validate_schema3_contract_structure

_NE = _load_light_module("standalone_stage1_normal_envelope", "stage1_normal_envelope.py")
FORMAL_CLIP_ORDER = _NE.FORMAL_CLIP_ORDER
derive_stage1_normal_envelope = _NE.derive_stage1_normal_envelope

_AO = _load_light_module("standalone_atomic_output", "atomic_output.py")
atomic_output_path = _AO.atomic_output_path

# A stdlib+NumPy subprocess probe used on machines that intentionally have no Isaac packages.
# The formal export path below uses these exact file-loaded modules; passing this flag proves the
# package __init__ was not touched before ONNX/Torch are imported.
if "--contract-import-smoke" in sys.argv:
    imported_package = any(
        name == "whole_body_tracking" or name.startswith("whole_body_tracking.")
        for name in sys.modules
    )
    imported_isaac = any(
        name == "isaaclab" or name.startswith(("isaaclab.", "omni.", "pxr."))
        for name in sys.modules
    )
    imported_export_runtime = any(
        name in ("onnx", "torch") or name.startswith(("onnx.", "torch."))
        for name in sys.modules
    )
    print(
        "[standalone-export-import] OK "
        f"whole_body_tracking_package_imported={str(imported_package).lower()} "
        f"isaac_modules_imported={str(imported_isaac).lower()} "
        f"onnx_torch_imported={str(imported_export_runtime).lower()} "
        f"training_schema={TRAINING_CONTRACT_SCHEMA_VERSION} "
        f"envelope_schema={_NE.ENVELOPE_SCHEMA_VERSION}"
    )
    raise SystemExit(1 if imported_package or imported_isaac or imported_export_runtime else 0)

import onnx
import torch
import torch.nn as nn

MOTION_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def _require(condition: bool, message: str) -> None:
    """Optimization-safe contract check (unlike assert, survives ``python -O``)."""
    if not condition:
        raise SystemExit(f"[FATAL] {message}")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_stage1_question_bank_module():
    """Load only the dependency-light bank module, bypassing the Isaac-heavy mdp package init."""
    path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "source", "whole_body_tracking", "whole_body_tracking",
        "tasks", "tracking", "mdp", "stage1_question_bank.py",
    ))
    spec = importlib.util.spec_from_file_location("standalone_stage1_question_bank", path)
    _require(spec is not None and spec.loader is not None, "cannot load stage1_question_bank.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _csv(meta: dict[str, str], key: str) -> list[str]:
    _require(key in meta and str(meta[key]).strip(), f"donor onnx missing metadata key {key}")
    return [item.strip() for item in str(meta[key]).split(",")]


def _csv_floats(meta: dict[str, str], key: str) -> list[float]:
    try:
        values = [float(item) for item in _csv(meta, key)]
    except ValueError as exc:
        raise SystemExit(f"[FATAL] donor metadata {key} is not numeric") from exc
    _require(all(np.isfinite(values)), f"donor metadata {key} contains NaN/Inf")
    return values


def _flat_pairs(pairs) -> list[float]:
    return [float(value) for pair in pairs for value in pair]


def _require_exact_floats(meta: dict[str, str], key: str, expected) -> None:
    actual = _csv_floats(meta, key)
    wanted = [float(value) for value in expected]
    _require(
        len(actual) == len(wanted) and np.array_equal(np.asarray(actual), np.asarray(wanted)),
        f"donor metadata {key} != checkpoint training contract",
    )


def _bind_schema3_donor_metadata(donor_meta: dict[str, str], contract: dict) -> None:
    """Prove and canonicalize every donor-sourced execution value against schema 3."""
    motion_body_lin_vel_points = resolve_motion_body_lin_vel_points(
        contract["motion_kinematics_contracts"]
    )
    exact_csv = {
        "joint_names": contract["joint_names"],
        "articulation_joint_names": contract["articulation_joint_names"],
        "joint_actuator_types": contract["joint_actuator_types"],
        "observation_names": contract["actor_obs_term_names"],
        "actor_obs_term_names": contract["actor_obs_term_names"],
        "body_names": contract["body_names"],
    }
    for key, expected in exact_csv.items():
        _require(_csv(donor_meta, key) == [str(value) for value in expected], (
            f"donor metadata {key} != checkpoint training contract"
        ))

    numeric_csv = {
        "action_joint_ids": contract["action_joint_ids"],
        "default_joint_pos": contract["default_joint_pos"],
        "action_scale": contract["action_scale"],
        "joint_stiffness": contract["joint_stiffness"],
        "joint_damping": contract["joint_damping"],
        "joint_effort_limits": contract["joint_effort_limits"],
        "joint_armature": contract["joint_armature"],
        "joint_friction_coefficients": contract["joint_friction_coefficients"],
        "joint_velocity_limits": contract["joint_velocity_limits"],
        "qdes_joint_pos_limits": _flat_pairs(contract["qdes_joint_pos_limits"]),
        "observation_history_lengths": contract["observation_history_lengths"],
        "actor_obs_term_dims": contract["actor_obs_term_dims"],
        "body_indices": contract["body_indices"],
        "clip_seg_lengths": contract["motion_segment_lengths"],
        "racket_control_point_offset_wrist_m": contract["racket_control_point_offset_wrist_m"],
    }
    for key, expected in numeric_csv.items():
        _require_exact_floats(donor_meta, key, expected)

    exact_scalars = {
        "actor_obs_contract": contract["actor_obs_contract"],
        "actor_obs_mode": contract["actor_obs_mode"],
        "actor_obs_total_dim": contract["actor_obs_total_dim"],
        "anchor_body_name": contract["anchor_body_name"],
        "anchor_body_index": contract["anchor_body_index"],
        "control_decimation": contract["control_decimation"],
        "qdes_clamp": "1" if contract["qdes_clamp"] else "0",
        "action_use_default_offset": (
            "1" if contract["action_use_default_offset"] else "0"
        ),
        "racket_control_point": contract["racket_control_point"],
        "joint_friction_backend": contract["joint_friction_backend"],
        "joint_friction_semantics": contract["joint_friction_semantics"],
        "joint_friction_units": contract["joint_friction_units"],
        "motion_kinematics_exact": (
            "1" if contract["motion_kinematics_exact"] else "0"
        ),
    }
    if "face_command_pairing" in contract:
        exact_scalars["face_command_pairing"] = contract["face_command_pairing"]
    if "face_command_enabled" in contract:
        exact_scalars["face_command_enabled"] = (
            "1" if contract["face_command_enabled"] else "0"
        )
    if "motion_allow_legacy_link_origin_velocity" in contract:
        exact_scalars["motion_allow_legacy_link_origin_velocity"] = (
            "1" if contract["motion_allow_legacy_link_origin_velocity"] else "0"
        )
    for key, expected in exact_scalars.items():
        _require(str(donor_meta.get(key, "")) == str(expected), (
            f"donor metadata {key} != checkpoint training contract"
        ))
    for key in ("physics_step_dt_s", "policy_step_dt_s"):
        _require_exact_floats(donor_meta, key, [contract[key]])

    # Re-serialize from the checkpoint contract after comparison. The donor is a checked carrier,
    # never the authority for values that decode actions or advance the reference clock.
    def csv_values(values) -> str:
        return ",".join(format(float(value), ".17g") for value in values)

    for key, values in exact_csv.items():
        donor_meta[key] = ",".join(str(value) for value in values)
    # This field was introduced after the currently reusable donor graphs were produced.  Its
    # authority is the content-bound checkpoint contract above, not an older donor that cannot
    # attest it.  Bind it from that contract so old same-task donors remain usable without ever
    # trusting or propagating their value.
    donor_meta["motion_body_lin_vel_points"] = ",".join(motion_body_lin_vel_points)
    for key, values in numeric_csv.items():
        donor_meta[key] = csv_values(values)
    for key, value in exact_scalars.items():
        donor_meta[key] = str(value)
    donor_meta["physics_step_dt_s"] = format(float(contract["physics_step_dt_s"]), ".17g")
    donor_meta["policy_step_dt_s"] = format(float(contract["policy_step_dt_s"]), ".17g")


def _donor_activation(donor) -> tuple[nn.Module, int]:
    """Recover the actor activation from the donor graph instead of assuming ELU.

    Activations have no checkpoint parameters, so weight shapes alone cannot identify them.  The
    exact donor graph is already a required input to this export path and is the only reliable
    source available without booting Isaac.
    """
    supported = {
        "Elu": lambda node: nn.ELU(
            alpha=float(next((a.f for a in node.attribute if a.name == "alpha"), 1.0))
        ),
        "Relu": lambda _node: nn.ReLU(),
        "Selu": lambda _node: nn.SELU(),
        "Tanh": lambda _node: nn.Tanh(),
        "Sigmoid": lambda _node: nn.Sigmoid(),
        "Softplus": lambda _node: nn.Softplus(),
        "LeakyRelu": lambda node: nn.LeakyReLU(
            negative_slope=float(
                next((a.f for a in node.attribute if a.name == "alpha"), 0.01)
            )
        ),
    }
    nodes = [node for node in donor.graph.node if node.op_type in supported]
    kinds = {node.op_type for node in nodes}
    _require(bool(nodes), "donor graph has no supported actor activation")
    _require(
        len(kinds) == 1,
        f"donor graph has ambiguous activation operators {sorted(kinds)}",
    )
    prototype = supported[nodes[0].op_type](nodes[0])
    return prototype, len(nodes)


def build_actor(msd: dict, activation: nn.Module, *, expected_activations: int) -> nn.Sequential:
    idx = sorted({int(k.split(".")[1]) for k in msd if k.startswith("actor.") and k.endswith(".weight")})
    layers, prev_end = [], -1
    activation_count = 0
    for i in idx:
        w = msd[f"actor.{i}.weight"]
        if prev_end >= 0:
            for _ in range(i - prev_end - 1):
                layers.append(copy.deepcopy(activation))
                activation_count += 1
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(msd[f"actor.{i}.bias"])
        layers.append(lin)
        prev_end = i
    _require(bool(layers), "checkpoint actor has no linear layers")
    _require(
        activation_count == expected_activations,
        "checkpoint actor layer gaps disagree with donor activation count: "
        f"checkpoint={activation_count}, donor={expected_activations}",
    )
    return nn.Sequential(*layers)


class _StandaloneExporter(nn.Module):
    def __init__(self, actor, normalizer, motion):
        super().__init__()
        self.actor = actor
        self.normalizer = normalizer
        for k in MOTION_KEYS:
            setattr(self, k, motion[k])
        self.time_step_total = motion["joint_pos"].shape[0]

    def forward(self, x, time_step):
        t = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.actor(self.normalizer(x)),
            self.joint_pos[t],
            self.joint_vel[t],
            self.body_pos_w[t],
            self.body_quat_w[t],
            self.body_lin_vel_w[t],
            self.body_ang_vel_w[t],
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--fh", required=True)
    p.add_argument("--bh", required=True)
    p.add_argument("--donor", required=True, help="same-task-config exported policy.onnx to copy metadata from")
    p.add_argument("--harvest", required=True, help="motion-buffer npz produced by harvest_onnx_motion.py from the donor")
    p.add_argument(
        "--train-bank",
        help="exact schema-3 train NPZ; mandatory for a formal 179-D face export",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--run-path", default="standalone-export")
    p.add_argument("--bake-obs-norm", action="store_true",
                   help="bake the checkpoint's empirical obs normalization into the graph (DEPLOY artifacts)")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    msd = ckpt["model_state_dict"]
    donor = onnx.load(args.donor)
    donor_activation, donor_activation_count = _donor_activation(donor)
    actor = build_actor(
        msd, donor_activation, expected_activations=donor_activation_count
    )
    in_dim = actor[0].in_features
    training_contract_path = os.path.join(
        os.path.dirname(os.path.abspath(args.ckpt)), "params", "training_contract.json"
    )
    training_contract = None
    training_contract_sha256 = None
    training_contract_lineage_exact = False
    if os.path.isfile(training_contract_path):
        try:
            with open(training_contract_path, encoding="utf-8") as stream:
                training_contract = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"[FATAL] invalid checkpoint training contract {training_contract_path}: {exc}"
            ) from exc
        _require(isinstance(training_contract, dict), "training contract root must be an object")
        training_contract_sha256 = _sha256_file(training_contract_path)
        try:
            contract_schema = int(training_contract.get("schema_version", 0))
        except (TypeError, ValueError):
            raise SystemExit("[FATAL] invalid training-contract schema version")
        if contract_schema == TRAINING_CONTRACT_SCHEMA_VERSION:
            try:
                validate_schema3_contract_structure(training_contract)
                require_actor_leg_ref_mask_provenance(training_contract)
                require_checkpoint_contract_binding(
                    ckpt,
                    schema=contract_schema,
                    sha256=training_contract_sha256,
                    require_lineage_exact=False,
                )
            except ValueError as exc:
                raise SystemExit(f"[FATAL] {exc}") from exc
            training_contract_lineage_exact = checkpoint_contract_lineage_exact(ckpt)
            if training_contract_lineage_exact:
                try:
                    validate_schema3_contract(training_contract)
                except ValueError as exc:
                    raise SystemExit(f"[FATAL] {exc}") from exc
        elif contract_schema in (1, 2):
            if checkpoint_claims_contract(ckpt):
                try:
                    require_checkpoint_contract_binding(
                        ckpt,
                        schema=contract_schema,
                        sha256=training_contract_sha256,
                        require_lineage_exact=False,
                    )
                except ValueError as exc:
                    raise SystemExit(f"[FATAL] {exc}") from exc
        else:
            raise SystemExit(
                f"[FATAL] unsupported training-contract schema {contract_schema}; "
                f"formal schema is {TRAINING_CONTRACT_SCHEMA_VERSION}"
            )
    elif checkpoint_claims_contract(ckpt):
        raise SystemExit(
            "[FATAL] checkpoint claims a training-contract binding but the adjacent sidecar "
            f"is missing: {training_contract_path}"
        )

    # The existing export chain (play.py -> _OnnxPolicyExporter -> hardware/MuJoCo) bakes the RAW
    # actor with NO empirical normalizer — verified 2026-07-04 by zero-point matching a donor ONNX
    # against its own checkpoint (matches ONLY without the normalizer). Identity keeps bit-parity
    # with that chain, so consumers treat every ONNX the same way.
    # P0 CORRECTION (2026-07-04): training DOES use empirical_normalization=true, so a raw-actor
    # ONNX fed raw obs is out-of-distribution and scores ~0 — the earlier "baking it in scores 0"
    # conclusion was an artifact of the then-broken eval harness (it normalized nothing, so ONLY a
    # raw export could match its raw obs at the zero point). The training-time transform now ships
    # as an `obs_norm.npz` SIDECAR next to the ONNX (scripts/make_std_sidecar.py), applied by
    # scripts/mujoco_eval_onnx.py before inference. If you ever bake the normalizer INTO an export,
    # do NOT ship the sidecar next to it (it would be applied twice).
    if args.bake_obs_norm:
        # DEPLOY FIX (2026-07-04 P0): training uses empirical_normalization=true, so the raw actor
        # is lobotomized on raw obs. Baking (x-mean)/(std+eps) INTO the graph makes the artifact
        # self-contained for the C++ deploy runner (which has no normalization stage). Metadata
        # obs_norm_baked=1 tells eval harnesses NOT to also apply the sidecar (double-norm guard).
        from rsl_rl.modules import EmpiricalNormalization

        normalizer = EmpiricalNormalization(shape=[in_dim])
        normalizer.load_state_dict(ckpt["obs_norm_state_dict"])
        normalizer.eval()
    else:
        normalizer = nn.Identity()

    clips = [dict(np.load(args.fh)), dict(np.load(args.bh))]
    seg_lengths = [c["joint_pos"].shape[0] for c in clips]
    # Motion buffers come from a HARVEST of the donor ONNX (see harvest_onnx_motion.py): the env
    # subsets body arrays to cfg.body_names (14 tracked bodies) in cfg order, while the npz holds
    # all robot bodies — the donor already carries the exact baked buffers (same clip pair,
    # asserted below). Harvest runs in the onnxruntime venv; this script runs in the torch venv.
    h = dict(np.load(args.harvest))
    _require(int(h.get("total", -1)) == sum(seg_lengths), "harvest length != clip pair length")
    donor_sha256 = _sha256_file(args.donor)
    harvest_donor_sha = str(np.asarray(h.get("donor_sha256", "")).item())
    _require(
        harvest_donor_sha == donor_sha256,
        "harvest was not produced from this exact donor ONNX (SHA256 mismatch)",
    )
    _require(int(h.get("donor_obs_dim", -1)) == in_dim, "harvest donor obs dim != checkpoint actor")
    for key in MOTION_KEYS:
        _require(key in h, f"harvest missing {key}")
        _require(h[key].shape[0] == sum(seg_lengths), f"harvest {key} frame count mismatch")
        _require(np.isfinite(h[key]).all(), f"harvest {key} contains NaN/Inf")
    motion = {k: torch.as_tensor(h[k], dtype=torch.float32) for k in MOTION_KEYS}

    donor_meta = {e.key: e.value for e in donor.metadata_props}
    for req in (
        "hope_metadata_schema_version", "joint_names", "joint_stiffness", "clip_seg_lengths",
        "clip_strike_phases", "motion_clip_sha256", "actor_obs_contract",
        "actor_obs_total_dim", "actor_obs_term_dims", "observation_names",
    ):
        _require(req in donor_meta, f"donor onnx missing metadata key {req} — pick a newer donor")
    _require(donor_meta["hope_metadata_schema_version"] == "2", "donor metadata schema must be v2")
    donor_segs = [int(s) for s in donor_meta["clip_seg_lengths"].split(",")]
    _require(donor_segs == seg_lengths, (
        f"donor clip layout {donor_segs} != provided clips {seg_lengths} — donor must come from the "
        f"same motion pair (otherwise gains/obs metadata may not match either)"
    ))
    donor_hashes = donor_meta["motion_clip_sha256"].split(",")
    actual_hashes = [_sha256_file(args.fh), _sha256_file(args.bh)]
    _require(donor_hashes == actual_hashes, (
        "donor clip SHA256 does not match --fh/--bh; equal frame counts are insufficient"
    ))
    donor_obs = next(v for v in donor.graph.input if v.name == "obs")
    donor_obs_dim = donor_obs.type.tensor_type.shape.dim[1].dim_value
    donor_actions = next(v for v in donor.graph.output if v.name == "actions")
    donor_action_dim = donor_actions.type.tensor_type.shape.dim[1].dim_value
    _require(donor_obs_dim == in_dim == int(donor_meta["actor_obs_total_dim"]), (
        f"donor/checkpoint obs mismatch: donor={donor_obs_dim}, checkpoint={in_dim}, "
        f"metadata={donor_meta['actor_obs_total_dim']}"
    ))
    _require(donor_action_dim == actor[-1].out_features == 31, "donor/checkpoint action dim mismatch")
    # A donor is a source of reference buffers and static layout only. It is never authority for
    # which bank/protocol trained this checkpoint; strip those labels before rebuilding them from
    # the checkpoint-neighbour training contract.
    for key in list(donor_meta):
        if key.startswith("stage1_") or key.startswith("training_contract_"):
            donor_meta.pop(key, None)
    donor_meta["training_contract_exact"] = "0"

    contract_schema = 0
    if training_contract is not None:
        try:
            contract_schema = int(training_contract.get("schema_version", 0))
        except (TypeError, ValueError):
            raise SystemExit("[FATAL] invalid training-contract schema version")
        donor_meta["training_contract_schema_version"] = str(contract_schema)
        donor_meta["training_contract_sha256"] = str(training_contract_sha256)
    # Donor metadata is never authority for observation semantics.  A schema-3 checkpoint may
    # legitimately use an older unmasked donor, while an unmasked/legacy checkpoint must not
    # inherit actor_leg_ref_mask=1 from a masked donor.
    bind_actor_leg_ref_mask_metadata(donor_meta, None)
    if contract_schema == TRAINING_CONTRACT_SCHEMA_VERSION:
        _bind_schema3_donor_metadata(donor_meta, training_contract)
        donor_effort = [float(v) for v in donor_meta["joint_effort_limits"].split(",")]
        _require(
            len(donor_effort) == 31 and all(value > 0.0 for value in donor_effort),
            "donor joint effort limits are invalid",
        )
        trained_motion = list(training_contract.get("motion_clips") or [])
        trained_motion_sha = [str(item.get("sha256", "")) for item in trained_motion]
        _require(
            trained_motion_sha == actual_hashes,
            "checkpoint training-contract motion SHA does not match --fh/--bh",
        )
        _require(
            [int(item.get("segment_length", -1)) for item in trained_motion] == seg_lengths,
            "checkpoint training-contract segment lengths do not match --fh/--bh",
        )
        actor_contract = str(training_contract.get("actor_obs_contract", ""))
        actor_total = int(training_contract.get("actor_obs_total_dim", -1))
        _require(actor_total == in_dim, "training-contract actor dimension != checkpoint")
        _require(
            donor_meta.get("actor_obs_contract") == actor_contract,
            "donor actor observation contract != checkpoint training contract",
        )
        expected_mode = {
            "full": "full",
            "deploy_parity": "deploy_parity",
            "deploy_parity_face179": "deploy_parity",
            "deploy_parity_station181": "deploy_parity",
            "hitter_footwork": "hitter_footwork",
            "hitter_pure": "hitter_pure",
        }.get(actor_contract)
        _require(expected_mode is not None, f"unknown training actor contract {actor_contract!r}")
        _require(
            donor_meta.get("actor_obs_mode") == expected_mode,
            "donor actor obs mode != checkpoint training contract",
        )
        phases = [float(v) for v in training_contract.get("strike_phase_per_clip", [])]
        _require(len(phases) == len(seg_lengths), "training contract lacks one strike phase per clip")
        donor_meta.update({
            "training_contract_exact": "1" if training_contract_lineage_exact else "0",
            "training_contract_schema_version": str(contract_schema),
            "training_contract_sha256": training_contract_sha256,
            "episode_length_s": format(float(training_contract["episode_length_s"]), ".17g"),
            "qdes_clamp": "1" if bool(training_contract["qdes_clamp"]) else "0",
            "wrap_teleport": "1" if bool(training_contract["motion_wrap_teleport"]) else "0",
            "motion_hold_steps_range": ",".join(
                str(int(v)) for v in training_contract["motion_hold_steps_range"]
            ),
            "motion_hold_reference": str(training_contract["motion_hold_reference"]),
            "motion_stand_start_prob": str(training_contract["motion_stand_start_prob"]),
            "motion_stand_start_min_hold": str(
                int(training_contract["motion_stand_start_min_hold"])
            ),
            "motion_stand_start_yaw_range": ",".join(
                format(float(v), ".17g")
                for v in training_contract["motion_stand_start_yaw_range"]
            ),
            "motion_rsi_hold_root_stand_z": (
                "1" if bool(training_contract["motion_rsi_hold_root_stand_z"]) else "0"
            ),
            "face_command_pairing": str(
                training_contract.get("face_command_pairing", "shared_plus_y")
            ),
            "motion_allow_legacy_link_origin_velocity": (
                "1"
                if bool(training_contract.get(
                    "motion_allow_legacy_link_origin_velocity", False
                ))
                else "0"
            ),
            "motion_kinematics_exact": (
                "1" if bool(training_contract["motion_kinematics_exact"]) else "0"
            ),
            "clip_strike_phases": ",".join(format(v, ".17g") for v in phases),
            "motion_clip_sha256": ",".join(actual_hashes),
            "actor_obs_contract": actor_contract,
            "actor_obs_mode": expected_mode,
            "actor_obs_total_dim": str(in_dim),
        })
        if "face_command_enabled" in training_contract:
            donor_meta["face_command_enabled"] = (
                "1" if bool(training_contract["face_command_enabled"]) else "0"
            )
        signs = training_contract.get("mount_normal_sign_per_clip")
        checkpoint_mount_signs = None
        if signs is not None:
            checkpoint_mount_signs = tuple(float(value) for value in signs)
            donor_meta["mount_normal_sign_per_clip"] = ",".join(
                format(value, ".17g") for value in checkpoint_mount_signs
            )
        bank_contract = training_contract.get("question_bank")
        if actor_contract == "deploy_parity_face179":
            _require(
                donor_meta.get("mount_normal_sign_per_clip") == "1,-1",
                "formal face179 standalone export requires checkpoint-contract "
                "mount_normal_sign_per_clip=[+1,-1]",
            )
            _require(
                bank_contract is not None,
                "formal face179 standalone export requires an exact schema-3 train-bank binding",
            )
        if bank_contract is not None:
            expected_bank = {
                "schema_version": 3,
                "split": "train",
                "exact": True,
            }
            _require(
                all(bank_contract.get(k) == v for k, v in expected_bank.items()),
                "formal standalone export needs an exact schema-v3 train-bank contract",
            )
            for key in ("sha256", "source_family_sha256"):
                _require(
                    len(str(bank_contract.get(key, ""))) == 64,
                    f"training bank contract has invalid {key}",
                )
            donor_meta.update({
                "stage1_train_bank_sha256": bank_contract["sha256"],
                "stage1_bank_schema_version": "3",
                "stage1_bank_split": "train",
                "stage1_source_family_sha256": bank_contract["source_family_sha256"],
                "stage1_question_bank_exact": "1",
            })
            if actor_contract == "deploy_parity_face179":
                _require(
                    bool(args.train_bank),
                    "formal face179 standalone export requires --train-bank; donor metadata or "
                    "the checkpoint-side JSON cannot substitute for the exact normal rows",
                )
                qb_module = _load_stage1_question_bank_module()
                try:
                    train_bank = qb_module.load_question_bank(
                        os.path.abspath(args.train_bank),
                        device="cpu",
                        clip_names=FORMAL_CLIP_ORDER,
                        allow_legacy=False,
                        expected_split="train",
                    )
                    qb_module.validate_runtime_motion_contract(
                        train_bank.metadata,
                        [os.path.abspath(args.fh), os.path.abspath(args.bh)],
                        seg_lengths,
                        phases,
                    )
                except (KeyError, OSError, ValueError) as exc:
                    raise SystemExit(f"[FATAL] invalid formal face179 --train-bank: {exc}") from exc
                donor_meta.update(
                    derive_stage1_normal_envelope(
                        train_bank.source_path,
                        expected_train_bank_sha256=bank_contract["sha256"],
                        expected_source_family_sha256=bank_contract["source_family_sha256"],
                        mount_normal_sign_per_clip=checkpoint_mount_signs,
                        clip_order=FORMAL_CLIP_ORDER,
                    )
                )
    else:
        # Older/no checkpoint contracts cannot prove per-clip point semantics.  Never propagate a
        # donor's per-clip claim into a newly exported actor.  The evaluator may apply its narrow
        # pre-field exact-schema-2 all-COM rule; all other old aggregate contracts stay ambiguous
        # for teacher-reference reset while stand-keyframe diagnostics remain loadable.
        donor_meta.pop("motion_body_lin_vel_points", None)
    donor_meta["run_path"] = args.run_path
    # Always overwrite donor provenance. A no-bake re-export from a baked donor must clear the
    # old value or evaluators will skip the required sidecar.
    donor_meta["obs_norm_baked"] = "1" if args.bake_obs_norm else "0"
    donor_meta["empirical_normalization"] = "1" if "obs_norm_state_dict" in ckpt else "0"
    donor_meta["trained_with_obs_norm"] = donor_meta["empirical_normalization"]
    donor_meta["source_checkpoint_sha256"] = _sha256_file(args.ckpt)
    bind_actor_leg_ref_mask_metadata(
        donor_meta,
        training_contract if contract_schema == TRAINING_CONTRACT_SCHEMA_VERSION else None,
    )
    donor_meta["motion_harvest_donor_sha256"] = donor_sha256

    # Every checkpoint, donor, clip, harvest, bank, contract and envelope input has now been
    # validated. Only now create an export artifact, always at a same-directory temporary path.
    # A failed graph export/check/save leaves an existing policy.onnx byte-identical and removes
    # the owned temporary file; successful installation is one fsync + atomic replace.
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "policy.onnx")
    module = _StandaloneExporter(actor, normalizer, motion).eval()
    with atomic_output_path(out_path) as temporary_path:
        torch.onnx.export(
            module,
            (torch.zeros(1, in_dim), torch.zeros(1, 1)),
            str(temporary_path),
            export_params=True,
            opset_version=11,
            input_names=["obs", "time_step"],
            output_names=[
                "actions", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                "body_lin_vel_w", "body_ang_vel_w",
            ],
            dynamic_axes={},
        )
        model = onnx.load(temporary_path)
        del model.metadata_props[:]
        for key, value in donor_meta.items():
            entry = model.metadata_props.add()
            entry.key, entry.value = key, str(value)
        onnx.checker.check_model(model)
        onnx.save(model, temporary_path)
        installed_candidate = onnx.load(temporary_path)
        onnx.checker.check_model(installed_candidate)
        installed_meta = {entry.key: entry.value for entry in installed_candidate.metadata_props}
        _require(installed_meta == {key: str(value) for key, value in donor_meta.items()}, (
            "temporary ONNX metadata round-trip disagrees before atomic install"
        ))

    print(f"[standalone-export] SUCCESS {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
    print(f"[standalone-export] in_dim={in_dim} actions={actor[-1].out_features} "
          f"clips={seg_lengths} iter={ckpt.get('iter', '?')} metadata_keys={len(donor_meta)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
