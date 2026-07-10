#!/usr/bin/env python3
"""Freeze Isaac termination terms into a simulator-independent formal-eval contract.

The source of truth is a completed run's ``params/env.yaml``.  This module never infers behavior
from a run name.  Unknown active terms, unexpected functions, or missing parameters are rejected
because silently approximating them in MuJoCo would change which attempts enter the denominator.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml


CONTRACT_SCHEMA = "hope.phase1.termination-contract.v2"
KNOWN_TERMS = (
    "time_out",
    "anchor_pos",
    "anchor_ori",
    "ee_body_pos",
    "base_fell_tilt",
    "base_too_low",
)
TRACKING_TERMS = ("anchor_pos", "anchor_ori", "ee_body_pos")
PHYSICAL_TERMS = ("base_fell_tilt", "base_too_low")
FUNCTION_IDENTITIES = {
    "time_out": (
        "isaaclab.envs.mdp.terminations:time_out",
        "isaaclab.envs.mdp.terminations.time_out",
    ),
    "bad_anchor_pos_z_only": (
        "whole_body_tracking.tasks.tracking.mdp.terminations:bad_anchor_pos_z_only",
        "whole_body_tracking.tasks.tracking.mdp.terminations.bad_anchor_pos_z_only",
    ),
    "bad_anchor_ori": (
        "whole_body_tracking.tasks.tracking.mdp.terminations:bad_anchor_ori",
        "whole_body_tracking.tasks.tracking.mdp.terminations.bad_anchor_ori",
    ),
    "bad_motion_body_pos_z_only": (
        "whole_body_tracking.tasks.tracking.mdp.terminations:bad_motion_body_pos_z_only",
        "whole_body_tracking.tasks.tracking.mdp.terminations.bad_motion_body_pos_z_only",
    ),
    "bad_orientation": (
        "isaaclab.envs.mdp.terminations:bad_orientation",
        "isaaclab.envs.mdp.terminations.bad_orientation",
    ),
    "root_height_below_minimum": (
        "isaaclab.envs.mdp.terminations:root_height_below_minimum",
        "isaaclab.envs.mdp.terminations.root_height_below_minimum",
    ),
}


class TerminationContractError(RuntimeError):
    """Frozen termination config cannot be represented faithfully by the evaluator."""


class _FrozenEnvLoader(yaml.SafeLoader):
    """Read Isaac's data-only YAML tags without enabling Python object construction.

    Isaac's config dump uses exactly two Python-specific tags for ordinary container values.
    Registering those exact tags keeps the loader non-executing and leaves every other unknown
    Python tag rejected by ``SafeLoader``.
    """

    def construct_mapping(self, node: yaml.Node, deep: bool = False) -> dict[Any, Any]:
        if not isinstance(node, yaml.MappingNode):
            raise yaml.constructor.ConstructorError(
                None, None, f"expected a mapping node, got {node.id}", node.start_mark
            )
        self.flatten_mapping(node)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as exc:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable key",
                    key_node.start_mark,
                ) from exc
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _construct_python_tuple(loader: _FrozenEnvLoader, node: yaml.Node) -> tuple[Any, ...]:
    return tuple(loader.construct_sequence(node, deep=True))


def _construct_builtin_slice_data(loader: _FrozenEnvLoader, node: yaml.Node) -> list[Any]:
    # The termination parser does not use slice values.  Preserve their three scalar fields as
    # plain data; never call ``slice`` (or any constructor named by the input document).
    return list(loader.construct_sequence(node, deep=True))


_FrozenEnvLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", _construct_python_tuple
)
_FrozenEnvLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice",
    _construct_builtin_slice_data,
)


def _canonical_bytes(document: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise TerminationContractError(f"termination contract is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def _content_id(document: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(document)).hexdigest()


def _hash_file(path: Path) -> tuple[str, int]:
    before = path.stat()
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    after = path.stat()
    before_id = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_id = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_id != after_id or size != after.st_size:
        raise TerminationContractError(f"env config changed while hashing: {path}")
    return digest.hexdigest(), size


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TerminationContractError(
            f"{label} must be a YAML number, not {type(value).__name__}"
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise TerminationContractError(f"{label} must be finite and > 0, got {value!r}")
    return result


def _function_matches(value: Any, expected: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TerminationContractError(f"{label}.func must identify {expected}, got {value!r}")
    text = value.strip()
    identities = FUNCTION_IDENTITIES.get(expected)
    if identities is None or text not in identities:
        raise TerminationContractError(
            f"{label}.func must be one of {list(identities or ())}, got {value!r}; "
            "evaluator will not trust a matching function-name suffix from another module"
        )


def _params(term: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    value = term.get("params")
    if not isinstance(value, Mapping):
        raise TerminationContractError(f"{label}.params must be an object")
    return value


def _tracking_term(
    raw: Mapping[str, Any], name: str, function: str, *, body_names: bool = False,
    require_robot_asset: bool = False,
) -> dict[str, Any]:
    _function_matches(raw.get("func"), function, f"terminations.{name}")
    params = _params(raw, f"terminations.{name}")
    if params.get("command_name") != "motion":
        raise TerminationContractError(
            f"terminations.{name}.params.command_name must be 'motion', "
            f"got {params.get('command_name')!r}"
        )
    if require_robot_asset:
        asset_cfg = params.get("asset_cfg")
        if not isinstance(asset_cfg, Mapping) or asset_cfg.get("name") != "robot":
            raise TerminationContractError(
                f"terminations.{name}.params.asset_cfg.name must be 'robot', got "
                f"{getattr(asset_cfg, 'get', lambda *_: None)('name')!r}"
            )
    result: dict[str, Any] = {
        "active": True,
        "function": function,
        "threshold": _finite_positive(params.get("threshold"), f"terminations.{name}.threshold"),
    }
    if body_names:
        names = params.get("body_names")
        if not isinstance(names, Sequence) or isinstance(names, (str, bytes)) or not names:
            raise TerminationContractError(
                "terminations.ee_body_pos.params.body_names must be a non-empty list"
            )
        if any(not isinstance(item, str) or not item for item in names):
            raise TerminationContractError(
                f"terminations.ee_body_pos body_names must contain only non-empty strings, got {names}"
            )
        normalized = list(names)
        if len(set(normalized)) != len(normalized):
            raise TerminationContractError(
                f"terminations.ee_body_pos body_names must be unique non-empty names, got {normalized}"
            )
        result["body_names"] = normalized
    return result


def _parse_terms(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_terms = document.get("terminations")
    if not isinstance(raw_terms, Mapping):
        raise TerminationContractError("params/env.yaml must contain a terminations object")
    for name, value in raw_terms.items():
        if name not in KNOWN_TERMS and value is not None:
            raise TerminationContractError(
                f"unsupported active termination {name!r}; formal MuJoCo evaluation cannot "
                "faithfully reproduce it"
            )

    terms: dict[str, dict[str, Any]] = {
        name: {"active": False} for name in KNOWN_TERMS
    }
    for name in KNOWN_TERMS:
        raw = raw_terms.get(name)
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            raise TerminationContractError(f"terminations.{name} must be an object or null")
        if name == "time_out":
            _function_matches(raw.get("func"), "time_out", "terminations.time_out")
            if raw.get("time_out") is not True:
                raise TerminationContractError("terminations.time_out.time_out must be true")
            params = _params(raw, "terminations.time_out")
            if params:
                raise TerminationContractError("terminations.time_out.params must be empty")
            terms[name] = {"active": True, "function": "time_out", "time_out": True}
            continue
        if raw.get("time_out") is not False:
            raise TerminationContractError(f"terminations.{name}.time_out must be false")
        if name == "anchor_pos":
            terms[name] = _tracking_term(raw, name, "bad_anchor_pos_z_only")
        elif name == "anchor_ori":
            terms[name] = _tracking_term(
                raw, name, "bad_anchor_ori", require_robot_asset=True
            )
        elif name == "ee_body_pos":
            terms[name] = _tracking_term(
                raw, name, "bad_motion_body_pos_z_only", body_names=True
            )
        elif name == "base_fell_tilt":
            _function_matches(raw.get("func"), "bad_orientation", f"terminations.{name}")
            params = _params(raw, f"terminations.{name}")
            terms[name] = {
                "active": True,
                "function": "bad_orientation",
                "limit_angle": _finite_positive(
                    params.get("limit_angle"), f"terminations.{name}.limit_angle"
                ),
            }
        elif name == "base_too_low":
            _function_matches(
                raw.get("func"), "root_height_below_minimum", f"terminations.{name}"
            )
            params = _params(raw, f"terminations.{name}")
            terms[name] = {
                "active": True,
                "function": "root_height_below_minimum",
                "minimum_height": _finite_positive(
                    params.get("minimum_height"), f"terminations.{name}.minimum_height"
                ),
            }
    if not terms["time_out"]["active"]:
        raise TerminationContractError(
            "formal training-like scorecards require the frozen time_out termination to be active"
        )
    return terms


def _parse_runtime_context(document: Mapping[str, Any]) -> dict[str, Any]:
    sim = document.get("sim")
    if not isinstance(sim, Mapping):
        raise TerminationContractError("params/env.yaml must contain a sim object")
    sim_dt = _finite_positive(sim.get("dt"), "sim.dt")
    decimation = document.get("decimation")
    if isinstance(decimation, bool) or not isinstance(decimation, int) or decimation <= 0:
        raise TerminationContractError(
            f"decimation must be a positive YAML integer, got {decimation!r}"
        )
    episode_length_s = _finite_positive(
        document.get("episode_length_s"), "episode_length_s"
    )
    control_dt = sim_dt * decimation
    max_episode_steps = int(math.ceil(episode_length_s / control_dt - 1e-12))
    if max_episode_steps <= 0:
        raise TerminationContractError("computed max_episode_steps must be positive")

    commands = document.get("commands")
    motion = commands.get("motion") if isinstance(commands, Mapping) else None
    if not isinstance(motion, Mapping):
        raise TerminationContractError("params/env.yaml must contain commands.motion")
    anchor_body_name = motion.get("anchor_body_name")
    if anchor_body_name != "torso_Link":
        raise TerminationContractError(
            "commands.motion.anchor_body_name must be 'torso_Link' for the current MuJoCo "
            f"evaluator, got {anchor_body_name!r}"
        )
    return {
        "sim_dt": sim_dt,
        "decimation": decimation,
        "control_dt": control_dt,
        "episode_length_s": episode_length_s,
        "max_episode_steps": max_episode_steps,
        "motion_command_name": "motion",
        "anchor_body_name": anchor_body_name,
    }


def build_termination_contract(env_yaml: str | Path) -> dict[str, Any]:
    """Parse and hash one completed run's frozen environment config."""

    path = Path(env_yaml).expanduser().resolve()
    if not path.is_file():
        raise TerminationContractError(f"params/env.yaml is not a regular file: {path}")
    raw = path.read_bytes()
    digest, size = _hash_file(path)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        raise TerminationContractError(
            f"params/env.yaml changed between reading and hashing: {path}"
        )
    try:
        document = yaml.load(raw.decode("utf-8"), Loader=_FrozenEnvLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise TerminationContractError(f"cannot safely parse {path}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise TerminationContractError("params/env.yaml root must be an object")
    payload: dict[str, Any] = {
        "schema": CONTRACT_SCHEMA,
        "source_env": {"path": str(path), "sha256": digest, "size_bytes": size},
        "runtime": _parse_runtime_context(document),
        "terms": _parse_terms(document),
    }
    return {**payload, "contract_id": _content_id(payload)}


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise TerminationContractError(
            f"{label} fields must be exactly {sorted(expected)}, got {sorted(actual)}"
        )


def _normalized_positive_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise TerminationContractError(
            f"{label} must be a normalized finite positive JSON float, got {value!r}"
        )
    return value


def _validate_normalized_terms(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise TerminationContractError("termination contract terms must be an object")
    if set(value) != set(KNOWN_TERMS):
        raise TerminationContractError(
            f"termination contract terms must be exactly {list(KNOWN_TERMS)}, got {sorted(value)}"
        )
    expected_functions = {
        "time_out": "time_out",
        "anchor_pos": "bad_anchor_pos_z_only",
        "anchor_ori": "bad_anchor_ori",
        "ee_body_pos": "bad_motion_body_pos_z_only",
        "base_fell_tilt": "bad_orientation",
        "base_too_low": "root_height_below_minimum",
    }
    value_fields = {
        "anchor_pos": "threshold",
        "anchor_ori": "threshold",
        "ee_body_pos": "threshold",
        "base_fell_tilt": "limit_angle",
        "base_too_low": "minimum_height",
    }
    for name in KNOWN_TERMS:
        term = value[name]
        if not isinstance(term, Mapping) or type(term.get("active")) is not bool:
            raise TerminationContractError(f"termination contract term {name} is malformed")
        if not term["active"]:
            if name == "time_out":
                raise TerminationContractError("termination contract time_out must be active")
            _require_exact_keys(term, {"active"}, f"terms.{name}")
            continue
        if name == "time_out":
            _require_exact_keys(
                term, {"active", "function", "time_out"}, "terms.time_out"
            )
            if term.get("function") != "time_out" or term.get("time_out") is not True:
                raise TerminationContractError(
                    "terms.time_out must have function='time_out' and time_out=true"
                )
            continue
        fields = {"active", "function", value_fields[name]}
        if name == "ee_body_pos":
            fields.add("body_names")
        _require_exact_keys(term, fields, f"terms.{name}")
        if term.get("function") != expected_functions[name]:
            raise TerminationContractError(
                f"terms.{name}.function must be {expected_functions[name]!r}"
            )
        _normalized_positive_float(term[value_fields[name]], f"terms.{name}.{value_fields[name]}")
        if name == "ee_body_pos":
            names = term["body_names"]
            if not isinstance(names, list) or not names \
                    or any(type(item) is not str or not item for item in names) \
                    or len(set(names)) != len(names):
                raise TerminationContractError(
                    "terms.ee_body_pos.body_names must be a non-empty unique string list"
                )


def _validate_normalized_runtime(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise TerminationContractError("termination contract runtime must be an object")
    fields = {
        "sim_dt", "decimation", "control_dt", "episode_length_s",
        "max_episode_steps", "motion_command_name", "anchor_body_name",
    }
    _require_exact_keys(value, fields, "runtime")
    sim_dt = _normalized_positive_float(value["sim_dt"], "runtime.sim_dt")
    control_dt = _normalized_positive_float(value["control_dt"], "runtime.control_dt")
    episode_length_s = _normalized_positive_float(
        value["episode_length_s"], "runtime.episode_length_s"
    )
    decimation = value["decimation"]
    max_steps = value["max_episode_steps"]
    if type(decimation) is not int or decimation <= 0:
        raise TerminationContractError("runtime.decimation must be a positive JSON integer")
    if type(max_steps) is not int or max_steps <= 0:
        raise TerminationContractError("runtime.max_episode_steps must be a positive JSON integer")
    expected_dt = sim_dt * decimation
    if not math.isclose(control_dt, expected_dt, rel_tol=0.0, abs_tol=1e-15):
        raise TerminationContractError(
            f"runtime.control_dt {control_dt} != sim_dt*decimation {expected_dt}"
        )
    expected_steps = int(math.ceil(episode_length_s / control_dt - 1e-12))
    if max_steps != expected_steps:
        raise TerminationContractError(
            f"runtime.max_episode_steps {max_steps} != computed {expected_steps}"
        )
    if value["motion_command_name"] != "motion":
        raise TerminationContractError("runtime.motion_command_name must be 'motion'")
    if value["anchor_body_name"] != "torso_Link":
        raise TerminationContractError("runtime.anchor_body_name must be 'torso_Link'")


def verify_termination_contract(
    contract: Mapping[str, Any], *, env_yaml: str | Path | None = None
) -> None:
    """Validate canonical shape/id and optionally re-derive it from the source YAML."""

    if not isinstance(contract, Mapping) or contract.get("schema") != CONTRACT_SCHEMA:
        raise TerminationContractError(
            f"unsupported termination contract schema: {getattr(contract, 'get', lambda *_: None)('schema')!r}"
        )
    _require_exact_keys(
        contract, {"schema", "source_env", "runtime", "terms", "contract_id"},
        "termination contract",
    )
    supplied_id = contract.get("contract_id")
    if not isinstance(supplied_id, str):
        raise TerminationContractError("termination contract_id is missing")
    payload = dict(contract)
    del payload["contract_id"]
    expected_id = _content_id(payload)
    if supplied_id != expected_id:
        raise TerminationContractError(
            f"termination contract_id mismatch: expected {expected_id}, got {supplied_id}"
        )
    source = contract.get("source_env")
    if not isinstance(source, Mapping):
        raise TerminationContractError("termination contract source_env must be an object")
    _require_exact_keys(source, {"path", "sha256", "size_bytes"}, "source_env")
    source_path_text = source.get("path")
    source_hash = source.get("sha256")
    source_size = source.get("size_bytes")
    if type(source_path_text) is not str or not Path(source_path_text).is_absolute():
        raise TerminationContractError("source_env.path must be a non-empty absolute path")
    if type(source_hash) is not str or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None:
        raise TerminationContractError("source_env.sha256 must be 64 lowercase hexadecimal digits")
    if type(source_size) is not int or source_size <= 0:
        raise TerminationContractError("source_env.size_bytes must be a positive integer")
    _validate_normalized_runtime(contract.get("runtime"))
    _validate_normalized_terms(contract.get("terms"))
    source_path = Path(str(source.get("path", ""))).expanduser().resolve()
    requested = source_path if env_yaml is None else Path(env_yaml).expanduser().resolve()
    if env_yaml is not None and requested != source_path:
        raise TerminationContractError(
            f"termination contract source {source_path} != runtime env config {requested}"
        )
    if env_yaml is not None:
        rebuilt = build_termination_contract(requested)
        if rebuilt != dict(contract):
            raise TerminationContractError(
                "termination contract no longer matches the frozen params/env.yaml"
            )


def _runtime_function_identity(value: Any, label: str) -> str:
    if isinstance(value, str):
        return value.strip()
    module = getattr(value, "__module__", None)
    name = getattr(value, "__name__", None)
    if not isinstance(module, str) or not isinstance(name, str) or not module or not name:
        raise TerminationContractError(
            f"{label}.func must be a named function or full identity string, got {value!r}"
        )
    return f"{module}:{name}"


def _runtime_env_cfg_document(env_cfg: Any) -> dict[str, Any]:
    """Project the trusted env.pkl object onto the exact fields frozen from env.yaml."""

    try:
        sim_dt = env_cfg.sim.dt
        decimation = env_cfg.decimation
        episode_length_s = env_cfg.episode_length_s
        anchor_body_name = env_cfg.commands.motion.anchor_body_name
        terminations = env_cfg.terminations
    except AttributeError as exc:
        raise TerminationContractError(
            f"restored params/env.pkl is missing a required runtime field: {exc}"
        ) from exc

    raw_terms: dict[str, Any] = {}
    for name in KNOWN_TERMS:
        raw = getattr(terminations, name, None)
        if raw is None:
            raw_terms[name] = None
            continue
        try:
            func = _runtime_function_identity(raw.func, f"runtime terminations.{name}")
            params_raw = raw.params
            is_timeout = raw.time_out
        except AttributeError as exc:
            raise TerminationContractError(
                f"runtime terminations.{name} is missing func/params/time_out: {exc}"
            ) from exc
        if not isinstance(params_raw, Mapping):
            raise TerminationContractError(
                f"runtime terminations.{name}.params must be a mapping"
            )
        params = dict(params_raw)
        if "asset_cfg" in params:
            asset_cfg = params["asset_cfg"]
            if isinstance(asset_cfg, Mapping):
                params["asset_cfg"] = dict(asset_cfg)
            else:
                params["asset_cfg"] = {"name": getattr(asset_cfg, "name", None)}
        if "body_names" in params and isinstance(params["body_names"], tuple):
            params["body_names"] = list(params["body_names"])
        raw_terms[name] = {
            "func": func,
            "params": params,
            "time_out": is_timeout,
        }
    return {
        "sim": {"dt": sim_dt},
        "decimation": decimation,
        "episode_length_s": episode_length_s,
        "commands": {"motion": {"anchor_body_name": anchor_body_name}},
        "terminations": raw_terms,
    }


def verify_runtime_env_cfg(contract: Mapping[str, Any], env_cfg: Any) -> None:
    """Fail if the lossless env.pkl runtime differs from the human-readable env.yaml contract."""

    verify_termination_contract(contract)
    document = _runtime_env_cfg_document(env_cfg)
    runtime = _parse_runtime_context(document)
    terms = _parse_terms(document)
    if runtime != dict(contract["runtime"]):
        raise TerminationContractError(
            f"params/env.pkl runtime {runtime} does not match params/env.yaml {contract['runtime']}"
        )
    if terms != dict(contract["terms"]):
        raise TerminationContractError(
            f"params/env.pkl terminations {terms} do not match params/env.yaml {contract['terms']}"
        )


def runtime_termination_settings(
    contract: Mapping[str, Any], tracked_body_names: Sequence[str]
) -> dict[str, Any]:
    """Compile a verified contract into the small structure used on every MuJoCo step."""

    verify_termination_contract(contract)
    tracked = [str(name) for name in tracked_body_names]
    if len(set(tracked)) != len(tracked):
        raise TerminationContractError("tracked body names must be unique")
    terms = contract["terms"]
    ee = terms["ee_body_pos"]
    if ee["active"]:
        missing = [name for name in ee["body_names"] if name not in tracked]
        if missing:
            raise TerminationContractError(
                f"active ee_body_pos names are absent from MuJoCo tracked bodies: {missing}"
            )
        ee_indices = [tracked.index(name) for name in ee["body_names"]]
    else:
        ee_indices = []
    runtime = contract["runtime"]
    return {
        "contract_id": contract["contract_id"],
        "sim_dt": float(runtime["sim_dt"]),
        "decimation": int(runtime["decimation"]),
        "control_dt": float(runtime["control_dt"]),
        "episode_length_s": float(runtime["episode_length_s"]),
        "max_episode_steps": int(runtime["max_episode_steps"]),
        "motion_command_name": str(runtime["motion_command_name"]),
        "anchor_body_name": str(runtime["anchor_body_name"]),
        "time_out": bool(terms["time_out"]["active"]),
        "anchor_pos_z": (
            float(terms["anchor_pos"]["threshold"])
            if terms["anchor_pos"]["active"] else None
        ),
        "anchor_ori": (
            float(terms["anchor_ori"]["threshold"])
            if terms["anchor_ori"]["active"] else None
        ),
        "ee_body_pos_z": (
            {
                "threshold": float(ee["threshold"]),
                "body_names": list(ee["body_names"]),
                "indices": ee_indices,
            }
            if ee["active"] else None
        ),
        "fall_tilt_rad": (
            float(terms["base_fell_tilt"]["limit_angle"])
            if terms["base_fell_tilt"]["active"] else None
        ),
        "root_height_min": (
            float(terms["base_too_low"]["minimum_height"])
            if terms["base_too_low"]["active"] else None
        ),
        "active_terms": [name for name in KNOWN_TERMS if terms[name]["active"]],
    }


def tracking_reset_reasons(
    settings: Mapping[str, Any], *, anchor_pos_z_error: float,
    anchor_ori_error: float, ee_body_pos_z_errors: Sequence[float]
) -> list[str]:
    """Pure threshold application shared by CPU tests and the MuJoCo runtime."""

    reasons: list[str] = []
    if settings.get("anchor_pos_z") is not None \
            and float(anchor_pos_z_error) > float(settings["anchor_pos_z"]):
        reasons.append("anchor_pos")
    if settings.get("anchor_ori") is not None \
            and float(anchor_ori_error) > float(settings["anchor_ori"]):
        reasons.append("anchor_ori")
    ee = settings.get("ee_body_pos_z")
    if ee is not None and any(float(value) > float(ee["threshold"])
                              for value in ee_body_pos_z_errors):
        reasons.append("ee_body_pos")
    return reasons


def physical_reset_reasons(
    settings: Mapping[str, Any], *, tilt_rad: float, root_height: float
) -> list[str]:
    reasons: list[str] = []
    if settings.get("fall_tilt_rad") is not None \
            and float(tilt_rad) > float(settings["fall_tilt_rad"]):
        reasons.append("fall_tilt")
    if settings.get("root_height_min") is not None \
            and float(root_height) < float(settings["root_height_min"]):
        reasons.append("fall_root_z")
    return reasons
