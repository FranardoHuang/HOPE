#!/usr/bin/env python3
"""Replay the tracked EPA24-overflow/EPA48-contact fixture on one CUDA GPU.
Only the fixed pair is used; CPU MuJoCo is not an oracle and results cannot authorize training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_JSON = ROOT / "configs/fixtures/mujoco_warp_epa48_ellipsoid_cylinder_cross_v1.json"
FIXTURE_JSON_SHA256 = "5bd5fdc9d48cdfce775d360f0fe44b5de5562b8aaf6de9483b56fe5e5ee8d6e6"
FIXTURE_ID = "mujoco_warp_epa48_ellipsoid_cylinder_cross_v1"
MJCF_FILE = "mujoco_warp_epa48_ellipsoid_cylinder_cross_v1.xml"
MJCF_SHA256 = "f611bbf5189a5eb87b0b9da58261d7f6d1d31302757daea106cfbc22a4fc58ce"
SCIENTIFIC_SCOPE = {"diagnostic_unauthorized": True, "training_authorized": False,
                    "stock_cpu_mujoco_is_oracle": False}
EPA_BIT = 256
ROLE_SPECS = {
    "stock24": {
        "version": "3.10.0.3", "horizon": 24,
        "types_sha256": "712e76f495d3dedcb45acc7c248e226f56144b5fef5d9841d5e04d279fa7fd4f",
        "package_file_count": 281, "package_manifest_sha256": "6fb7b2849955d952e69d67c534b96816f43db5914805ff2732692e70011ea3c2",
    },
    "fork48": {
        "version": "3.10.0.3+hope.epa48.1", "horizon": 48,
        "types_sha256": "391e421eeede84389d6c7daeae39b19ce43132d29c11f7f3c328a50011c7a696",
        "package_file_count": 281, "package_manifest_sha256": "90ae9570e2e4e0dc45fd28315aae15b156802a5c2f92c0ba33dad3aac4385036",
    },
}


class ReplayError(RuntimeError):
    pass


def _stable_regular_bytes(path, label):
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ReplayError(label + " is not a regular non-symlink file")
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise ReplayError(label + " is unreadable") from exc
    if before != after or len(payload) != after.st_size:
        raise ReplayError(label + " changed while being read")
    return payload


def load_fixture():
    try:
        payload = _stable_regular_bytes(FIXTURE_JSON, "tracked fixture JSON")
        if hashlib.sha256(payload).hexdigest() != FIXTURE_JSON_SHA256:
            raise ReplayError("tracked fixture JSON SHA differs")
        value = json.loads(payload)
    except ValueError as exc:
        raise ReplayError("tracked fixture JSON is malformed") from exc
    expected = (("schema_version", 1), ("fixture_id", FIXTURE_ID), ("mjcf_file", MJCF_FILE), ("mjcf_sha256", MJCF_SHA256),
                ("scientific_scope", SCIENTIFIC_SCOPE))
    if not isinstance(value, dict) or any(value.get(key) != wanted for key, wanted in expected):
        raise ReplayError("tracked fixture top-level identity differs")
    pose, probe = value.get("pose"), value.get("probe")
    if (not isinstance(pose, dict) or not isinstance(pose.get("translation_m"), list)
            or len(pose["translation_m"]) != 3 or not isinstance(pose.get("quat_wxyz"), list)
            or len(pose["quat_wxyz"]) != 4 or not isinstance(probe, dict)):
        raise ReplayError("tracked fixture pose/probe schema differs")
    probe_types = (("nworld", int), ("ccd_iterations", int), ("ccd_tolerance", float), ("disable_multiccd", bool),
                   ("nconmax", int), ("nccdmax", int),
                   ("njmax", int), ("gravity_m_s2", list), ("joint", str))
    if any(type(probe.get(key)) is not kind for key, kind in probe_types):
        raise ReplayError("tracked fixture probe types differ")
    xml = FIXTURE_JSON.parent / MJCF_FILE
    if hashlib.sha256(_stable_regular_bytes(xml, "tracked fixture MJCF")).hexdigest() != MJCF_SHA256:
        raise ReplayError("tracked fixture MJCF SHA differs")
    return value, xml


def _package_manifest(package_root):
    digest = hashlib.sha256()
    count = 0
    for path in sorted(package_root.rglob("*")):
        relative = path.relative_to(package_root)
        if "__pycache__" in relative.parts or path.suffix in (".pyc", ".pyo"):
            continue
        if path.is_symlink():
            raise ReplayError("mujoco_warp package contains a symlink")
        if path.is_file():
            count += 1
            name = "mujoco_warp/" + relative.as_posix()
            digest.update(name.encode("utf-8") + b"\0" + hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii") + b"\n")
    return count, digest.hexdigest()


def _under(path, parent, label):
    try:
        path.resolve(strict=True).relative_to(parent.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ReplayError(label + " is outside the worker interpreter environment") from exc


def _worker_identity(role, expected_python, cache_dir, device_name):
    import warp as wp

    wp.config.kernel_cache_dir = str(cache_dir.resolve())
    wp.config.quiet = True
    wp.init()
    import mujoco
    import mujoco_warp as mjwarp
    from mujoco_warp._src import types as mjwarp_types

    package_root = Path(mjwarp.__file__).resolve().parent
    types_path = Path(mjwarp_types.__file__).resolve()
    prefix = Path(sys.prefix).resolve()
    dist_root = Path(importlib.metadata.distribution("mujoco-warp").locate_file("")).resolve()
    _under(package_root, prefix, "mujoco_warp package")
    _under(types_path, package_root, "mujoco_warp types module")
    _under(dist_root, prefix, "mujoco-warp distribution")
    count, manifest = _package_manifest(package_root)
    spec = ROLE_SPECS[role]
    actual = {
        "role": role,
        "python_executable": str(Path(sys.executable).absolute()),
        "python_prefix": str(prefix),
        "package_root": str(package_root),
        "distribution_version": importlib.metadata.version("mujoco-warp"),
        "module_version": mjwarp.__version__,
        "epa_horizon": int(mjwarp_types.MJ_MAX_EPAHORIZON),
        "epa_horizon_bit": int(mjwarp_types.OverflowType.EPA_HORIZON),
        "types_sha256": hashlib.sha256(types_path.read_bytes()).hexdigest(),
        "package_file_count": count,
        "package_manifest_sha256": manifest,
        "mujoco_version": getattr(mujoco, "__version__", "unknown"),
        "warp_version": getattr(wp, "__version__", "unknown"),
        "warp_cache_dir": str(Path(wp.config.kernel_cache_dir).resolve()),
    }
    expected = {
        "python_executable": str(expected_python.absolute()),
        "distribution_version": spec["version"], "module_version": spec["version"],
        "epa_horizon": spec["horizon"], "epa_horizon_bit": EPA_BIT,
        "types_sha256": spec["types_sha256"], "package_file_count": spec["package_file_count"],
        "package_manifest_sha256": spec["package_manifest_sha256"],
    }
    for key, wanted in expected.items():
        if actual[key] != wanted:
            raise ReplayError("%s identity mismatch for %s" % (role, key))
    _under(Path(actual["warp_cache_dir"]), cache_dir, "Warp cache")
    device = wp.get_device(device_name)
    wp.set_device(device)
    if not device.is_cuda or device.is_cpu:
        raise ReplayError("worker device is not CUDA")
    actual["device"] = {
        "requested": device_name, "alias": str(device.alias), "ordinal": int(device.ordinal),
        "name": str(device.name), "uuid": str(device.uuid), "pci_bus_id": str(device.pci_bus_id),
        "is_cuda": bool(device.is_cuda), "is_cpu": bool(device.is_cpu),
    }
    return actual, wp, mujoco, mjwarp


def _worker(role, expected_python, output, cache_dir, device_name, repeats):
    fixture, xml = load_fixture()
    identity, wp, mujoco, mjwarp = _worker_identity(role, expected_python, cache_dir, device_name)
    model = mujoco.MjModel.from_xml_path(str(xml))
    if (model.nq, model.nv, model.ngeom) != (7, 6, 2):
        raise ReplayError("fixture must compile to one free joint and two geoms")
    probe = fixture["probe"]
    model.opt.ccd_iterations = probe["ccd_iterations"]
    model.opt.ccd_tolerance = probe["ccd_tolerance"]
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD
    host_data = mujoco.MjData(model)
    warp_model = mjwarp.put_model(model)
    pose = fixture["pose"]["translation_m"] + fixture["pose"]["quat_wxyz"]
    observations = []
    for repeat_index in range(repeats):
        data = mjwarp.put_data(
            model, host_data, nworld=1, nconmax=probe["nconmax"],
            nccdmax=probe["nccdmax"], njmax=probe["njmax"], nvmax=model.nv,
        )
        data.qpos.assign([pose])
        data.qvel.zero_()
        data.overflow.zero_()
        mjwarp.kinematics(warp_model, data)
        mjwarp.collision(warp_model, data)
        wp.synchronize_device(data.qpos.device)
        mask = int(data.overflow.numpy().reshape(-1)[0])
        nacon = int(data.nacon.numpy().reshape(-1)[0])
        if nacon < 0 or nacon > data.naconmax:
            raise ReplayError("worker produced an invalid active contact count")
        worldids = data.contact.worldid.numpy()[:nacon].reshape(-1).tolist()
        indices = [index for index, worldid in enumerate(worldids) if int(worldid) == 0]
        distances = [float(data.contact.dist.numpy()[index]) for index in indices]
        positions = [[float(x) for x in data.contact.pos.numpy()[index].reshape(-1)] for index in indices]
        frames = [[float(x) for x in data.contact.frame.numpy()[index].reshape(-1)] for index in indices]
        values = distances + [x for row in positions + frames for x in row]
        observations.append({
            "repeat_index": repeat_index, "overflow_mask": mask, "contact_count": len(indices),
            "contact_distances": distances, "contact_positions": positions, "contact_frames": frames,
            "active_contact_finite": bool(values) and all(math.isfinite(x) for x in values),
        })
    result = {
        "schema_version": 1, "fixture_id": fixture["fixture_id"], "nworld": 1,
        "repeats": repeats, "runtime": identity, "observations": observations,
        "scientific_scope": fixture["scientific_scope"],
    }
    _write_json_x(output, result)


def _write_json_x(path, value):
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise ReplayError("refusing to overwrite " + str(path)) from exc


def _observation_errors(result, role, repeats):
    errors = []
    if result.get("schema_version") != 1 or result.get("fixture_id") != FIXTURE_ID:
        raise ReplayError(role + " worker result fixture/schema mismatch")
    if result.get("nworld") != 1 or result.get("repeats") != repeats:
        raise ReplayError(role + " worker result world/repeat mismatch")
    if result.get("scientific_scope") != SCIENTIFIC_SCOPE:
        raise ReplayError(role + " worker scientific scope mismatch")
    runtime = result.get("runtime")
    spec = ROLE_SPECS[role]
    for key, wanted in {
        "role": role, "distribution_version": spec["version"], "module_version": spec["version"],
        "epa_horizon": spec["horizon"], "epa_horizon_bit": EPA_BIT,
        "types_sha256": spec["types_sha256"], "package_file_count": spec["package_file_count"],
        "package_manifest_sha256": spec["package_manifest_sha256"],
    }.items():
        if not isinstance(runtime, dict) or runtime.get(key) != wanted:
            raise ReplayError(role + " worker runtime mismatch for " + key)
    observations = result.get("observations")
    if not isinstance(observations, list) or len(observations) != repeats:
        raise ReplayError(role + " worker observation count mismatch")
    counts, masks = [], []
    for index, item in enumerate(observations):
        if not isinstance(item, dict) or item.get("repeat_index") != index:
            raise ReplayError(role + " worker repeat order mismatch")
        count, mask = item.get("contact_count"), item.get("overflow_mask")
        distances, positions, frames = item.get("contact_distances"), item.get("contact_positions"), item.get("contact_frames")
        if type(count) is not int or count < 0 or type(mask) is not int or mask < 0:
            raise ReplayError(role + " worker count/mask is malformed")
        if not isinstance(distances, list) or not isinstance(positions, list) or not isinstance(frames, list):
            raise ReplayError(role + " worker contact vectors are malformed")
        if len(distances) != count or len(positions) != count or len(frames) != count:
            raise ReplayError(role + " worker contact vector length mismatch")
        values = [float(x) for x in distances] + [float(x) for row in positions + frames for x in row]
        finite = bool(values) and all(math.isfinite(x) for x in values)
        if any(len(row) != size for row, size in [(row, 3) for row in positions] + [(row, 9) for row in frames]):
            raise ReplayError(role + " worker contact vector width mismatch")
        if item.get("active_contact_finite") is not finite:
            raise ReplayError(role + " worker finite flag disagrees with raw contacts")
        counts.append(count)
        masks.append(mask)
    if role == "stock24" and masks != [EPA_BIT] * repeats:
        errors.append("stock24 did not report exact overflow mask 256 on every repeat")
    if role == "fork48":
        if masks != [0] * repeats:
            errors.append("fork48 did not report exact overflow mask 0 on every repeat")
        if any(count <= 0 for count in counts):
            errors.append("fork48 did not produce an active contact on every repeat")
        if any(not item["active_contact_finite"] for item in observations):
            errors.append("fork48 contact dist/pos/frame was nonfinite")
        if len(set(counts)) != 1:
            errors.append("fork48 contact count changed across repeats")
    return errors, counts, masks


def classify_results(stock, fork, repeats, expected_gpu_uuid):
    errors, stock_counts, stock_masks = _observation_errors(stock, "stock24", repeats)
    more, fork_counts, fork_masks = _observation_errors(fork, "fork48", repeats)
    errors.extend(more)
    sr, fr = stock["runtime"], fork["runtime"]
    stock_device, fork_device = sr.get("device"), fr.get("device")
    if not isinstance(stock_device, dict) or not isinstance(fork_device, dict):
        raise ReplayError("worker CUDA device record is missing")
    if not all(isinstance(device.get(key), str) and device[key] for device in (stock_device, fork_device)
               for key in ("uuid", "pci_bus_id")):
        errors.append("worker physical GPU UUID/PCI record is malformed")
    if stock_device.get("uuid") != expected_gpu_uuid or fork_device.get("uuid") != expected_gpu_uuid:
        errors.append("worker GPU UUID differs from --expected-gpu-uuid")
    if stock_device.get("uuid") != fork_device.get("uuid") or stock_device.get("pci_bus_id") != fork_device.get("pci_bus_id"):
        errors.append("workers did not use the same physical GPU UUID/PCI bus")
    if sr.get("mujoco_version") != fr.get("mujoco_version") or sr.get("warp_version") != fr.get("warp_version"):
        errors.append("workers used different mujoco or warp versions")
    if sr.get("python_prefix") == fr.get("python_prefix"):
        errors.append("stock24 and fork48 did not use independent Python environments")
    if sr.get("warp_cache_dir") == fr.get("warp_cache_dir"):
        errors.append("stock24 and fork48 did not use independent Warp caches")
    return {
        "schema_version": 1,
        "verdict": "PASS_EPA48_FIXED_FIXTURE_REPLAY" if not errors else "FAIL_EPA48_FIXED_FIXTURE_REPLAY",
        "fixture_id": FIXTURE_ID, "repeats": repeats,
        "physical_gpu": {"uuid": stock_device.get("uuid"), "pci_bus_id": stock_device.get("pci_bus_id")},
        "mujoco_version": sr.get("mujoco_version"), "warp_version": sr.get("warp_version"),
        "stock24": {"overflow_masks": stock_masks, "contact_counts": stock_counts},
        "fork48": {"overflow_masks": fork_masks, "contact_counts": fork_counts},
        "raw_results": {"stock24": "stock24_raw_result.json", "fork48": "fork48_raw_result.json"},
        "errors": errors,
        "scientific_scope": SCIENTIFIC_SCOPE,
    }


def _python(path, label):
    value = path.absolute()
    if not value.is_file() or not os.access(str(value), os.X_OK):
        raise ReplayError(label + " must be an executable Python file")
    return value


def _create_output_root(path):
    try:
        path.mkdir()
    except FileExistsError as exc:
        raise ReplayError("output root already exists; replay is no-clobber") from exc
    except OSError as exc:
        raise ReplayError("cannot create fresh output root") from exc


def _invoke(python, role, output, cache, device, repeats, timeout):
    command = [
        str(python), "-I", str(Path(__file__).resolve()), "_worker", "--role", role,
        "--expected-python", str(python), "--output", str(output), "--cache-dir", str(cache),
        "--device", device, "--repeats", str(repeats),
    ]
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"):
        env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(command, env=env, text=True, capture_output=True, timeout=timeout, check=False)
    if completed.returncode != 0 or not output.is_file():
        detail = (completed.stdout + "\n" + completed.stderr).strip()[-4000:]
        raise ReplayError("%s worker failed rc=%d: %s" % (role, completed.returncode, detail))
    return json.loads(output.read_text(encoding="utf-8"))


def _replay(args):
    load_fixture()
    if args.repeats < 3:
        raise ReplayError("--repeats must be at least 3")
    if args.timeout_seconds < 1:
        raise ReplayError("--timeout-seconds must be positive")
    if not args.expected_gpu_uuid.startswith("GPU-"):
        raise ReplayError("--expected-gpu-uuid must be one NVIDIA GPU UUID")
    stock_python = _python(args.stock_python, "stock24 Python")
    fork_python = _python(args.fork_python, "fork48 Python")
    if stock_python == fork_python:
        raise ReplayError("stock24 and fork48 Python entries must differ")
    _create_output_root(args.output_root)
    stock_cache, fork_cache = args.output_root / "warp_cache_stock24", args.output_root / "warp_cache_fork48"
    stock_cache.mkdir()
    fork_cache.mkdir()
    stock = _invoke(stock_python, "stock24", args.output_root / "stock24_raw_result.json", stock_cache,
                    args.device, args.repeats, args.timeout_seconds)
    fork = _invoke(fork_python, "fork48", args.output_root / "fork48_raw_result.json", fork_cache,
                   args.device, args.repeats, args.timeout_seconds)
    summary = classify_results(stock, fork, args.repeats, args.expected_gpu_uuid)
    _write_json_x(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["verdict"].startswith("PASS_") else 3


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    replay = commands.add_parser("replay", help="replay the tracked fixture in stock24 and fork48 environments")
    for name in ("stock-python", "fork-python", "output-root"):
        replay.add_argument("--" + name, type=Path, required=True)
    replay.add_argument("--expected-gpu-uuid", required=True)
    replay.add_argument("--device", default="cuda:0")
    replay.add_argument("--repeats", type=int, default=10)
    replay.add_argument("--timeout-seconds", type=int, default=900)
    return parser


def _private_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--role", choices=sorted(ROLE_SPECS), required=True)
    for name in ("expected-python", "output", "cache-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--repeats", type=int, required=True)
    return parser


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    try:
        if raw[:1] == ["_worker"]:
            args = _private_parser().parse_args(raw[1:])
            _worker(args.role, args.expected_python, args.output, args.cache_dir, args.device, args.repeats)
            return 0
        args = _parser().parse_args(raw)
        return _replay(args)
    except (ReplayError, subprocess.TimeoutExpired, ValueError, OSError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
