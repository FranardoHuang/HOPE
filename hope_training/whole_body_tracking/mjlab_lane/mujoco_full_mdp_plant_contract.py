"""One path-free plant contract for FullMDP launch, run, and evidence.

``source_plant`` pins the vendor MJCF closure. ``runtime_attach`` pins the
actual MJLab-augmented MuJoCo model plus the few runtime values that are not
stored in an MJB (policy decimation and MuJoCo-Warp allocation capacity).
Paths are locators only and never durable identity.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile


EXPECTED_MANIFEST_RELATIVE = Path("configs/a3_mujoco_identity_v2_20260803.json")
EXPECTED_GEOMETRY_RELATIVE = Path(
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/table_tennis/geometry.py"
)
TRUSTED_EXPECTED_MANIFEST_SHA256 = (
    "b8fc5deaaff8d213c2d077a0e7892b30d7f5a6c77c3d06dc029e3a2616d54d91"
)
TRUSTED_GEOMETRY_SOURCE_SHA256 = (
    "df71f12a21fedb4b8caed182906288f573b2f05c731441a5f529996baaf056b2"
)
RUNTIME_ATTACH_CONTRACT_TYPE = "action_ball_mjlab_runtime_attach_v2"
RUNTIME_MJB_RELATIVE_LOCATOR = "runtime.mjb"

# Pinned diagnostic candidate from exact-Pod CPU constructions.  The artifact
# chain below proves that a run used these bytes; it does not turn the still
# pending independent N=1/N=2 and semantic-mutation receipts into formal plant
# authority.  Every downstream record remains ``diagnostic_unauthorized``.
EXPECTED_FINAL_AUGMENTED_MJB_SHA256 = (
    "1ef4bb9e52b0b46afd422d2fe712ae38628853a1704b324b20a8ec3f26030c0b"
)
EXPECTED_FINAL_AUGMENTED_MJB_SIZE_BYTES = 72_260_546

POLICY_CLOCK = {
    "decimation": 20,
    "step_dt": 0.02,
}
WARP_CAPACITY = {
    "njmax_per_world": 572,
    "nconmax_per_world": 128,
}


class PlantContractError(ValueError):
    pass


def _digest(value) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _same_exact(value, expected) -> bool:
    if type(value) is not type(expected):
        return False
    if type(value) is dict:
        return set(value) == set(expected) and all(
            _same_exact(value[key], expected[key]) for key in expected
        )
    if type(value) is list:
        return len(value) == len(expected) and all(
            _same_exact(left, right) for left, right in zip(value, expected)
        )
    return value == expected


def _stable_regular_bytes(path: Path, *, limit: int, label: str) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        resolved = absolute.resolve(strict=True)
        row = absolute.lstat()
        descriptor = os.open(
            absolute,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise PlantContractError(f"cannot open {label}: {exc}") from exc
    try:
        before, chunks, size = os.fstat(descriptor), [], 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, limit + 1))
            if not block:
                break
            size += len(block)
            if size > limit:
                raise PlantContractError(f"{label} is too large")
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = os.stat(absolute, follow_symlinks=False)
    except OSError as exc:
        raise PlantContractError(f"cannot restat {label}") from exc
    state = lambda item: (
        item.st_dev, item.st_ino, item.st_mode, item.st_size,
        item.st_mtime_ns, item.st_ctime_ns,
    )
    if (
        absolute != resolved
        or stat.S_ISLNK(row.st_mode)
        or not stat.S_ISREG(row.st_mode)
        or state(before) != state(after)
        or state(after) != state(current)
        or size != after.st_size
    ):
        raise PlantContractError(f"{label} changed during verification")
    return b"".join(chunks)


def expected_manifest_path() -> Path:
    return Path(__file__).resolve().parents[3] / EXPECTED_MANIFEST_RELATIVE


def expected_geometry_source_path() -> Path:
    return Path(__file__).resolve().parents[3] / EXPECTED_GEOMETRY_RELATIVE


def verify_geometry_source(path: Path) -> str:
    """Bind the one checkout geometry module allowed to build FullMDP."""

    selected = Path(path)
    expected = expected_geometry_source_path()
    if selected != expected:
        raise PlantContractError("MuJoCo FullMDP geometry source path differs")
    payload = _stable_regular_bytes(
        selected, limit=1024 * 1024, label="MuJoCo FullMDP geometry source"
    )
    digest = hashlib.sha256(payload).hexdigest()
    if digest != TRUSTED_GEOMETRY_SOURCE_SHA256:
        raise PlantContractError("MuJoCo FullMDP geometry source bytes differ")
    return digest


def load_pinned_manifest(path: Path | None = None) -> dict:
    """Project fields from pinned bytes; the canonical verifier owns schema."""

    selected = expected_manifest_path() if path is None else Path(path)
    payload = _stable_regular_bytes(
        selected, limit=64 * 1024, label="MuJoCo identity manifest"
    )
    if hashlib.sha256(payload).hexdigest() != TRUSTED_EXPECTED_MANIFEST_SHA256:
        raise PlantContractError("MuJoCo identity manifest bytes differ")
    try:
        manifest = json.loads(payload.decode("utf-8"))
        expected = manifest["expected"]
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlantContractError("MuJoCo identity manifest projection differs") from exc
    digest_keys = (
        "portable_identity_sha256", "root_mjcf_sha256",
        "source_closure_sha256", "compiled_mjb_sha256",
        "compiler_toolchain_sha256", "model_contract_sha256",
    )
    integer_keys = (
        "source_member_count", "source_total_bytes", "compiled_mjb_size_bytes",
    )
    if (
        type(manifest) is not dict
        or type(expected) is not dict
        or manifest.get("identity_type") != "exact_mujoco_identity_v1"
        or manifest.get("root_filename") != "a3_pingpong.xml"
        or any(not _digest(expected.get(key)) for key in digest_keys)
        or any(
            type(expected.get(key)) is not int or expected[key] <= 0
            for key in integer_keys
        )
    ):
        raise PlantContractError("MuJoCo identity manifest projection differs")
    return manifest


def expected_plant_model_identity() -> dict:
    """Return expected path-free layers, without the two live receipts."""

    manifest = load_pinned_manifest()
    expected = manifest["expected"]
    return {
        "source_plant": {
            "model_scope": "pre_registered_vendor_base",
            "identity_type": manifest["identity_type"],
            "manifest_sha256": TRUSTED_EXPECTED_MANIFEST_SHA256,
            "portable_identity_sha256": expected["portable_identity_sha256"],
            "root_filename": manifest["root_filename"],
            **{key: expected[key] for key in (
                "root_mjcf_sha256", "source_closure_sha256",
                "source_member_count", "source_total_bytes",
                "compiled_mjb_sha256", "compiled_mjb_size_bytes",
                "compiler_toolchain_sha256", "model_contract_sha256",
            )},
        },
        "runtime_attach": {
            "model_scope": "mjlab_augmented_court_ball_runtime",
            "contract_type": RUNTIME_ATTACH_CONTRACT_TYPE,
            "geometry_source_sha256": TRUSTED_GEOMETRY_SOURCE_SHA256,
            "policy_clock": dict(POLICY_CLOCK),
            "warp_capacity": dict(WARP_CAPACITY),
            "final_augmented_mjb": {
                "relative_locator": RUNTIME_MJB_RELATIVE_LOCATOR,
                "sha256": EXPECTED_FINAL_AUGMENTED_MJB_SHA256,
                "size_bytes": EXPECTED_FINAL_AUGMENTED_MJB_SIZE_BYTES,
            },
        },
    }


def runtime_attach_is_exact(value) -> bool:
    return _same_exact(value, expected_plant_model_identity()["runtime_attach"])


def _save_augmented_mjb(mujoco_module, model, path: Path) -> None:
    saved = False
    for arguments in ((model, str(path), None, 0), (model, str(path))):
        try:
            mujoco_module.mj_saveModel(*arguments)
            saved = True
            break
        except TypeError:
            continue
    if not saved:
        raise PlantContractError("MuJoCo exposes no supported mj_saveModel API")


def _stable_regular_inventory(path: Path, *, limit: int, label: str) -> dict:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        resolved = absolute.resolve(strict=True)
        row = absolute.lstat()
        descriptor = os.open(
            absolute,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise PlantContractError(f"cannot open {label}: {exc}") from exc
    try:
        before, digest, size = os.fstat(descriptor), hashlib.sha256(), 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            size += len(block)
            if size > limit:
                raise PlantContractError(f"{label} is too large")
            digest.update(block)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = os.stat(absolute, follow_symlinks=False)
    except OSError as exc:
        raise PlantContractError(f"cannot restat {label}") from exc
    state = lambda item: (
        item.st_dev, item.st_ino, item.st_mode, item.st_nlink, item.st_size,
        item.st_mtime_ns, item.st_ctime_ns,
    )
    if (
        absolute != resolved
        or stat.S_ISLNK(row.st_mode)
        or not stat.S_ISREG(row.st_mode)
        or before.st_nlink != 1
        or state(before) != state(after)
        or state(after) != state(current)
        or size != after.st_size
        or size == 0
    ):
        raise PlantContractError(f"{label} changed during verification")
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def serialize_augmented_mjb_identity(mujoco_module, model) -> dict:
    """Serialize one already-built live model without compiling it again."""

    root = Path(tempfile.gettempdir()).resolve(strict=True)
    stage = Path(tempfile.mkdtemp(prefix="fullmdp-augmented-mjb-", dir=str(root)))
    path = stage / "runtime.mjb"
    try:
        _save_augmented_mjb(mujoco_module, model, path)
        inventory = _stable_regular_inventory(
            path, limit=256 * 1024 * 1024, label="augmented runtime MJB"
        )
        return {
            "relative_locator": RUNTIME_MJB_RELATIVE_LOCATOR,
            **inventory,
        }
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        try:
            stage.rmdir()
        except FileNotFoundError:
            pass


def persist_augmented_runtime_mjb(mujoco_module, model, run_root: Path) -> dict:
    """Persist the live model once under the run root without overwriting.

    MuJoCo's filename API does not expose an exclusive-create mode.  Save into
    a private staging directory, verify those exact bytes, then hard-link them
    into the fixed run-relative locator.  ``link(2)`` supplies the required
    atomic no-clobber publication; the staging link is removed afterwards.
    """

    root = Path(run_root)
    root_fd = -1
    try:
        row = root.lstat()
        if (
            not root.is_absolute()
            or root.resolve(strict=True) != root
            or not stat.S_ISDIR(row.st_mode)
            or stat.S_ISLNK(row.st_mode)
        ):
            raise PlantContractError("MuJoCo run artifact root differs")
        root_fd = os.open(
            root,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(root_fd)
        if (
            (row.st_dev, row.st_ino, row.st_mode)
            != (opened.st_dev, opened.st_ino, opened.st_mode)
        ):
            os.close(root_fd)
            raise PlantContractError("MuJoCo run artifact root differs")
    except OSError as exc:
        if root_fd >= 0:
            os.close(root_fd)
        raise PlantContractError("MuJoCo run artifact root differs") from exc

    stage = None
    staged_path = None
    published = False
    try:
        stage = Path(tempfile.mkdtemp(prefix=".runtime-mjb-stage-", dir=str(root)))
        staged_path = stage / RUNTIME_MJB_RELATIVE_LOCATOR
        _save_augmented_mjb(mujoco_module, model, staged_path)
        inventory = _stable_regular_inventory(
            staged_path,
            limit=256 * 1024 * 1024,
            label="augmented runtime MJB",
        )
        observed = {
            "relative_locator": RUNTIME_MJB_RELATIVE_LOCATOR,
            **inventory,
        }
        expected = expected_plant_model_identity()["runtime_attach"][
            "final_augmented_mjb"
        ]
        if not _same_exact(observed, expected):
            raise PlantContractError(
                "MuJoCo augmented runtime MJB receipt differs"
            )
        try:
            os.link(
                staged_path,
                RUNTIME_MJB_RELATIVE_LOCATOR,
                dst_dir_fd=root_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise PlantContractError(
                "augmented runtime MJB locator already exists"
            ) from exc
        published = True
        os.fsync(root_fd)
        staged_path.unlink()
        stage.rmdir()
        os.fsync(root_fd)
        staged_path = stage = None
        current = os.stat(
            RUNTIME_MJB_RELATIVE_LOCATOR,
            dir_fd=root_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or current.st_size != inventory["size_bytes"]
        ):
            raise PlantContractError("published augmented runtime MJB differs")
        return observed
    except Exception:
        if published:
            try:
                os.unlink(RUNTIME_MJB_RELATIVE_LOCATOR, dir_fd=root_fd)
                os.fsync(root_fd)
            except OSError:
                pass
        raise
    finally:
        if staged_path is not None:
            try:
                staged_path.unlink()
            except FileNotFoundError:
                pass
        if stage is not None:
            try:
                stage.rmdir()
            except FileNotFoundError:
                pass
        os.close(root_fd)


def verified_plant_model_identity(
    *, verification_receipt_sha256: str, owner_local_frame_sha256: str,
    final_augmented_mjb: dict,
) -> dict:
    if not _digest(verification_receipt_sha256):
        raise PlantContractError("MuJoCo live plant receipt digest differs")
    if not _digest(owner_local_frame_sha256):
        raise PlantContractError("MuJoCo owner-local frame receipt differs")
    expected_mjb = expected_plant_model_identity()["runtime_attach"][
        "final_augmented_mjb"
    ]
    if not _same_exact(final_augmented_mjb, expected_mjb):
        raise PlantContractError("MuJoCo augmented runtime MJB receipt differs")
    identity = expected_plant_model_identity()
    identity["source_plant"]["verification_receipt_sha256"] = (
        verification_receipt_sha256
    )
    identity["runtime_attach"]["owner_local_frame_sha256"] = (
        owner_local_frame_sha256
    )
    identity["runtime_attach"]["final_augmented_mjb"] = dict(
        final_augmented_mjb
    )
    return identity


def plant_model_identity_is_exact(value) -> bool:
    if (
        type(value) is not dict
        or set(value) != {"source_plant", "runtime_attach"}
        or type(value["source_plant"]) is not dict
        or type(value["runtime_attach"]) is not dict
    ):
        return False
    source = dict(value["source_plant"])
    attach = json.loads(json.dumps(value["runtime_attach"]))
    verification = source.pop("verification_receipt_sha256", None)
    owner = attach.pop("owner_local_frame_sha256", None)
    return (
        _digest(verification)
        and _digest(owner)
        and _same_exact(
            {"source_plant": source, "runtime_attach": attach},
            expected_plant_model_identity(),
        )
    )


def clone_plant_model_identity(value) -> dict:
    if not plant_model_identity_is_exact(value):
        raise PlantContractError("MuJoCo plant model identity differs")
    return json.loads(json.dumps(value))


__all__ = [
    "EXPECTED_FINAL_AUGMENTED_MJB_SHA256",
    "EXPECTED_FINAL_AUGMENTED_MJB_SIZE_BYTES",
    "POLICY_CLOCK",
    "PlantContractError",
    "RUNTIME_ATTACH_CONTRACT_TYPE",
    "RUNTIME_MJB_RELATIVE_LOCATOR",
    "TRUSTED_EXPECTED_MANIFEST_SHA256",
    "TRUSTED_GEOMETRY_SOURCE_SHA256",
    "WARP_CAPACITY",
    "clone_plant_model_identity",
    "expected_geometry_source_path",
    "expected_manifest_path",
    "expected_plant_model_identity",
    "load_pinned_manifest",
    "plant_model_identity_is_exact",
    "persist_augmented_runtime_mjb",
    "runtime_attach_is_exact",
    "serialize_augmented_mjb_identity",
    "verified_plant_model_identity",
    "verify_geometry_source",
]
