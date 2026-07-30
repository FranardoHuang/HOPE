#!/usr/bin/env python3
"""Create a fail-closed clip subset of a schema-3 stage-1 question bank.

This is intentionally a *projection*, not a new question generator.  Every
selected clip keeps its original question tensors and per-clip provenance.  The
only semantic change is the ordered clip set in ``meta_json`` and its
``source_family_contract`` digest.

The output archive is deterministic: selected ``.npy`` members are copied from
the source archive, the rewritten metadata uses canonical JSON, and every ZIP
member has fixed metadata.  Existing outputs are never overwritten.

Example::

    python scripts/subset_stage1_question_bank.py \
      --bank /path/fivebind_upper4_train.npz \
      --source-sha256 db3b0ee5... \
      --clip bh_loop_c --clip bh_block --clip s0_highpress \
      --out /new/path/fivebind_upper3_backhand_train.npz \
      --manifest /new/path/fivebind_upper3_backhand_train.subset.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import io
import json
import os
import stat
import sys
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


TOOL_ID = "subset_stage1_question_bank.py:v1"
EXIT_FAIL = 2
HERE = Path(__file__).resolve().parent
QB_MODULE_PATH = (
    HERE.parent
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/stage1_question_bank.py"
)
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


class SubsetBankError(RuntimeError):
    """The requested projection is unsafe or internally inconsistent."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_file(path_like: str | os.PathLike[str], label: str) -> Path:
    try:
        path = Path(path_like).expanduser().resolve(strict=True)
        info = path.stat()
    except OSError as exc:
        raise SubsetBankError(f"{label} is not readable: {path_like}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise SubsetBankError(f"{label} must be a non-empty regular file: {path}")
    return path


def _validate_sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SubsetBankError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _decode_meta(array: np.ndarray) -> dict[str, Any]:
    raw = np.asarray(array)
    if raw.dtype != np.dtype("uint8") or raw.ndim != 1:
        raise SubsetBankError("meta_json must be a one-dimensional uint8 array")
    try:
        value = json.loads(raw.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SubsetBankError(f"meta_json is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SubsetBankError("meta_json must decode to an object")
    return value


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)
    return stream.getvalue()


def _fixed_zip_info(member_name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(member_name, date_time=_FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100444 << 16
    return info


def _deterministic_npz_bytes(
    ordered_members: Sequence[tuple[str, bytes]],
) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in ordered_members:
            archive.writestr(_fixed_zip_info(name), payload)
    return stream.getvalue()


def build_subset_meta(
    source_meta: Mapping[str, Any],
    selected_clips: Sequence[str],
    *,
    source_bank_sha256: str,
) -> dict[str, Any]:
    """Project the bank metadata onto an ordered, source-preserving clip subset."""

    selected = [str(value) for value in selected_clips]
    if not selected or len(set(selected)) != len(selected):
        raise SubsetBankError("selected clips must be a non-empty unique ordered list")
    if int(source_meta.get("schema_version", 0)) != 3:
        raise SubsetBankError(
            f"source bank schema_version={source_meta.get('schema_version')!r}; expected 3"
        )
    source_order = list(source_meta.get("clip_order") or [])
    if not source_order or len(set(source_order)) != len(source_order):
        raise SubsetBankError("source clip_order must be a non-empty unique list")
    unknown = [name for name in selected if name not in source_order]
    if unknown:
        raise SubsetBankError(
            f"selected clips are absent from source clip_order: {unknown!r}"
        )
    source_relative_order = [name for name in source_order if name in set(selected)]
    if selected != source_relative_order:
        raise SubsetBankError(
            "selected clips must preserve their source relative order; "
            f"requested={selected!r}, expected={source_relative_order!r}"
        )

    clips = source_meta.get("clips")
    family = source_meta.get("source_family_contract")
    if not isinstance(clips, Mapping) or not isinstance(family, Mapping):
        raise SubsetBankError("source bank lacks clips/source_family_contract objects")
    family_clips = family.get("clips")
    if not isinstance(family_clips, Mapping):
        raise SubsetBankError("source_family_contract lacks a clips object")
    for name in source_order:
        if name not in clips or name not in family_clips:
            raise SubsetBankError(f"source metadata is incomplete for clip {name!r}")

    projected = copy.deepcopy(dict(source_meta))
    projected["clip_order"] = selected
    projected["clips"] = {name: copy.deepcopy(clips[name]) for name in selected}

    for map_key in ("grip_applied_per_clip", "rally_yaw_applied_per_clip"):
        source_map = source_meta.get(map_key)
        if not isinstance(source_map, Mapping):
            raise SubsetBankError(f"source bank lacks {map_key}")
        projected[map_key] = {
            name: copy.deepcopy(source_map[name]) for name in selected
        }

    projected_family = copy.deepcopy(dict(family))
    projected_family["clip_order"] = selected
    projected_family["clips"] = {
        name: copy.deepcopy(family_clips[name]) for name in selected
    }
    projected["source_family_contract"] = projected_family
    projected["source_family_sha256"] = _canonical_sha256(projected_family)
    projected["clip_subset"] = {
        "tool": TOOL_ID,
        "source_bank_sha256": _validate_sha256(
            source_bank_sha256, "source_bank_sha256"
        ),
        "source_clip_order": source_order,
        "selected_clip_order": selected,
        "dropped_clips": [name for name in source_order if name not in set(selected)],
        "question_arrays_bitwise_preserved": True,
    }
    return projected


def _write_exclusive(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise SubsetBankError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(str(path), flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    finally:
        os.close(descriptor)


def _load_runtime_bank_module():
    spec = importlib.util.spec_from_file_location("subset_stage1_qb", str(QB_MODULE_PATH))
    if spec is None or spec.loader is None:
        raise SubsetBankError(f"cannot import runtime bank loader: {QB_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def subset_bank(
    bank_path: Path,
    selected_clips: Sequence[str],
    out_path: Path,
    manifest_path: Path,
    *,
    expected_source_sha256: str,
    runtime_validator: Callable[[Path, Sequence[str], str], None] | None = None,
) -> dict[str, Any]:
    """Build, re-open, and validate one immutable subset bank plus receipt."""

    source = _require_regular_file(bank_path, "source bank")
    expected_source_sha256 = _validate_sha256(
        expected_source_sha256, "expected source bank SHA-256"
    )
    actual_source_sha256 = _sha256_file(source)
    if actual_source_sha256 != expected_source_sha256:
        raise SubsetBankError(
            f"source bank SHA-256 mismatch: expected {expected_source_sha256}, "
            f"got {actual_source_sha256}"
        )
    out = Path(out_path).expanduser().absolute()
    receipt = Path(manifest_path).expanduser().absolute()
    if out == receipt:
        raise SubsetBankError("--out and --manifest must be different paths")
    for target in (out, receipt):
        if target.exists() or target.is_symlink():
            raise SubsetBankError(f"refusing to overwrite existing output: {target}")

    with np.load(source, allow_pickle=False) as loaded:
        key_order = list(loaded.files)
        if "meta_json" not in key_order:
            raise SubsetBankError("source bank lacks meta_json")
        source_meta = _decode_meta(np.array(loaded["meta_json"], copy=True))
        projected_meta = build_subset_meta(
            source_meta,
            selected_clips,
            source_bank_sha256=actual_source_sha256,
        )
        selected = list(projected_meta["clip_order"])
        allowed_prefixes = tuple(f"{name}/" for name in selected)
        selected_keys = [
            key
            for key in key_order
            if key == "meta_json" or key.startswith(allowed_prefixes)
        ]
        unexpected_global = [
            key
            for key in key_order
            if key != "meta_json" and "/" not in key
        ]
        if unexpected_global:
            raise SubsetBankError(
                f"source bank has unsupported global arrays: {unexpected_global!r}"
            )
        source_arrays = {
            key: np.array(loaded[key], copy=True)
            for key in selected_keys
            if key != "meta_json"
        }

    with zipfile.ZipFile(source, mode="r") as archive:
        member_names = archive.namelist()
        if len(member_names) != len(set(member_names)):
            raise SubsetBankError("source bank contains duplicate ZIP members")
        source_members = {name: archive.read(name) for name in member_names}

    members: list[tuple[str, bytes]] = []
    for key in selected_keys:
        member_name = f"{key}.npy"
        if key == "meta_json":
            meta_array = np.frombuffer(
                _canonical_json_bytes(projected_meta), dtype=np.uint8
            ).copy()
            members.append((member_name, _npy_bytes(meta_array)))
        else:
            if member_name not in source_members:
                raise SubsetBankError(
                    f"source archive lacks the expected member {member_name!r}"
                )
            members.append((member_name, source_members[member_name]))
    output_payload = _deterministic_npz_bytes(members)
    _write_exclusive(out, output_payload)

    try:
        with np.load(out, allow_pickle=False) as written:
            if list(written.files) != selected_keys:
                raise SubsetBankError(
                    "output NPZ key order differs from the reviewed projection"
                )
            written_meta = _decode_meta(np.array(written["meta_json"], copy=True))
            if written_meta != projected_meta:
                raise SubsetBankError("output metadata differs from the reviewed projection")
            for key, source_array in source_arrays.items():
                candidate = np.asarray(written[key])
                if (
                    candidate.dtype != source_array.dtype
                    or candidate.shape != source_array.shape
                    or candidate.tobytes(order="C") != source_array.tobytes(order="C")
                ):
                    raise SubsetBankError(
                        f"output array {key!r} is not bitwise-identical to source"
                    )

        if runtime_validator is None:
            runtime = _load_runtime_bank_module()

            def runtime_validator(
                path: Path, clips: Sequence[str], split: str
            ) -> None:
                runtime.load_question_bank(
                    str(path),
                    device="cpu",
                    clip_names=list(clips),
                    allow_legacy=False,
                    expected_split=split,
                )

        split = str(projected_meta.get("split"))
        runtime_validator(out, selected, split)
        output_sha256 = _sha256_file(out)
        content = {
            "tool": TOOL_ID,
            "tool_sha256": _sha256_file(Path(__file__).resolve()),
            "source_bank": {
                "basename": source.name,
                "bytes": int(source.stat().st_size),
                "sha256": actual_source_sha256,
                "meta_canonical_sha256": _canonical_sha256(source_meta),
                "clip_order": list(source_meta["clip_order"]),
            },
            "output_bank": {
                "basename": out.name,
                "bytes": int(out.stat().st_size),
                "sha256": output_sha256,
                "meta_canonical_sha256": _canonical_sha256(projected_meta),
                "clip_order": selected,
            },
            "dropped_clips": list(projected_meta["clip_subset"]["dropped_clips"]),
            "question_arrays_bitwise_preserved": True,
            "runtime_schema3_loader_accepted": True,
        }
        report = {
            "artifact_kind": "stage1_question_bank_clip_subset",
            "schema_version": 1,
            "content": content,
            "content_sha256": _canonical_sha256(content),
        }
        _write_exclusive(
            receipt,
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n",
        )
    except BaseException:
        try:
            out.unlink()
        except OSError:
            pass
        try:
            receipt.unlink()
        except OSError:
            pass
        raise

    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project a strict schema-3 stage-1 bank onto an ordered clip subset"
    )
    parser.add_argument("--bank", required=True, help="source schema-3 train bank")
    parser.add_argument(
        "--source-sha256",
        required=True,
        help="required SHA-256 of the source bank",
    )
    parser.add_argument(
        "--clip",
        action="append",
        default=[],
        help="selected clip name, repeated in source-relative order",
    )
    parser.add_argument("--out", required=True, help="new immutable subset bank")
    parser.add_argument("--manifest", required=True, help="new immutable subset receipt")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = subset_bank(
            Path(args.bank),
            list(args.clip),
            Path(args.out),
            Path(args.manifest),
            expected_source_sha256=args.source_sha256,
        )
    except (SubsetBankError, ValueError, KeyError, OSError) as exc:
        print(f"[subset-stage1-bank] ERROR: {exc}", file=sys.stderr)
        return EXIT_FAIL
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
