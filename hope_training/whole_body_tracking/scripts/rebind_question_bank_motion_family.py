#!/usr/bin/env python3
"""把 TOPP 烤入变速片段登记进 schema-3 题库的"每族允许 motion SHA 列表"。

人话:题库(gen_stage1_questions.py 产物)的答案是在正/反手两份 cal 原件上解出来的,运行时
按 SHA-256 对账、认错文件就 fail-closed。6-clip 变速列表里的烤入片段(bake_topp_strike_speed.py
产物)重排了挥拍时间律但**触球那一帧逐位不变**(manifest 里 contact.row_bitwise=true),所以同
一族的题目/答案照用是合法的——本脚本做的就是把这几份烤入文件的 SHA 追加进对应族的
``motion_sha256_allowed`` 列表,让 validate_runtime_motion_contract 的族寻址对账放行它们。

铁律:
* 题目张量一个字节都不改——只动 meta_json,且只加不减;写完回读逐数组指纹核对。
* 每份烤入必须自带 bake manifest,且逐条核过才收:
  - contact.row_bitwise 必须是 true(触球行逐位相同 = 答案按族复用的前提;false/缺失一律拒绝);
  - feasibility.verdict 必须是 "feasible"(判卷不过的资产禁止进列表);
  - source.sha256 必须等于题库该族记录的 motion_sha256(烤入确实出自题库作答的那份原件);
  - output.frames == 该族 n_frames 且 output.contact_frame == anchor_frame(时间锚没动);
  - output.sha256 必须等于烤入 npz 现场重算的 SHA(资产没被换过)。
* 输出 npz / manifest 一律 O_EXCL 拒绝覆盖;manifest 单层内容 SHA(照 bake 工具先例,不搞
  多层审批链)。
* 写完用当前运行时 loader(load_question_bank,strict schema-3)整卷重验,过不了就删输出。

用法::

    python rebind_question_bank_motion_family.py \
        --bank s1_schema3_train.npz \
        --clip forehand:fh_speed0p80.npz:fh_speed0p80.json \
        --clip forehand:fh_speed1p20.npz:fh_speed1p20.json \
        --clip backhand:bh_speed0p80.npz:bh_speed0p80.json \
        --out s1_schema3_train_family.npz --manifest s1_train_family_rebind.json

退出码:0 成功;2 任一前提/校验失败(fail loud, 不出半成品)。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
QB_MODULE_PATH = (
    HERE.parent
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/stage1_question_bank.py"
)
TOOL_NAME = "rebind_question_bank_motion_family.py v1 (per-family allowed motion SHA rebind)"
EXIT_FAIL = 2


class FamilyRebindError(RuntimeError):
    """任何一条重绑前提没核过(fail-closed;绝不出半成品资产)。"""


def _load_qb_module():
    """按文件路径加载 stage1_question_bank(mdp 包 __init__ 会拉 isaaclab,不能整包 import)。"""
    spec = importlib.util.spec_from_file_location("family_rebind_qbank", str(QB_MODULE_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_evidence(path_like, label: str) -> dict:
    """{path, bytes, sha256}(非空普通文件;照 bake/topp 工具先例)。"""
    try:
        path = Path(path_like).expanduser().resolve(strict=True)
        info = path.stat()
    except OSError as exc:
        raise FamilyRebindError(f"{label} 不可读: {path_like}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise FamilyRebindError(f"{label} 必须是非空普通文件: {path}")
    return {"path": str(path), "bytes": int(info.st_size), "sha256": _sha256_file(path)}


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_fingerprint(value: np.ndarray) -> dict:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise FamilyRebindError("题库里不允许 object 数组")
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "c_order_sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _decode_meta(arrays: dict) -> dict:
    if "meta_json" not in arrays:
        raise FamilyRebindError("源题库没有 meta_json——不是 schema-3 题库,拒绝重绑")
    raw = np.asarray(arrays["meta_json"])
    if raw.dtype != np.dtype("uint8") or raw.ndim != 1:
        raise FamilyRebindError("meta_json 必须是一维 uint8 数组")
    try:
        meta = json.loads(raw.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FamilyRebindError(f"meta_json 不是合法 JSON: {exc}") from exc
    if not isinstance(meta, dict):
        raise FamilyRebindError("meta_json 必须解码成对象")
    return meta


def parse_clip_specs(specs: list[str]) -> list[tuple[str, Path, Path]]:
    """--clip family:baked.npz:manifest.json → [(family, npz 路径, manifest 路径)]。"""
    parsed = []
    for spec in specs:
        parts = spec.split(":", 2)
        if len(parts) != 3 or not all(parts):
            raise FamilyRebindError(
                f"--clip 必须是 family:baked_npz:bake_manifest_json 三段,得到 {spec!r}"
            )
        family, npz_path, manifest_path = parts
        parsed.append((family, Path(npz_path).expanduser(), Path(manifest_path).expanduser()))
    return parsed


def validate_baked_input(
    family: str, npz_path: Path, manifest_path: Path, clip_info: dict
) -> dict:
    """核一份烤入资产:manifest 前提逐条对,过了返回登记条目(进输出 manifest 用)。"""
    npz_evidence = _file_evidence(npz_path, f"{family} 烤入 npz")
    manifest_evidence = _file_evidence(manifest_path, f"{family} 烤入 manifest")
    try:
        manifest = json.loads(Path(manifest_evidence["path"]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FamilyRebindError(f"{family} 烤入 manifest 读不出 JSON: {manifest_path}: {exc}") from exc
    if manifest.get("mode") != "bake":
        raise FamilyRebindError(
            f"{family} 烤入 manifest mode={manifest.get('mode')!r},只收 bake 模式产物"
        )
    contact = manifest.get("contact")
    if not isinstance(contact, dict) or contact.get("row_bitwise") is not True:
        raise FamilyRebindError(
            f"{family} 烤入 manifest 的 contact.row_bitwise 不是 true —— 触球行没有逐位保真,"
            f"题目答案按族复用的前提不成立,拒绝登记 ({manifest_path})"
        )
    feasibility = manifest.get("feasibility")
    verdict = feasibility.get("verdict") if isinstance(feasibility, dict) else None
    if verdict != "feasible":
        raise FamilyRebindError(
            f"{family} 烤入 manifest 判卷不是 feasible (verdict={verdict!r}),"
            f"拒绝登记 ({manifest_path})"
        )
    source = manifest.get("source")
    source_sha = source.get("sha256") if isinstance(source, dict) else None
    if source_sha != clip_info.get("motion_sha256"):
        raise FamilyRebindError(
            f"{family} 烤入的源 motion SHA {source_sha!r} != 题库该族记录的 "
            f"{clip_info.get('motion_sha256')!r} —— 烤入不是出自题库作答的那份原件"
        )
    output = manifest.get("output")
    if not isinstance(output, dict):
        raise FamilyRebindError(f"{family} 烤入 manifest 没有 output 块(判卷未通过或未落盘)")
    if output.get("sha256") != npz_evidence["sha256"]:
        raise FamilyRebindError(
            f"{family} 烤入 npz 现场 SHA {npz_evidence['sha256']} != manifest 记录的 "
            f"{output.get('sha256')!r} —— 资产和判卷记录对不上"
        )
    if int(output.get("frames", -1)) != int(clip_info.get("n_frames", -2)):
        raise FamilyRebindError(
            f"{family} 烤入帧数 {output.get('frames')!r} != 题库该族 n_frames "
            f"{clip_info.get('n_frames')!r} —— 时间锚变了,答案不可复用"
        )
    if int(output.get("contact_frame", -1)) != int(clip_info.get("anchor_frame", -2)):
        raise FamilyRebindError(
            f"{family} 烤入触球帧 {output.get('contact_frame')!r} != 题库该族 anchor_frame "
            f"{clip_info.get('anchor_frame')!r} —— 击球锚帧变了,答案不可复用"
        )
    return {
        "family": family,
        "baked_npz": npz_evidence,
        "bake_manifest": manifest_evidence,
        "row_bitwise": True,
        "frames": int(output["frames"]),
        "contact_frame": int(output["contact_frame"]),
        "source_motion_sha256": str(source_sha),
        "speed_ratio": (manifest.get("speed") or {}).get("ratio"),
    }


def build_rebound_meta(meta: dict, entries: list[dict], qb) -> dict:
    """产出新 meta:逐族扩 motion_sha256_allowed(只加不减)+ 记来源摘要。源 meta 不动。"""
    new_meta = copy.deepcopy(meta)
    clips_meta = new_meta.get("clips") or {}
    for entry in entries:
        family = entry["family"]
        info = clips_meta[family]
        allowed = list(qb.allowed_motion_shas(info))  # 现有列表(或单值退化),已自洽校验
        sha = entry["baked_npz"]["sha256"]
        if sha in allowed:
            raise FamilyRebindError(
                f"{family} 烤入 SHA {sha} 已在允许列表里 —— 重复登记按配置错误拒绝"
            )
        allowed.append(sha)
        info["motion_sha256_allowed"] = allowed
    new_meta["motion_family_rebind"] = {
        "tool": TOOL_NAME,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "inputs": [
            {
                "family": entry["family"],
                "baked_sha256": entry["baked_npz"]["sha256"],
                "bake_manifest_sha256": entry["bake_manifest"]["sha256"],
                "row_bitwise": True,
                "frames": entry["frames"],
                "contact_frame": entry["contact_frame"],
                "speed_ratio": entry["speed_ratio"],
            }
            for entry in entries
        ],
    }
    return new_meta


def _write_exclusive(path: Path, writer) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags, 0o444)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    finally:
        os.close(fd)


def rebind(bank_path: Path, clip_specs: list[tuple[str, Path, Path]],
           out_path: Path, manifest_path: Path) -> dict:
    qb = _load_qb_module()
    bank_evidence = _file_evidence(bank_path, "源题库")

    with np.load(bank_evidence["path"], allow_pickle=False) as loaded:
        key_order = list(loaded.files)
        arrays = {key: np.array(loaded[key], copy=True) for key in key_order}
    meta = _decode_meta(arrays)
    if int(meta.get("schema_version", 0)) != int(qb.SCHEMA_VERSION):
        raise FamilyRebindError(
            f"源题库 schema_version={meta.get('schema_version')!r},只收 schema-{qb.SCHEMA_VERSION}"
            f"(legacy 题库没有可扩的族级 SHA 合同)"
        )
    clip_order = list(meta.get("clip_order") or [])
    clips_meta = meta.get("clips") or {}
    if not clip_order or any(family not in clips_meta for family in clip_order):
        raise FamilyRebindError(f"源题库 clip_order/clips 元数据不完整: {clip_order!r}")

    if not clip_specs:
        raise FamilyRebindError("至少要一条 --clip family:baked_npz:bake_manifest_json")
    entries = []
    seen = set()
    for family, npz_path, bake_manifest_path in clip_specs:
        if family not in clip_order:
            raise FamilyRebindError(
                f"--clip 族名 {family!r} 不在题库 clip_order {clip_order!r} 里"
            )
        entry = validate_baked_input(family, npz_path, bake_manifest_path, clips_meta[family])
        key = (family, entry["baked_npz"]["sha256"])
        if key in seen:
            raise FamilyRebindError(f"{family} 烤入 SHA {key[1]} 在输入里重复出现")
        seen.add(key)
        entries.append(entry)

    new_meta = build_rebound_meta(meta, entries, qb)

    # 题目张量逐字节不变的证据:先记源指纹,写完回读逐数组核对。
    fingerprints = {
        key: _array_fingerprint(value) for key, value in arrays.items() if key != "meta_json"
    }
    out_arrays = dict(arrays)
    out_arrays["meta_json"] = np.frombuffer(
        json.dumps(
            new_meta, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("utf-8"),
        dtype=np.uint8,
    ).copy()

    out_path = out_path.expanduser().absolute()
    manifest_path = manifest_path.expanduser().absolute()
    for target in (out_path, manifest_path):
        if target.exists() or target.is_symlink():
            raise FamilyRebindError(f"拒绝覆盖已有文件: {target}")
    if out_path == manifest_path:
        raise FamilyRebindError("--out 和 --manifest 必须是两个不同路径")

    _write_exclusive(
        out_path,
        lambda stream: np.savez(stream, **{key: out_arrays[key] for key in key_order}),
    )
    try:
        # 回读:非 meta 数组指纹必须逐个等于源指纹;meta 必须恰好是审好的 new_meta。
        with np.load(out_path, allow_pickle=False) as written:
            written_keys = list(written.files)
            actual = {
                key: _array_fingerprint(written[key])
                for key in written_keys
                if key != "meta_json"
            }
            written_meta = _decode_meta({"meta_json": np.array(written["meta_json"], copy=True)})
        if written_keys != key_order:
            raise FamilyRebindError("输出题库改变了 npz 键序")
        if actual != fingerprints:
            raise FamilyRebindError("输出题库的题目张量和源不逐字节一致 —— 拒绝发布")
        if _canonical_sha256(written_meta) != _canonical_sha256(new_meta):
            raise FamilyRebindError("输出题库 meta 与审好的新 meta 不一致")
        # 终验:当前运行时 loader 必须整卷收下重绑后的题库(strict schema-3,不走 legacy)。
        qb.load_question_bank(
            str(out_path), device="cpu", expected_split=meta.get("split"), allow_legacy=False
        )
        allowed_by_family = {
            family: list(qb.allowed_motion_shas(new_meta["clips"][family]))
            for family in clip_order
        }
        content = {
            "tool": TOOL_NAME,
            "generated_utc": new_meta["motion_family_rebind"]["generated_utc"],
            "source_bank": bank_evidence,
            "baked_inputs": entries,
            "output_bank": {
                "path": str(out_path),
                "bytes": int(out_path.stat().st_size),
                "sha256": _sha256_file(out_path),
            },
            "motion_sha256_allowed": allowed_by_family,
            "question_arrays_bitwise_identical": True,
            "runtime_loader_accepted": True,
        }
        report = {
            "artifact_kind": "stage1_question_bank_motion_family_rebind",
            "schema_version": 1,
            "content": content,
            "content_sha256": _canonical_sha256(content),
        }
        payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
        _write_exclusive(manifest_path, lambda stream: stream.write(payload))
        if json.loads(manifest_path.read_text(encoding="utf-8")) != report:
            raise FamilyRebindError("manifest 回读不一致 —— refusing")
    except BaseException:
        try:
            out_path.unlink()
        except OSError:
            pass
        raise
    return {
        "status": "published",
        "bank": str(out_path),
        "bank_sha256": content["output_bank"]["sha256"],
        "manifest": str(manifest_path),
        "families": {entry["family"]: entry["baked_npz"]["sha256"] for entry in entries},
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把烤入变速片段的 SHA 登记进 schema-3 题库的族级允许列表(题目张量逐字节不变)"
    )
    parser.add_argument("--bank", required=True, help="源 schema-3 题库 npz")
    parser.add_argument(
        "--clip", action="append", default=[],
        help="family:baked_npz:bake_manifest_json(可重复;族名须在题库 clip_order 里)",
    )
    parser.add_argument("--out", required=True, help="输出题库 npz(拒绝覆盖)")
    parser.add_argument("--manifest", required=True, help="输出 manifest JSON(拒绝覆盖,单层内容 SHA)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        result = rebind(
            Path(args.bank).expanduser(),
            parse_clip_specs(list(args.clip)),
            Path(args.out),
            Path(args.manifest),
        )
    except (FamilyRebindError, ValueError, KeyError) as exc:
        print(f"[family-rebind] ERROR: {exc}", file=sys.stderr)
        return EXIT_FAIL
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
