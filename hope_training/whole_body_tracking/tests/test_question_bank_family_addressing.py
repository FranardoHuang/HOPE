"""题库族寻址 + motion 重绑脚本单测(spdmix v2 硬绑定二;NO Isaac imports)。

人话:6-clip 变速列表里,题库还是只有正/反手两族题;每个 clip 先查族、再拿族号当题库下标,
SHA 对账改成"该 clip 的 motion SHA ∈ 其族的允许列表"。本文件钉死四件事:

* legacy 逐字节不变:不传 clip_families / 题库没有 motion_sha256_allowed 时,
  validate_runtime_motion_contract 的行为和报错文本与旧实现完全一致;
* family 寻址正确:6-clip 族表对账通过;族配错 / 下标越界 / 非整数族号 / 长度不齐当场拒绝;
* 允许 SHA 列表合同:allowed_motion_shas 的通过/拒绝面;strict-v3 loader 对坏列表 fail-closed;
* rebind 脚本:端到端 CLI 产出题目张量逐字节不变的新题库 + 单层内容 SHA manifest;
  row_bitwise=false / 源 SHA 不对 / 帧锚漂移 / infeasible / 重复登记 / 拒绝覆盖全部拒收。

Run:  pytest hope_training/whole_body_tracking/tests/test_question_bank_family_addressing.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
QB_MODULE_PATH = os.path.abspath(os.path.join(
    HERE, "..", "source", "whole_body_tracking", "whole_body_tracking",
    "tasks", "tracking", "mdp", "stage1_question_bank.py",
))
REBIND_SCRIPT_PATH = os.path.abspath(os.path.join(
    HERE, "..", "scripts", "rebind_question_bank_motion_family.py",
))
HOPE_COMMANDS_PATH = os.path.abspath(os.path.join(
    HERE, "..", "source", "whole_body_tracking", "whole_body_tracking",
    "tasks", "tracking", "mdp", "hope_commands.py",
))


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def qb():
    return _load_by_path("famaddr_qbank", QB_MODULE_PATH)


@pytest.fixture(scope="module")
def rb():
    return _load_by_path("famaddr_rebind", REBIND_SCRIPT_PATH)


# ------------------------------------------------------------------------------------------- #
# 造件:motion 原件/烤入文件、runtime-contract 用的轻量 meta、完整 schema-3 题库、bake manifest
# ------------------------------------------------------------------------------------------- #

N_FRAMES = {"forehand": 11, "backhand": 13}
ANCHOR = {"forehand": 5, "backhand": 6}


def _write_file(path, payload: bytes) -> str:
    with open(path, "wb") as fh:
        fh.write(payload)
    return path


def _make_motions(tmpdir, qb):
    """两份 cal 原件 + 每族两份烤入(内容互不相同 → SHA 互不相同)。"""
    files = {}
    for family in ("forehand", "backhand"):
        files[family] = _write_file(
            os.path.join(tmpdir, f"{family}_cal.npz"), f"cal-{family}".encode()
        )
        for tag in ("0p80", "1p20"):
            files[f"{family}_{tag}"] = _write_file(
                os.path.join(tmpdir, f"{family}_speed{tag}.npz"), f"bake-{family}-{tag}".encode()
            )
    shas = {key: qb.sha256_file(path) for key, path in files.items()}
    return files, shas


def _contract_meta(shas, allowed=None):
    """validate_runtime_motion_contract 只读 clip_order/clips 的轻量 meta。"""
    clips = {}
    for family in ("forehand", "backhand"):
        info = {
            "motion_sha256": shas[family],
            "n_frames": N_FRAMES[family],
            "anchor_frame": ANCHOR[family],
        }
        if allowed is not None:
            info["motion_sha256_allowed"] = list(allowed[family])
        clips[family] = info
    return {"clip_order": ["forehand", "backhand"], "clips": clips}


def _phase(family):
    return ANCHOR[family] / (N_FRAMES[family] - 1)


def _rows_for_split(qb, split, n, offset=0):
    rows = []
    i = int(offset)
    while len(rows) < n:
        row = np.array([-3.0 - 0.01 * i, 0.01 * i, -0.2], dtype=np.float64)
        if qb.question_split(row) == split:
            rows.append(row)
        i += 1
    return np.stack(rows)


def _write_schema3_bank(path, qb, motion_shas, split="train", mutate_meta=None):
    """完整可过 strict loader 的 schema-3 题库(照 test_stage1_wiring 先例,motion SHA 可指定)。"""
    arrays = {}
    clips = {}
    clip_order = ["forehand", "backhand"]
    for c, name in enumerate(clip_order):
        incoming = _rows_for_split(qb, split, 3, offset=100 * c)
        normal = np.tile(np.array([0.0, 1.0, 0.0]), (3, 1))
        contact = np.array([0.5, -0.2 + 0.4 * c, 0.9])
        clip_normal = np.array([0.0, 1.0, 0.0])
        clip_vel = np.array([1.0, 0.1 * c, 0.2])
        arrays.update({
            f"{name}/contact_pos_env": contact,
            f"{name}/clip_normal": clip_normal,
            f"{name}/clip_vel": clip_vel,
            f"{name}/incoming_vel": incoming,
            f"{name}/incoming_spin": np.zeros((3, 3)),
            f"{name}/demanded_vel": np.stack([
                np.array([1.0 + i, 0.1 * c, 0.2]) for i in range(3)
            ]),
            f"{name}/demanded_normal": normal,
            f"{name}/difficulty_deg": np.zeros(3),
        })
        clips[name] = {
            "motion_basename": f"{name}_cal.npz", "motion_sha256": motion_shas[name],
            "n_frames": N_FRAMES[name], "anchor_frame": ANCHOR[name],
            "anchor_phase": _phase(name), "question_count": 3,
            "grip_mode": "baked", "grip_rotation_matrix": None,
            "rally_yaw_deg": 0.0,
            "contact_pos_env": np.round(contact, 12).tolist(),
            "clip_normal": np.round(clip_normal, 12).tolist(),
            "clip_vel": np.round(clip_vel, 12).tolist(),
        }
    repo = qb.find_repo_root()
    physics_sha = qb.sha256_file(os.path.join(repo, "configs", "ball_physics_venue.yaml"))
    physics_contract = qb.runtime_physics_contract(repo)
    physics_contract_sha = qb.canonical_sha256(physics_contract)
    family = {
        "contract": qb.SOURCE_FAMILY_CONTRACT, "stage": "S1",
        "face_frame": "mount_plusY_A", "incoming_spin_mode": "zero",
        "split_algorithm": qb.SPLIT_ALGORITHM,
        "clip_order": clip_order,
        "clips": {name: {
            "motion_sha256": info["motion_sha256"],
            "anchor_frame": info["anchor_frame"], "anchor_phase": info["anchor_phase"],
            "grip_mode": info["grip_mode"],
            "grip_rotation_matrix": info["grip_rotation_matrix"],
            "rally_yaw_deg": info["rally_yaw_deg"],
            "contact_pos_env": info["contact_pos_env"],
            "clip_normal": info["clip_normal"],
            "clip_vel": info["clip_vel"],
        } for name, info in clips.items()},
        "incoming": {"speed_range": [2.0, 5.0], "vy_max": 0.6, "vz_range": [-2.0, 0.3]},
        "landing_env": [0.4, 0.0],
        "near_x": -1.37,
        "table_surface_z": 0.76,
        "speed_budget": None,
        "physics_sha256": physics_sha,
        "physics_contract_sha256": physics_contract_sha,
    }
    meta = {
        "schema_version": 3, "stage": "S1", "face_frame": "mount_plusY_A",
        "incoming_spin_mode": "zero", "spin_units": "rad/s", "split": split,
        "split_algorithm": qb.SPLIT_ALGORITHM, "clip_order": clip_order,
        "speed_range": [2.0, 5.0], "vy_max": 0.6, "vz_range": [-2.0, 0.3],
        "landing_env": [0.4, 0.0], "near_x": -1.37,
        "table_surface_z": 0.76, "speed_budget": None,
        "source_family_contract": family,
        "source_family_sha256": qb.canonical_sha256(family),
        "physics_sha256": physics_sha,
        "physics_contract": physics_contract,
        "physics_contract_sha256": physics_contract_sha,
        "validation": {"torch_closed_loop_pass": True, "net_clearance_all_pass": True,
                       "max_landing_error_m": 0.01},
        "grip_applied": True, "rally_yaw_applied": True,
        "grip_applied_per_clip": {"forehand": True, "backhand": True},
        "rally_yaw_applied_per_clip": {"forehand": True, "backhand": True},
        "clips": clips,
    }
    if mutate_meta is not None:
        mutate_meta(meta)
    arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    np.savez(path, **arrays)
    return arrays


def _write_bake_manifest(path, source_path, baked_path, family, qb,
                         row_bitwise=True, verdict="feasible", mode="bake",
                         frames=None, contact_frame=None, source_sha=None, output_sha=None):
    frames = N_FRAMES[family] if frames is None else frames
    contact_frame = ANCHOR[family] if contact_frame is None else contact_frame
    manifest = {
        "tool": "bake_topp_strike_speed.py v1 (axis A: strike-speed retiming bake)",
        "mode": mode,
        "source": {
            "path": os.path.abspath(source_path),
            "bytes": os.path.getsize(source_path),
            "sha256": qb.sha256_file(source_path) if source_sha is None else source_sha,
        },
        "speed": {"ratio": 1.2, "bucket": "plus20"},
        "feasibility": {"verdict": verdict, "reasons": []},
        "contact": {
            "frame": contact_frame, "phase": contact_frame / (frames - 1),
            "registered_phase": contact_frame / (frames - 1), "row_bitwise": row_bitwise,
        },
        "output": {
            "path": os.path.abspath(baked_path),
            "bytes": os.path.getsize(baked_path),
            "sha256": qb.sha256_file(baked_path) if output_sha is None else output_sha,
            "frames": frames, "fps": 50, "contact_frame": contact_frame,
            "phase": round(contact_frame / (frames - 1), 6),
        },
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return path


SIX_CLIP_FAMILIES = [0, 0, 0, 1, 1, 1]


def _six_clip_args(files, key_order=("forehand", "forehand_0p80", "forehand_1p20",
                                     "backhand", "backhand_0p80", "backhand_1p20")):
    motion_files = [files[key] for key in key_order]
    seg_lens = [N_FRAMES["forehand"]] * 3 + [N_FRAMES["backhand"]] * 3
    phases = [_phase("forehand")] * 3 + [_phase("backhand")] * 3
    return motion_files, seg_lens, phases


# ------------------------------------------------------------------------------------------- #
# A. legacy 逐字节不变
# ------------------------------------------------------------------------------------------- #

def test_legacy_contract_passes_and_error_texts_unchanged(qb, tmp_path):
    """不传 clip_families 时,通过面与报错文本和旧实现一致(现役所有在跑臂的路径)。"""
    tmpdir = str(tmp_path)
    files, shas = _make_motions(tmpdir, qb)
    meta = _contract_meta(shas)
    two_files = [files["forehand"], files["backhand"]]
    lens = [N_FRAMES["forehand"], N_FRAMES["backhand"]]
    phases = [_phase("forehand"), _phase("backhand")]
    qb.validate_runtime_motion_contract(meta, two_files, lens, phases)
    with pytest.raises(ValueError, match="runtime contract length mismatch"):
        qb.validate_runtime_motion_contract(meta, [two_files[1]], [lens[1]], [phases[1]])
    with pytest.raises(ValueError, match="loaded motion SHA"):
        qb.validate_runtime_motion_contract(meta, list(reversed(two_files)), lens, phases)
    with pytest.raises(ValueError, match="anchored at frame"):
        qb.validate_runtime_motion_contract(meta, two_files, lens, [0.1, phases[1]])
    with pytest.raises(ValueError, match="runtime has"):
        qb.validate_runtime_motion_contract(meta, two_files, [7, lens[1]], phases)
    # 旧单值报错文本必须原样保留("!= bank" 措辞),下游剧本抓这行
    with open(two_files[0], "ab") as fh:
        fh.write(b"tampered")
    with pytest.raises(ValueError, match="!= bank"):
        qb.validate_runtime_motion_contract(meta, two_files, lens, phases)


def test_legacy_identity_bank_row_addressing(qb, tmp_path):
    """族表缺席 = clip_id 直接当题库下标:select_questions 行为与旧实现逐字节一致。"""
    path = str(tmp_path / "bank.npz")
    flat = {}
    for c, name in enumerate(("forehand", "backhand")):
        q = 3 - c
        flat.update({
            f"{name}/contact_pos_env": np.array([0.5, 0.1 * c, 0.9]),
            f"{name}/incoming_vel": np.full((q, 3), -3.0 + c),
            f"{name}/incoming_spin": np.zeros((q, 3)),
            f"{name}/demanded_vel": np.full((q, 3), 100.0 * (c + 1)),
            f"{name}/demanded_normal": np.full((q, 3), 1.0 + c),
        })
    meta = np.frombuffer(json.dumps({
        "schema_version": 2, "stage": "S1", "face_frame": "mount_plusY_A",
        "incoming_spin_mode": "zero", "spin_units": "rad/s",
        "grip_applied": True, "rally_yaw_applied": True,
        "grip_applied_per_clip": {"forehand": True, "backhand": True},
        "rally_yaw_applied_per_clip": {"forehand": True, "backhand": True},
    }).encode(), dtype=np.uint8)
    np.savez(path, meta_json=meta, **flat)
    bank = qb.load_question_bank(path, allow_legacy=True)
    clip_ids = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    u = torch.tensor([0.0, 0.5, 0.99, 0.2])
    got = qb.select_questions(bank, clip_ids, u)
    # clip 0 行是 100、clip 1 行是 200 —— 直接下标语义未被族寻址改动
    assert torch.all(got[3][clip_ids == 0] == 100.0)
    assert torch.all(got[3][clip_ids == 1] == 200.0)


def test_allowed_motion_shas_helper(qb):
    """允许列表合同:单值退化 / 合法列表 / 缺原件 / 重复 / 非 hex / 空表。"""
    primary = "a" * 64
    baked = "b" * 64
    assert qb.allowed_motion_shas({"motion_sha256": primary}) == (primary,)
    assert qb.allowed_motion_shas({}) == (None,)  # legacy: 缺 SHA 时由调用方按不等报错
    assert qb.allowed_motion_shas(
        {"motion_sha256": primary, "motion_sha256_allowed": [primary, baked]}
    ) == (primary, baked)
    with pytest.raises(ValueError, match="must contain the primary"):
        qb.allowed_motion_shas({"motion_sha256": primary, "motion_sha256_allowed": [baked]})
    with pytest.raises(ValueError, match="duplicates"):
        qb.allowed_motion_shas(
            {"motion_sha256": primary, "motion_sha256_allowed": [primary, primary]}
        )
    with pytest.raises(ValueError, match="64-hex"):
        qb.allowed_motion_shas({"motion_sha256": primary, "motion_sha256_allowed": ["nope"]})
    with pytest.raises(ValueError, match="64-hex"):
        qb.allowed_motion_shas({"motion_sha256": primary, "motion_sha256_allowed": []})


# ------------------------------------------------------------------------------------------- #
# B. family 寻址对账
# ------------------------------------------------------------------------------------------- #

def test_family_contract_six_clips_pass(qb, tmp_path):
    """6-clip 族表 + 每族允许列表齐全 → 对账通过(核心放行面)。"""
    tmpdir = str(tmp_path)
    files, shas = _make_motions(tmpdir, qb)
    allowed = {
        "forehand": [shas["forehand"], shas["forehand_0p80"], shas["forehand_1p20"]],
        "backhand": [shas["backhand"], shas["backhand_0p80"], shas["backhand_1p20"]],
    }
    meta = _contract_meta(shas, allowed=allowed)
    motion_files, seg_lens, phases = _six_clip_args(files)
    qb.validate_runtime_motion_contract(
        meta, motion_files, seg_lens, phases, clip_families=SIX_CLIP_FAMILIES
    )
    # 2-clip 原件配族表 (0,1) 也照走族路径(允许列表含原件)
    qb.validate_runtime_motion_contract(
        meta, [files["forehand"], files["backhand"]],
        [N_FRAMES["forehand"], N_FRAMES["backhand"]],
        [_phase("forehand"), _phase("backhand")], clip_families=[0, 1]
    )


def test_family_contract_rejects_wrong_family_sha(qb, tmp_path):
    """反手烤入挂在正手族下 → SHA 不在该族允许列表,当场拒绝。"""
    tmpdir = str(tmp_path)
    files, shas = _make_motions(tmpdir, qb)
    allowed = {
        "forehand": [shas["forehand"], shas["forehand_0p80"]],
        "backhand": [shas["backhand"], shas["backhand_0p80"]],
    }
    meta = _contract_meta(shas, allowed=allowed)
    with pytest.raises(ValueError, match="not in the family's allowed list"):
        qb.validate_runtime_motion_contract(
            meta,
            [files["forehand"], files["backhand_0p80"]],
            [N_FRAMES["forehand"], N_FRAMES["forehand"]],
            [_phase("forehand"), _phase("forehand")],
            clip_families=[0, 0],
        )
    # 没登记过的烤入(1p20 不在列表)同样拒绝
    with pytest.raises(ValueError, match="not in the family's allowed list"):
        qb.validate_runtime_motion_contract(
            meta,
            [files["forehand"], files["forehand_1p20"]],
            [N_FRAMES["forehand"]] * 2,
            [_phase("forehand")] * 2,
            clip_families=[0, 0],
        )


def test_family_contract_rejects_bad_family_indices(qb, tmp_path):
    """族号越界 / 非整数 / bool 一律 fail-closed。"""
    tmpdir = str(tmp_path)
    files, shas = _make_motions(tmpdir, qb)
    meta = _contract_meta(shas)
    args = ([files["forehand"]], [N_FRAMES["forehand"]], [_phase("forehand")])
    with pytest.raises(ValueError, match="outside the bank's clip_order"):
        qb.validate_runtime_motion_contract(meta, *args, clip_families=[2])
    with pytest.raises(ValueError, match="outside the bank's clip_order"):
        qb.validate_runtime_motion_contract(meta, *args, clip_families=[-1])
    with pytest.raises(ValueError, match="must be integers"):
        qb.validate_runtime_motion_contract(meta, *args, clip_families=[0.5])
    with pytest.raises(ValueError, match="must be integers"):
        qb.validate_runtime_motion_contract(meta, *args, clip_families=[True])


def test_family_contract_length_mismatch(qb, tmp_path):
    """族表长度和文件/段长/相位不齐 → 专用报错(不静默截断)。"""
    tmpdir = str(tmp_path)
    files, shas = _make_motions(tmpdir, qb)
    meta = _contract_meta(shas)
    with pytest.raises(ValueError, match="family contract length mismatch"):
        qb.validate_runtime_motion_contract(
            meta, [files["forehand"], files["backhand"]],
            [N_FRAMES["forehand"], N_FRAMES["backhand"]],
            [_phase("forehand"), _phase("backhand")],
            clip_families=[0],
        )


def test_family_contract_frame_and_anchor_still_strict(qb, tmp_path):
    """族寻址不放松时间锚:帧数 / 击球相位错照旧当场炸(答案可复用的前提)。"""
    tmpdir = str(tmp_path)
    files, shas = _make_motions(tmpdir, qb)
    allowed = {
        "forehand": [shas["forehand"], shas["forehand_0p80"]],
        "backhand": [shas["backhand"]],
    }
    meta = _contract_meta(shas, allowed=allowed)
    with pytest.raises(ValueError, match="runtime has"):
        qb.validate_runtime_motion_contract(
            meta, [files["forehand_0p80"]], [N_FRAMES["forehand"] + 1],
            [_phase("forehand")], clip_families=[0]
        )
    with pytest.raises(ValueError, match="anchored at frame"):
        qb.validate_runtime_motion_contract(
            meta, [files["forehand_0p80"]], [N_FRAMES["forehand"]],
            [0.1], clip_families=[0]
        )


def test_family_row_addressing_matches_direct_selection(qb, tmp_path):
    """clip→族→题库行:6-clip 查表选题 == 直接用族号选题(同 u 逐元素一致)。"""
    path = str(tmp_path / "bank_rows.npz")
    _write_schema3_bank(path, qb, {"forehand": "a" * 64, "backhand": "b" * 64})
    bank = qb.load_question_bank(path, expected_split="train")
    table = torch.tensor(SIX_CLIP_FAMILIES, dtype=torch.long)
    clip_ids = torch.tensor([0, 1, 2, 3, 4, 5, 2, 5], dtype=torch.long)
    u = torch.rand(8)
    via_table = qb.select_questions(bank, table[clip_ids], u)
    direct = qb.select_questions(bank, torch.tensor([0, 0, 0, 1, 1, 1, 0, 1]), u)
    for got, want in zip(via_table, direct):
        assert torch.equal(got, want)


# ------------------------------------------------------------------------------------------- #
# C. strict-v3 loader 对允许列表 fail-closed
# ------------------------------------------------------------------------------------------- #

def test_loader_accepts_valid_allowed_list_and_rejects_malformed(qb, tmp_path):
    baked_sha = "c" * 64
    good = str(tmp_path / "good.npz")
    _write_schema3_bank(
        good, qb, {"forehand": "a" * 64, "backhand": "b" * 64},
        mutate_meta=lambda m: m["clips"]["forehand"].__setitem__(
            "motion_sha256_allowed", ["a" * 64, baked_sha]),
    )
    bank = qb.load_question_bank(good, expected_split="train")
    assert bank.metadata["clips"]["forehand"]["motion_sha256_allowed"] == ["a" * 64, baked_sha]
    cases = {
        "missing_primary": (["c" * 64], "must contain the primary"),
        "dupes": (["a" * 64, "a" * 64], "duplicates"),
        "not_hex": (["zz"], "64-hex"),
        "empty": ([], "64-hex"),
    }
    for name, (bad_list, match) in cases.items():
        bad = str(tmp_path / f"bad_{name}.npz")
        _write_schema3_bank(
            bad, qb, {"forehand": "a" * 64, "backhand": "b" * 64},
            mutate_meta=lambda m, lst=bad_list: m["clips"]["forehand"].__setitem__(
                "motion_sha256_allowed", lst),
        )
        with pytest.raises(ValueError, match=match):
            qb.load_question_bank(bad, expected_split="train")


# ------------------------------------------------------------------------------------------- #
# D. rebind 脚本
# ------------------------------------------------------------------------------------------- #

def _rebind_inputs(tmp_path, qb, split="train"):
    tmpdir = str(tmp_path)
    files, shas = _make_motions(tmpdir, qb)
    bank = os.path.join(tmpdir, "bank.npz")
    _write_schema3_bank(bank, qb, {"forehand": shas["forehand"], "backhand": shas["backhand"]},
                        split=split)
    manifests = {}
    for family in ("forehand", "backhand"):
        for tag in ("0p80", "1p20"):
            key = f"{family}_{tag}"
            manifests[key] = _write_bake_manifest(
                os.path.join(tmpdir, f"{key}.json"), files[family], files[key], family, qb,
            )
    return tmpdir, files, shas, bank, manifests


def _run_cli(rb, args):
    return rb.main(args)


def test_rebind_end_to_end_cli(qb, rb, tmp_path, capsys):
    """CLI 端到端:产出可过 strict loader 的题库,允许列表扩了,6-clip 对账放行。"""
    tmpdir, files, shas, bank, manifests = _rebind_inputs(tmp_path, qb)
    out = os.path.join(tmpdir, "bank_family.npz")
    report = os.path.join(tmpdir, "bank_family_rebind.json")
    rc = _run_cli(rb, [
        "--bank", bank,
        "--clip", f"forehand:{files['forehand_0p80']}:{manifests['forehand_0p80']}",
        "--clip", f"forehand:{files['forehand_1p20']}:{manifests['forehand_1p20']}",
        "--clip", f"backhand:{files['backhand_0p80']}:{manifests['backhand_0p80']}",
        "--clip", f"backhand:{files['backhand_1p20']}:{manifests['backhand_1p20']}",
        "--out", out, "--manifest", report,
    ])
    assert rc == 0, capsys.readouterr().err
    rebound = qb.load_question_bank(out, expected_split="train")
    fh = rebound.metadata["clips"]["forehand"]["motion_sha256_allowed"]
    bh = rebound.metadata["clips"]["backhand"]["motion_sha256_allowed"]
    assert fh == [shas["forehand"], shas["forehand_0p80"], shas["forehand_1p20"]]
    assert bh == [shas["backhand"], shas["backhand_0p80"], shas["backhand_1p20"]]
    # 6-clip 变速列表现在能过运行时对账
    motion_files, seg_lens, phases = _six_clip_args(files)
    qb.validate_runtime_motion_contract(
        rebound.metadata, motion_files, seg_lens, phases, clip_families=SIX_CLIP_FAMILIES
    )
    # 单层内容 SHA manifest 自洽
    with open(report, encoding="utf-8") as fh_json:
        published = json.load(fh_json)
    content = published["content"]
    assert published["content_sha256"] == rb._canonical_sha256(content)
    assert content["question_arrays_bitwise_identical"] is True
    assert content["output_bank"]["sha256"] == rb._sha256_file(rb.Path(out))
    assert {entry["family"] for entry in content["baked_inputs"]} == {"forehand", "backhand"}


def test_rebind_question_tensors_bitwise_identical(qb, rb, tmp_path):
    """铁律:输出题库所有非 meta 数组与源逐字节一致(逐数组 tobytes 对比)。"""
    tmpdir, files, shas, bank, manifests = _rebind_inputs(tmp_path, qb)
    out = os.path.join(tmpdir, "bank_family.npz")
    rc = _run_cli(rb, [
        "--bank", bank,
        "--clip", f"forehand:{files['forehand_0p80']}:{manifests['forehand_0p80']}",
        "--out", out, "--manifest", os.path.join(tmpdir, "r.json"),
    ])
    assert rc == 0
    with np.load(bank, allow_pickle=False) as src, np.load(out, allow_pickle=False) as dst:
        assert list(src.files) == list(dst.files)
        for key in src.files:
            if key == "meta_json":
                continue
            a, b = np.asarray(src[key]), np.asarray(dst[key])
            assert a.dtype == b.dtype and a.shape == b.shape, key
            assert a.tobytes(order="C") == b.tobytes(order="C"), f"{key} 张量被改动"


def test_rebind_rejects_row_bitwise_false(qb, rb, tmp_path):
    """row_bitwise 不是 true = 触球行没逐位保真 → 拒收,不留半成品输出。"""
    tmpdir, files, shas, bank, _ = _rebind_inputs(tmp_path, qb)
    bad_manifest = _write_bake_manifest(
        os.path.join(tmpdir, "bad_rb.json"), files["forehand"], files["forehand_0p80"],
        "forehand", qb, row_bitwise=False,
    )
    out = os.path.join(tmpdir, "nope.npz")
    rc = _run_cli(rb, [
        "--bank", bank,
        "--clip", f"forehand:{files['forehand_0p80']}:{bad_manifest}",
        "--out", out, "--manifest", os.path.join(tmpdir, "nope.json"),
    ])
    assert rc == rb.EXIT_FAIL
    assert not os.path.exists(out) and not os.path.exists(os.path.join(tmpdir, "nope.json"))
    # row_bitwise 缺失同样拒绝
    with open(bad_manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)
    del manifest["contact"]["row_bitwise"]
    missing = os.path.join(tmpdir, "missing_rb.json")
    with open(missing, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    rc = _run_cli(rb, [
        "--bank", bank,
        "--clip", f"forehand:{files['forehand_0p80']}:{missing}",
        "--out", out, "--manifest", os.path.join(tmpdir, "nope2.json"),
    ])
    assert rc == rb.EXIT_FAIL and not os.path.exists(out)


def test_rebind_rejects_wrong_source_sha(qb, rb, tmp_path):
    """烤入不是出自题库作答的那份原件(source.sha256 不匹配)→ 拒收。"""
    tmpdir, files, shas, bank, _ = _rebind_inputs(tmp_path, qb)
    wrong_source = _write_bake_manifest(
        os.path.join(tmpdir, "wrong_src.json"), files["backhand"], files["forehand_0p80"],
        "forehand", qb,  # source 指向反手原件,但登记到正手族
    )
    out = os.path.join(tmpdir, "nope.npz")
    rc = _run_cli(rb, [
        "--bank", bank,
        "--clip", f"forehand:{files['forehand_0p80']}:{wrong_source}",
        "--out", out, "--manifest", os.path.join(tmpdir, "nope.json"),
    ])
    assert rc == rb.EXIT_FAIL and not os.path.exists(out)


def test_rebind_rejects_frame_or_anchor_drift(qb, rb, tmp_path):
    """帧数或触球帧和题库记录不一致 → 时间锚变了,拒收。"""
    tmpdir, files, shas, bank, _ = _rebind_inputs(tmp_path, qb)
    out = os.path.join(tmpdir, "nope.npz")
    drift_frames = _write_bake_manifest(
        os.path.join(tmpdir, "drift_frames.json"), files["forehand"], files["forehand_0p80"],
        "forehand", qb, frames=N_FRAMES["forehand"] + 2,
    )
    rc = _run_cli(rb, [
        "--bank", bank, "--clip", f"forehand:{files['forehand_0p80']}:{drift_frames}",
        "--out", out, "--manifest", os.path.join(tmpdir, "n1.json"),
    ])
    assert rc == rb.EXIT_FAIL and not os.path.exists(out)
    drift_anchor = _write_bake_manifest(
        os.path.join(tmpdir, "drift_anchor.json"), files["forehand"], files["forehand_0p80"],
        "forehand", qb, contact_frame=ANCHOR["forehand"] + 1,
    )
    rc = _run_cli(rb, [
        "--bank", bank, "--clip", f"forehand:{files['forehand_0p80']}:{drift_anchor}",
        "--out", out, "--manifest", os.path.join(tmpdir, "n2.json"),
    ])
    assert rc == rb.EXIT_FAIL and not os.path.exists(out)


def test_rebind_rejects_infeasible_tampered_and_legacy(qb, rb, tmp_path):
    """判卷 infeasible / 资产被换过(SHA 不符)/ legacy 题库 → 全部拒收。"""
    tmpdir, files, shas, bank, manifests = _rebind_inputs(tmp_path, qb)
    out = os.path.join(tmpdir, "nope.npz")
    infeasible = _write_bake_manifest(
        os.path.join(tmpdir, "infeasible.json"), files["forehand"], files["forehand_0p80"],
        "forehand", qb, verdict="physical-infeasible",
    )
    rc = _run_cli(rb, [
        "--bank", bank, "--clip", f"forehand:{files['forehand_0p80']}:{infeasible}",
        "--out", out, "--manifest", os.path.join(tmpdir, "n1.json"),
    ])
    assert rc == rb.EXIT_FAIL
    # manifest 写完后资产被换 → 现场 SHA 对不上
    with open(files["forehand_0p80"], "ab") as fh:
        fh.write(b"swapped")
    rc = _run_cli(rb, [
        "--bank", bank,
        "--clip", f"forehand:{files['forehand_0p80']}:{manifests['forehand_0p80']}",
        "--out", out, "--manifest", os.path.join(tmpdir, "n2.json"),
    ])
    assert rc == rb.EXIT_FAIL
    # legacy(schema-2)题库没有族级合同可扩
    legacy = os.path.join(tmpdir, "legacy.npz")
    np.savez(legacy, meta_json=np.frombuffer(
        json.dumps({"schema_version": 2}).encode(), dtype=np.uint8))
    rc = _run_cli(rb, [
        "--bank", legacy,
        "--clip", f"backhand:{files['backhand_0p80']}:{manifests['backhand_0p80']}",
        "--out", out, "--manifest", os.path.join(tmpdir, "n3.json"),
    ])
    assert rc == rb.EXIT_FAIL and not os.path.exists(out)


def test_rebind_rejects_overwrite_duplicates_and_unknown_family(qb, rb, tmp_path):
    """拒绝覆盖已有输出 / 输入重复 SHA / 已登记 SHA 重复登记 / 未知族名。"""
    tmpdir, files, shas, bank, manifests = _rebind_inputs(tmp_path, qb)
    out = os.path.join(tmpdir, "bank_family.npz")
    report = os.path.join(tmpdir, "report.json")
    ok_args = [
        "--bank", bank,
        "--clip", f"forehand:{files['forehand_0p80']}:{manifests['forehand_0p80']}",
        "--out", out, "--manifest", report,
    ]
    assert _run_cli(rb, ok_args) == 0
    # 同路径再跑 = 拒绝覆盖
    assert _run_cli(rb, ok_args) == rb.EXIT_FAIL
    # 同一 SHA 在一次输入里重复
    rc = _run_cli(rb, [
        "--bank", bank,
        "--clip", f"forehand:{files['forehand_0p80']}:{manifests['forehand_0p80']}",
        "--clip", f"forehand:{files['forehand_0p80']}:{manifests['forehand_0p80']}",
        "--out", os.path.join(tmpdir, "d.npz"), "--manifest", os.path.join(tmpdir, "d.json"),
    ])
    assert rc == rb.EXIT_FAIL
    # 对已重绑过的题库重复登记同一 SHA
    rc = _run_cli(rb, [
        "--bank", out,
        "--clip", f"forehand:{files['forehand_0p80']}:{manifests['forehand_0p80']}",
        "--out", os.path.join(tmpdir, "e.npz"), "--manifest", os.path.join(tmpdir, "e.json"),
    ])
    assert rc == rb.EXIT_FAIL
    # 未知族名
    rc = _run_cli(rb, [
        "--bank", bank,
        "--clip", f"midhand:{files['forehand_0p80']}:{manifests['forehand_0p80']}",
        "--out", os.path.join(tmpdir, "f.npz"), "--manifest", os.path.join(tmpdir, "f.json"),
    ])
    assert rc == rb.EXIT_FAIL


def test_rebind_incremental_second_pass(qb, rb, tmp_path):
    """增量重绑:先登记 0.8 档,再对产物登记 1.2 档 → 列表按序累加,loader 照收。"""
    tmpdir, files, shas, bank, manifests = _rebind_inputs(tmp_path, qb)
    step1 = os.path.join(tmpdir, "s1.npz")
    step2 = os.path.join(tmpdir, "s2.npz")
    assert _run_cli(rb, [
        "--bank", bank,
        "--clip", f"forehand:{files['forehand_0p80']}:{manifests['forehand_0p80']}",
        "--out", step1, "--manifest", os.path.join(tmpdir, "s1.json"),
    ]) == 0
    assert _run_cli(rb, [
        "--bank", step1,
        "--clip", f"forehand:{files['forehand_1p20']}:{manifests['forehand_1p20']}",
        "--out", step2, "--manifest", os.path.join(tmpdir, "s2.json"),
    ]) == 0
    rebound = qb.load_question_bank(step2, expected_split="train")
    assert rebound.metadata["clips"]["forehand"]["motion_sha256_allowed"] == [
        shas["forehand"], shas["forehand_0p80"], shas["forehand_1p20"]
    ]
    # 两轮重绑的来源摘要都留在 meta 里可审(最后一轮覆盖 motion_family_rebind 块,
    # 逐轮完整链在各自的输出 manifest 里)
    assert rebound.metadata["motion_family_rebind"]["inputs"][0]["family"] == "forehand"


def test_rebind_cli_subprocess_smoke(qb, tmp_path):
    """真子进程跑一遍 CLI(退出码 0 + stdout JSON),防 main()/argparse 接线回归。"""
    tmpdir, files, shas, bank, manifests = _rebind_inputs(tmp_path, qb)
    out = os.path.join(tmpdir, "cli.npz")
    proc = subprocess.run(
        [sys.executable, REBIND_SCRIPT_PATH,
         "--bank", bank,
         "--clip", f"backhand:{files['backhand_0p80']}:{manifests['backhand_0p80']}",
         "--out", out, "--manifest", os.path.join(tmpdir, "cli.json")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["status"] == "published" and payload["bank"] == out
    assert os.path.exists(out)


# ------------------------------------------------------------------------------------------- #
# E. hope_commands 消费点接线(源码级卫兵;mdp 包引 isaaclab,不能实例化)
# ------------------------------------------------------------------------------------------- #

def test_hope_commands_bank_consumption_is_family_aware():
    """题库消费点必须走族表:select_questions 吃 bank_clip,两处 runtime 对账带 clip_families,
    事件/考卷安装按 bank_clips 下标。防止后续改动把某一处退回 clip_id 直连。"""
    with open(HOPE_COMMANDS_PATH, encoding="utf-8") as fh:
        source = fh.read()
    assert "def _qb_bank_family_table(" in source
    assert source.count("clip_families=(family_table.tolist() if family_table is not None else None)") == 2, \
        "两处 validate_runtime_motion_contract 调用都必须传 clip_families"
    assert "self._question_bank, bank_clip, torch.rand(n, device=self.device)" in source, \
        "_apply_question_bank_targets 必须用族折算后的 bank_clip 选题"
    assert "bank.demanded_vel[bank_clips, rows]" in source, \
        "_install_event_training_questions 必须按 bank_clips 下标"
    assert "exam_bank.incoming_vel.to(self.device)[bank_clips, rows]" in source, \
        "install_external_exam_questions 必须按 bank_clips 下标"
    # 族表缺席的 legacy 短路必须保留(现役行为逐字节不变)
    assert 'if getattr(getattr(motion, "cfg", None), "clip_family_per_clip", None) is None:' in source
