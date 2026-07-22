"""判卷链 task-revision 代际修复的机制测试(全部无 GPU、无 Isaac、mock 层面)。

覆盖三件事(2026-07-21 判卷欠账事故,两条 Pod1 真 traceback 实证):
1. revision env 重建:planner_revision run 的导出命令必须整块回搬 env.yaml 持久化的
   planner 配置,并锁零 legacy hold clocks(16:41Z 崩溃 = 基础任务 yaml 的 [0,100]
   在 planner 清零之后又被写回)。
2. 老代际不变:无 planner 旗标的 run,导出命令逐字节不新增任何覆盖。
3. 看门狗:judge.sh 拿 Kit boot 锁带超时,导出子进程组带 15 分钟看门狗,超时
   TERM 自家 setsid 进程组、释放锁、落 .aborted 日志(6 小时占锁事故的结构性修复)。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
JUDGE = ROOT / "scripts" / "judge.sh"

PROFILE = {
    "policy_dt_s": 0.02,
    "min_tts_s": 0.02,
    "max_tts_s": 2.0,
    "max_phase_rate_per_s": 4.0,
    "max_phase_acceleration_per_s2": 20.0,
    "max_deadline_revision_delta_s": 0.25,
    "max_position_revision_delta_m": 0.1,
    "max_velocity_revision_delta_mps": 0.5,
    "max_normal_revision_delta_rad": 0.2,
    "normal_unit_tolerance": 0.0001,
    "early_deadline_tolerance_s": 1e-06,
    "contract_version": "phase_governor_v1",
    "schema_version": 1,
}
MIXTURE = {
    "contract_version": "initial_tts_mixture_v1",
    "components": [
        {"name": "late_stress", "range_s": [0.36, 0.49], "weight": 0.1},
        {"name": "baseline_0p5", "range_s": [0.5, 0.5], "weight": 0.45},
        {"name": "fast_deploy", "range_s": [0.5, 0.9], "weight": 0.4},
        {"name": "broad_arrival", "range_s": [0.9, 1.7], "weight": 0.05},
    ],
}
TTS_RANGE = [0.36, 1.7]
STDS = {
    "planner_revision_position_std_m": 0.02,
    "planner_revision_velocity_std_mps": 0.1,
    "planner_revision_normal_std_rad": 0.05,
    "planner_revision_tts_std_s": 0.08,
}


def _planner_command_fields():
    return {
        "planner_revision_enabled": True,
        "planner_revision_profile": dict(PROFILE),
        "planner_revision_initial_tts_range_s": list(TTS_RANGE),
        "planner_revision_initial_tts_mixture": json.loads(json.dumps(MIXTURE)),
    }


def _contract_planner_block():
    return {
        "enabled": True,
        "revision_schema_version": 1,
        "governor": {
            "contract_version": "phase_governor_v1",
            "schema_version": 1,
            "profile": dict(PROFILE),
            "profile_sha256": "0" * 64,
        },
        "initial_tts_range_s": list(TTS_RANGE),
    }


def _make_run_dir(
    tmp_path: Path,
    *,
    planner: bool,
    contract_planner=None,
    with_contract: bool = True,
    env_mutator=None,
    contract_mutator=None,
) -> Path:
    run_dir = tmp_path / "agibot_a3_hope_virtualball" / "run"
    params = run_dir / "params"
    params.mkdir(parents=True)
    (run_dir / "model_1.pt").touch()
    fh = tmp_path / "fh.npz"
    bh = tmp_path / "bh.npz"
    train_bank = tmp_path / "questions_train.npz"
    exam_bank = tmp_path / "questions_exam.npz"
    for path in (fh, bh, train_bank, exam_bank):
        path.touch()

    motion = {
        "motion_file": [str(fh), str(bh)],
        "allow_legacy_link_origin_velocity": False,
    }
    racket = {
        "strike_phase_per_clip": [0.4, 0.6],
        "question_bank": str(train_bank),
        "face_command": True,
        "face_command_pairing": "shared_plus_y",
    }
    if planner:
        motion.update(_planner_command_fields())
        motion.update(
            {
                "hold_steps_range": [0, 0],
                "stand_start_min_hold": 0,
                "post_swing_min_hold": 0,
                "clip_switch_prob": 0.0,
            }
        )
        racket.update(_planner_command_fields())
        racket.update(STDS)
    env = {
        "commands": {"motion": motion, "racket_target": racket},
        "face_command_obs": True,
        "station_obs": False,
    }
    if env_mutator is not None:
        env_mutator(env)
    # 真 env.yaml 是 yaml dump(1e-06 写成 1.0e-06 才是 float);json.dumps 的 1e-06
    # 会被 YAML 1.1 读成字符串,fixture 必须与真工件同形。
    (params / "env.yaml").write_text(yaml.safe_dump(env), encoding="utf-8")

    if with_contract:
        contract = {
            "schema_version": 3,
            "joint_friction_coefficients": [0.0] * 31,
            "actor_obs_contract": "deploy_parity_face179",
            "actor_obs_total_dim": 179,
            "motion_kinematics_exact": True,
            "face_command_pairing": "shared_plus_y",
        }
        if contract_planner is not None:
            contract["planner_task_revision"] = contract_planner
        if contract_mutator is not None:
            contract_mutator(contract)
        (params / "training_contract.json").write_text(
            json.dumps(contract), encoding="utf-8"
        )
    return run_dir


def _dry_run(run_dir: Path):
    return subprocess.run(
        [
            "bash",
            str(JUDGE),
            str(run_dir),
            str(run_dir / "model_1.pt"),
            "--dry-run",
            "--gpu",
            "0",
            "--task",
            "HOPEPingPongVirtualBall",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


# ---------------------------------------------------------------- ① revision env 重建


def test_revision_run_rebuilds_training_planner_env(tmp_path):
    run_dir = _make_run_dir(tmp_path, planner=True, contract_planner=_contract_planner_block())
    result = _dry_run(run_dir)
    assert result.returncode == 0, result.stdout
    out = result.stdout
    assert "'++task.planner_revision={enabled: true" in out
    assert "task-revision 代际" in out
    # 训练队列必传的单时钟三件套 + 任务同一性,一根不能少(16:41Z 崩溃机理)
    assert "'task.motion.hold_steps_range=[0,0]'" in out
    assert "task.motion.stand_start_min_hold=0" in out
    assert "task.motion.post_swing_min_hold=0" in out
    assert "task.motion.clip_switch_prob=0.0" in out


def test_revision_override_round_trips_the_exact_training_block(tmp_path):
    """发出去的 hydra dict 必须能反解析回 env.yaml 的同一套配置(不多不少不变形)。"""
    run_dir = _make_run_dir(tmp_path, planner=True, contract_planner=_contract_planner_block())
    result = _dry_run(run_dir)
    assert result.returncode == 0, result.stdout
    match = re.search(r"\+\+task\.planner_revision=(.*?)'", result.stdout)
    assert match, result.stdout
    literal = match.group(1)
    assert "'" not in literal
    parsed = yaml.safe_load(literal)
    assert parsed == {
        "enabled": True,
        "profile": PROFILE,
        "initial_tts_range_s": TTS_RANGE,
        "initial_tts_mixture": MIXTURE,
        "position_std_m": 0.02,
        "velocity_std_mps": 0.1,
        "normal_std_rad": 0.05,
        "tts_std_s": 0.08,
    }


# ---------------------------------------------------------------- ② 老代际逐字节不变


@pytest.mark.parametrize("explicit_false", [False, True])
def test_legacy_run_gets_no_new_overrides(tmp_path, explicit_false):
    def _mutator(env):
        if explicit_false:
            env["commands"]["motion"]["planner_revision_enabled"] = False
            env["commands"]["racket_target"]["planner_revision_enabled"] = False

    run_dir = _make_run_dir(tmp_path, planner=False, env_mutator=_mutator)
    result = _dry_run(run_dir)
    assert result.returncode == 0, result.stdout
    assert "++task.planner_revision" not in result.stdout
    assert "task.motion.hold_steps_range" not in result.stdout
    assert "task.motion.stand_start_min_hold" not in result.stdout
    assert "task.motion.clip_switch_prob" not in result.stdout
    assert "legacy/OFF 代际" in result.stdout


# ---------------------------------------------------------------- fail-loud 拒判矩阵


def test_one_sided_planner_flags_fail_loud(tmp_path):
    def _mutator(env):
        env["commands"]["racket_target"]["planner_revision_enabled"] = False

    run_dir = _make_run_dir(
        tmp_path, planner=True, contract_planner=_contract_planner_block(), env_mutator=_mutator
    )
    result = _dry_run(run_dir)
    assert result.returncode != 0
    assert "planner_revision_enabled 不一致" in result.stdout


@pytest.mark.parametrize(
    "drop_key",
    [
        "planner_revision_profile",
        "planner_revision_initial_tts_mixture",
        "planner_revision_position_std_m",
        "planner_revision_tts_std_s",
    ],
)
def test_enabled_planner_with_missing_pieces_fails_loud(tmp_path, drop_key):
    def _mutator(env):
        env["commands"]["racket_target"].pop(drop_key)
        # motion 侧同步删掉,免得先撞「原子块不一致」而不是「缺件」
        env["commands"]["motion"].pop(drop_key, None)

    run_dir = _make_run_dir(
        tmp_path, planner=True, contract_planner=_contract_planner_block(), env_mutator=_mutator
    )
    result = _dry_run(run_dir)
    assert result.returncode != 0
    assert drop_key in result.stdout


def test_motion_and_racket_profile_disagreement_fails_loud(tmp_path):
    def _mutator(env):
        env["commands"]["motion"]["planner_revision_profile"]["max_tts_s"] = 3.0

    run_dir = _make_run_dir(
        tmp_path, planner=True, contract_planner=_contract_planner_block(), env_mutator=_mutator
    )
    result = _dry_run(run_dir)
    assert result.returncode != 0
    assert "原子块" in result.stdout


def test_enabled_planner_with_nonzero_legacy_hold_clock_fails_loud(tmp_path):
    def _mutator(env):
        env["commands"]["motion"]["hold_steps_range"] = [0, 100]

    run_dir = _make_run_dir(
        tmp_path, planner=True, contract_planner=_contract_planner_block(), env_mutator=_mutator
    )
    result = _dry_run(run_dir)
    assert result.returncode != 0
    assert "非 [0,0]" in result.stdout


def test_enabled_planner_without_schema3_contract_backing_fails_loud(tmp_path):
    # (a) schema-3 合同存在但没有 planner_task_revision 块
    run_dir = _make_run_dir(tmp_path / "nokey", planner=True, contract_planner=None)
    result = _dry_run(run_dir)
    assert result.returncode != 0
    assert "硬合同背书" in result.stdout

    # (b) 完全没有合同文件
    run_dir = _make_run_dir(tmp_path / "nofile", planner=True, with_contract=False)
    result = _dry_run(run_dir)
    assert result.returncode != 0
    assert "硬合同背书" in result.stdout


def test_contract_enabled_but_legacy_env_fails_loud(tmp_path):
    run_dir = _make_run_dir(
        tmp_path, planner=False, contract_planner=_contract_planner_block()
    )
    result = _dry_run(run_dir)
    assert result.returncode != 0
    assert "互相矛盾" in result.stdout


def test_contract_profile_drift_fails_loud(tmp_path):
    block = _contract_planner_block()
    block["governor"]["profile"]["max_tts_s"] = 3.0
    run_dir = _make_run_dir(tmp_path, planner=True, contract_planner=block)
    result = _dry_run(run_dir)
    assert result.returncode != 0
    assert "governor profile 与 env.yaml" in result.stdout


# ---------------------------------------------------------------- ③ 锁 + 看门狗


def test_judge_lock_and_watchdog_structure():
    """结构合同:拿锁带超时、导出走 setsid 进程组、锁 fd 不外泄、超时落 .aborted。"""
    subprocess.run(["bash", "-n", str(JUDGE)], check=True)
    script = JUDGE.read_text(encoding="utf-8")
    assert 'flock -w "$JUDGE_LOCK_WAIT_S" -x 8' in script
    assert "\nflock -x 8\n" not in script  # 旧的无限等待拿锁绝不许回潮
    assert "JUDGE_LOCK_WAIT_S=${JUDGE_LOCK_WAIT_S:-900}" in script
    assert "JUDGE_EXPORT_WATCHDOG_S=${JUDGE_EXPORT_WATCHDOG_S:-900}" in script
    assert "export_play.log.aborted" in script
    assert 'kill -TERM -- "-$EXPORT_PID"' in script
    assert "8>&-" in script  # 导出/看门狗子进程绝不继承锁 fd(挂死子进程不许继续占锁)
    lock_open = script.index('exec 8>"$JUDGE_KIT_BOOT_LOCK"')
    lock_acquire = script.index('flock -w "$JUDGE_LOCK_WAIT_S" -x 8')
    export_spawn = script.index("setsid bash -c")
    export_eval = script.index('eval "$EXPORT_CMD"', export_spawn)
    abort_check = script.index('if [ -f "$ABORT_LOG" ]')
    assert lock_open < lock_acquire < export_spawn < export_eval < abort_check


def _make_stub_bin(tmp_path: Path, *, flock_acquire_rc: int, export_python: str) -> dict:
    """搭一套 stub 工具链,把 judge.sh 跑到导出阶段而不碰 GPU/Isaac。

    - flock/setsid:macOS 上没有,pod 上不许真拿全局锁 —— 都用可控 stub。
    - python3:转发到当前测试解释器(env.yaml 解析要 PyYAML)。
    - isaac venv 的 python:由各测试注入(挂死/秒退),mjeval venv 的 python 打依赖行。
    """
    stub = tmp_path / "stubbin"
    isaac_bin = tmp_path / "isaacbin"
    mj_bin = tmp_path / "mjbin"
    for d in (stub, isaac_bin, mj_bin):
        d.mkdir()

    def _script(path: Path, body: str):
        path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
        path.chmod(0o755)

    _script(
        stub / "flock",
        f'[ "$1" = "-u" ] && exit 0\nexit {int(flock_acquire_rc)}\n',
    )
    _script(stub / "setsid", 'exec "$@"\n')
    _script(stub / "python3", f'exec "{sys.executable}" "$@"\n')
    _script(mj_bin / "python", 'echo "onnx=stub onnxruntime=stub"\nexit 0\n')
    _script(isaac_bin / "python", export_python)

    isaac_act = tmp_path / "isaac_activate"
    isaac_act.write_text(f'export PATH="{isaac_bin}:$PATH"\n', encoding="utf-8")
    mj_act = tmp_path / "mj_activate"
    mj_act.write_text(f'export PATH="{mj_bin}:$PATH"\n', encoding="utf-8")

    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{stub}:{env['PATH']}",
            "JUDGE_ISAAC_ENV": str(isaac_act),
            "JUDGE_MJEVAL_ACT": str(mj_act),
            "JUDGE_KIT_BOOT_LOCK": str(tmp_path / "kit_boot.lock"),
            "JUDGE_LOCK_WAIT_S": "5",
            "JUDGE_EXPORT_WATCHDOG_S": "2",
            "JUDGE_EXPORT_KILL_GRACE_S": "1",
        }
    )
    return env


def _run_judge_real(run_dir: Path, env: dict):
    return subprocess.run(
        [
            "bash",
            str(JUDGE),
            str(run_dir),
            str(run_dir / "model_1.pt"),
            "--gpu",
            "0",
            "--task",
            "HOPEPingPongVirtualBall",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=60,
    )


def test_watchdog_aborts_hung_export_and_writes_aborted_log(tmp_path):
    run_dir = _make_run_dir(tmp_path, planner=False)
    env = _make_stub_bin(tmp_path, flock_acquire_rc=0, export_python="sleep 45\n")
    start = time.monotonic()
    result = _run_judge_real(run_dir, env)
    elapsed = time.monotonic() - start
    assert result.returncode != 0
    assert "看门狗超时" in result.stdout
    aborted = list(run_dir.glob("judge/*/export_play.log.aborted"))
    assert len(aborted) == 1, result.stdout
    body = aborted[0].read_text(encoding="utf-8")
    assert "export 超时 2s" in body
    assert "不碰任何训练进程" in body
    # 挂死导出必须在看门狗窗口(2s)+宽限(1s)量级内被打断,不许陪 sleep 45
    assert elapsed < 30, f"watchdog 未生效,耗时 {elapsed:.1f}s"


def test_fast_export_failure_propagates_rc_without_false_abort(tmp_path):
    run_dir = _make_run_dir(tmp_path, planner=False)
    env = _make_stub_bin(tmp_path, flock_acquire_rc=0, export_python="exit 7\n")
    result = _run_judge_real(run_dir, env)
    assert result.returncode != 0
    assert "play.py export-only 失败(rc=" in result.stdout
    assert "看门狗超时" not in result.stdout
    assert list(run_dir.glob("judge/*/export_play.log.aborted")) == []


def test_lock_acquisition_timeout_fails_loud_before_export(tmp_path):
    run_dir = _make_run_dir(tmp_path, planner=False)
    env = _make_stub_bin(tmp_path, flock_acquire_rc=1, export_python="exit 0\n")
    result = _run_judge_real(run_dir, env)
    assert result.returncode != 0
    assert "等 Kit boot 锁" in result.stdout and "超时" in result.stdout
    assert list(run_dir.glob("judge/*/export_play.log")) == []
