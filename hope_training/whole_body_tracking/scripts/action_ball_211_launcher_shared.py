#!/usr/bin/env python3
"""A211 / C211 两个发射器之间**逐字相同**部分的共享库(准备件,尚未接线)。

人话
====
`launch_action_ball_a211_four_arm_diagnostic.py`(4770 行)和
`launch_action_ball_c211_diagnostic.py`(4448 行)里有一大堆一模一样的代码:
同样的常量、同样的小工具函数。本文件把其中**经过 difflib 验证、A/C 两边一个字节都不差**的
那部分集中放一处,让下一步两个发射器可以直接 import,不必各抄一份。

**本轮只建库,不改发射器。** 接线(让两个发射器真的 import 本文件)是下一步、
必须在没有其他人并行改这两个文件时单独做。所以现在导入本文件对现役行为零影响。

量化(difflib,A/C 字母折叠后比较;数字取自 A 侧行数)
====================================================
同名模块级函数 56 个:
  - 逐字相同 12 个 / 125 行
  - 只差 A211↔C211 字面量 6 个 / 71 行(不能直接共享,见下"不能共享")
  - 相似度 >= 0.85 但有真实差异 17 个
  - 相似度 < 0.85(真的不同) 21 个
同名模块级常量 108 个:
  - 逐字相同 72 个 / 151 行
  - 只差 A211↔C211 字面量 14 个 / 22 行(**这些恰恰是最不能共享的**,见下)
  - 有真实差异 22 个

本文件搬了什么
==============
1) 50 个**纯字面量/纯路径**常量(123 行)——直接 import 即可,无副作用。
   路径常量靠 ``SCRIPT_DIR = Path(__file__).resolve().parent``,本文件与两个发射器
   同在 ``scripts/`` 目录下,所以 SCRIPT_DIR 取值与发射器里完全一致。
2) 2 个**零依赖**纯函数:``_load_helper`` / ``_whole_body_state_sha256``(25 行)。
3) ``bind(base=..., task_wait=..., four_grid=...)`` —— 需要发射器自己那份
   ``_B`` / ``_W`` / ``_F`` 的工具函数(71 行 + WAIT_SCHEDULE 7 行 + 四宫格转出 7 行)。
4) ``bind_admission(admission)`` —— 需要发射器自己那份 ``_ADMISSION`` 的
   GPU 准入转出(``_verify_gpu_admission`` 18 行 + 10 个一行转出)。
5) ``runtime_sources(...)`` —— 函数体逐字相同但读的全局表 A/C 不同,
   这里改成把表当参数传进来(见下"参数化"一节)。

为什么必须用 bind() 而不是本文件自己去 _load_helper
===================================================
**这是最关键的一条,别绕过去。**
两个发射器都用 ``_load_helper`` 把 ``launch_n1_reward_screen_diagnostic.py``
加载成 ``_B``,但各自登记成不同的 ``sys.modules`` 名字
(``_a211_four_arm_base`` / ``_c211_diagnostic_base``)——也就是**两个互相独立的模块实例**。
``LaunchRefused`` 是在那个文件里定义的类,所以
``a211 的 LaunchRefused is not c211 的 LaunchRefused``。

如果本文件自己再 ``_load_helper`` 一份 ``_B``,就会造出**第三个** ``LaunchRefused`` 类;
发射器里 ``except LaunchRefused:`` 就抓不住本文件抛出的异常 —— 失败会从 fail-closed
变成裸崩,甚至改变退出码语义。
所以本文件**任何**会抛 ``LaunchRefused`` 的函数,都必须通过 ``bind()`` 拿到**调用方自己那份**
``base`` 模块,用 ``base.LaunchRefused`` 抛。本文件在 import 时不加载任何 helper、不碰文件系统。

哪些"看起来一样"但**绝对不能共享**
===================================
下面这 14 个常量在 A/C 折叠比较下相似度 = 1.000,看着像重复,**其实是每条臂的身份**。
实验要求 A/C 各自独立的 lineage / normalizer / checkpoint,共享 helper 不等于共享 lineage。
把它们提到公共模块 = 两条臂身份合流,是严重错误:

  ACTOR_CONTRACT                       action_ball_a211        / action_ball_c211
  ACTOR_NORMALIZER_IDENTITY            ..._a211_actor_norm_v2  / ..._c211_...
  CRITIC_CONTRACT                      action_ball_a211_critic_v1 / ..._c211_...
  CRITIC_NORMALIZER_IDENTITY           ..._a211_critic_norm_v1 / ..._c211_...
  GYM_TASK_ID                          HOPE-...-A211Learnability-... / ...-C211...-...
  TASK_PROFILE_ID / TASK_PROFILE_SOURCE
  RETAINED_TASK_PROFILE_PARENT_SOURCE
  POLICY_RECIPE_FILENAME               a211_..._policy_recipe.json / c211_...
  REWARD_RECIPE_FILENAME               a211_effective_reward_recipe.json / c211_...
  SCALE4096_TERMINAL_ACCEPTANCE_KIND
  _P / _S / _W                         (_load_helper 的 sys.modules 名字必须按臂区分)

另外这些在 difflib 下就已经不同,更不能共享(列出来是免得下一步有人"顺手统一"):
  LINEAGE_KIND      A: action_ball_a211_split_ready_online_question_dr_l0_lineage_v5
                    C: action_ball_c211_direct_ball_split_ready_lineage_v4
  TARGET_SEMANTICS  A: a211_desired_contact_v1
                    C: c211_incoming_ball_p_v_spin_outcome_dense_v1
  MATERIALIZATION_KIND / POLICY_MATERIALIZATION_KIND / RESULT_KIND / CLAIM_KIND /
  SPEC_KIND / ORACLE32_KIND / EXPERIMENT_NAME / TRAINABILITY_CONTRACT / LAUNCHER_SOURCE
  RUNTIME_SOURCE_PATHS  —— A 有 question-cache 一行,C 多了 evidence/live-oracle/
                           reward/mdp-export 五行,而且 C 的 DR_L0_MANIFEST_SOURCE 走 _FRAME0
  BUDGETS / _ADMISSION / FRAME0_RECEIPT_PROBE_SOURCE_PATHS

还有 6 个函数只差 A211↔C211 报错字面量(``_four_grid_cell`` /
``_four_grid_prelong_receipt_pin`` / ``_isaac_four_grid_manifest`` /
``_ordered_terminal_events`` / ``_prelong_semantics_exec_environment`` /
``_validate_four_grid_prelong_receipt``)。它们**没有**放进本文件:
错误消息里的臂名是运维读日志时区分是哪条臂在拒绝的唯一线索,合并要先决定
"把臂名做成参数"还是"允许日志里不写臂名",那是下一步的设计决定,不是这轮的搬运。

参数化(函数体逐字相同,但读的全局不同)
========================================
  _runtime_sources    —— 12 个逐字相同函数之一,但它读模块级 ``RUNTIME_SOURCE_PATHS``,
                         而该表 A/C 不同(相似度 0.866)。本文件给出 ``runtime_sources()``,
                         把表当第三个参数传入,行为等价。
  _verify_gpu_admission —— 也是逐字相同,但依赖各臂自己的 ``_ADMISSION``
                         (相似度 0.964,真有差异)。走 ``bind_admission()``。

下一步接线怎么写(示例,本轮不要动发射器)
==========================================
    SHARED_FILE = SCRIPT_DIR / "action_ball_211_launcher_shared.py"
    _SH = _load_helper("_a211_launcher_shared", SHARED_FILE)   # C 侧换成 _c211_...
    _H = _SH.bind(base=_B, task_wait=_W, four_grid=_F)
    _exact_dict = _H._exact_dict
    _external_pin = _H._external_pin
    canonical_sha256 = _H.canonical_sha256
    _isaac_python_entry = _H._isaac_python_entry
    _update_profile_exec_environment = _H._update_profile_exec_environment
    _update_profile_contract = _H._update_profile_contract
    _termination_contract = _H._termination_contract
    _wait_contract = _H._wait_contract
    WAIT_SCHEDULE = _H.WAIT_SCHEDULE
    _GA = _SH.bind_admission(_ADMISSION)
    _verify_gpu_admission = _GA._verify_gpu_admission
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# 1) 纯常量 —— A/C 逐字相同,50 个 / 123 行。原样搬自
#    launch_action_ball_a211_four_arm_diagnostic.py(与 C211 侧逐字校验一致)。
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

BASE_FILE = SCRIPT_DIR / "launch_n1_reward_screen_diagnostic.py"

ADMISSION_FILE = SCRIPT_DIR / "vendor_v2_gpu_admission.py"

EXACT_GROUP_FILE = SCRIPT_DIR / "exact_process_group.py"

OLD_VALIDATOR_FILE = SCRIPT_DIR / "launch_n1_measured_vendor_v2_diagnostic.py"

FOUR_GRID_FILE = SCRIPT_DIR / "action_ball_211_four_grid_contract.py"

FOUR_GRID_BARRIER_FILE = (
    SCRIPT_DIR / "action_ball_211_four_grid_prelong_barrier.py"
)

PRELONG_GATE_FILE = SCRIPT_DIR / "action_ball_4096x5_prelong_gate.py"

PRELONG_SEMANTICS_FILE = (
    SCRIPT_DIR.parent
    / "source/whole_body_tracking/whole_body_tracking/utils/"
    "action_ball_prelong_semantics.py"
)

TASK_WAIT_FILE = (
    SCRIPT_DIR.parent
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/action_ball_task_wait.py"
)

SCHEMA_VERSION = 2

FRAME0_LIVE_RECEIPT_KIND = "isaac_action_ball_nominal_hold_v1"

PRELONG_SEMANTICS_ENABLE_ENV = (
    "HOPE_ACTION_BALL_4096X5_PRELONG_SEMANTICS"
)

PRELONG_REWARD_RECIPE_SHA_ENV = (
    "HOPE_ACTION_BALL_4096X5_PRELONG_REWARD_RECIPE_SHA256"
)

REWARD_PPO_ECONOMY_ENABLE_ENV = "HOPE_ACTION_BALL_REWARD_PPO_ECONOMY_GATE"

UPDATE_PROFILE_ENV = "HOPE_ACTION_BALL_UPDATE_PROFILE"

UPDATE_PROFILE_JSON_PREFIX = "HOPE_ACTION_BALL_UPDATE_PROFILE_JSON="

_DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS = {
    "left_sole_floor_slack_m": 1.0e-4,
    "right_sole_floor_slack_m": 1.0e-4,
    "left_contact_load_slack_n": 1.0e-1,
    "right_contact_load_slack_n": 1.0e-1,
    "support_margin_slack_m": 1.0e-3,
    "joint_position_slack_rad": 2.0e-2,
    "qdes_slack_rad": 2.0e-2,
    "torque_slack_nm": 2.0,
    "table_clearance_slack_m": 1.0e-2,
    "root_height_slack_m": 2.0e-2,
    "root_tilt_slack_rad": 2.0e-2,
    "collision_slack_m": 5.0e-3,
    "ground_lp_residual_slack": 5.0e-8,
}

ACTOR_WIDTH = 211

CRITIC_WIDTH = 319

ACTION_ID = "take_061_unit04_bh"

ACTION_UID = 5527597793770800

TEACHER_ID = "Take_061_unit04_BH"

PHYSICAL_BALL_SEMANTICS = "analytic_virtual_ball_authoritative_physx_disabled"

REWARD_MATERIALIZATION_PROFILE = "measured_vendor_v2_n1_static_v1"

RECIPE_SENTINEL_POLICY_SHA256 = "0" * 64

POLICY_DT_S = 0.02

COLOCATION_SPEC_KEY = "allow_vendor_v2_colocation"

COLOCATED_STAGES = ("scale4096", "long4096")

MAX_COLOCATED_PROCESSES_PER_GPU = 2

HARD_TERMINATION_UNION = (
    "base_fell_tilt",
    "base_too_low",
    "joint_actual_forbidden",
    "joint_qdes_forbidden",
    "robot_hit_table",
)

STRICT_HARD_TERMINATION_UNION = (
    "joint_actual_forbidden",
    "joint_qdes_forbidden",
)

PHYSICAL_FALL_REASONS = ("base_fell_tilt", "base_too_low")

PHYSICAL_FALL_PHASES = (
    "hidden_wait",
    "revealed_pre_strike",
    "post_strike",
)

TASK_WAIT_STARTED_COUNTER = "task_wait_started_count"

TASK_REVEAL_REACHED_COUNTER = "task_reveal_reached_count"

PROHIBITED_HOLD_REFERENCE_TERMINATIONS = (
    "anchor_pos",
    "anchor_ori",
    "ee_body_pos",
)

BASE_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "launch_n1_reward_screen_diagnostic.py"
)

TRAIN_SOURCE = "hope_training/whole_body_tracking/scripts/train.py"

OLD_VALIDATOR_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "launch_n1_measured_vendor_v2_diagnostic.py"
)

TASK_WAIT_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/action_ball_task_wait.py"
)

KIT_LAUNCHER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/launch_kit_training_locked.sh"
)

FOUR_GRID_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_211_four_grid_contract.py"
)

PRELONG_GATE_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_4096x5_prelong_gate.py"
)

FOUR_GRID_BARRIER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_211_four_grid_prelong_barrier.py"
)

PRELONG_SEMANTICS_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/utils/action_ball_prelong_semantics.py"
)

ACTION_BALL_SAMPLING_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/action_ball_sampling.py"
)

ACTION_BALL_COMMAND_SOURCE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/hope_commands.py"
)

SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

PIN_KEYS = ("path", "sha256")


# 本文件不定义 WAIT_SCHEDULE 的字面构造以外的东西;WAIT_SCHEDULE 需要 _W(按臂命名的
# task-wait 模块实例),见 bind()。下面是 A/C 逐字相同的构造参数。
WAIT_SCHEDULE_KWARGS = {
    "seed": 20260804,
    "min_wait_ticks": 5,
    "max_wait_ticks": 25,
    "episode_horizon_ticks": 500,
    "required_active_ticks": 200,
}


# ---------------------------------------------------------------------------
# 2) 零依赖纯函数 —— A/C 逐字相同,不碰 LaunchRefused,可以直接 import 用。
# ---------------------------------------------------------------------------


def _load_helper(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("cannot import helper %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _whole_body_state_sha256(
    joint_pos: Sequence[float],
    root_pos: Sequence[float],
    root_quat: Sequence[float],
) -> str:
    digest = hashlib.sha256()
    for label, values in (
        ("joint_pos", joint_pos),
        ("root_pos_w", root_pos),
        ("root_quat_wxyz", root_quat),
    ):
        array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
        digest.update(label.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 3) 需要调用方自己那份 helper 模块的工具 —— 走 bind(),保证 LaunchRefused 同一性。
# ---------------------------------------------------------------------------


class BoundLauncherHelpers:
    """把 A/C 逐字相同的工具函数绑到**调用方自己那份** ``_B`` 上。

    属性名刻意与两个发射器里现有的模块级名字一一对应,接线时可以直接
    ``_exact_dict = _H._exact_dict``,所有调用点一个字都不用改。
    """

    def __init__(self, base, task_wait=None, four_grid=None):
        if base is None:
            raise ValueError("bind() requires the caller's own base helper module")
        self._base = base
        self._task_wait = task_wait
        self._four_grid = four_grid

        # LaunchRefused 必须来自调用方那份 base,否则调用方 except 抓不住。
        self.LaunchRefused = base.LaunchRefused

        if task_wait is not None:
            self.WAIT_SCHEDULE = task_wait.ActionBallTaskWaitSchedule(
                **WAIT_SCHEDULE_KWARGS
            ).to_dict()

        if four_grid is not None:
            self.ISAAC_FOUR_GRID_KIND = four_grid.KIND
            self.ISAAC_FOUR_GRID_CELL_IDS = four_grid.CELL_IDS
            self.FORMAL_GRID_STAGE_ORDER = four_grid.FORMAL_STAGE_ORDER
            self.A_BOOTSTRAP_CELL_ID = four_grid.A_BOOTSTRAP_CELL_ID
            self.A_STANDARD_INIT_CELL_ID = four_grid.A_STANDARD_INIT_CELL_ID
            self.C_BOOTSTRAP_CELL_ID = four_grid.C_BOOTSTRAP_CELL_ID
            self.C_STANDARD_INIT_CELL_ID = four_grid.C_STANDARD_INIT_CELL_ID
            self.ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS = (
                four_grid.ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS
            )
            self.ACTOR_INIT_MODE_DEFAULT = four_grid.ACTOR_INIT_MODE_DEFAULT

    # -- 逐字相同:_exact_dict ---------------------------------------------
    def _exact_dict(self, value: Any, keys: Sequence[str], *, name: str) -> dict:
        return self._base._exact_dict(value, tuple(keys), name=name)

    # -- 逐字相同:canonical_sha256 ----------------------------------------
    def canonical_sha256(self, value: Any) -> str:
        return self._base.canonical_sha256(value)

    # -- 逐字相同:_external_pin -------------------------------------------
    def _external_pin(self, value: Any, *, name: str):
        base = self._base
        row = self._exact_dict(value, PIN_KEYS, name=name)
        path = base._absolute_path(row["path"], name="%s.path" % name, must_exist=True)
        base._stable_regular_file(path, name=name)
        digest = base._sha256(row["sha256"], name="%s.sha256" % name)
        if base.sha256_file(path) != digest:
            raise self.LaunchRefused("%s file SHA differs" % name)
        return {"path": str(path), "sha256": digest}, path

    # -- 逐字相同:_isaac_python_entry -------------------------------------
    def _isaac_python_entry(self, value: Any) -> Path:
        base = self._base
        entry = base._absolute_path(value, name="source.isaac_python", must_exist=True)
        try:
            real = entry.resolve(strict=True)
            info = real.stat()
        except OSError as exc:
            raise self.LaunchRefused(
                "source.isaac_python cannot resolve to a real file"
            ) from exc
        if not stat.S_ISREG(info.st_mode) or not os.access(real, os.X_OK):
            raise self.LaunchRefused(
                "source.isaac_python must resolve to an executable regular file"
            )
        return entry

    # -- 逐字相同:_update_profile_exec_environment -------------------------
    def _update_profile_exec_environment(
        self, environ: Mapping[str, str]
    ) -> dict:
        """Pass only the exact diagnostic profiler switch across both execs."""

        value = environ.get(UPDATE_PROFILE_ENV)
        if value is None:
            return {}
        if value not in ("0", "1"):
            raise self.LaunchRefused(
                "%s must be exactly 0 or 1 when set" % UPDATE_PROFILE_ENV
            )
        return {UPDATE_PROFILE_ENV: value}

    # -- 逐字相同:_update_profile_contract ---------------------------------
    def _update_profile_contract(self, environ: Mapping[str, str]) -> dict:
        forwarded = self._update_profile_exec_environment(environ)
        value = forwarded.get(UPDATE_PROFILE_ENV)
        mode = (
            "not_requested"
            if value is None
            else "profile_on_attribution_only"
            if value == "1"
            else "explicit_profiler_off"
        )
        return {
            "environment_variable": UPDATE_PROFILE_ENV,
            "forwarded_value": value,
            "mode": mode,
            "profile_json_prefix": UPDATE_PROFILE_JSON_PREFIX,
            "speed_evidence_eligible": False,
            "gpu_kernel_attribution_claimed": False,
            "gpu_attribution_reason": (
                "host perf-counter spans add no CUDA synchronization and cannot "
                "delimit asynchronous GPU kernels"
            ),
        }

    # -- 逐字相同:_termination_contract ------------------------------------
    def _termination_contract(self) -> dict:
        return {
            "hard_union": list(HARD_TERMINATION_UNION),
            "single_stroke_terminal": "action_ball_single_stroke_complete",
            "finite_horizon_terminal": "time_out",
        }

    # -- 逐字相同:_wait_contract -------------------------------------------
    def _wait_contract(self) -> dict:
        if self._task_wait is None:
            raise ValueError("bind(task_wait=...) is required for _wait_contract")
        return {
            "policy_dt_s": POLICY_DT_S,
            "schedule": dict(self.WAIT_SCHEDULE),
            "in_loop_expansion_prohibited": True,
        }

    # -- 逐字相同函数体,但全局表 A/C 不同 -> 参数化 ------------------------
    def runtime_sources(
        self,
        checkout: Path,
        commit: str,
        source_paths: Sequence,
    ) -> dict:
        """等价于两侧的 ``_runtime_sources``,只是把 RUNTIME_SOURCE_PATHS 传进来。

        A 的表有 question-cache 一行,C 的表多了 evidence / live-oracle /
        reward / mdp-export 四行且 DR_L0_MANIFEST_SOURCE 走 _FRAME0,
        所以表本身**不能**共享,只有遍历逻辑能共享。
        """

        base = self._base
        output = {}
        for relative, label in source_paths:
            normalized, _path = base._verify_tracked_file(
                checkout,
                commit,
                {"path": relative, "sha256": base.sha256_file(checkout / relative)},
                name=label,
            )
            output[label] = normalized
        return output


def bind(base, task_wait=None, four_grid=None) -> BoundLauncherHelpers:
    """给发射器绑一套共享工具。``base`` 必须是调用方自己 _load_helper 出来的 ``_B``。"""

    return BoundLauncherHelpers(base, task_wait=task_wait, four_grid=four_grid)


# ---------------------------------------------------------------------------
# 4) GPU 准入转出 —— 转出语句 A/C 逐字相同,但 _ADMISSION 本身 A/C 不同
#    (相似度 0.964:A 多一个 output_contract_from_payload,peer 列表互指对方),
#    所以 _ADMISSION 的构造必须留在各自发射器里,只有这层转出可以共享。
# ---------------------------------------------------------------------------


class BoundGpuAdmission:
    """把 A/C 逐字相同的 10 个转出 + ``_verify_gpu_admission`` 绑到调用方的 ``_ADMISSION``。"""

    # 注意:``_ADMISSION`` 是 ``_A.VendorV2GPUAdmission`` 的实例,下面取到的是**绑定方法**,
    # 每次属性访问都是新对象。核对等价性时用 ``==`` 而不是 ``is``(底层函数与实例相同即相等)。
    def __init__(self, admission):
        if admission is None:
            raise ValueError("bind_admission() requires the caller's own _ADMISSION")
        self._admission = admission
        self._live_reservations = admission._live_reservations
        self._lock_gpu_admission = admission._lock_gpu_admission
        self._open_gpu_shared_lock = admission._open_gpu_shared_lock
        self._query_gpu_processes = admission._query_gpu_processes
        self._release_reservation = admission._release_reservation
        self._reservation_document = admission._reservation_document
        self._runtime_namespace_receipt = admission._runtime_namespace_receipt
        self._unlock_gpu_admission = admission._unlock_gpu_admission
        self._validate_runtime_gpu_process = admission._validate_runtime_gpu_process
        self._write_reservation = admission._write_reservation

    def _verify_gpu_admission(
        self,
        spec: Mapping[str, Any],
        *,
        phase: str,
        current_namespace: Optional[Path],
        require_current_compute: bool = False,
        proc_root: Path = Path("/proc"),
    ) -> dict:
        return self._admission._verify_gpu_admission(
            spec,
            phase=phase,
            current_namespace=current_namespace,
            require_current_compute=require_current_compute,
            proc_root=proc_root,
            query_gpu_processes=self._query_gpu_processes,
            validate_runtime_gpu_process=self._validate_runtime_gpu_process,
            live_reservations=self._live_reservations,
        )


def bind_admission(admission) -> BoundGpuAdmission:
    """给发射器绑 GPU 准入转出。``admission`` 是调用方自己的 ``_ADMISSION``。"""

    return BoundGpuAdmission(admission)
