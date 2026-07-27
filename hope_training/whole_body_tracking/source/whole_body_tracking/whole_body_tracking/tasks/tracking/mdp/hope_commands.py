"""HOPE-specific command term: racket / base target tracking on top of BeyondMimic.

This adds the HITTER (arXiv:2508.21043) racket-target objective to the BeyondMimic motion
tracker. The base ``MotionCommand`` (in ``commands.py``) drives the imitation reward and owns the
per-env motion clock (``time_steps``). ``RacketTargetCommand`` rides on top of it:

* it samples a *desired* racket state (position, velocity, face normal) and a *desired* base XY
  position each swing — exactly the quantities the model-based planner emits at deploy time via
  ``hope_msgs/RacketCommand`` (position, velocity, normal). No ball is needed in simulation;
  training samples targets, the planner supplies them at runtime.
* it computes the *actual* racket state in simulation by forward kinematics through the fixed
  racket mount ``T_mount`` (wrist -> paddle center), so the reward can compare actual vs desired.
* it derives the strike timing from the motion clip phase, exposing ``time_to_strike`` plus the
  ``pre_strike`` / ``strike_window`` masks that gate the goal rewards.

Per the HOPE racket-tracking prohibition, there is NO measured racket feedback at deploy time:
``r_racket`` is a simulation-only signal; on hardware the policy runs open-loop on racket pose.

HITTER alignment notes (see the project HITTER verification):
* racket *position* is observed relative to the base; racket *velocity* is observed in world.
* HOPE currently also observes desired racket *normal* in the actor so the policy can respond to
  normal targets; actual racket normal remains a privileged simulation-only critic/reward signal.
* swing type is a *sampled* variable used here to (a) flag forehand/backhand and (b) select the
  reference clip; it is not required in the actor observation when separate forehand/backhand
  policies are trained (the HOPE default, reimplement.md step 17).
"""

from __future__ import annotations

import json
import math
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    matrix_from_quat,
    quat_apply,
    quat_mul,
    quat_rotate_inverse,
    sample_uniform,
    yaw_quat,
)

from whole_body_tracking.tasks.tracking.mdp.commands import (
    MotionCommand,
    resolve_clip_family_is_forehand,
)
from whole_body_tracking.tasks.tracking.mdp.planner_revision import (
    InitialTtsMixture,
    PhaseGovernorProfile,
)
from whole_body_tracking.tasks.tracking.mdp.stage1_question_bank import (
    load_question_bank,
    question_id,
    select_questions,
    validate_runtime_motion_contract,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Holds starting beyond this yaw (rad) contribute to the conditioned recovery metric.
# It sits just inside the deploy engage limit, so the metric measures states that matter.
_RECOVERY_START_YAW_THRESHOLD = 0.30

# Absolute, reference-independent balance guards from HOPEDeployParityTerminationsCfg.  A reset
# caused only by anchor/body tracking envelopes is a guard reset, not evidence that the robot fell.
_PHYSICAL_FALL_TERMINATION_TERMS = ("base_fell_tilt", "base_too_low")

_FACE_COMMAND_PAIRINGS = ("shared_plus_y", "legacy_signed_vs_A")

_TARGET_DELAY_TTS_MODES = (
    "live",
    "source_timestamp_compensated",
    "uncompensated",
)

# Initial planner deadline buckets used by the per-update behavior ledger.  The 0.5 s row is a
# real point mass in the configured mixture, so equality is intentionally exact; 0.9 s belongs to
# the middle bucket.  These names are part of the additive stdout-JSON counter interface.
_PLANNER_INITIAL_TTS_BUCKETS = (
    "lt_0p5",
    "eq_0p5",
    "gt_0p5_le_0p9",
    "gt_0p9",
)
_TIMING_BUCKET_SPARSE_EVENTS = (
    "strike_opportunity_count",
    "virtual_capture_count",
    "virtual_legal_return_count",
)


def _face_command_pairing(cfg: "RacketTargetCommandCfg") -> str:
    """Return a validated face-command grading convention.

    ``getattr`` preserves the historical default for small test doubles and old serialized
    configuration objects; real environments also validate the field during command construction.
    """
    pairing = str(getattr(cfg, "face_command_pairing", "shared_plus_y"))
    if pairing not in _FACE_COMMAND_PAIRINGS:
        raise ValueError(
            "face_command_pairing must be one of "
            f"{_FACE_COMMAND_PAIRINGS}, got {pairing!r}"
        )
    return pairing


def _validate_adaptive_sigma_cfg(cfg: "RacketTargetCommandCfg") -> None:
    """adaptive_sigma_normal(拍面第三通道)必须搭在 adaptive_sigma 上,单开 fail-loud。

    人话:normal 通道复用 pos/vel 的同一路 exact-strike 误差 EMA 驱动与更新节拍
    (sigma_update_every / exact_success_decay / sigma_ema_scale)。只开 normal 不开
    adaptive_sigma 时那套驱动根本不跑——sigma 永远不更新,却看起来"配置了自适应",
    这是典型的半配置静默失效,按 fail-loud 纪律在构造期就拒绝。
    """
    if getattr(cfg, "adaptive_sigma_normal", False) and not getattr(cfg, "adaptive_sigma", False):
        raise ValueError(
            "RacketTargetCommandCfg.adaptive_sigma_normal=True requires adaptive_sigma=True: "
            "the normal channel rides the pos/vel exact-strike EMA driver and update cadence; "
            "enabled alone it would silently never update. Enable adaptive_sigma or drop "
            "adaptive_sigma_normal."
        )


def _coupled_transport_mode(cfg: "RacketTargetCommandCfg") -> bool:
    """判定并校验"耦合传输"模式:planner 修订 + 目标延迟合并成一条 mocap→中继延迟流.

    人话:planner_revision_enabled 且 target_delay_steps=d>0 时,不再走旧的
    "只延迟 actor 观测"的 NO-LAUNCH 死路,而是把每步生成的修订元组(pos/vel/normal/tts,
    原子)压进在途环,d 步后才提交给相位调度器——接受记账、调度器看到的 desired_tts、
    actor 可见元组从此消费同一条延迟流(真实 mocap→relay 的语义)。

    fail-loud 的半配置组合:
    * ``target_delay_tts_mode='live'``——耦合传输里 tts 必须随元组一起延迟
      (调度器消费的就是元组里的 desired_tts),"元组晚到、时钟即时"是矛盾的传输语义;
      要么 source_timestamp_compensated(提交时按在途时长补偿,actor 时钟连续),
      要么 uncompensated(显式陈旧 tts 阴性对照)。
    d=0 时返回 False:一切走现役路径,逐字节不变。
    """
    coupled = bool(getattr(cfg, "planner_revision_enabled", False)) and (
        max(int(getattr(cfg, "target_delay_steps", 0)), 0) > 0
    )
    if coupled and _target_delay_tts_mode(cfg) == "live":
        raise ValueError(
            "planner revisions with target_delay_steps > 0 form ONE coupled transport tuple; "
            "target_delay_tts_mode='live' would deliver the tuple late but the clock instantly "
            "— incoherent. Use 'source_timestamp_compensated' (age-compensated at submission, "
            "continuous actor clock) or 'uncompensated' (explicit stale-TTS negative control)."
        )
    return coupled


def _target_delay_tts_mode(cfg: "RacketTargetCommandCfg") -> str:
    """Return the actor-visible time-to-strike delay convention.

    ``live`` is the historical behavior.  The other two modes put TTS in the same delayed planner
    tuple as position/velocity/normal/side; ``source_timestamp_compensated`` then advances the
    source value by the known transport age while ``uncompensated`` is the explicit stale-TTS
    negative control.
    """
    mode = str(getattr(cfg, "target_delay_tts_mode", "live"))
    if mode not in _TARGET_DELAY_TTS_MODES:
        raise ValueError(
            "target_delay_tts_mode must be one of "
            f"{_TARGET_DELAY_TTS_MODES}, got {mode!r}"
        )
    return mode


def face_tracking_pair(command: "RacketTargetCommand") -> tuple[torch.Tensor, torch.Tensor]:
    """Return the measured/target normals used by every face reward and success metric.

    Face-command targets always remain in the bank's A frame. ``shared_plus_y`` compares them to
    raw mount +Y, while the explicit diagnostic ``legacy_signed_vs_A`` reproduces the historical
    signed-measurement/A-target mismatch. Non-face tasks retain the signed clip-reference pair.
    Keeping the selection here makes rewards, privileged observations, and metrics grade the same
    convention.
    """
    if command.cfg.face_command:
        pairing = _face_command_pairing(command.cfg)
        if pairing == "shared_plus_y":
            return command.racket_normal_raw_w, command.target_normal_cmd
        return command.racket_normal_w, command.target_normal_cmd
    return command.racket_normal_w, command.racket_target_normal_w


def _peek_question_bank_clip_order(path) -> tuple:
    """Read a bank npz's own ``clip_order`` without loading it. () when it carries none (legacy).

    人话:题库自己知道它是按哪几个 clip 生成的。以前加载器把 clip 名默认写死成
    ("forehand","backhand"),于是任何别的动作集生成的题库**结构上就装不进来**,唯一的绕法是把
    一个反手 clip 改名叫 "forehand" —— 那会对下游每一个按家族分桶的消费者撒谎。所以这里先问题库。
    """
    try:
        import numpy as _np
        data = _np.load(str(path))
        if "meta_json" not in data:
            return ()
        meta = json.loads(bytes(_np.asarray(data["meta_json"], dtype=_np.uint8)).decode("utf-8"))
        return tuple(str(c) for c in (meta.get("clip_order") or ()))
    except Exception:
        return ()


#: Knobs that STOP DOING ANYTHING once a bank / inline solve produces the target. A run that sets
#: them looks like it is shaping the task while it is not, so setting one is a construction error.
#: NOTE ``racket_pos_range_per_clip`` is deliberately NOT here: under a solved target it is the
#: CONTINUOUS distribution the contact point (= the ball's arrival point) is drawn from, which is
#: still very much alive. Only the VELOCITY box dies, because the velocity is solved.
_DEAD_UNDER_SOLVED_TARGET = (
    ("racket_vel_range_per_clip", None, "the commanded velocity is SOLVED, not a box sample"),
    ("ref_vel_scale", 1.0, "the commanded velocity no longer comes from the reference clip"),
)
#: The reference-perturbation curriculum only has an effect in ``target_mode='reference_perturbed'``
#: — its cfg defaults are non-zero, so flagging it unconditionally would fire on every legal run.
_DEAD_UNDER_SOLVED_TARGET_IF_PERTURB_MODE = (
    ("ref_perturb_pos", (0.0, 0.0, 0.0)),
    ("ref_perturb_vel", (0.0, 0.0, 0.0)),
    ("ref_perturb_normal", 0.0),
    ("ref_perturb_curriculum_steps", 0),
)


def _assert_solved_target_recipe_is_coherent(cfg) -> None:
    """Every 'must be turned on' item in the launch recipe, as a fail-closed check.

    人话:一份必须靠人记住的发射清单本身就是缺陷。这里把清单里每一条变成"错组合当场炸,并且报错
    里直接写该设成什么"——以后哪次会话什么都没读,也照样不可能配错。
    """
    solved = bool(str(getattr(cfg, "question_bank", "") or "").strip()) or \
        str(getattr(cfg, "target_mode", "")) == "solved"
    if not solved:
        return

    # 1. FACE. Without face_command the inverse-solved face is computed, stored — and graded
    #    against by nothing, which is exactly how a system ends up lofting the ball upward
    #    instead of returning it flat while the landing reward notices nothing.
    if not bool(getattr(cfg, "face_command", False)):
        raise ValueError(
            "a solved/banked target computes a demanded racket FACE, but "
            "racket_target.face_command=False means nothing ever grades it — the face becomes a "
            "stored number with no consumer. Set racket.face_command: true (and, if you are "
            "adding the actor lane, racket.face_command_obs: true BEFORE any other observation "
            "term is attached — attaching it afterwards raises an override error that names the "
            "wrong term)."
        )
    pairing = str(getattr(cfg, "face_command_pairing", "shared_plus_y"))
    if pairing not in ("shared_plus_y", "legacy_signed_vs_A"):
        raise ValueError(
            f"racket_target.face_command_pairing={pairing!r} is not a known pairing; use "
            f"'shared_plus_y' (the A-frame convention every bank is generated in)"
        )

    # 2. FACE SIGN. A wrong sign silently inverts which PHYSICAL face is judged to strike.
    signs = tuple(getattr(cfg, "mount_normal_sign_per_clip", ()) or ())
    if not signs:
        raise ValueError(
            "a solved/banked target grades a signed physical face, so "
            "racket_target.mount_normal_sign_per_clip must be declared per clip (one +1/-1 per "
            "loaded clip, in motion_file order). It is absent, so every clip would fall back to "
            f"the scalar mount_normal_sign={getattr(cfg, 'mount_normal_sign', None)} — and a wrong "
            "sign silently inverts which physical rubber face is judged to strike."
        )
    bad = [(i, float(s)) for i, s in enumerate(signs) if float(s) not in (1.0, -1.0)]
    if bad:
        raise ValueError(
            f"racket_target.mount_normal_sign_per_clip entries must be +1 or -1, got {bad} as "
            f"(clip index, value)"
        )

    # 3. DEAD KNOBS. Setting one looks like shaping the task; it is not.
    checks = list(_DEAD_UNDER_SOLVED_TARGET)
    if str(getattr(cfg, "target_mode", "")) == "reference_perturbed":
        checks += [(n, v, "the reference-perturbation curriculum no longer shapes the target")
                   for n, v in _DEAD_UNDER_SOLVED_TARGET_IF_PERTURB_MODE]
    offenders = []
    for name, inert, why in checks:
        if not hasattr(cfg, name):
            continue
        value = getattr(cfg, name)
        if inert is None:
            live = value is not None
        elif isinstance(inert, tuple):
            live = tuple(float(v) for v in (value or inert)) != inert
        else:
            live = float(value) != float(inert)
        if live:
            offenders.append(f"{name}={value!r} ({why}; set it to {inert!r})")
    if offenders:
        raise ValueError(
            "these racket_target knobs are DEAD once a bank / inline solve produces the target, "
            "but this run sets them, so the yaml reads as if they were shaping the task:\n  - "
            + "\n  - ".join(offenders)
            + "\nRemove them (or set the inert value shown) so the config says what it does."
        )

    # 4. HER replay. The solved/banked override runs AFTER the HER block in _sample_targets_uniform,
    #    so every replayed target is clobbered — burned RNG and compute plus a lying
    #    achieved_replay_frac (the DeployParity yaml defaults the mix to 0.30).
    if float(getattr(cfg, "achieved_target_mix_prob", 0.0)) > 0.0:
        raise ValueError(
            f"racket_target.achieved_target_mix_prob={cfg.achieved_target_mix_prob} is "
            f"incompatible with a solved/banked target: the override runs AFTER the HER block in "
            f"_sample_targets_uniform, so every replayed target is clobbered. Set "
            f"racket.achieved_target_mix_prob: 0.0 in the task yaml."
        )
    if float(getattr(cfg, "midswing_resample_prob", 0.0)) > 0.0:
        raise ValueError(
            f"racket_target.midswing_resample_prob={cfg.midswing_resample_prob} is incompatible "
            f"with a solved/banked target: the mid-swing refinement path re-enters the samplers "
            f"(and therefore the solve seam) but never calls the shadow/physical ball's "
            f"on_resample, so the question would be swapped while the ball is still flying to the "
            f"old one. Set racket.midswing_resample_prob: 0.0."
        )

    # 5. CONTINUOUS-ONLY items. target_mode='solved' means the answer is produced INLINE from a
    #    continuous draw. 人话:开关只有这一个,剩下的必须自洽,否则当场炸并写清该设成什么。
    if str(getattr(cfg, "target_mode", "")) != "solved":
        return
    if str(getattr(cfg, "question_bank", "") or "").strip():
        raise ValueError(
            "racket_target.target_mode='solved' produces the target from a CONTINUOUS draw, but "
            "racket_target.question_bank is also set. Two producers cannot both own the atomic "
            "question/ball write. Pick one: drop question_bank (continuous training), or use "
            "target_mode 'uniform'/'reference_perturbed' with the bank. To keep a bank as the "
            "CONTRACT ANCHOR of a continuous run (validated, never trained on) put its path in "
            "racket.cq_anchor_bank instead."
        )
    if getattr(cfg, "cq_vel_range_per_clip", None) is None:
        raise ValueError(
            "racket_target.target_mode='solved' needs racket.cq_vel_range_per_clip — the "
            "INCOMING BALL's velocity box, one (x,y,z) lo/hi triple per loaded clip. That box IS "
            "the run's declared distribution; there is no shared fallback on purpose (a silent "
            "default would make every arm's task different from what its yaml says)."
        )
    if getattr(cfg, "racket_pos_range_per_clip", None) is None:
        raise ValueError(
            "racket_target.target_mode='solved' needs racket.racket_pos_range_per_clip — under a "
            "solved target it is the CONTACT-POINT DRAW BOX (the ball's arrival point), and its "
            "per-clip centre is also the pinned S2a base ready-anchor. It is not dead here; it is "
            "the only thing that says where the ball arrives."
        )
    # 5b. INCOMING BOX SANITY. vb_vel_range_per_clip 有 _assert_incoming_ball_boxes_are_sane 把关,
    #     但连续路径下那个框是死的、真正出球的是 cq_vel_range_per_clip —— 于是"球朝反方向飞"这个
    #     构造期错误在连续臂上完全没人看。同一套规则原样搬过来:-x 才是冲着机器人来。
    for clip_id, clip_rng in enumerate(tuple(cfg.cq_vel_range_per_clip)):
        (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = clip_rng
        if float(x_hi) >= 0.0:
            raise ValueError(
                f"racket_target.cq_vel_range_per_clip: clip {clip_id} has x_hi={float(x_hi):.4f} "
                f">= 0, but the incoming ball travels toward the robot (-x). A non-negative "
                f"ceiling lets the draw launch a ball that flies AWAY, which no stroke can ever "
                f"answer — the solver would just reject them and the accept rate would crater."
            )
        for name, (lo, hi) in (("x", (x_lo, x_hi)), ("y", (y_lo, y_hi)), ("z", (z_lo, z_hi))):
            if float(lo) > float(hi):
                raise ValueError(
                    f"racket_target.cq_vel_range_per_clip: clip {clip_id} axis {name} has "
                    f"lo={float(lo):.4f} > hi={float(hi):.4f} — an empty box"
                )

    aim = getattr(cfg, "cq_aim_xy", None)
    if aim is not None and len(tuple(aim)) != 2:
        raise ValueError(
            f"racket_target.cq_aim_xy must be ONE (x, y) point, got {aim!r}. A per-env aim range "
            f"is refused on purpose: hope_rewards.virtual_landing, virtual_land_err_m and the "
            f"physical ball's return-flight target all grade against the single fixed "
            f"vb_target_x/vb_target_y, so a varying aim would solve for A and score at B with "
            f"nothing asserting. Make _vb_target_xy per-env first, then open this."
        )
    # 瞄点必须就是被打分的那个点。人话:解题时瞄 A、打分时按 B 量,中间只有 10 cm 的闭环回读兜底,
    # 偏 9 cm 也照训不误 —— 那正是"解出来的目标最后被别的东西打分"。所以直接钉死相等。
    if aim is not None:
        graded = (float(cfg.vb_target_x), float(cfg.vb_target_y))
        if tuple(float(v) for v in aim) != graded:
            raise ValueError(
                f"racket_target.cq_aim_xy={tuple(float(v) for v in aim)!r} is not the point this "
                f"run is GRADED at, vb_target_x/vb_target_y={graded!r}. The landing reward, "
                f"virtual_land_err_m and the physical ball's return flight all read vb_target_*, "
                f"so the solver would answer one question and the reward would mark a different "
                f"one. Either drop cq_aim_xy (it defaults to vb_target_*) or move vb_target_*."
            )
    # 契约锚题库:连续臂没有 npz 行可扫,于是 load_question_bank 的物理契约 SHA 和
    # _check_question_bank_face_frame 里的 validate_runtime_motion_contract 两道开机对账全都不跑。
    # 这正是老板说的"必须开的东西要变成默认"——所以它是必填项,不是可选项,不是一行 WARN。
    if not str(getattr(cfg, "cq_anchor_bank", "") or "").strip():
        raise ValueError(
            "racket_target.target_mode='solved' requires racket.cq_anchor_bank — a schema-v3 npz "
            "used ONLY as the contract anchor (never trained on, never installed as a target). "
            "Without it a continuous arm boots with NO physics-contract SHA check, NO runtime "
            "motion contract (motion SHA / seg_len / strike_phase) and NO on-machine "
            "continuous-vs-bank parity gate — the three checks every bank arm gets for free. Set "
            "racket.cq_anchor_bank: cfg/<your clip>_train.npz."
        )
    if float(getattr(cfg, "cq_overdraw", 1.0)) < 1.0:
        raise ValueError("racket_target.cq_overdraw must be >= 1.0 (it is a first-pass margin)")
    if int(getattr(cfg, "cq_buffer_rows", 0)) <= 0:
        raise ValueError("racket_target.cq_buffer_rows must be > 0")


class RacketTargetCommand(CommandTerm):
    """Samples desired racket/base targets and computes the actual racket state by FK."""

    cfg: RacketTargetCommandCfg

    # 类级默认 = 连续题目关。人话:仓库里有一批源码级/纯张量单测用 __new__ 造对象、只塞它们关心
    # 的那几个字段(以前的判据 _question_bank 就是这么塞的),不走 __init__。判据换成 _cq_enabled
    # 之后,那些测试会在读属性时直接 AttributeError —— 门不是失效,是炸掉,而炸掉的门等于没门。
    # 放一个类级 False 兜底,任何没经过 __init__ 的实例都读到"关",默认安全,且以后新增读点不必
    # 逐个记得写 getattr。
    _cq_enabled: bool = False

    def __init__(self, cfg: RacketTargetCommandCfg, env: ManagerBasedRLEnv):
        # Validate even when face_command is currently disabled: a typo must fail at environment
        # construction instead of lying dormant until a later curriculum/override enables it.
        _face_command_pairing(cfg)
        _target_delay_tts_mode(cfg)
        _validate_adaptive_sigma_cfg(cfg)
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # Resolve the racket FK source: prefer the racket body if it survived the physics import,
        # otherwise fall back to (wrist body pose) * (constant mount offset).
        # NOTE: Articulation.find_bodies() RAISES (resolve_matching_names) when a name matches no
        # body — it does not return []. So gate on body_names membership before calling it, or the
        # wrist-offset fallback below becomes unreachable.
        if cfg.racket_body_name in self.robot.body_names:
            self._racket_mode = "body"
            self._racket_body_index = self.robot.find_bodies(cfg.racket_body_name, preserve_order=True)[0][0]
            self._wrist_body_index = -1
        else:
            self._racket_mode = "wrist_offset"
            self._racket_body_index = -1
            if cfg.wrist_body_name not in self.robot.body_names:
                raise ValueError(
                    f"RacketTargetCommand: neither racket body '{cfg.racket_body_name}' nor wrist "
                    f"body '{cfg.wrist_body_name}' found on asset '{cfg.asset_name}'."
                )
            self._wrist_body_index = self.robot.find_bodies(cfg.wrist_body_name, preserve_order=True)[0][0]

        self._mount_offset = torch.tensor(cfg.mount_offset, dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1
        )
        # Fixed wrist->racket rotation (used only in wrist_offset fallback mode so the face normal is
        # taken in the racket frame, not the bare wrist frame). Identity for the A3 mount (all mount
        # joints are rpy=0); set non-identity if your mount tilts the paddle relative to the wrist.
        self._mount_quat = torch.tensor(cfg.mount_quat, dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1
        )

        # The motion command (resolved lazily on first update; not guaranteed to exist at __init__).
        self._motion_term: MotionCommand | None = None
        self._event_timing_bound = False
        # Per-env motion phase at the last target resample; used to detect clip wraps (new swings).
        # Stamped on every resample so a reset-time resample is not double-triggered next step.
        self._prev_motion_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Desired (sampled) targets, world frame.
        self.racket_target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_target_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_target_normal_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_target_normal_w[:, 2] = 1.0
        self.base_target_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
        # R10c 站位锚(franco 2026-07-09 拍板"planner 的 p_base 应该加进去:就算不需要移动,
        # 它也是一个锚")——世界系常数锚点 = env origin + cfg.station_anchor_offset_xy。
        # 固定点阶段 clip 已重落地(frame-0 root xy ≈ origin),所以这就是"出生点常数"。
        # 纯观测用(station_obs 旗标,读 station_anchor_err_b),不进任何奖励;和 base_target_pos_w
        # (带耦合+抖动的奖励目标)是两回事,别混。躯干漂移时这 2 维误差自己变大 = 策略始终有
        # 世界系位置基准(R9a 删缰绳后拍随躯干漂移挥空的任务通道解法)。
        self.station_anchor_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.swing_sign = torch.ones(self.num_envs, device=self.device)

        # --- Stage-1 question bank (fixed contact point / inverse-solved face+velocity answers) ----
        # cfg.question_bank non-empty -> per-swing targets are OVERRIDDEN by a bank row (see
        # _apply_question_bank_targets); empty (default) -> the sampling paths below are untouched.
        # target_normal_cmd ALWAYS exists (zeros when off) so the face-command obs/reward reads are
        # unconditionally safe; it is only ever written when the bank is active.
        self.target_normal_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self._question_bank = None
        # --- CONTINUOUS producer (target_mode: "solved") -------------------------------------
        # 人话:这不是题库。这是一个生产者-消费者缓冲——一次批量解一大批题,游标**不放回**地发,
        # 每道题只发一次、发完就丢,发空了再解一批。统计上和"每次 reset 现场解一次"完全等价,
        # 没有任何一道题被重放;之所以批量,是因为求解成本按调用计(48 行要花 8192 行的 63%),
        # 所以正确的省法是让调用变稀,不是让调用变小。
        # 谁要是把它改成有放回抽取,它就退化成一张 8192 行的滚动题库 —— 也就是 owner 明确否掉的
        # 那个东西 —— 而且没有任何断言会响。别改。
        self._cq_enabled = str(getattr(cfg, "target_mode", "")) == "solved"
        self._cq = None                 # 缓冲状态(懒建:参考拍面/venue 参数在 __init__ 还不存在)
        self._cq_cfg = None
        self._cq_gen = None
        self._cq_anchor = None
        self._cq_boot_gate_done = False
        self._cq_last_exhausted_frac = 0.0
        # clip→题库族查表缓存((nseg,) long;forehand 族=0/backhand 族=1)。只有配了
        # cfg.motion.clip_family_per_clip(6-clip 变速列表)才会建;缺席 = None = 现役行为
        # 逐字节不变(clip_id 直接当题库下标)。见 _qb_bank_family_table。
        self._qb_family_table = None
        # face_command without a bank would leave target_normal_cmd all-zero: the re-anchored
        # racket_normal reward reads cos = <n_fk, 0> = 0 -> a CONSTANT kernel, i.e. the face reward
        # silently dead while looking configured. Loud error instead.
        # The predicate is "will target_normal_cmd actually be WRITTEN", not "is there an npz
        # path". Those two used to be the same sentence, and the difference deadlocked the
        # continuous path: this guard demanded a bank for face_command, while
        # _assert_solved_target_recipe_is_coherent demanded face_command for a solved target, so
        # target_mode='solved' with no npz was structurally unlaunchable.
        if cfg.face_command and not (cfg.question_bank or self._cq_enabled):
            raise ValueError(
                "RacketTargetCommandCfg.face_command=True requires a producer that WRITES "
                "target_normal_cmd — either question_bank (npz path) or target_mode='solved' "
                "(continuous inline solve). Without one, target_normal_cmd stays zeros and the "
                "re-anchored racket_normal reward is silently dead. Set racket.question_bank, or "
                "racket.target_mode: solved, or drop face_command."
            )
        # (The continuous producer cannot collide with hitter_pure: both live in target_mode, so
        # 'solved' and 'hitter_pure' are mutually exclusive by construction. The seam is also not
        # called from _sample_targets_hitter_pure, which is why the bank is refused here.)
        if cfg.question_bank and cfg.target_mode == "hitter_pure":
            raise ValueError(
                "RacketTargetCommandCfg.question_bank is incompatible with "
                "target_mode='hitter_pure': HitterPure samples a station-relative target, while "
                "the Stage-1 bank defines a fixed contact point and an atomic incoming-ball/answer "
                "row. Use target_mode='uniform' or 'reference_perturbed' for the bank."
            )
        if (cfg.question_bank or self._cq_enabled) and float(cfg.midswing_resample_prob) > 0.0:
            raise ValueError(
                "a solved/banked racket target (question_bank, or target_mode='solved') is "
                "incompatible with midswing_resample_prob>0: a redraw would replace the "
                "incoming-ball/answer pair without rescheduling the physical/shadow ball (the "
                "on_resample hooks only exist in _resample_command). Disable mid-swing redraws."
            )
        # GATE: every "you must remember to set this" item in the launch recipe.
        # 人话:靠人记住的清单迟早会漏。凡是"必须打开才对"的开关,要么变成默认值,要么把错误的
        # 组合当场拒掉并把该设什么写进报错里。下面这一坨就是把整张清单变成后者。
        _assert_solved_target_recipe_is_coherent(cfg)
        # GATE: landing/net rewards with NO solved answer behind the command.
        # 人话:虚拟球的 landing/pass_net 奖励只有在"命令的速度确实能把球送上台"时才和回球率同向。
        # 命令速度来自 uniform 球箱、又没有题库(逆解答案)时,两者是反的——实测 88% 的成功反手
        # 击球没通过速度阈值,因为它们靠"违抗命令速度"才把球打回去。这不是超参,是任务构造错误,
        # 所以当场拒绝,除非显式 allow_unbanked_landing_rewards=True(做对照臂时用)。
        # 判据是"命令的速度是不是被逆解出来的",不是"有没有 npz 路径"。连续臂天然满足,所以它
        # 不必征用 allow_unbanked_landing_rewards —— 那个旗标保住它原本"负对照臂"的含义。
        if (
            bool(cfg.virtual_ball)
            and not str(cfg.question_bank or "").strip()
            and not self._cq_enabled
            and not bool(getattr(cfg, "allow_unbanked_landing_rewards", False))
        ):
            raise ValueError(
                "racket_target.virtual_ball=True pays the landing/net reward terms "
                "(hope_rewards.virtual_landing / virtual_pass_net) but no question_bank is "
                "loaded, so the COMMANDED racket velocity is a uniform box sample that was never "
                "solved to land the ball. That construction makes obeying the velocity command "
                "anti-correlated with returning the ball. Load a question bank, or set "
                "racket_target.target_mode: solved (continuous inline solve), or set "
                "racket_target.allow_unbanked_landing_rewards=True to run it deliberately as a "
                "control arm."
            )
        if cfg.question_bank:
            # CLIP NAMES COME FROM THE BANK. The loader defaults clip_names to exactly
            # ("forehand","backhand"), which makes a bank built over any other clip set
            # STRUCTURALLY UNLOADABLE — and the only "workaround" is renaming a backhand clip to
            # "forehand", which then lies to every family-scoped consumer downstream. So: read the
            # bank's own clip_order, cross-check it against racket_target.clip_names_per_clip when
            # that is declared, and index the per-clip name map from it.
            _bank_order = _peek_question_bank_clip_order(cfg.question_bank)
            _declared = tuple(str(n) for n in (getattr(cfg, "clip_names_per_clip", ()) or ()))
            if _bank_order and _declared and tuple(_bank_order) != _declared:
                raise ValueError(
                    f"question bank {cfg.question_bank!r} was built over clips "
                    f"{list(_bank_order)} but racket_target.clip_names_per_clip declares "
                    f"{list(_declared)}. clip_index is positional — align them; do NOT rename a "
                    f"clip to make the loader happy (that lies to every family-scoped consumer)"
                )
            _names = tuple(_bank_order) if _bank_order else (
                _declared if _declared else ("forehand", "backhand"))
            if not _declared and _bank_order:
                # The bank is the authority when the YAML said nothing: adopt its clip order so
                # per-clip metric buckets are named after the clips actually being trained.
                self._metric_buckets_per_clip = len(_names) > 0
                self._clip_names = {i: n for i, n in enumerate(_names)}
                self._metric_bucket_rows_t = None
            self._question_bank = load_question_bank(
                cfg.question_bank,
                device=self.device,
                clip_names=_names,
                allow_legacy=bool(cfg.question_bank_allow_legacy),
                expected_split=(None if cfg.question_bank_allow_legacy else "train"),
            )
            self.metrics["question_difficulty_deg"] = torch.zeros(self.num_envs, device=self.device)
            # A-frame (+Y) guard runs lazily on the first bank application (the reference strike
            # state is unresolved at __init__) — see _check_question_bank_face_frame.
            self._qb_face_frame_checked = False
            if cfg.face_command:
                # Demanded-face tracking error (deg) in the face_command frame (raw +Y vs bank
                # target). racket_normal_error_deg cannot see the M3c 单翻病: it measures vs the
                # clip reference face and is flip-INVARIANT under the sign table (both sides carry
                # the sign). This metric closes that observation blind spot in the training ledger.
                self.metrics["face_cmd_normal_error_deg"] = torch.zeros(self.num_envs, device=self.device)
            # HER achieved-target replay is bank-incompatible: the bank override runs AFTER the HER
            # block in _sample_targets_uniform, so every replayed target would be clobbered — burned
            # RNG/compute and a lying achieved_replay_frac (the DeployParity yaml defaults the mix
            # to 0.30). Forced off, once, loudly; the HER block is also hard-gated on the bank.
            if float(cfg.achieved_target_mix_prob) > 0.0:
                raise ValueError(
                    f"racket_target.achieved_target_mix_prob={cfg.achieved_target_mix_prob} is "
                    f"incompatible with a question bank / solved target: the bank override runs "
                    f"AFTER the HER block in _sample_targets_uniform, so every replayed target is "
                    f"clobbered — burned RNG and compute plus a lying achieved_replay_frac. This "
                    f"used to be force-zeroed with a print, which meant a shipped task yaml could "
                    f"keep saying 0.30 forever. Set racket.achieved_target_mix_prob: 0.0 in the "
                    f"task yaml."
                )
            # S2a base pin: per-clip ready-anchor XY offset, evaluated ONCE from the bank's FIXED
            # contact point through the same coupling as the per-question path (built lazily in
            # _qb_base_anchor_off_xy — the reference reach offset needs the motion term, which is
            # unresolved at __init__).
            self._qb_base_anchor = None
            print(
                f"[RacketTargetCommand] stage-1 question bank {cfg.question_bank}: "
                f"questions per clip = {self._question_bank.counts.tolist()}",
                flush=True,
            )

        if self._cq_enabled:
            # Same metric buckets the bank path registers — question_difficulty_deg would otherwise
            # freeze at its init value and the reset-mean "question difficulty mix" readout becomes
            # a lie. The continuous path recomputes it as angle(demanded face, raw clip face), which
            # is the SAME quantity the bank stores (reproduced to <= 3.5 deg on the shipped banks).
            self.metrics["question_difficulty_deg"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["question_exhausted_frac"] = torch.zeros(self.num_envs, device=self.device)
            if cfg.face_command:
                self.metrics["face_cmd_normal_error_deg"] = torch.zeros(
                    self.num_envs, device=self.device)
            self._qb_base_anchor = None
            self._qb_face_frame_checked = True   # 连续路径没有 npz 行可扫,门在求解时就闭合了
            print(
                f"[RacketTargetCommand] CONTINUOUS questions ON (target_mode=solved): "
                f"buffer {int(cfg.cq_buffer_rows)} rows/clip, overdraw {float(cfg.cq_overdraw)}, "
                f"n_iters {int(cfg.cq_n_iters)}, tol {float(cfg.cq_tol_m)} m, "
                f"speed_budget {float(cfg.cq_speed_budget)} m/s (deploy pp_policy gate; the shipped "
                f"banks were generated at 4.0), spin_abs_max {float(cfg.cq_spin_abs_max)} rad/s, "
                f"exam_holdout={bool(cfg.cq_exam_holdout)}, "
                f"anchor_bank={cfg.cq_anchor_bank or '<none>'}",
                flush=True,
            )

        # Actual racket state, world frame (from FK).
        self.racket_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.racket_quat_w[:, 0] = 1.0
        self.racket_lin_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_normal_w = torch.zeros(self.num_envs, 3, device=self.device)
        # Raw (+Y, unsigned) twin of racket_normal_w — the face_command reward frame (_face_pair).
        # Same unit-+Z pre-FK init as the signed twin (both are overwritten by the first
        # _compute_racket_state; a degenerate zero "normal" would read as a silent 90°).
        self.racket_normal_raw_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_normal_raw_w[:, 2] = 1.0
        self.racket_normal_w[:, 2] = 1.0

        # --- Tier-1 virtual incoming ball (rewardDesign.md): per-swing sampled incoming state and
        # the at-strike outcome caches read by the one-shot virtual_* reward terms. vb_fired is
        # recomputed EVERY step in _update_metrics (true only on a gated exact-strike frame), so the
        # cached outcome is consumed exactly once per swing. Buffers exist even when the feature is
        # off (all-zero / all-False) so the reward terms are safely inert.
        self.vb_vel_in_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.vb_spin_in_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.vb_fired = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.vb_landing_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.vb_landing_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.vb_on_opponent = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.vb_depth_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.vb_net_z = torch.zeros(self.num_envs, device=self.device)
        self.vb_net_clear = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.vb_net_crossed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.vb_topspin = torch.zeros(self.num_envs, device=self.device)
        self.vb_spin_out_norm = torch.zeros(self.num_envs, device=self.device)
        self._vb_params = None  # lazy venue-yaml load on first evaluation
        # Derived table landmarks (env frame), from geometry.py ITTF constants.
        from whole_body_tracking.tasks.table_tennis import geometry as _tt_geom

        self._vb_net_x = float(cfg.vb_table_near_x) + _tt_geom.NET_X
        self._vb_far_x = float(cfg.vb_table_near_x) + _tt_geom.TABLE_LENGTH
        self._vb_half_w = _tt_geom.TABLE_WIDTH / 2.0
        self._vb_net_top_z = float(cfg.vb_table_surface_z) + _tt_geom.NET_HEIGHT
        self._vb_ball_r = _tt_geom.BALL_RADIUS
        self._vb_target_xy = torch.tensor(
            [float(cfg.vb_target_x), float(cfg.vb_target_y)], device=self.device
        )
        # Sample-weighted EMA accumulators (same decay/min-count discipline as the exact-strike block).
        self._vb_exact_acc = 0.0
        self._vb_hit_acc = 0.0
        self._vb_net_acc = 0.0
        self._vb_land_valid_acc = 0.0
        self._vb_inb_acc = 0.0
        # Per-clip (forehand/backhand) accumulators for the PRIMARY in-training metric
        # virtual_return_rate (franco 2026-07-06 "反手先行" needs the per-side number).
        # Literal keys: self._clip_names ({0:"forehand",1:"backhand"}) is defined later in __init__.
        self._vb_exact_acc_c = {0: 0.0, 1: 0.0}
        self._vb_inb_acc_c = {0: 0.0, 1: 0.0}
        self._vb_hit_acc_c = {0: 0.0, 1: 0.0}   # per-side 击球率 (franco 2026-07-08 报数格式)

        # Reference racket state at the strike frame (CONSTANT per clip): pos (env-origin relative),
        # world linear velocity, and face normal, computed by the SAME FK as the actual racket
        # (_compute_racket_state) but fed the reference MOTION's body poses. Used by the
        # "reference_perturbed" target mode so a sampled target is one the imitated swing can actually
        # reach (a perfect imitator hits it exactly). Cached lazily on first resample, after the motion
        # term is resolved and its motion_file is loaded.
        self._ref_strike_cached = False
        # Unified multi-clip: per-clip strike phase as a [num_segments] tensor (built lazily once the
        # motion term is resolved). None until then; falls back to the scalar strike_phase.
        self._strike_phase_per_clip_t = None
        # 每 clip 的击球面符号表([num_segments] 张量,懒构建:第一次用到时按加载的 clip 数 fail-loud
        # 校验后落地)。None = cfg 表为空 = 全部 clip 用标量 mount_normal_sign(现役行为,逐位不变)。
        # 病根见 cfg.mount_normal_sign_per_clip 的注释:正反手用拍子相反的两面,单一符号钉死反手拍面。
        self._mount_sign_per_clip_t = None
        # 每 clip 的"是不是正手家族"bool 表([num_segments] 张量,懒构建:第一次查表时从 motion 术语的
        # cfg.clip_family_per_clip 解析,fail-loud;见 _clip_family_is_forehand)。None = 还没解析。
        # 用途(spdmix v2 硬绑定一):swing_sign / uniform 目标 y 侧 / 事件与考卷安装原来四处写死
        # ``clips == 0`` = 正手,6-clip 变速列表里正手 1.0/1.2 变体会被误判成反手;现在全部按这张表取。
        # 缺席配置时表按"单 clip 正手 / 恰 2 clip = (正手, 反手)"推导,与写死判断逐字节同值。
        self._family_is_forehand_t = None
        # 家族行号表([num_segments] long,0=正手族/1=反手族,懒构建自 _clip_family_is_forehand)。
        # 所有"按族分桶"的下标(HER 缓存、per-族指标、每族一行的框表展开)单一来源。
        self._clip_family_rows_t = None
        # cfg per-clip 框表按加载 clip 数对齐后的缓存(键 = cfg 键名;见 _per_clip_range_rows)。
        self._per_clip_range_rows_cache = {}
        # Per-clip reference paddle FACE NORMAL at the strike frame ([num_segments, 3], built lazily). In
        # uniform mode the target normal is set to the imitated swing's actual paddle normal (which the
        # policy can achieve) — NOT the racket-velocity direction, which is ~18-110 deg off the +Y blade
        # face and makes the normal goal (and thus the composite success metric) unsatisfiable.
        self._ref_normal_per_clip = None
        # Optional per-clip racket target-velocity boxes (uniform mode). Built ONCE from cfg; stays None
        # when the shared box is used (backward compatible). Shape (num_clips, 3, 2): [clip][x/y/z][lo/hi].
        self._vel_range_per_clip_t = None
        if self.cfg.racket_vel_range_per_clip is not None:
            self._vel_range_per_clip_t = torch.tensor(
                [[[float(lo), float(hi)] for (lo, hi) in clip_rng]
                 for clip_rng in self.cfg.racket_vel_range_per_clip],
                dtype=torch.float32,
                device=self.device,
            )
            self._assert_target_velocity_points_forward()
        # Optional per-clip racket target-POSITION boxes (uniform mode). Same shape/semantics as the
        # velocity one above; None -> shared pos box (backward compatible). (num_clips, 3, 2): [clip][x/y/z][lo/hi].
        self._pos_range_per_clip_t = None
        if self.cfg.racket_pos_range_per_clip is not None:
            self._pos_range_per_clip_t = torch.tensor(
                [[[float(lo), float(hi)] for (lo, hi) in clip_rng]
                 for clip_rng in self.cfg.racket_pos_range_per_clip],
                dtype=torch.float32,
                device=self.device,
            )
            for _clip_id, _clip_rng in enumerate(self.cfg.racket_pos_range_per_clip):
                self._assert_contact_clears_table(
                    _clip_id,
                    self._commanded_target_x_hi(float(_clip_rng[0][1])),
                    float(_clip_rng[2][0]),
                    "racket_pos_range_per_clip",
                )
        # PER-CLIP INCOMING-BALL regime (a block gets fast balls, a loop slow ones, same run). Same
        # (num_clips, 3, 2) shape as the racket boxes; None -> the shared vb_vel_*_range box.
        self._vb_vel_range_per_clip_t = None
        if getattr(self.cfg, "vb_vel_range_per_clip", None) is not None:
            self._vb_vel_range_per_clip_t = torch.tensor(
                [[[float(lo), float(hi)] for (lo, hi) in clip_rng]
                 for clip_rng in self.cfg.vb_vel_range_per_clip],
                dtype=torch.float32,
                device=self.device,
            )
            self._assert_incoming_ball_boxes_are_sane()
        self._vb_spin_abs_max_per_clip_t = None
        if getattr(self.cfg, "vb_spin_abs_max_per_clip", None) is not None:
            _spin_rows = tuple(float(s) for s in self.cfg.vb_spin_abs_max_per_clip)
            _bad_spin = [(i, s) for i, s in enumerate(_spin_rows) if not s >= 0.0]
            if _bad_spin:
                raise ValueError(
                    f"vb_spin_abs_max_per_clip must be non-negative (rad/s); got {_bad_spin} as "
                    f"(clip index, value)"
                )
            self._vb_spin_abs_max_per_clip_t = torch.tensor(
                _spin_rows, dtype=torch.float32, device=self.device
            )
        self._ref_racket_pos_rel = torch.zeros(3, device=self.device)
        self._ref_racket_vel_w = torch.zeros(3, device=self.device)
        self._ref_racket_normal_w = torch.zeros(3, device=self.device)
        # Reference base (root) XY at the strike + the base->racket horizontal offset. Used to COUPLE
        # base_target to racket_target so standing at base_target keeps the racket reachable.
        self._ref_base_pos_rel = torch.zeros(3, device=self.device)
        self._ref_reach_offset_xy = torch.zeros(2, device=self.device)
        self._ref_racket_pos_rel_per_clip = None
        self._ref_racket_vel_w_per_clip = None
        self._ref_racket_normal_w_per_clip = None
        self._ref_racket_normal_raw_w_per_clip = None
        self._ref_base_pos_rel_per_clip = None
        self._ref_reach_offset_xy_per_clip = None

        # Success-gated curriculum: running perturbation scale, advanced only when the smoothed
        # exact-strike composite success clears the threshold (see _perturb_scale / _update_metrics).
        self._curr_perturb_scale = float(cfg.ref_perturb_curriculum_start)

        # Decayed accumulators for the CONDITIONAL exact-strike pass rates (see _update_metrics). The
        # logged strike_*_pass_exact / strike_composite_success_exact report the fraction of *exact-strike
        # samples* that clear each acceptance threshold — NOT a per-env value held over the long
        # non-strike portion of every episode (which diluted the old metric ~10x at reset-logging time).
        # Sample-weighted EMA: acc = decay*acc + this-step-count; rate = pass_acc / sample_acc.
        self._exact_n_acc = 0.0
        self._exact_pass_comp_acc = 0.0
        self._exact_pass_pos_acc = 0.0
        self._exact_pass_vel_acc = 0.0
        self._exact_pass_normal_acc = 0.0
        # 5 cm / 10 cm position-accuracy buckets on the SAME exact-strike mask + EMA denominator as the
        # pass metrics above (so they are comparable with the composite, unlike the old window-exit-held
        # strike_success_5cm/10cm which sampled the racket ~0.26 m past target).
        self._exact_pass_5cm_acc = 0.0
        self._exact_pass_10cm_acc = 0.0
        self._exact_composite_rate = 0.0
        # P2.3 adaptive-sigma driver: global decayed error-magnitude sums over exact-strike samples
        # (mean = sum / _exact_n_acc). Per-clip variants exist below; sigma needs one global signal.
        self._exact_pos_err_sum = 0.0
        self._exact_vel_err_sum = 0.0
        # 拍面(normal)通道的全局衰减误差和(弧度)——与 pos/vel 完全平行的第三路驱动。
        # 无条件累加(纯内部标量,不进任何输出),只有 adaptive_sigma_normal=True 时才被读。
        self._exact_nrm_err_sum = 0.0
        # Live adaptive sigmas (start at the cfg maxima = the hand-tuned YAML stds; only applied to
        # the reward terms when cfg.adaptive_sigma is on).
        self._adaptive_sigma_pos = float(cfg.sigma_pos_max)
        self._adaptive_sigma_vel = float(cfg.sigma_vel_max)
        # 第三通道(拍面法向)sigma:同样从 cfg 最大值起步(=YAML 手调 std 的 2x 验收宽度),
        # 只有 adaptive_sigma_normal=True 时才会被更新/落到奖励项上。
        self._adaptive_sigma_normal = float(getattr(cfg, "sigma_normal_max", 0.52))

        # Per-clip (forehand=clip 0 / backhand=clip 1) breakdown of the exact-strike metrics, so wandb
        # shows each swing separately (the aggregate composite can hide one swing lagging). Same
        # sample-weighted EMA as the global accumulators above, but each clip's exact-strike samples are
        # accumulated separately (selected by the motion command's clip_id). Populated in multiseg only.
        # 键 = 指标桶行号(_metric_bucket_rows)。cfg.clip_names_per_clip 为空(现役所有在跑臂)
        # 时桶 = 家族行号(0=正手族, 1=反手族),legacy 2-clip 下与 clip_id 恒等,逐字节不变;
        # 配了 clip_names_per_clip 时桶 = clip_id,每个动作一个桶——否则 fh_loop 和 fh_block_syn
        # 共用一个桶,挡球的失败会藏在拉球的数字里(正手回球率 0.0000 藏在 45% 合计里的老病)。
        self._family_names = {0: "forehand", 1: "backhand"}
        _names_cfg = tuple(str(n) for n in (getattr(cfg, "clip_names_per_clip", ()) or ()))
        if _names_cfg and len(set(_names_cfg)) != len(_names_cfg):
            raise ValueError(
                f"racket_target.clip_names_per_clip has duplicate name(s) {_names_cfg} — the "
                f"names key per-clip metric buckets, so two clips sharing a name would silently "
                f"merge their return rates"
            )
        self._metric_buckets_per_clip = bool(_names_cfg)
        self._clip_names = (
            {i: n for i, n in enumerate(_names_cfg)} if _names_cfg else dict(self._family_names)
        )
        self._metric_bucket_rows_t = None
        # Non-decayed, per-PPO-update eligibility ledger for sparse virtual-ball outcomes.  The
        # existing virtual_* rates are EMAs and deliberately suppress values before their
        # denominators warm up; they therefore cannot tell a milestone classifier whether a zero
        # means "failed" or "the reward was never eligible".  These integer counters book the
        # exact same masks as _vb_book_strike_step, globally and per action family, and are consumed
        # exactly once by MotionOnPolicyRunner.  They do not enter observations, rewards, resets or
        # sampling.
        _sparse_names = (
            "strike_opportunity_count",
            "virtual_capture_count",
            "virtual_net_clear_count",
            "virtual_landing_valid_count",
            "virtual_legal_return_count",
        )
        self._sparse_reward_eligibility_counters = {
            name: torch.zeros((), dtype=torch.long, device=self.device)
            for name in _sparse_names
        }
        for _family in self._clip_names.values():
            for _name in _sparse_names:
                self._sparse_reward_eligibility_counters[f"{_name}_{_family}"] = torch.zeros(
                    (), dtype=torch.long, device=self.device
                )
        # Exact per-PPO-update behavior ledger.  Event counters and ready-phase denominators are
        # integer, sums are float64, and nothing decays.  The runner consumes this transaction
        # together with the sparse strike ledger once per update, so two disjoint 100-update
        # windows can be reconstructed by summing their records (unlike the historical EMAs).
        self._exact_behavior_decision_counters = {}
        self._ensure_exact_behavior_decision_counters()
        self._exact_n_acc_c = {c: 0.0 for c in self._clip_names}
        self._exact_pass_pos_acc_c = {c: 0.0 for c in self._clip_names}
        self._exact_pass_vel_acc_c = {c: 0.0 for c in self._clip_names}
        self._exact_pass_normal_acc_c = {c: 0.0 for c in self._clip_names}
        self._exact_pass_comp_acc_c = {c: 0.0 for c in self._clip_names}
        self._exact_pos_err_sum_c = {c: 0.0 for c in self._clip_names}
        self._exact_vel_err_sum_c = {c: 0.0 for c in self._clip_names}
        self._exact_nrm_err_sum_c = {c: 0.0 for c in self._clip_names}

        # --- UNCONDITIONAL swing accounting (Phase A wandb fix) ------------------------------------
        # strike_composite_success_exact is CONDITIONAL: its denominator is exact-strike SAMPLES, so
        # an env that falls BEFORE the strike frame contributes nothing — composite ~1.0 coexists
        # with any pre-strike fall rate (exactly what happened in deploy). These accumulators count
        # every swing START (episode reset or clip wrap assigns a new swing) with the same EMA decay
        # as the exact accumulators, so:
        #   swing_completion_rate = exact_n_acc / swing_starts_acc   (unconditional; falls count)
        #   pre_strike_fall_rate  = pre-strike terminations / swing starts
        self._swing_starts_acc = 0.0
        self._swing_starts_acc_c = {c: 0.0 for c in self._clip_names}
        # --- Per-swing SAME-LEDGER rally accounting (metric-sync fix, 取证定案 2026-07-08) ----------
        # Disease: virtual_return_rate_rally's numerator (_vb_inb_acc) books at the exact-strike
        # frame and decays ONLY on strike-carrying steps (inside _vb_evaluate), while its
        # denominator (_swing_starts_acc) books at swing START and decays EVERY step — two ledgers
        # with different decay schedules plus a ~1-swing (~116-step) booking phase lag. When 4096
        # envs form a synchronized reset queue (same-instant resume + low fall rate -> mass
        # timeout; episode_length sawtooth 52->485), the ratio oscillates 0.31->1.48 and breaks 1.
        # Cure (per-swing 同刻入账): the exact-strike frame only LATCHES "this swing produced a
        # legal return" (_rally_returned); at the swing's END (wrap or true reset — i.e. inside
        # _count_swing_starts) the ended attempt books its start AND its returned flag TOGETHER,
        # into accumulators decayed TOGETHER once per step in _update_metrics. Paired bookings on
        # one ledger => returns can never outrun starts => the rate is <=1 by construction and
        # equals the true per-swing return rate under ANY reset synchronization. The old mixed-
        # ledger readout survives one transition period as *_legacy (cfg.rally_legacy_metrics)
        # for new/old comparison. 人话:改成"每拍打完才记账,回没回球和这一拍同时入同一本账"。
        self._rally_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._rally_returned = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Wrap-boundary parking latch (防御修 2026-07-09 取证副产品): a legal return whose strike
        # fires on the SAME step a clip wraps belongs to the swing STARTING at that wrap (the motion
        # term wraps before this term's metrics pass), but the OLD attempt is still unbooked at that
        # moment. _vb_book_strike_step parks such returns here; _count_swing_starts books the ended
        # attempt first, then hands the parked latch to the new attempt that owns it.
        self._rally_pending_return = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._rally_starts_acc = 0.0
        self._rally_returns_acc = 0.0
        self._rally_starts_acc_c = {c: 0.0 for c in self._clip_names}
        self._rally_returns_acc_c = {c: 0.0 for c in self._clip_names}
        # Exact decision-window swing ledger.  Starts and exact-strike opportunities occur at
        # different phases of an attempt, so their counts may legitimately straddle a PPO-window
        # boundary.  Latch completion during the attempt and book numerator+denominator together
        # only when that attempt closes (wrap or true reset).  This makes every independently
        # aggregated window satisfy completion <= outcome without clipping or boundary slack.
        self._exact_attempt_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._exact_attempt_completed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # A defensive wrap-step exact strike belongs to the newly starting attempt because Motion
        # has already advanced before this command's metrics pass.  Park it until close-out books
        # the old attempt, mirroring the legal-return rally latch above.
        self._exact_pending_completion = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # The initial planner deadline belongs to the whole swing attempt, not to the live TTS
        # countdown.  Latch one bucket per attempt so its eventual close-out and sparse strike
        # outcomes can be compared without reconstructing state from EMA curves.  -1 means this is
        # a legacy/non-planner attempt and is deliberately absent from bucket denominators.
        self._exact_attempt_initial_tts_bucket = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        # A defensive exact strike may fire on the same step Motion wraps into the next clip,
        # before RacketTargetCommand samples that new task's initial TTS.  Park those three sparse
        # outcomes until _begin_same_ball_planner_task assigns the new bucket.
        self._exact_pending_timing_bucket_events = {
            name: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for name in _TIMING_BUCKET_SPARSE_EVENTS
        }
        self._prestrike_fall_acc = 0.0
        # POST-strike falls (fall AFTER reaching the strike frame — the follow-through/recovery fall that
        # swing_completion_rate + pre_strike_fall_rate are both blind to; it was the actual backhand
        # deploy failure mode). Same swing-starts denominator as pre_strike_fall_rate.
        self._poststrike_fall_acc = 0.0
        # Per-clip fall attribution. NOTE: at reset time the MOTION command has already resampled
        # clip_id to the NEW swing (motion resets before racket_target), so falls are attributed via
        # _prev_clip_id — the clip snapshot taken at the END of the previous _update_command, i.e. the
        # clip the env was actually swinging when it fell.
        self._prestrike_fall_acc_c = {c: 0.0 for c in self._clip_names}
        self._poststrike_fall_acc_c = {c: 0.0 for c in self._clip_names}
        self._prev_clip_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Post-wrap recovery latch: >=0 = the clip whose swing JUST finished while this env sits in the
        # post-wrap hold. A fall during that hold is physically the PREVIOUS swing's recovery fall, but
        # at the wrap the timing already describes the NEXT swing (pre_strike=True, clip_id=new random
        # clip) — without the latch such falls book as pre-strike falls of a 50%-wrong clip, which would
        # invert exactly the backhand-recovery diagnosis these metrics exist for. -1 = not recovering.
        self._recover_from_clip = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        # True only while _resample_command is invoked from the intra-episode WRAP path (see
        # _update_command): wraps start a new swing but never count a pre-strike fall (a wrapped
        # env necessarily passed its strike frame alive).
        self._resample_is_wrap = False

        # --- Rally drift accounting (2026-07-07 continuous-rally upgrade) -------------------------
        # Deploy P7 failure mode: each walk-and-strike lunges forward; over consecutive swings the
        # displacement ACCUMULATES until a swing starts from an untrained stance and falls. These
        # track exactly that: per-swing base displacement (closed out at WRAPS = completed swings
        # only), its forward (x) component (drift is directional: forward), and the base->station
        # error at each swing start (how far the new station is when the swing begins — the recovery
        # debt the previous swing left behind). Same EMA decay/denominator discipline as the exact
        # accumulators; drift uses its own wrap-count denominator (resets don't close out a swing).
        self._swing_start_base_xy = torch.zeros(self.num_envs, 2, device=self.device)
        # Stamp lazily on the FIRST _update_metrics after a swing start: at reset time the cached
        # base_pos_w still holds the PRE-reset pose (events teleport the root after the snapshot),
        # so an eager stamp would book the teleport as drift of the episode's first swing.
        self._swing_start_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._drift_n_acc = 0.0
        self._drift_sum_acc = 0.0
        self._drift_fwd_sum_acc = 0.0
        self._station_offset_start_sum_acc = 0.0
        self._heading_expiry_sum_acc = 0.0
        self._heading_expiry_n_acc = 0.0
        self._recovery_spawn_sum_acc = 0.0
        self._recovery_expiry_sum_acc = 0.0
        self._recovery_n_acc = 0.0
        self._previous_in_hold = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._hold_start_yaw = torch.zeros(self.num_envs, device=self.device)
        # A true reset can replace one held state with another without a boolean falling/rising
        # edge. Force a fresh stamp on the first metrics tick after every resample.
        self._hold_edge_pending = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # --- HER-style achieved-target replay buffers (see RacketTargetCommandCfg) -----------------
        # Per-clip ring buffers of the racket state the policy ACTUALLY produced at exact-strike frames:
        # position env-origin-relative (world minus env origin), velocity world. Written in
        # _update_metrics on the exact_strike mask (alive envs only — terminated envs were reset before
        # the command computes); read in _sample_targets_uniform with prob achieved_target_mix_prob.
        _absize = max(int(cfg.achieved_buffer_size), 1)
        self._ach_pos = {c: torch.zeros(_absize, 3, device=self.device) for c in self._clip_names}
        self._ach_vel = {c: torch.zeros(_absize, 3, device=self.device) for c in self._clip_names}
        # R14: playback speed of the swing that PRODUCED each achieved state (1.0 when retiming off),
        # so replay can rescale the velocity to the replaying swing's own speed.
        self._ach_spd = {c: torch.ones(_absize, device=self.device) for c in self._clip_names}
        self._ach_fill = {c: 0 for c in self._clip_names}
        self._ach_ptr = {c: 0 for c in self._clip_names}
        # R14: one-shot exact-strike latch per swing (armed at every target resample). Only consulted
        # when retiming is active — the float clock's ~1e-4/swing double-fire guard.
        self._exact_fired = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Decayed counters for the logged replay fraction (same EMA timescale as the exact accumulators).
        self._resample_n_acc = 0.0
        self._replay_n_acc = 0.0

        # Live strike timing is the reward/critic/physics truth.  A separately materialized actor
        # TTS exists only in the two explicit atomic-planner-tuple modes below.
        self.time_to_strike = torch.zeros(self.num_envs, device=self.device)

        # Same-ball planner revisions.  Unlike the retired midswing redraw knob, this never
        # replaces the question row or physical ball.  It maintains one explicit
        # (control_epoch, task_id), emits strictly increasing revisions of one atomic actor tuple,
        # and asks MotionCommand's phase governor to meet the revised deadline.
        self.planner_revision_enabled = bool(
            getattr(cfg, "planner_revision_enabled", False)
        )
        self._planner_revision_profile: PhaseGovernorProfile | None = None
        self._planner_initial_tts_mixture: InitialTtsMixture | None = None
        if self.planner_revision_enabled:
            raw_profile = getattr(cfg, "planner_revision_profile", None)
            if not isinstance(raw_profile, dict):
                raise ValueError(
                    "planner_revision_enabled requires a complete planner_revision_profile mapping"
                )
            self._planner_revision_profile = PhaseGovernorProfile.from_mapping(raw_profile)
            if not cfg.face_command or self._question_bank is None:
                # DELIBERATE HARD BLOCK, continuous included. This one really does want npz ROWS
                # to reconcile against, not merely "a solved target", so it is not switched to
                # _solved_targets_active. 响亮的不可发射 >> 悄悄降级的可发射。
                raise ValueError(
                    "planner revisions require the formal signed-face question-bank task; "
                    "otherwise the atomic target normal has no defined contract. This includes "
                    "target_mode='solved': a continuously drawn ball has no bank row for the "
                    "revision contract to address."
                )
            if float(cfg.midswing_resample_prob) != 0.0:
                raise ValueError(
                    "planner revisions replace midswing_resample_prob; truth/question redraws "
                    "during one physical ball are forbidden"
                )
            initial_tts = tuple(
                float(value)
                for value in getattr(cfg, "planner_revision_initial_tts_range_s", ())
            )
            if (
                len(initial_tts) != 2
                or not (
                    self._planner_revision_profile.min_tts_s
                    <= initial_tts[0]
                    < initial_tts[1]
                    <= self._planner_revision_profile.max_tts_s
                )
            ):
                raise ValueError(
                    "planner_revision_initial_tts_range_s must be a non-degenerate ordered pair "
                    "inside the "
                    "profile TTS envelope"
                )
            raw_mixture = getattr(
                cfg, "planner_revision_initial_tts_mixture", None
            )
            if not isinstance(raw_mixture, dict):
                raise ValueError(
                    "planner_revision_enabled requires a complete "
                    "planner_revision_initial_tts_mixture mapping"
                )
            self._planner_initial_tts_mixture = InitialTtsMixture.from_mapping(
                raw_mixture
            )
            self._planner_initial_tts_mixture.validate_support(
                lo_s=initial_tts[0], hi_s=initial_tts[1]
            )
            self._planner_initial_tts_component_lo = torch.tensor(
                [component.lo_s for component in self._planner_initial_tts_mixture.components],
                dtype=torch.float32,
                device=self.device,
            )
            self._planner_initial_tts_component_hi = torch.tensor(
                [component.hi_s for component in self._planner_initial_tts_mixture.components],
                dtype=torch.float32,
                device=self.device,
            )
            self._planner_initial_tts_component_weight = torch.tensor(
                [component.weight for component in self._planner_initial_tts_mixture.components],
                dtype=torch.float64,
                device=self.device,
            )
            for name in (
                "planner_revision_position_std_m",
                "planner_revision_velocity_std_mps",
                "planner_revision_normal_std_rad",
                "planner_revision_tts_std_s",
            ):
                value = float(getattr(cfg, name, 0.0))
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"{name} must be finite and non-negative")
            n = self.num_envs
            self._planner_control_epoch = torch.zeros(n, dtype=torch.long, device=self.device)
            self._planner_task_id = torch.zeros(n, dtype=torch.long, device=self.device)
            self._planner_task_revision = torch.full(
                (n,), -1, dtype=torch.long, device=self.device
            )
            self._planner_visible_pos = self.racket_target_pos_w.clone()
            self._planner_visible_vel = self.racket_target_vel_w.clone()
            self._planner_visible_normal = self.target_normal_cmd.clone()
            self._planner_visible_tts = self.time_to_strike.clone()
            self._planner_visible_last_precontact = torch.zeros(
                n, dtype=torch.bool, device=self.device
            )
            # The planner-owned clock deliberately disables the legacy hold counters: preparation
            # time comes from the task deadline, not from a second frozen-reference clock.  Keep a
            # separate instrumentation-only identity latch so the exact behavior ledger can still
            # sample the robot once on the first metrics tick after every physical-ball task is
            # installed.  Without this, defining "ready" exclusively as ``motion.in_hold`` makes
            # every planner-revision run report a silent zero denominator even though the first
            # post-install state is observable.
            self._exact_ready_sampled_control_epoch = torch.zeros(
                n, dtype=torch.long, device=self.device
            )
            self._exact_ready_sampled_task_id = torch.zeros(
                n, dtype=torch.long, device=self.device
            )
            # Actor delivery is a different event from producer acceptance.  Keep the complete
            # task identity so a delayed ring cannot confuse revision 2 of two adjacent balls.
            self._planner_actor_control_epoch = torch.zeros(
                n, dtype=torch.long, device=self.device
            )
            self._planner_actor_task_id = torch.zeros(
                n, dtype=torch.long, device=self.device
            )
            self._planner_actor_task_revision = torch.full(
                (n,), -1, dtype=torch.long, device=self.device
            )
            self.metrics["planner_task_revision"] = torch.zeros(n, device=self.device)
            self.metrics["planner_same_task_revision_active"] = torch.zeros(n, device=self.device)

        # --- A1 target latency & time-variance (mocap->planner->runner realism) --------------------
        # MOTIVATION: training previously handed the actor a PERFECT, instantly-updated target; the
        # real loop (mocap -> planner -> runner) delivers it LATE (transport + planning latency),
        # NOISY (ball-prediction error that SHRINKS as the strike approaches — SMASH Eq. 14), and
        # REFINED mid-swing (the planner re-plans WHERE while the swing clock keeps running — PACE
        # injects sensor delays for the same reason). Without modeling this, the mocap-closed-loop
        # deployment faces out-of-distribution target dynamics. ALL knobs default OFF and the default
        # path is byte-identical: delay==0 & jitter==0 make the actor-visible views ALIAS the live
        # tensors (zero overhead, no extra RNG); midswing_resample_prob==0 short-circuits before any
        # RNG draw. Only the ACTOR-visible view is degraded — rewards, metrics, the privileged critic,
        # and the achieved-target-replay WRITE always use the TRUE live target.
        self._delay_steps = max(int(cfg.target_delay_steps), 0)
        self._delay_tts_mode = _target_delay_tts_mode(cfg)
        # 耦合传输(2026-07-25,取代旧 NO-LAUNCH 守卫):planner 修订 + 延迟时,延迟发生在
        # "提交"这道工序——每步生成的修订元组(pos/vel/normal/tts,原子)先进在途环压 d 步,
        # d 步后才提交给相位调度器。接受记账、调度器看到的 desired_tts、actor 可见元组因此
        # 消费同一条延迟流(mocap→relay 语义);actor 端不再叠观测延迟环(否则总延迟成 2d)。
        # 任务安装(begin)不是 mocap 流,保持即时。d=0 = 耦合不激活 = 现役路径逐字节不变。
        # 半配置组合(live tts 模式)在 _coupled_transport_mode 里 fail-loud;event timing 与
        # 耦合传输的交互未定义,在 _revise_same_ball_actor_tuple 运行期 fail-loud(motion 术语
        # 构造期还未解析,只能推迟到第一次使用)。
        self._coupled_transport = _coupled_transport_mode(cfg)
        self._delay_tts_active = self._delay_tts_mode != "live"
        self._atomic_tts_active = self._delay_tts_active or self.planner_revision_enabled
        # actor 观测端延迟环的步数:legacy 模式 = d(现役行为);耦合模式 = 0(提交已经晚了
        # d 步,actor 直接读被接受的 planner_visible 流,调度器/actor 同拍看到同一元组)。
        self._actor_ring_steps = 0 if self._coupled_transport else self._delay_steps
        if self._coupled_transport:
            # 在途环(长度 d,单指针):槽位 p 在第 t 步写入、第 t+d 步弹出提交。valid=False 的
            # 槽(当步没生成 / 新任务安装时被作废)不提交——中继丢弃已结束任务的消息。
            self._pend_ptr = 0
            self._pend_valid = torch.zeros(
                self._delay_steps, self.num_envs, dtype=torch.bool, device=self.device
            )
            self._pend_pos = torch.zeros(self._delay_steps, self.num_envs, 3, device=self.device)
            self._pend_vel = torch.zeros(self._delay_steps, self.num_envs, 3, device=self.device)
            self._pend_normal = torch.zeros(self._delay_steps, self.num_envs, 3, device=self.device)
            self._pend_tts = torch.zeros(self._delay_steps, self.num_envs, device=self.device)
        self._jitter_pos = max(float(cfg.target_jitter_pos_per_s), 0.0)
        self._jitter_vel = max(float(cfg.target_jitter_vel_per_s), 0.0)
        # Calibrated MEASUREMENT noise (ball_physics_venue.yaml `capture:` block, 2026-07-03 fit):
        # white + AR(1)-colored position error of the mocap link. Unlike the tts-scaled jitter above
        # (which models PREDICTION convergence), measurement noise does NOT shrink as the strike
        # approaches, so no tts scaling. rho is per POLICY step: venue 0.946/frame @300 Hz -> ^6 @50 Hz.
        self._mnoise_white = max(float(cfg.target_noise_white), 0.0)
        self._mnoise_ar1_sigma = max(float(cfg.target_noise_ar1_sigma), 0.0)
        self._mnoise_ar1_rho = min(max(float(cfg.target_noise_ar1_rho), 0.0), 0.9999)
        if self._mnoise_ar1_sigma > 0.0:
            self._mnoise_ar1_state = torch.zeros(self.num_envs, 3, device=self.device)
        self._drop_prob = max(float(cfg.target_dropout_prob), 0.0)
        self._post_strike_drop_steps = max(int(round(float(cfg.target_post_strike_dropout_s) * 50.0)), 0)
        self._bias_per_swing = max(float(cfg.target_bias_per_swing), 0.0)
        self._a1v2_active = self._drop_prob > 0.0 or self._post_strike_drop_steps > 0 or self._bias_per_swing > 0.0
        if self._a1v2_active:
            self._swing_bias = torch.zeros(self.num_envs, 3, device=self.device)
            self._drop_cd = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._prev_pre_strike = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self._held_pos = torch.zeros(self.num_envs, 3, device=self.device)
            self._held_vel = torch.zeros(self.num_envs, 3, device=self.device)
            # A planner command is one atomic message.  A dropped frame must therefore hold the
            # face command and side selector together with position/velocity; holding only the
            # first two used to expose question N+1's normal/sign next to question N's target.
            self._held_normal = torch.zeros(self.num_envs, 3, device=self.device)
            self._held_sign = torch.ones(self.num_envs, device=self.device)
            if self._atomic_tts_active:
                self._held_tts = torch.zeros(self.num_envs, device=self.device)
            if self.planner_revision_enabled:
                self._held_planner_epoch = torch.zeros(
                    self.num_envs, dtype=torch.long, device=self.device
                )
                self._held_planner_task = torch.zeros(
                    self.num_envs, dtype=torch.long, device=self.device
                )
                self._held_planner_revision = torch.full(
                    (self.num_envs,), -1, dtype=torch.long, device=self.device
                )
                self._held_planner_last_precontact = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
        # The actor view is materialized (separate tensors) whenever latency OR jitter is on;
        # otherwise the delayed_* attributes ARE the live tensors (which are only ever index-assigned
        # after __init__, so the alias stays valid for the whole run).
        self._actor_view_active = (
            self.planner_revision_enabled
            or
            self._delay_steps > 0 or self._jitter_pos > 0.0 or self._jitter_vel > 0.0
            or self._mnoise_white > 0.0 or self._mnoise_ar1_sigma > 0.0
            or float(cfg.target_dropout_prob) > 0.0
            or float(cfg.target_post_strike_dropout_s) > 0.0
            or float(cfg.target_bias_per_swing) > 0.0
            or self._atomic_tts_active
        )
        if self._actor_ring_steps > 0:
            # Ring buffers (length delay+1) over the ACTOR-VISIBLE target quantities: the slot
            # written this step is read back `delay` pushes later (see _push_actor_target).
            # 耦合传输模式下 _actor_ring_steps==0:观测环不建,延迟由提交侧在途环承担。
            _L = self._actor_ring_steps + 1
            self._delay_buf_pos = torch.zeros(_L, self.num_envs, 3, device=self.device)
            self._delay_buf_vel = torch.zeros(_L, self.num_envs, 3, device=self.device)
            self._delay_buf_normal = torch.zeros(_L, self.num_envs, 3, device=self.device)
            self._delay_buf_sign = torch.ones(_L, self.num_envs, device=self.device)
            if self._atomic_tts_active:
                self._delay_buf_tts = torch.zeros(_L, self.num_envs, device=self.device)
            if self.planner_revision_enabled:
                self._delay_buf_planner_epoch = torch.zeros(
                    _L, self.num_envs, dtype=torch.long, device=self.device
                )
                self._delay_buf_planner_task = torch.zeros(
                    _L, self.num_envs, dtype=torch.long, device=self.device
                )
                self._delay_buf_planner_revision = torch.full(
                    (_L, self.num_envs), -1, dtype=torch.long, device=self.device
                )
                self._delay_buf_planner_last_precontact = torch.zeros(
                    _L, self.num_envs, dtype=torch.bool, device=self.device
                )
            self._delay_ptr = 0
        if self._actor_view_active:
            self.delayed_racket_target_pos_w = self.racket_target_pos_w.clone()
            self.delayed_racket_target_vel_w = self.racket_target_vel_w.clone()
            self.delayed_target_normal_cmd = self.target_normal_cmd.clone()
            self.delayed_swing_sign = self.swing_sign.clone()
            if self._atomic_tts_active:
                self.delayed_time_to_strike = self.time_to_strike.clone()
        else:
            # Flags off: zero-overhead aliases of the live tensors (byte-identical baseline).
            self.delayed_racket_target_pos_w = self.racket_target_pos_w
            self.delayed_racket_target_vel_w = self.racket_target_vel_w
            self.delayed_target_normal_cmd = self.target_normal_cmd
            self.delayed_swing_sign = self.swing_sign
        # A1 metrics: per-step per-env redraw indicator (wandb reset-mean = per-step mid-swing
        # refinement fraction) + the constant delay-in-effect broadcast (refreshed every step in
        # _update_metrics because CommandTerm.reset() zeros metric entries of resetting envs).
        self.metrics["midswing_resample_count"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["target_delay_steps_in_effect"] = torch.full(
            (self.num_envs,), float(self._delay_steps), device=self.device
        )
        self.metrics["actor_time_to_strike_s"] = torch.zeros(self.num_envs, device=self.device)

        # Strike timing / gating.
        self.pre_strike = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.strike_window = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Wall/control-time recovery clock.  It cannot be reconstructed from the reference phase
        # under the planner governor because playback speed is time-varying.
        self._post_strike_elapsed_s = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._post_strike_elapsed_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._post_strike_elapsed_last_step = -1
        # 1c split windows: refreshed alongside strike_window in _compute_strike_timing. With the
        # cfg fields at their None defaults these are recomputed from strike_window_s each step,
        # i.e. numerically identical to strike_window (byte-identical default path).
        self.strike_window_pos = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.strike_window_wide = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # --- R-b envelope-violation accounting (cfg.track_envelope_violation; default OFF) --------
        # Counts the tracking-envelope violations that no longer terminate under
        # terminations.envelope_as_penalty: tracking_loss_rate = violation RISING EDGES / swing
        # starts (same EMA timescale/denominator as pre_strike_fall_rate, per-clip variants too) +
        # envelope_violated_frac = per-step violation fraction (loafing/挂机 monitor). Cross-arm
        # bookkeeping: old-arm pre_strike_fall ≈ new-arm (fall + tracking_loss).
        self._envelope_track = bool(getattr(cfg, "track_envelope_violation", False))
        if self._envelope_track:
            self._envelope_body_idx: list[int] | None = None  # resolved lazily vs motion.cfg.body_names
            self._prev_envelope_viol = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._tracking_loss_acc = 0.0
            self._tracking_loss_acc_c = {c: 0.0 for c in self._clip_names}
            self.metrics["envelope_violated_frac"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["tracking_loss_rate"] = torch.zeros(self.num_envs, device=self.device)
            for _cname in self._clip_names.values():
                self.metrics[f"tracking_loss_rate_{_cname}"] = torch.zeros(self.num_envs, device=self.device)

        # Episode-wide tracking errors (instantaneous; averaged over terminating envs at reset).
        self.metrics["racket_pos_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_vel_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["adaptive_sigma_pos"] = torch.full((self.num_envs,), float(cfg.sigma_pos_max), device=self.device)
        self.metrics["adaptive_sigma_vel"] = torch.full((self.num_envs,), float(cfg.sigma_vel_max), device=self.device)
        # 第三通道曲线只在旗标开启时注册:默认跑的 wandb 指标键集合逐字节不变。
        if getattr(cfg, "adaptive_sigma_normal", False):
            self.metrics["adaptive_sigma_normal"] = torch.full(
                (self.num_envs,), float(cfg.sigma_normal_max), device=self.device
            )
        self.metrics["racket_normal_error_deg"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["base_pos_error"] = torch.zeros(self.num_envs, device=self.device)
        # How far the (coupled) base target sits from spawn — i.e. how much repositioning is commanded.
        # ~0 when base_couple_blend=0; grows with the coupling toward far racket targets.
        self.metrics["base_target_offset_norm"] = torch.zeros(self.num_envs, device=self.device)
        # Strike-window metrics: hold the value from the MOST RECENT strike — these map directly to
        # the acceptance criteria (racket pos < 7.5 cm, vel < 0.5 m/s, normal < 15 deg AT strike) and
        # are the real "is the policy learning to hit" signal (the episode-wide ones above are diluted
        # by the long non-strike portion of each swing).
        self.metrics["racket_pos_error_at_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_vel_error_at_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_normal_error_deg_at_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["strike_success"] = torch.zeros(self.num_envs, device=self.device)
        # Exact-strike metrics: sampled only on the nearest control frame to the configured strike
        # step. These avoid the "within-window" dilution from the +/- strike reward window.
        self.metrics["racket_pos_error_exact_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_vel_error_exact_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_normal_error_deg_exact_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["strike_composite_success_exact"] = torch.zeros(self.num_envs, device=self.device)
        # Per-axis position error AT the exact strike frame (which axis is the miss?).
        self.metrics["racket_pos_error_x_exact_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_pos_error_y_exact_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_pos_error_z_exact_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["exact_strike_hit_rate"] = torch.zeros(self.num_envs, device=self.device)
        # Conditional exact-strike pass rates (broadcast scalar; the trustworthy success signal). Each is
        # the fraction of exact-strike samples that clear that acceptance threshold, undiluted by the
        # non-strike steps that wrecked the old held metric. strike_composite_success_exact requires all
        # three (pos & vel & normal) and drives the success-gated perturbation curriculum.
        self.metrics["strike_pos_pass_exact"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["strike_vel_pass_exact"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["strike_normal_pass_exact"] = torch.zeros(self.num_envs, device=self.device)
        # Position-accuracy buckets + error distribution on the exact-strike sample (comparable w/ composite).
        self.metrics["exact_strike_pos_success_5cm"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["exact_strike_pos_success_10cm"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["exact_strike_pos_err_mean"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["exact_strike_pos_err_p90"] = torch.zeros(self.num_envs, device=self.device)
        # Per-clip (forehand/backhand) versions of the exact-strike pass rates + errors (multiseg only;
        # stay 0 for a single-clip run). Updated in _update_metrics.
        for _cname in self._clip_names.values():
            for _key in (
                "strike_pos_pass_exact", "strike_vel_pass_exact", "strike_normal_pass_exact",
                "strike_composite_success_exact", "racket_pos_error_exact_strike",
                "racket_vel_error_exact_strike", "racket_normal_error_deg_exact_strike",
            ):
                self.metrics[f"{_key}_{_cname}"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["exact_strike_sample_count_decayed"] = torch.zeros(self.num_envs, device=self.device)
        # Tier-1 virtual-ball outcome rates (broadcast sample-weighted EMAs, exact-strike denominator
        # for hit rate; hit (captured) denominator for the outcome rates). Logged when the virtual
        # ball is enabled for rewards (virtual_ball) OR as pure metrics (vb_metrics_only).
        # virtual_return_rate is the PRIMARY in-training curve (franco 2026-07-06): legal returns
        # per exact-strike sample (上台率); virtual_hit_rate (击球率) is the auxiliary;
        # strike_composite (三合格) is a diagnostic only. Canonical bookkeeping stays MuJoCo.
        if cfg.virtual_ball or cfg.vb_metrics_only:
            for _vk in (
                "virtual_return_rate", "virtual_return_rate_forehand", "virtual_return_rate_backhand",
                "virtual_hit_rate_forehand", "virtual_hit_rate_backhand",
                "virtual_return_rate_rally", "virtual_return_rate_rally_forehand",
                "virtual_return_rate_rally_backhand",
                "virtual_hit_rate", "virtual_net_clear_rate", "virtual_land_valid_rate",
                "virtual_land_inbounds_rate", "virtual_land_err_m", "virtual_topspin_revs",
                "virtual_approach_speed",
            ):
                self.metrics[_vk] = torch.zeros(self.num_envs, device=self.device)
            # Transition-period *_legacy rally curves: the OLD mixed-ledger readout (can spike >1
            # under synchronized reset queues — see the rally block in __init__) kept alongside the
            # fixed virtual_return_rate_rally* for new/old comparison. Drop by turning the cfg off.
            if cfg.rally_legacy_metrics:
                for _vk in (
                    "virtual_return_rate_rally_legacy", "virtual_return_rate_rally_forehand_legacy",
                    "virtual_return_rate_rally_backhand_legacy",
                ):
                    self.metrics[_vk] = torch.zeros(self.num_envs, device=self.device)
        # UNCONDITIONAL swing accounting (Phase A): completion_rate = exact-strike arrivals / swing
        # STARTS (falls count against it, unlike the conditional composite above); fall rate before
        # the strike frame. Broadcast scalars like the pass rates.
        self.metrics["swing_completion_rate"] = torch.zeros(self.num_envs, device=self.device)
        for _cname in self._clip_names.values():
            self.metrics[f"swing_completion_rate_{_cname}"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pre_strike_fall_rate"] = torch.zeros(self.num_envs, device=self.device)
        # Post-strike (follow-through/recovery) falls + per-clip fall attribution: the multi-swing
        # episode's real recovery signal. pre_strike_fall_rate alone hides a policy that hits and THEN
        # falls (100% completion, 0% pre-strike falls — the actual backhand deploy failure signature).
        self.metrics["post_strike_fall_rate"] = torch.zeros(self.num_envs, device=self.device)
        for _cname in self._clip_names.values():
            self.metrics[f"pre_strike_fall_rate_{_cname}"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[f"post_strike_fall_rate_{_cname}"] = torch.zeros(self.num_envs, device=self.device)
        # HER-style achieved-target replay diagnostics: fraction of resampled targets drawn from the
        # achieved buffer (EMA; ~achieved_target_mix_prob once the buffers are filled) + per-clip fill.
        self.metrics["achieved_replay_frac"] = torch.zeros(self.num_envs, device=self.device)
        for _cname in self._clip_names.values():
            self.metrics[f"achieved_buffer_fill_{_cname}"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["strike_window_hit_rate"] = torch.zeros(self.num_envs, device=self.device)
        # Base-position error while the base target is active (pre-strike), held at its last value.
        self.metrics["base_pos_error_pre_strike"] = torch.zeros(self.num_envs, device=self.device)
        # Rally drift metrics (2026-07-07): EMA-broadcast per-swing drift + per-step recovery signals.
        self.metrics["base_drift_per_swing"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["base_drift_fwd_per_swing"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["base_station_offset_at_swing_start"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["base_heading_abs_at_swing_start"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["base_heading_hold_expiry_count"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["heading_recovery_spawn_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["heading_recovery_expiry_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["heading_recovery_count"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["post_strike_base_speed_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["base_dist_from_origin"] = torch.zeros(self.num_envs, device=self.device)
        # Swing-quality detail at the most recent strike: actual paddle speed, per-axis position error,
        # and success at tighter/looser thresholds (5 cm / 10 cm) for a fuller accuracy distribution.
        self.metrics["racket_speed_at_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_target_speed_at_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_pos_error_x_at_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_pos_error_y_at_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_pos_error_z_at_strike"] = torch.zeros(self.num_envs, device=self.device)
        # DEPRECATED semantics: these hold the value at the WINDOW-EXIT frame (racket ~0.26 m past target),
        # NOT at contact, and use the diluting reset-mean denominator. Renamed so they stop reading as
        # "success". Use exact_strike_pos_success_5cm/10cm above for the real contact-frame accuracy.
        self.metrics["strike_success_5cm_window_exit"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["strike_success_10cm_window_exit"] = torch.zeros(self.num_envs, device=self.device)
        # Robot-health diagnostics (episode-wide, instantaneous) — logged here because this term already
        # holds ``self.robot``. Useful for sim2real: standing height, peak joint speed, actuator effort.
        self.metrics["base_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["base_upright"] = torch.zeros(self.num_envs, device=self.device)
        # Stability diagnostics (instability shows up here BEFORE a termination): absolute base roll/pitch
        # (deg; 0 = level — base_upright only gives the combined tilt), and foot contact + slip (a planted
        # foot should be ~still; horizontal foot speed while in contact = slip = the robot shuffling/sliding).
        self.metrics["base_roll_deg"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["base_pitch_deg"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["foot_contact_frac"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["foot_slip_speed"] = torch.zeros(self.num_envs, device=self.device)
        # Foot-slip magnitude (sum over feet of horizontal speed WHILE in contact) used by the
        # pre_strike_foot_slip reward. Recomputed each step in _update_metrics; stays 0 if the feet /
        # contact sensor cannot be resolved, so the reward is a safe no-op in that case.
        self.foot_slip_in_contact = torch.zeros(self.num_envs, device=self.device)
        # Fraction of feet in contact (mean over the 2 feet): 1.0 = both planted, 0.5 = one foot
        # unloaded, 0.0 = airborne. Clean attribute for the feet_contact stance reward (real_sensor
        # variant). Same value as metrics["foot_contact_frac"]; 0 (safe) if the sensor cannot resolve.
        self.feet_contact_frac = torch.zeros(self.num_envs, device=self.device)
        # ---- footwork-to-strike signals (base-FREE; reward/metric only, NEVER in the observation) ----
        # racket_target_distance = ||racket_FK - racket_target|| (frame-invariant, no base position).
        # racket_progress = prev_distance - current_distance (>0 = the WHOLE body moved the racket closer
        # to the target). This dense progress term is what drives footwork WITHOUT a base target.
        z = lambda: torch.zeros(self.num_envs, device=self.device)  # noqa: E731
        self.racket_target_distance = z()
        self.racket_progress = z()
        self._prev_racket_dist = z()
        self._progress_reset_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.foot_slip_sq = z()  # sum_feet contact * ||foot_xy_vel||^2
        self.foot_vel_sq = z()  # sum_feet ||foot_vel||^2 (excessive/violent foot motion)
        self.foot_drag = z()  # sum_feet ||foot_xy_vel|| while the foot is LOW (near ground -> dragging)
        self.arm_overreach_frac = z()  # fraction of ARM joints within 10% of a position limit
        self.waist_twist = z()  # |waist_yaw - default| + |waist_roll - default| (anti twist-instead-of-step)
        self.proj_grav_xy = z()  # ||projected_gravity_xy|| = base tilt (strike-window stability)
        self.base_ang_vel_xy_norm = z()  # ||base_ang_vel_xy|| (strike-window stability)
        self.vertical_speed = z()  # |base_lin_vel_z| (vertical bob)
        self._drag_height = 0.10  # m: foot below this counts as "near ground" for the drag penalty
        self.metrics["joint_vel_abs_max"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["time_to_strike_s"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pre_strike_flag"] = torch.zeros(self.num_envs, device=self.device)
        # P2.4 watch-metric: planar |v_base| during the approach (the base_decel_tracking reward's
        # subject). 0 outside pre_strike -> the reset-mean dilutes like the other *_prestrike metrics.
        self.metrics["base_speed_xy_prestrike"] = torch.zeros(self.num_envs, device=self.device)
        # Curriculum perturbation scale (reference_perturbed mode): 0 at start ramping to 1; lets you
        # watch the reachable target ball widen in wandb. Stays 0 in "uniform" mode.
        self.metrics["ref_perturb_scale"] = torch.zeros(self.num_envs, device=self.device)
        self._has_jpos_limits = hasattr(self.robot.data, "soft_joint_pos_limits") or hasattr(
            self.robot.data, "joint_pos_limits"
        )
        if self._has_jpos_limits:
            self.metrics["joint_pos_near_limit_frac"] = torch.zeros(self.num_envs, device=self.device)
        self._has_torque = hasattr(self.robot.data, "applied_torque")
        if self._has_torque:
            self.metrics["joint_torque_abs_mean"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["joint_torque_abs_max"] = torch.zeros(self.num_envs, device=self.device)
        # Policy action magnitude (saturation check for sim2real).
        self.metrics["action_abs_mean"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["action_abs_max"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["action_delta_abs_mean"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["action_delta_abs_max"] = torch.zeros(self.num_envs, device=self.device)
        for axis in ("x", "y"):
            self.metrics[f"base_pos_{axis}"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[f"base_pos_error_{axis}"] = torch.zeros(self.num_envs, device=self.device)
        for axis in ("x", "y", "z"):
            self.metrics[f"racket_pos_error_{axis}"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[f"racket_vel_error_{axis}"] = torch.zeros(self.num_envs, device=self.device)

        # --- SHADOW physical ball (+ optional table): flag-gated, METRICS-ONLY measurement --------
        # A real PhysX ball per env that flies the question in, takes the SAME contact model as the
        # reward path at a captured strike, and lands under engine integration — an online
        # engine-vs-analytic cross-check of the virtual-ball prediction. Zero coupling to rewards/
        # observations/bank-target logic (see shadow_ball.py's module docstring for the honesty
        # notes: linear cosmetic pre-strike path, no bounce-before-strike, engine-fidelity baseline
        # from scripts/isaac_ball_inloop_check.py). Default OFF = byte-identical (no driver, no
        # scene entity, no physics callback, no metrics keys).
        self._shadow = None
        # Mid-swing target refinement redraws the question without notifying a ball already in
        # flight.  Combining the two would make the truth instrument measure a target jump rather
        # than engine delivery, so fail before constructing either driver.
        if float(getattr(cfg, "midswing_resample_prob", 0.0)) > 0.0 and (
            cfg.shadow_ball or cfg.physical_ball
        ):
            raise ValueError(
                "RacketTargetCommandCfg.midswing_resample_prob > 0 cannot be combined with "
                "shadow_ball/physical_ball: the served ball would realize the old question."
            )
        if cfg.shadow_ball:
            # The shadow ball mirrors the virtual-ball question stream (per-swing incoming
            # velocity/spin + the capture gate live in the vb machinery) — without it there is
            # nothing to fly in and nothing to cross-check against. Loud error, not silent no-op.
            if not cfg.virtual_ball:
                raise ValueError(
                    "RacketTargetCommandCfg.shadow_ball=True requires virtual_ball=True: the "
                    "shadow ball flies the vb-sampled incoming ball and cross-checks the vb "
                    "landing prediction. Enable the virtual-ball task variant or drop shadow_ball."
                )
            from whole_body_tracking.tasks.tracking.mdp.shadow_ball import ShadowBallDriver

            self._shadow = ShadowBallDriver(self, env)

        # --- PHYSICAL ball + table (Phase A truth instrument): flag-gated, METRICS-ONLY ------------
        # A real PhysX ball per env + a real static table collider: each swing's question-bank
        # incoming ball is realized physically (reverse-integrated venue-model launch so it arrives
        # at the question contact point with the question incoming velocity exactly at the strike
        # frame), flies under the per-substep venue aero wrench, takes the CODE-DRIVEN fitted table
        # bounce, and passes THROUGH the robot (collider off — the fitted racket impulse is Phase B).
        # Zero coupling to rewards/observations/bank-target logic (see physical_ball.py docstring).
        # Default OFF = byte-identical (no manager, no scene entity, no physics callback, no metrics
        # keys, no RNG consumption — the serve is deterministic from the question).
        self._physical = None
        if not cfg.physical_ball and (
            getattr(cfg, "physical_ball_impulse", False)
            or int(getattr(cfg, "physical_ball_substep", 1)) != 1
        ):
            raise ValueError(
                "RacketTargetCommandCfg.physical_ball_impulse / physical_ball_substep require "
                "physical_ball=True; Phase B rides on the truth instrument."
            )
        if cfg.physical_ball:
            # The physical ball realizes the virtual-ball question stream (per-swing incoming
            # velocity/spin live in the vb machinery) — without it there is nothing to serve.
            if not cfg.virtual_ball:
                raise ValueError(
                    "RacketTargetCommandCfg.physical_ball=True requires virtual_ball=True: the "
                    "physical ball serves the vb-sampled incoming ball (contact point + incoming "
                    "velocity + spin). Enable the virtual-ball task variant or drop physical_ball."
                )
            from whole_body_tracking.tasks.tracking.mdp.physical_ball import PhysicalBallManager

            self._physical = PhysicalBallManager(self, env)

        # --- DEBUG: swing-through sign check + raw/gated reward kernels (cfg.debug_reward_logging) ---
        # err_minus uses the CURRENT (correct) swing-through form target - vel*t_to_strike; err_plus uses
        # the FLIPPED form target + vel*t_to_strike. In-window we expect err_minus < err_plus (sign OK) and
        # at the exact strike (t_to_strike~0) the two collapse together. raw/gated are written by the reward
        # terms in hope_rewards.py. All held over the relevant mask so the reset-mean is the in-window value.
        if self.cfg.debug_reward_logging:
            for _k in (
                "dbg_err_minus_win", "dbg_err_plus_win", "dbg_err_minus_exact", "dbg_err_plus_exact",
                "dbg_racket_pos_raw", "dbg_racket_pos_gated",
                "dbg_racket_vel_raw", "dbg_racket_vel_gated",
                "dbg_racket_normal_raw", "dbg_racket_normal_gated",
                "dbg_base_raw", "dbg_base_gated",
            ):
                self.metrics[_k] = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------------ #
    # CommandTerm API
    # ------------------------------------------------------------------ #
    @property
    def command(self) -> torch.Tensor:
        """Raw desired-target vector (world frame): [pos(3), vel(3), normal(3), t_left(1), base_xy(2), swing(1)]."""
        return torch.cat(
            [
                self.racket_target_pos_w,
                self.racket_target_vel_w,
                self.racket_target_normal_w,
                self.time_to_strike.unsqueeze(-1),
                self.base_target_pos_w,
                self.swing_sign.unsqueeze(-1),
            ],
            dim=-1,
        )

    @property
    def base_pos_w(self) -> torch.Tensor:
        return self.robot.data.root_pos_w

    @property
    def base_quat_w(self) -> torch.Tensor:
        return self.robot.data.root_quat_w

    def _motion(self) -> MotionCommand:
        if self._motion_term is None:
            self._motion_term = self._env.command_manager.get_term(self.cfg.motion_command_name)
        if (
            getattr(self._motion_term, "event_timing_enabled", False)
            and not self._event_timing_bound
            and hasattr(self, "_question_bank")
        ):
            self._bind_event_timing_contract()
        return self._motion_term

    def _bind_event_timing_contract(self) -> None:
        """Bind every immutable schedule row to this exact train bank and native clip timing."""

        motion = self._motion_term
        if motion is None or not motion.event_timing_enabled:
            self._event_timing_bound = True
            return
        bank = self._question_bank
        if bank is None:
            # Hard block for continuous too: the immutable event schedule is content-addressed by
            # question_id(incoming_vel) against bank ROWS, and a continuously drawn ball has none.
            raise ValueError(
                "post_strike_t1 requires a schema-v3 train question bank; it is mutually "
                "exclusive with target_mode='solved' (continuous draws have no bank row for the "
                "immutable schedule to reconcile against)"
            )
        metadata = getattr(bank, "metadata", None)
        if (
            not isinstance(metadata, dict)
            or int(metadata.get("schema_version", 0)) != 3
            or metadata.get("split") != "train"
        ):
            raise ValueError("post_strike_t1 requires a validated schema-v3 train-split bank")
        if float(self.cfg.midswing_resample_prob) != 0.0:
            raise ValueError("post_strike_t1 requires midswing_resample_prob=0")

        schedule = motion.event_schedule
        if schedule is None:
            raise RuntimeError("post_strike_t1 has no loaded immutable schedule")
        counts = bank.counts.to(device=self.device, dtype=torch.long)
        # 族寻址(clip_family_per_clip 配了才有;None = 现役行为逐字节不变):日程行的 clip_id
        # 是加载 clip 序号,题库下标要过族表折算。内容寻址的 question_id 对账保持不变——族表
        # 配错时哪怕越界检查侥幸通过,问题指纹也对不上,照样当场炸。
        family_table = self._qb_bank_family_table(motion, bank)
        for flat_index, row in enumerate(schedule.rows):
            if family_table is None:
                bank_clip = row.clip_id
            elif 0 <= row.clip_id < int(family_table.shape[0]):
                bank_clip = int(family_table[row.clip_id])
            else:
                raise ValueError(
                    f"event schedule row {flat_index} references clip {row.clip_id} outside "
                    f"the loaded motion's {int(family_table.shape[0])} clip(s)"
                )
            if bank_clip >= len(counts) or row.bank_row >= int(counts[bank_clip]):
                raise ValueError(
                    f"event schedule row {flat_index} references unavailable train-bank row "
                    f"clip={row.clip_id} bank_row={row.bank_row}"
                )
            incoming = bank.incoming_vel[bank_clip, row.bank_row].detach().cpu().numpy()
            actual_id = question_id(incoming)
            if actual_id != row.question_id:
                raise ValueError(
                    f"event schedule row {flat_index} question_id={row.question_id} does not "
                    f"match train bank clip={row.clip_id} row={row.bank_row} id={actual_id}"
                )

        ml = motion.motion
        nseg = int(ml.num_segments)
        phases_cfg = self._strike_phases_cfg(nseg)
        phases = (
            [float(value) for value in phases_cfg]
            if phases_cfg
            else [float(self.cfg.strike_phase)] * nseg
        )
        files = (
            [motion.cfg.motion_file]
            if isinstance(motion.cfg.motion_file, str)
            else list(motion.cfg.motion_file)
        )
        validate_runtime_motion_contract(
            metadata,
            files,
            [int(value) for value in ml.seg_len.tolist()],
            phases,
            clip_families=(family_table.tolist() if family_table is not None else None),
        )
        native_ticks = []
        for clip_id in range(nseg):
            strike_step, _, seg_start, _ = self._strike_frame_for_clip(ml, clip_id)
            native_ticks.append(strike_step - seg_start)
        motion.bind_event_native_strike_ticks(native_ticks)
        self._event_timing_bound = True

    def _clip_label(self, clip_id: int) -> str:
        """Human clip name for error text. Safe before ``_clip_names`` is assigned in ``__init__``."""
        return getattr(self, "_clip_names", {}).get(int(clip_id), f"clip{clip_id}")

    def _commanded_target_x_hi(self, box_x_hi: float) -> float:
        """Farthest env-frame x a commanded racket target built from ``box_x_hi`` can reach.

        ``hitter_pure`` position boxes are STATION-relative (see :meth:`_sample_targets_hitter_pure`),
        so the commanded station's own forward span rides on top of the box; every other mode adds the
        box straight to the env origin.
        """
        x_hi = float(box_x_hi)
        if self.cfg.target_mode == "hitter_pure":
            x_hi += float(max(self.cfg.base_target_x_range))
        return x_hi

    def _assert_contact_clears_table(
        self, clip_id: int, x_hi: float, z_lo: float, source: str
    ) -> None:
        """Fail closed when a commanded contact point out over the table sits below the ball line.

        Physical rule, from the SAME constants the virtual ball is integrated against
        (``cfg.vb_table_surface_z`` + ``geometry.BALL_RADIUS``): a ball centre can never be lower than
        the table surface plus one ball radius. A commanded strike point past the near table edge
        (``cfg.vb_table_near_x``) with z under that line is unreachable BY CONSTRUCTION — its
        signature is a per-side return rate pinned at exactly 0 with nothing else complaining, which
        is why this is a construction-time error and not a warning.
        人话:桌面以下没有球。目标框/参考击球点掉到桌面以下就当场报错,别再让某一侧的回球率
        无声地钉在 0.0000。
        """
        if float(x_hi) <= float(self.cfg.vb_table_near_x):
            return
        z_min = float(self.cfg.vb_table_surface_z) + self._vb_ball_r
        if float(z_lo) < z_min - 1e-9:
            raise ValueError(
                f"{source}: clip {self._clip_label(clip_id)} commands racket contact down to "
                f"z={float(z_lo):.4f} m, below the minimum legal contact height {z_min:.4f} m "
                f"(vb_table_surface_z={float(self.cfg.vb_table_surface_z):.4f} + ball radius "
                f"{self._vb_ball_r:.4f}). The commanded point reaches x={float(x_hi):.4f} m, past the "
                f"near table edge vb_table_near_x={float(self.cfg.vb_table_near_x):.4f} m, so it lies "
                f"inside/below the table top where no ball can ever be — raise the z floor to "
                f"{z_min:.2f} m"
            )

    def _assert_target_velocity_points_forward(self) -> None:
        """Fail closed when a per-clip target-velocity box can demand a non-forward return.

        +x is toward the opponent (geometry.py world frame), so ``x_lo <= 0`` lets the sampler command
        a return that stalls or travels back at the robot — a shot that can never land on the far half
        however well the policy tracks it.
        """
        if self.cfg.allow_non_forward_target_velocity:
            return
        for clip_id, clip_rng in enumerate(self.cfg.racket_vel_range_per_clip):
            x_lo = float(clip_rng[0][0])
            if x_lo <= 0.0:
                raise ValueError(
                    f"racket_vel_range_per_clip: clip {self._clip_label(clip_id)} has "
                    f"x_lo={x_lo:.4f} <= 0, so the commanded return velocity can point away from the "
                    f"opponent (+x is toward the opponent). Set "
                    f"allow_non_forward_target_velocity=True to command this deliberately"
                )

    def _assert_incoming_ball_boxes_are_sane(self) -> None:
        """Fail closed when a per-clip INCOMING-ball box cannot describe a ball coming at us.

        人话:来球得朝机器人飞。``vb_vel_x_range`` 的 -x 是"冲着机器人来";per-clip 表里哪个 clip
        写成了 x_hi >= 0,那个 clip 的球就可能永远不到拍子跟前,回球率会无声钉在 0——和目标框掉到
        桌面以下同一类的构造期错误,所以当场报错,不 warning。
        """
        for clip_id, clip_rng in enumerate(self.cfg.vb_vel_range_per_clip):
            (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = clip_rng
            if float(x_hi) >= 0.0:
                raise ValueError(
                    f"vb_vel_range_per_clip: clip {self._clip_label(clip_id)} has "
                    f"x_hi={float(x_hi):.4f} >= 0, but the incoming ball travels toward the robot "
                    f"(-x). A non-negative ceiling lets the sampler launch a ball that flies AWAY, "
                    f"which no stroke can ever answer"
                )
            for name, (lo, hi) in (("x", (x_lo, x_hi)), ("y", (y_lo, y_hi)), ("z", (z_lo, z_hi))):
                if float(lo) > float(hi):
                    raise ValueError(
                        f"vb_vel_range_per_clip: clip {self._clip_label(clip_id)} axis {name} has "
                        f"lo={float(lo):.4f} > hi={float(hi):.4f} — an empty box"
                    )

    def _assert_reference_strike_can_return_its_own_regime(
        self, clip_id: int, p_contact_w, v_racket_w, n_racket_w, face_sign: float = 1.0
    ) -> None:
        """Fail closed when a bound strike frame cannot return ANY ball from THAT clip's regime.

        人话(最重要的一道闸):把这个 clip 自己的来球箱采一批球,用仓库自己的 NumPy 记分器
        (virtual_return_scorer,和训练里评分用的是同一套接触+落点模型)算一下:这个 clip 绑定的
        参考击球状态到底能不能把自己的球打回台上。一个都打不回就当场报错、点名是哪个 clip、
        打回率多少——不要再让某一侧的回球率无声地钉在 0.0000。

        Scored with the repo's OWN NumPy return scorer so the gate and the in-training metric cannot
        disagree; the acceptance threshold is ``reference_return_gate_min_rate`` (0.0 = gate off).
        """
        min_rate = float(getattr(self.cfg, "reference_return_gate_min_rate", 0.0))
        if min_rate <= 0.0:
            return
        try:
            from reference_return_gate import score_reference_returns
        except Exception as exc:                            # pragma: no cover - import path varies
            raise ValueError(
                "reference_return_gate_min_rate > 0 but the repo's NumPy return scorer wrapper "
                "(scripts/reference_return_gate.py: score_reference_returns, built on "
                f"scripts/virtual_return_scorer.py) could not be imported ({exc}). The gate must "
                "not be silently skipped — put scripts/ on sys.path or set "
                "reference_return_gate_min_rate=0.0 deliberately"
            )
        # 采哪个箱子必须跟着"谁在出球"走。连续路径下 vb_vel_range_per_clip 是死的(通用采样器被
        # _solved_targets_active 关掉了),真正出球的是 cq_vel_range_per_clip;不改这里,闸门会拿一
        # 个一颗球都不来自的分布判绿 —— 它不拒绝也不报警,它换了个问题回答,比失效更糟。
        if self._cq_enabled:
            box = self.cfg.cq_vel_range_per_clip[clip_id]
            spin_max = float(self.cfg.cq_spin_abs_max)
        else:
            box = self.cfg.vb_vel_range_per_clip[clip_id] if (
                getattr(self.cfg, "vb_vel_range_per_clip", None) is not None
            ) else (self.cfg.vb_vel_x_range, self.cfg.vb_vel_y_range, self.cfg.vb_vel_z_range)
            spin_max = (
                float(self.cfg.vb_spin_abs_max_per_clip[clip_id])
                if getattr(self.cfg, "vb_spin_abs_max_per_clip", None) is not None
                else float(self.cfg.vb_spin_abs_max)
            )
        rate = float(score_reference_returns(
            p_contact_w=p_contact_w, v_racket_w=v_racket_w, n_racket_w=n_racket_w,
            vel_box=box, spin_abs_max=spin_max,
            surface_z=float(self.cfg.vb_table_surface_z),
            near_x=float(self.cfg.vb_table_near_x),
            n_samples=int(getattr(self.cfg, "reference_return_gate_samples", 256)),
            seed=int(getattr(self.cfg, "reference_return_gate_seed", 0)),
            face_sign=float(face_sign),
        ))
        if rate < min_rate:
            raise ValueError(
                f"reference strike gate: clip {self._clip_label(clip_id)} returns only "
                f"{rate:.4f} of the balls in its OWN incoming regime "
                f"(vel box {tuple(tuple(float(v) for v in ax) for ax in box)}, |spin| <= "
                f"{spin_max:.1f} rad/s), below reference_return_gate_min_rate={min_rate:.4f}. "
                f"The bound strike frame cannot answer the balls this clip will be posed — fix the "
                f"strike phase, the clip, or that clip's ball box; a 0.0000 per-clip return rate "
                f"must never be discovered from a training curve"
            )

    def _strike_phases_cfg(self, nseg: int) -> tuple:
        """cfg.strike_phase_per_clip validated against the loaded motion's segment count (fail-loud).

        Empty tuple = use the scalar strike_phase for every clip (the documented default). Any OTHER
        length means the per-clip table and the loaded clips no longer line up — every env would
        strike at a wrong frame — so raise instead of silently falling back to the global phase.
        人话:每个 clip 的击球点表和实际加载的 clip 数对不上就当场报错,不再悄悄用全局值凑合。
        """
        spc = tuple(self.cfg.strike_phase_per_clip)
        if spc and len(spc) != nseg:
            raise ValueError(
                f"strike_phase_per_clip has {len(spc)} entries but the loaded motion has {nseg} "
                f"segment(s) — align it with the motion_file clip order, or set () to use "
                f"strike_phase={self.cfg.strike_phase} for every clip"
            )
        # _strike_frame_for_clip adds round(p * (seg_len - 1)) to the segment start with NO clamp, so a
        # phase outside [0, 1] silently reads a NEIGHBOURING clip's pose as this clip's strike frame.
        # TODO: the stronger check — the phase must land inside the clip's compiled contact window —
        # needs the binding manifest, which the plain motion_file path does not carry.
        out_of_range = [(i, float(p)) for i, p in enumerate(spc) if not 0.0 <= float(p) <= 1.0]
        if out_of_range:
            raise ValueError(
                f"strike_phase_per_clip entries must lie in [0, 1] (fraction along the clip); "
                f"got {out_of_range} as (clip index, value)"
            )
        return spc

    def _mount_signs_cfg(self, nseg: int) -> tuple:
        """cfg.mount_normal_sign_per_clip validated against the loaded motion's segment count (fail-loud).

        人话:每个 clip 的击球面符号表和实际加载的 clip 数对不上就当场报错(照 _strike_phases_cfg
        先例),不悄悄退回标量符号凑合——那样反手又会被按错误的一面判分还不吭声。空表 = 全部 clip 用
        标量 mount_normal_sign(文档化的默认,现役行为逐位不变)。符号只认 ±1:0 会把法向悄悄清零,
        其他值会把"单位法向"变成带模长的向量,奖励核和角度误差全被污染,所以也当场报错。
        """
        mns = tuple(self.cfg.mount_normal_sign_per_clip)
        if mns and len(mns) != nseg:
            raise ValueError(
                f"mount_normal_sign_per_clip has {len(mns)} entries but the loaded motion has {nseg} "
                f"segment(s) — align it with the motion_file clip order, or set () to use "
                f"mount_normal_sign={self.cfg.mount_normal_sign} for every clip"
            )
        if any(float(s) not in (1.0, -1.0) for s in mns):
            raise ValueError(
                f"mount_normal_sign_per_clip entries must be +1 or -1 (which paddle FACE strikes), "
                f"got {mns}"
            )
        return mns

    def _clip_family_is_forehand(self) -> torch.Tensor:
        """[num_segments] bool 表:True = 该 clip 属正手家族(懒构建 + fail-loud)。

        人话:替换四处写死的"clips==0 才是正手"判断(spdmix v2 硬绑定一)。没配
        task.motion.clip_family_per_clip 时按老规矩推——单 clip 当正手、恰好 2 clip = (正手, 反手),
        和写死判断逐字节同值(现役所有在跑臂);≥3 clip 缺表当场报错,不猜。配了表按表取:同一家族的
        变速 clip(如正手 0.8/1.0/1.2)swing_sign、swing_type 观测、目标侧全一致。解析/校验规则的
        单一来源是 commands.resolve_clip_family_is_forehand(和 MotionCommand 开机校验同一份)。
        """
        if self._family_is_forehand_t is None:
            motion = self._motion()
            self._family_is_forehand_t = torch.tensor(
                resolve_clip_family_is_forehand(
                    getattr(getattr(motion, "cfg", None), "clip_family_per_clip", None),
                    int(motion.motion.num_segments),
                ),
                dtype=torch.bool,
                device=self.device,
            )
        return self._family_is_forehand_t

    def _clip_family_rows(self) -> torch.Tensor:
        """[num_segments] long 表:每个加载 clip 的家族行号(0=正手族, 1=反手族;懒构建缓存)。

        人话(spdmix v2 硬绑定四配套):HER 缓存、per-族指标累计器、"每族一行"的框表全都按
        家族分桶。legacy 2-clip 下行号恒等于 clip_id(正手=0、反手=1),所有消费点逐字节不变;
        6-clip 变速列表里正手 0.8/1.0/1.2 三档同落 0 行、反手三档同落 1 行。单一来源 =
        _clip_family_is_forehand()(boot 已整表校验;≥3 clip 缺族表在那里当场报人话错误)。
        """
        rows = getattr(self, "_clip_family_rows_t", None)
        if rows is None:
            rows = (~self._clip_family_is_forehand()).to(dtype=torch.long)
            self._clip_family_rows_t = rows
        return rows

    def _metric_bucket_rows(self) -> torch.Tensor:
        """[num_segments] long:每个 clip 的**指标桶**行号(缓存)。

        人话(硬绑定四的下一步):`cfg.clip_names_per_clip` 没配时,桶就是家族行号——现役所有在跑
        臂逐字节不变。配了名字表,桶就是 clip_id 本身,每个动作一个桶。家族语义仍归
        `_clip_family_rows()`:回放/框表按族展开/题库按族寻址都不动。
        """
        # getattr defaults, like _clip_family_rows: the older unit-test fakes build the command with
        # __new__ and set only a handful of fields, and a metric bucket must never be the thing that
        # explodes on them.
        rows = getattr(self, "_metric_bucket_rows_t", None)
        if rows is not None:
            return rows
        if not getattr(self, "_metric_buckets_per_clip", False):
            rows = self._clip_family_rows()
        else:
            nseg = int(self._motion().motion.num_segments)
            if nseg != len(self._clip_names):
                raise ValueError(
                    f"racket_target.clip_names_per_clip has {len(self._clip_names)} name(s) "
                    f"{sorted(self._clip_names.values())} but the loaded motion has {nseg} "
                    f"clip(s) — give one name per loaded clip, in motion_file order; a short "
                    f"name table would put two clips in one metric bucket"
                )
            rows = torch.arange(nseg, dtype=torch.long, device=self.device)
        self._metric_bucket_rows_t = rows
        return rows

    def _per_clip_range_rows(self, table: torch.Tensor, cfg_key: str) -> torch.Tensor:
        """把 cfg 的 per-clip 框表对齐成"每个加载 clip 一行"((num_segments, 3, 2),缓存)。

        人话(spdmix v2 硬绑定四 = 六 clip boot 炸的真凶):racket_pos/vel_range_per_clip 只有
        正/反手两行,六 clip 列表拿 clip_id(0..5)直接进 GPU gather,越界不当场报错,而是异步
        CUDA device-side assert,尸体倒在下一个同步点(_strike_frame_for_clip 的 .item()),
        traceback 指向无辜的 seg_start。这里在任何 gather 之前按行数核对:

        * 行数 == 加载段数:按 clip 逐行用(legacy 2-clip 原张量原样返回,逐字节不变);
        * 行数 == 2 且段数 != 2:按 clip_family_per_clip 族表把"每族一行"展开成"每 clip 一行"
          (同族变速档共享该族的框;族表缺席时 _clip_family_is_forehand 自己报人话错误);
        * 其他组合:当场 ValueError 报人话,绝不让越界下标进 GPU。
        """
        cache = getattr(self, "_per_clip_range_rows_cache", None)
        if cache is None:
            cache = {}
            self._per_clip_range_rows_cache = cache
        cached = cache.get(cfg_key)
        if cached is not None:
            return cached
        nseg = int(self._motion().motion.num_segments)
        rows = int(table.shape[0])
        if rows == nseg:
            expanded = table
        elif rows == 2:
            # 每族一行:按族表展开(0=正手行, 1=反手行)。族表缺席时下一行当场报人话错误。
            expanded = table[self._clip_family_rows()]
        else:
            raise ValueError(
                f"task.racket.{cfg_key} has {rows} row(s) but the loaded motion has {nseg} "
                f"clip(s) — give one row per loaded clip (motion_file then motion_file_2 order), "
                f"or exactly 2 rows (forehand, backhand) plus task.motion.clip_family_per_clip "
                f"so each speed variant reuses its family's box"
            )
        cache[cfg_key] = expanded
        return expanded

    # ------------------------------------------------------------------ continuous questions ---
    @property
    def _solved_targets_active(self) -> bool:
        """目标是不是"逆解出来的答案"(题库行 或 连续缓冲行)。

        人话:仓库里有一批门,写的是"有没有题库",想问的其实是"目标是不是解出来的"。那批门必须
        改问这个属性,否则连续路径下它们会**静默失效** —— 而一个静默失效的门,正是"解出来的目标
        最后被什么都不打分"的走法。判据用 cfg 级的 ``_cq_enabled`` 而不是缓冲对象,因为缓冲是
        懒建的:第一次 resample 之前它还是 None,而那些门在更早就要给出正确答案。
        """
        return self._question_bank is not None or self._cq_enabled

    def _cq_prm(self):
        """Venue ball params, shared with the scorer's own lazy load (one model, one object)."""
        if self._vb_params is None:
            from whole_body_tracking.tasks.tracking.mdp import virtual_ball as _vb
            self._vb_params = _vb.load_venue_params()
            print(
                f"[RacketTargetCommand] virtual ball ON (continuous solve): venue constants from "
                f"{self._vb_params.source_path}",
                flush=True,
            )
        return self._vb_params

    def _cq_planes(self) -> tuple[float, float, float]:
        """(surface_z, net_x, net_top_z) — THE SAME NUMBERS THE SCORER USES, read from the same
        fields, never re-derived.

        人话:"你解的那个面,就是你被打分的那个面"是靠共用同一个属性保证的,不是靠两处各写一遍
        同一个数字。表面用 surface + 球半径(球心过平面),和 _vb_evaluate 的 :4493 / 题库生成器
        逐字一致;裸 0.76 会让求解落在离评分面 20 mm 的另一张桌子上。
        """
        prm = self._cq_prm()
        return (
            float(self.cfg.vb_table_surface_z) + float(prm.ball_radius),
            float(self._vb_net_x),
            float(self._vb_net_top_z) + float(self._vb_ball_r),
        )

    def _cq_ref_normals(self) -> torch.Tensor:
        """(nseg, 3) RUNTIME raw +Y clip face — what the continuous answers are sign-aligned to.

        题库是离线对齐到 npz 里烤死的 clip_normal、再由 _check_question_bank_face_frame 在运行时
        核 dot>0;连续路径直接对齐到**运行时**这张 FK 面,所以那条 dot>0 卫兵在这里恒真,漂移
        无从发生(没有可以对不上的第二份数)。
        """
        self._ensure_reference_strike_state()
        ref = self._ref_racket_normal_raw_w_per_clip
        if ref is None:
            raise RuntimeError("reference strike-state initialization produced no raw face normals")
        return ref

    def _cq_build_cfg(self, nseg: int):
        """Freeze the ContinuousQuestionCfg for this run (once). Every field comes from cfg."""
        if self._cq_cfg is not None:
            return self._cq_cfg
        from .continuous_questions import ContinuousQuestionCfg

        if self._pos_range_per_clip_t is None:
            raise ValueError(
                "target_mode='solved' needs racket.racket_pos_range_per_clip as the contact-point "
                "draw box"
            )
        pos_rows = self._per_clip_range_rows(
            self._pos_range_per_clip_t, "racket_pos_range_per_clip")
        if self.cfg.cq_vel_range_per_clip is None:
            raise ValueError("target_mode='solved' needs racket.cq_vel_range_per_clip")
        vel_t = torch.tensor(
            [[[float(lo), float(hi)] for (lo, hi) in clip_rng]
             for clip_rng in self.cfg.cq_vel_range_per_clip],
            dtype=torch.float32, device=self.device,
        )
        vel_rows = self._per_clip_range_rows(vel_t, "cq_vel_range_per_clip")
        aim = self.cfg.cq_aim_xy
        aim_x, aim_y = (
            (float(self.cfg.vb_target_x), float(self.cfg.vb_target_y)) if aim is None
            else (float(aim[0]), float(aim[1]))
        )
        self._cq_cfg = ContinuousQuestionCfg(
            vel_range_per_clip=vel_rows,
            pos_range_per_clip=pos_rows,
            spin_abs_max=float(self.cfg.cq_spin_abs_max),
            # DEGENERATE aim on purpose — the single point the landing reward / virtual_land_err_m /
            # the physical ball's return flight all grade against (see cq_aim_xy's cfg comment).
            aim_x_range=(aim_x, aim_x),
            aim_y_range=(aim_y, aim_y),
            tol_m=float(self.cfg.cq_tol_m),
            n_iters=int(self.cfg.cq_n_iters),
            speed_budget=float(self.cfg.cq_speed_budget),
            max_redraw_rounds=int(self.cfg.cq_max_redraw_rounds),
            fixed_direction=False,
        )
        del nseg
        return self._cq_cfg

    def _cq_bucket_of(self, v_in_x: torch.Tensor, clip_rows: torch.Tensor) -> torch.Tensor:
        """Per-regime bucket index = |v_in_x| bin crossed with the clip. (Both axes are a CHOICE.)

        人话:这是唯一能看见"重画循环把难的那条尾巴悄悄削掉"的仪表 —— 有 overdraw 之后
        ``exhausted`` 会整轮读 0,而全局 reject 直方图看不出是哪一档球消失了。
        """
        nb = max(1, int(self.cfg.cq_accept_buckets))
        lo = self._cq["vx_lo"][clip_rows]
        hi = self._cq["vx_hi"][clip_rows]
        u = ((v_in_x.abs() - lo) / (hi - lo).clamp(min=1e-6)).clamp(0.0, 1.0 - 1e-6)
        return clip_rows * nb + (u * nb).long()

    def _cq_note(self, name: str, amount) -> None:
        ledger = self._ensure_exact_behavior_decision_counters()
        if name in ledger:
            ledger[name] += int(amount)

    @torch.no_grad()
    def _cq_refill(self) -> None:
        """Solve ONE big batch and refill the buffer. Only ``ok=True`` rows are ever admitted.

        人话:失败策略在代码里的落点只有这一条线 —— 缓冲只收解出来的行。生产者把无解行整行填
        NaN,所以就算这条掩码哪天被删,装进去的也是 NaN 而不是一个看起来很像答案的东西;两层都
        在,才叫结构性。
        """
        from .continuous_questions import generate

        if self._cq is not None and "count" in self._cq:
            # A refill throws away whatever the old buffer had left (a drought refill can happen
            # before the cursor reaches the end). Book it, so "the reservoir is provably not a
            # small bank" stays auditable from the hourly log rather than being assumed.
            self._cq_note(
                "continuous_question_rows_discarded_count",
                int((self._cq["count"] - self._cq["cursor"]).clamp(min=0).sum()),
            )
        motion = self._motion()
        nseg = int(motion.motion.num_segments)
        cqcfg = self._cq_build_cfg(nseg)
        prm = self._cq_prm()
        surface_z, net_x, net_top_z = self._cq_planes()
        ref_all = self._cq_ref_normals()[:nseg]                      # (nseg,3)
        P = int(self.cfg.cq_buffer_rows)
        per = int(math.ceil(P * float(self.cfg.cq_overdraw)))
        clip_ids = torch.arange(nseg, device=self.device).repeat_interleave(per)
        if self._cq_gen is None:
            # Dedicated stream so the question draw is reproducible independently of every other
            # sampler's consumption pattern (the resume path stores its state; see _cq_state_dict).
            self._cq_gen = torch.Generator(device=self.device)
            self._cq_gen.manual_seed(int(self.cfg.cq_seed))
        res = generate(
            clip_ids, prm, surface_z=surface_z, net_x=net_x, cfg=cqcfg,
            ref_normal=ref_all[clip_ids], net_top_z=net_top_z, generator=self._cq_gen,
            # ONE source for the rollout parameters: the scorer's. They happen to equal the
            # solver's defaults today, which is exactly the kind of coincidence that rots.
            h=float(self.cfg.vb_rollout_h), n_steps=int(self.cfg.vb_rollout_steps),
        )
        n_draw = int(clip_ids.shape[0])
        self._cq_note("continuous_question_draw_count", n_draw)
        self._cq_note("continuous_question_exhausted_count", int(res.exhausted))
        self._cq_note("continuous_question_refill_count", 1)
        self._cq_note("continuous_question_redraw_round_sum", int(res.rounds_used))
        for name, cnt in (res.reason_counts or {}).items():
            self._cq_note(f"continuous_question_reject_{name}_count", int(cnt))

        keep = res.ok.clone()
        # FACE CAP. After sign alignment dot>0 is tautological, so the residual risk is "same side
        # but wildly off"; this is the knob that watches it. Shipped banks: p50 7.8-9.1, max 20.3.
        ang = torch.rad2deg(torch.arccos(
            torch.sum(torch.nan_to_num(res.n_racket) * ref_all[clip_ids], dim=-1).clamp(-1.0, 1.0)))
        face_bad = keep & (ang > float(self.cfg.cq_max_face_deg))
        self._cq_note("continuous_question_reject_face_deg_over_cap_count", int(face_bad.sum()))
        keep = keep & ~face_bad
        # EXAM HOLDOUT. Keep the repo invariant that no TRAINED question is an exam question, so a
        # paired exam comparison is not grading the policy on balls it trained on.
        if bool(self.cfg.cq_exam_holdout):
            from .stage1_question_bank import question_split

            v_np = torch.nan_to_num(res.v_ball_in).detach().cpu().numpy()
            is_exam = torch.tensor(
                [question_split(row) == "exam" for row in v_np],
                dtype=torch.bool, device=self.device,
            ) & keep
            self._cq_note("continuous_question_reject_exam_holdout_count", int(is_exam.sum()))
            keep = keep & ~is_exam

        exh_frac = float(res.exhausted) / max(1, n_draw)
        self._cq_last_exhausted_frac = exh_frac
        if exh_frac > float(self.cfg.cq_abort_exhausted_frac):
            raise RuntimeError(
                f"continuous question refill: {res.exhausted}/{n_draw} drawn balls have NO legal "
                f"answer after {res.rounds_used} redraw round(s) ({exh_frac * 100:.1f}% > "
                f"{float(self.cfg.cq_abort_exhausted_frac) * 100:.0f}%). reasons={res.reason_counts}. "
                f"This is a task-construction error (an incoming box that cannot be returned to the "
                f"aim point within the speed budget), not a hyperparameter — fix "
                f"racket.cq_vel_range_per_clip / racket_pos_range_per_clip / cq_speed_budget."
            )
        if exh_frac > float(self.cfg.cq_max_exhausted_frac):
            print(
                f"[RacketTargetCommand] WARN continuous refill: exhausted {res.exhausted}/{n_draw} "
                f"({exh_frac * 100:.1f}%) reasons={res.reason_counts}",
                flush=True,
            )

        # --- per-clip buffers, cursor at 0 ------------------------------------------------------
        vel_rows = cqcfg.vel_range_per_clip
        vx_lo = torch.minimum(vel_rows[:, 0, 0].abs(), vel_rows[:, 0, 1].abs())
        vx_hi = torch.maximum(vel_rows[:, 0, 0].abs(), vel_rows[:, 0, 1].abs())
        if self._cq is None:
            self._cq = {"vx_lo": vx_lo, "vx_hi": vx_hi}
        self._cq["vx_lo"], self._cq["vx_hi"] = vx_lo, vx_hi

        nb = max(1, int(self.cfg.cq_accept_buckets))
        asked_b = self._cq_bucket_of(torch.nan_to_num(res.attempted_v_ball_in)[:, 0], clip_ids)
        asked = torch.bincount(asked_b, minlength=nseg * nb)
        kept_b = self._cq_bucket_of(torch.nan_to_num(res.v_ball_in)[keep][:, 0], clip_ids[keep])
        kept = torch.bincount(kept_b, minlength=nseg * nb)
        for c in range(nseg):
            for b in range(nb):
                idx = c * nb + b
                self._cq_note(f"continuous_question_bucket{c}_{b}_asked_count", int(asked[idx]))
                self._cq_note(f"continuous_question_bucket{c}_{b}_kept_count", int(kept[idx]))
                a_i, k_i = int(asked[idx]), int(kept[idx])
                if a_i >= 2000 and k_i == 0:
                    raise RuntimeError(
                        f"continuous question refill: clip {self._clip_label(c)} |v_in_x| bucket "
                        f"{b}/{nb} asked {a_i} balls and kept ZERO. The declared incoming box has "
                        f"a regime with no legal answer at all, so the run's real distribution is "
                        f"not the one its yaml declares. Narrow cq_vel_range_per_clip or raise "
                        f"cq_speed_budget."
                    )
                if a_i >= 500 and k_i > 0 and (k_i / a_i) < float(self.cfg.cq_min_accept_rate):
                    print(
                        f"[RacketTargetCommand] WARN continuous accept: clip "
                        f"{self._clip_label(c)} |v_in_x| bucket {b}/{nb} accept "
                        f"{k_i}/{a_i} = {k_i / a_i:.3f} < {float(self.cfg.cq_min_accept_rate)} — "
                        f"the hard tail of the declared box is being thinned out by the redraw "
                        f"loop, and `exhausted` cannot show it",
                        flush=True,
                    )

        buf_pos, buf_vin, buf_win, buf_vr, buf_nr, buf_diff, buf_cnt = [], [], [], [], [], [], []
        discarded = 0
        for c in range(nseg):
            m = keep & (clip_ids == c)
            idx = torch.nonzero(m, as_tuple=False).flatten()
            if idx.numel() > P:
                discarded += int(idx.numel() - P)
                idx = idx[:P]
            if idx.numel() == 0:
                raise RuntimeError(
                    f"continuous question refill produced ZERO usable rows for clip "
                    f"{self._clip_label(c)} out of {per} draws — refusing to hand out a target "
                    f"that was never solved. reasons={res.reason_counts}"
                )
            buf_pos.append(res.p_contact[idx])
            buf_vin.append(res.v_ball_in[idx])
            buf_win.append(res.w_ball_in[idx])
            buf_vr.append(res.v_racket[idx])
            buf_nr.append(res.n_racket[idx])
            buf_diff.append(ang[idx])
            buf_cnt.append(int(idx.numel()))
        width = max(buf_cnt)

        def _pad(rows_list, dim):
            out = torch.zeros(nseg, width, dim, device=self.device) if dim > 1 else \
                torch.zeros(nseg, width, device=self.device)
            for c, rows in enumerate(rows_list):
                out[c, : rows.shape[0]] = rows
            return out

        self._cq.update({
            "pos": _pad(buf_pos, 3), "v_in": _pad(buf_vin, 3), "w_in": _pad(buf_win, 3),
            "v_r": _pad(buf_vr, 3), "n_r": _pad(buf_nr, 3), "diff": _pad(buf_diff, 1),
            "count": torch.tensor(buf_cnt, dtype=torch.long, device=self.device),
            "cursor": torch.zeros(nseg, dtype=torch.long, device=self.device),
        })
        n_admit = int(sum(buf_cnt))
        self._cq_note("continuous_question_admitted_count", n_admit)
        self._cq_note("continuous_question_rows_discarded_count", discarded)
        self._cq_note(
            "continuous_question_resid_um_sum",
            int((torch.nan_to_num(res.resid_m[keep], posinf=0.0) * 1e6).sum()),
        )
        self._cq_closed_loop_check()

    @torch.no_grad()
    def _cq_closed_loop_check(self) -> None:
        """Re-roll a subsample of the ADMITTED rows through the SCORER's own call and hard-fail.

        人话:题库是离线验一次(``validation.max_landing_error_m <= 0.10``,
        stage1_question_bank.py:402-413);连续路径每换一批都验一次。诚实说明范围:求解器和评分器
        今天用同一套 h/n_steps(下面 boot 时断言),所以这条检查抓的是**平面/帧/参数漂移**,不是
        积分器分歧 —— 它是唯一能抓到"求解的那张桌子和打分的那张桌子不是同一张"的东西,而 resid_m
        结构上抓不到(它是自评的)。
        """
        rows = int(self.cfg.cq_closed_loop_rows)
        if rows <= 0 or self._cq is None:
            return
        from whole_body_tracking.tasks.tracking.mdp import virtual_ball as _vb

        prm = self._cq_prm()
        surface_z, net_x, _net_top = self._cq_planes()
        pos = self._cq["pos"].reshape(-1, 3)
        cnt_mask = torch.zeros(pos.shape[0], dtype=torch.bool, device=self.device)
        width = int(self._cq["pos"].shape[1])
        for c, k in enumerate(self._cq["count"].tolist()):
            cnt_mask[c * width: c * width + int(k)] = True
        idx = torch.nonzero(cnt_mask, as_tuple=False).flatten()
        if idx.numel() == 0:
            return
        take = idx[torch.randperm(idx.numel(), device=self.device)[:rows]]
        v_plus, w_plus = _vb.predict_paddle_contact(
            self._cq["v_in"].reshape(-1, 3)[take], self._cq["v_r"].reshape(-1, 3)[take],
            self._cq["n_r"].reshape(-1, 3)[take], self._cq["w_in"].reshape(-1, 3)[take], prm,
        )
        land = _vb.coarse_landing(
            pos[take], v_plus, w_plus, prm, surface_z=surface_z, net_x=net_x,
            h=float(self.cfg.vb_rollout_h), n_steps=int(self.cfg.vb_rollout_steps),
        )
        err = torch.linalg.norm(land["land_xy"] - self._vb_target_xy.unsqueeze(0), dim=-1)
        err = torch.where(land["land_valid"], err, torch.full_like(err, float("inf")))
        worst = float(err.max())
        self._cq_note("continuous_question_closed_loop_um_sum",
                      int((torch.nan_to_num(err, posinf=0.0) * 1e6).sum()))
        self._cq_note("continuous_question_closed_loop_row_count", int(take.numel()))
        bad = int((err > float(self.cfg.cq_closed_loop_max_err_m)).sum())
        self._cq_note("continuous_question_closed_loop_fail_count", bad)
        if bad:
            raise RuntimeError(
                f"continuous question refill failed its own closed-loop check: {bad}/"
                f"{int(take.numel())} admitted answers land more than "
                f"{float(self.cfg.cq_closed_loop_max_err_m)} m from the aim when re-rolled through "
                f"the SCORER's call (worst {worst:.4f} m). The solver and the scorer are not "
                f"looking at the same table/net/frame."
            )

    @torch.no_grad()
    def _cq_take(self, clip_rows: torch.Tensor, n: int):
        """Hand out ``n`` rows WITHOUT REPLACEMENT, same 6-tuple shape as ``select_questions``.

        人话:每道题只发一次,发完就丢。这不是题库,是缓冲。改成有放回抽取 = 它立刻退化成一张
        8192 行的滚动题库,也就是 owner 明确否掉的那个东西,而且没有任何断言会响。
        """
        if self._cq is None or "count" not in self._cq:
            self._cq_refill()
        nseg = int(self._cq["count"].shape[0])
        need = torch.bincount(clip_rows, minlength=nseg)
        left = self._cq["count"] - self._cq["cursor"]
        if bool((need > left).any()):
            # DROUGHT: refill synchronously for the shortfall. No branch reuses a row, reorders by
            # resid, or substitutes — if the fresh batch is still short, that is a raise.
            self._cq_note("continuous_question_pool_dry_count", 1)
            self._cq_refill()
            left = self._cq["count"] - self._cq["cursor"]
            if bool((need > left).any()):
                self._cq_note("continuous_question_pool_underflow_count", 1)
                raise RuntimeError(
                    f"continuous question buffer underflow after a fresh refill: need "
                    f"{need.tolist()} rows per clip, have {left.tolist()}. Raise "
                    f"racket.cq_buffer_rows / cq_overdraw. Nothing was reused or substituted."
                )
        out_pos = torch.zeros(n, 3, device=self.device)
        out_vin = torch.zeros(n, 3, device=self.device)
        out_win = torch.zeros(n, 3, device=self.device)
        out_vr = torch.zeros(n, 3, device=self.device)
        out_nr = torch.zeros(n, 3, device=self.device)
        out_diff = torch.zeros(n, device=self.device)
        for c in range(nseg):
            sel = torch.nonzero(clip_rows == c, as_tuple=False).flatten()
            k = int(sel.numel())
            if k == 0:
                continue
            start = int(self._cq["cursor"][c])
            rows = torch.arange(start, start + k, device=self.device)
            out_pos[sel] = self._cq["pos"][c, rows]
            out_vin[sel] = self._cq["v_in"][c, rows]
            out_win[sel] = self._cq["w_in"][c, rows]
            out_vr[sel] = self._cq["v_r"][c, rows]
            out_nr[sel] = self._cq["n_r"][c, rows]
            out_diff[sel] = self._cq["diff"][c, rows]
            self._cq["cursor"][c] = start + k          # 游标只前进,行发过就永远不再出现
        self._cq_note("continuous_question_install_count", n)
        return out_pos, out_vin, out_win, out_vr, out_nr, out_diff

    @torch.no_grad()
    def _cq_boot_gate(self) -> None:
        """Boot-time contract anchor + parity gate for the continuous path. Runs ONCE, lazily.

        人话:连续训练把题库的四条溯源丢了两条,这里把它们拿回来 —— 用一份**同族的、完整校验过
        的**题库当**契约锚**:加载它就跑物理契约(ball physics 改了 = 和 bank 臂一模一样的开机
        报错,而不是训练分布悄悄变了)和运行时动作契约(motion SHA / 段长 / strike_phase)。它
        **一行都不参与训练**,只做两件事:锁契约,和给下面这个 parity 断言当数据。

        parity 用的是 pytest 里那**同一个函数**(continuous_questions.parity_report),所以"两条
        路同物理"从一句离线 pytest 声明,变成这台机器、这个 commit 上的既成事实。
        """
        self._cq_boot_gate_done = True          # 先置位:失败要炸,不要变成每步重试
        path = str(getattr(self.cfg, "cq_anchor_bank", "") or "").strip()
        if not path:
            # 到不了这里:_assert_solved_target_recipe_is_coherent 已经把空锚拦在构造期了。留着是
            # 因为"必须开的东西"只该有一条政策 —— 不能上面报错、这里降级成一行 WARN。
            raise ValueError(
                "continuous questions cannot run WITHOUT a contract anchor "
                "(racket.cq_anchor_bank is empty): no physics-contract SHA, no runtime motion "
                "contract, and no boot parity gate. Point it at a same-family schema-v3 train "
                "bank — it is never trained on."
            )
        from .continuous_questions import parity_report

        motion = self._motion()
        nseg = int(motion.motion.num_segments)
        _order = _peek_question_bank_clip_order(path)
        _names = tuple(_order) if _order else tuple(
            str(x) for x in (getattr(self.cfg, "clip_names_per_clip", ()) or ())
        ) or ("forehand", "backhand")
        # The loader itself runs validate_runtime_physics_contract + the split check.
        anchor = load_question_bank(path, device=self.device, clip_names=_names,
                                    allow_legacy=False, expected_split="train")
        family_table = self._qb_bank_family_table(motion, anchor)
        if anchor.metadata:
            files = ([motion.cfg.motion_file] if isinstance(motion.cfg.motion_file, str)
                     else list(motion.cfg.motion_file))
            phases_cfg = self._strike_phases_cfg(nseg)
            phases = ([float(v) for v in phases_cfg] if phases_cfg
                      else [float(self.cfg.strike_phase)] * nseg)
            validate_runtime_motion_contract(
                anchor.metadata, files,
                [int(v) for v in motion.motion.seg_len.tolist()], phases,
                clip_families=(family_table.tolist() if family_table is not None else None),
            )
        self._cq_anchor = anchor

        prm = self._cq_prm()
        surface_z, net_x, net_top_z = self._cq_planes()
        ref_all = self._cq_ref_normals()[:nseg]
        meta = anchor.metadata or {}
        land = meta.get("landing_env") or [float(self.cfg.vb_target_x), float(self.cfg.vb_target_y)]
        aim_pt = torch.tensor([float(land[0]), float(land[1])], device=self.device)
        budget = float(meta.get("speed_budget") or 4.0)
        K = 128
        for c in range(nseg):
            fam = c if family_table is None else int(family_table[c])
            q = int(anchor.counts[fam])
            if q <= 0:
                continue
            k = min(K, q)
            rows = torch.arange(k, device=self.device)
            p_c = anchor.contact_pos[fam].unsqueeze(0).expand(k, 3)
            rep = parity_report(
                p_c, anchor.incoming_vel[fam, rows], anchor.incoming_spin[fam, rows],
                aim_pt.unsqueeze(0).expand(k, 2), ref_all[c].unsqueeze(0).expand(k, 3),
                anchor.demanded_vel[fam, rows], anchor.demanded_normal[fam, rows],
                prm, surface_z=surface_z, net_x=net_x, net_top_z=net_top_z,
                speed_budget=budget, tol_m=0.005, n_iters=int(self.cfg.cq_n_iters),
                h=float(self.cfg.vb_rollout_h), n_steps=int(self.cfg.vb_rollout_steps),
            )
            if rep["failures"]:
                raise RuntimeError(
                    f"continuous-vs-bank boot parity gate FAILED on anchor {path!r} clip "
                    f"{self._clip_label(c)}:\n  - " + "\n  - ".join(rep["failures"])
                    + f"\nstats={rep['stats']}"
                )
            print(
                f"[RacketTargetCommand] continuous boot parity OK: clip {self._clip_label(c)} "
                f"{k} anchor rows, {rep['stats']}",
                flush=True,
            )

    def _cq_hard_contract(self) -> dict | None:
        """What produced this run's targets, for the checkpoint hard contract. None when off.

        人话:没有这一块,checkpoint 对"目标是怎么产生的"一个字都不记 —— 因为 bank_path_cfg 是
        空的,现在的 question_bank 直接是 None,可复现性会掉到前题库时代以下。
        """
        if not self._cq_enabled:
            return None
        from .stage1_question_bank import (
            canonical_sha256, runtime_physics_contract, sha256_file,
        )

        motion = self._motion()
        nseg = int(motion.motion.num_segments)
        cqcfg = self._cq_build_cfg(nseg)
        surface_z, net_x, net_top_z = self._cq_planes()
        declared = {
            "pos_range_per_clip": cqcfg.pos_range_per_clip.tolist(),
            "vel_range_per_clip": cqcfg.vel_range_per_clip.tolist(),
            "spin_abs_max": float(cqcfg.spin_abs_max),
            "aim_xy": [float(cqcfg.aim_x_range[0]), float(cqcfg.aim_y_range[0])],
            "exam_holdout": bool(self.cfg.cq_exam_holdout),
            "max_face_deg": float(self.cfg.cq_max_face_deg),
            "seed": int(self.cfg.cq_seed),
        }
        anchor = self._cq_anchor
        return {
            "producer": "continuous_inline_solve_v1",
            "cfg_sha256": canonical_sha256(declared),
            "declared": declared,
            "physics_contract_sha256": canonical_sha256(runtime_physics_contract()),
            "solver": {
                "n_iters": int(cqcfg.n_iters), "tol_m": float(cqcfg.tol_m),
                "speed_budget": float(cqcfg.speed_budget),
                "max_redraw_rounds": int(cqcfg.max_redraw_rounds),
                "rollout_h": float(self.cfg.vb_rollout_h),
                "rollout_steps": int(self.cfg.vb_rollout_steps),
            },
            # The three planes ACTUALLY handed to the solver, not the cfg values they came from.
            "planes": {"surface_z": surface_z, "net_x": net_x, "net_top_z": net_top_z},
            "ref_normal_source": "runtime_ref_racket_normal_raw_w_per_clip",
            "anchor_bank": (None if anchor is None else {
                "sha256": sha256_file(anchor.source_path),
                "schema_version": int((anchor.metadata or {}).get("schema_version", 0)),
                "split": (anchor.metadata or {}).get("split"),
                "source_family_sha256": (anchor.metadata or {}).get("source_family_sha256"),
                "trained_on": False,
            }),
        }

    def _cq_state_dict(self) -> dict:
        """Cursor/fill state for the exact-resume path (the buffer is run state, not derived)."""
        if self._cq is None or "count" not in self._cq:
            return {"filled": False}
        return {
            "filled": True,
            "cursor": self._cq["cursor"].tolist(),
            "count": self._cq["count"].tolist(),
            "generator": (None if self._cq_gen is None else self._cq_gen.get_state()),
        }

    def _qb_bank_family_table(self, motion, bank) -> torch.Tensor | None:
        """clip→题库族查表 ((nseg,) long;forehand 族=0 行, backhand 族=1 行),或 None。

        人话(spdmix v2 硬绑定二):题库永远只有正/反手两族题;6-clip 变速列表里每个 clip 先查
        自己属于哪族,再拿族号当题库下标。cfg.motion.clip_family_per_clip 没配(现役所有在跑臂)
        返回 None,调用方直接拿 clip_id 当题库下标——行为逐字节不变。配了表就复用
        _clip_family_is_forehand()(和 swing_sign 同一份 boot 已整表校验的表)折成题库行号;
        族寻址的允许 motion SHA 列表住在 schema-v3 元数据里,legacy 题库配族表当场报错,不猜。
        """
        if getattr(getattr(motion, "cfg", None), "clip_family_per_clip", None) is None:
            return None
        if self._qb_family_table is None:
            metadata = getattr(bank, "metadata", None)
            if not metadata:
                raise ValueError(
                    "clip_family_per_clip requires a schema-v3 question bank: a legacy bank "
                    "carries no per-family motion-SHA contract to reconcile the extra speed "
                    "clips against (allow_legacy 题库不配变速族表)"
                )
            order = list(metadata.get("clip_order") or [])
            if order != ["forehand", "backhand"]:
                raise ValueError(
                    f"question bank clip_order {order!r} does not match the (forehand, "
                    f"backhand) family convention required by clip_family_per_clip"
                )
            is_forehand = self._clip_family_is_forehand()
            self._qb_family_table = (~is_forehand).to(dtype=torch.long)
        return self._qb_family_table

    def _strike_frame_for_clip(self, motion, clip_id: int) -> tuple[int, float, int, int]:
        """Return (global strike frame, phase, segment start, segment len) for one reference clip."""
        nseg = int(motion.num_segments)
        if clip_id < 0 or clip_id >= nseg:
            raise IndexError(f"clip_id {clip_id} out of range for {nseg} segments")
        seg_start = int(motion.seg_start[clip_id].item())
        seg_len = int(motion.seg_len[clip_id].item())
        spc = self._strike_phases_cfg(nseg)
        phase = float(spc[clip_id]) if spc else float(self.cfg.strike_phase)
        strike_step = seg_start + round(phase * (seg_len - 1))
        return int(strike_step), phase, seg_start, seg_len

    def _ref_racket_pos_at(
        self,
        motion,
        f: int,
        *,
        clip_start: int = 0,
        clip_end: int | None = None,
    ) -> torch.Tensor:
        """Racket-center FK position (env-origin rel) at reference frame ``f``.

        Uses the SAME FK as :meth:`_compute_racket_state` /
        :meth:`_ensure_reference_strike_state` (racket body, or wrist + constant mount offset) but reads
        the reference MOTION's body poses. ``f`` is clamped to the requested clip segment so clean
        centered-difference velocities never leak across a concatenated forehand/backhand boundary.
        """
        total = max(int(motion.time_step_total), 1)
        hi = total - 1 if clip_end is None else int(clip_end)
        lo = int(clip_start)
        f = int(max(lo, min(hi, f)))
        if self._racket_mode == "body":
            return motion._body_pos_w[f, self._racket_body_index]
        widx = self._wrist_body_index
        wpos = motion._body_pos_w[f, widx]
        wquat = motion._body_quat_w[f, widx]
        offset_w = quat_apply(wquat.unsqueeze(0), self._mount_offset[0:1]).squeeze(0)
        return wpos + offset_w

    def _ensure_reference_strike_state(self):
        """Cache per-clip reference racket/base states at each clip's strike frame.

        Target sampling in ``reference_perturbed`` mode must be centered on the exact teacher clip the
        env is imitating. The old single cached state was fine for one clip, but a unified forehand+
        backhand policy needs separate strike position, velocity, face normal, and base->racket reach
        offsets for each concatenated MotionLoader segment.
        """
        if self._ref_strike_cached:
            return
        motion = self._motion().motion  # MotionLoader
        nseg = int(motion.num_segments)
        pos_all = torch.zeros(nseg, 3, device=self.device)
        vel_all = torch.zeros(nseg, 3, device=self.device)
        nrm_all = torch.zeros(nseg, 3, device=self.device)
        nrm_raw_all = torch.zeros(nseg, 3, device=self.device)
        base_all = torch.zeros(nseg, 3, device=self.device)
        reach_all = torch.zeros(nseg, 2, device=self.device)
        W = max(1, int(self.cfg.clean_strike_vel_window))
        dt = float(self._env.step_dt)
        report_lines = []
        # 每 clip 的击球面符号(空表 = 标量,现役行为不变)。参考拍面法向也要按 clip 翻面:诊断报表
        # 和参考锁定的拍面目标(_ref_normal_per_clip)都从这里出,必须和判分的那一面(实际击球面)一致。
        _mount_signs = self._mount_signs_cfg(nseg)

        for clip_id in range(nseg):
            strike_step, phase, seg_start, seg_len = self._strike_frame_for_clip(motion, clip_id)
            seg_end = seg_start + seg_len - 1
            if self._racket_mode == "body":
                idx = self._racket_body_index
                pos, quat, lin, _ = self._reference_body_state(motion, strike_step, idx)
            else:
                widx = self._wrist_body_index
                wpos, wquat, wlin, wang = self._reference_body_state(motion, strike_step, widx, require_ang_vel=True)
                offset_w = quat_apply(wquat.unsqueeze(0), self._mount_offset[0:1]).squeeze(0)
                pos = wpos + offset_w
                lin = wlin + torch.cross(wang, offset_w, dim=-1)
                quat = quat_mul(wquat.unsqueeze(0), self._mount_quat[0:1]).squeeze(0)
            # 击球面符号按 clip 取(正手一面、反手另一面);表为空用标量符号(现役行为逐位不变)。
            _sgn = float(_mount_signs[clip_id]) if _mount_signs else float(self.cfg.mount_normal_sign)
            normal_raw = matrix_from_quat(quat.unsqueeze(0))[0, :, self.cfg.mount_normal_axis]
            normal = normal_raw * _sgn

            # --- clean reference strike velocity --------------------------------------------------
            # Recompute the strike target velocity from the FINAL racket FK position by a centered
            # finite difference (clamped to this clip's segment), so it is consistent with the
            # position the policy actually tracks (the stored body_lin_vel_w is FD'd/interpolated and
            # ~1 m/s inconsistent at the racket tip — see cfg docs). raw_lin is retained only as
            # a diagnostic: the NPZ channel is a COM-point velocity and is not the controlled site.
            raw_lin = lin.detach().clone()
            fd1 = (
                self._ref_racket_pos_at(motion, strike_step + 1, clip_start=seg_start, clip_end=seg_end)
                - self._ref_racket_pos_at(motion, strike_step - 1, clip_start=seg_start, clip_end=seg_end)
            ) / (2.0 * dt)
            clean_lin = (
                self._ref_racket_pos_at(motion, strike_step + W, clip_start=seg_start, clip_end=seg_end)
                - self._ref_racket_pos_at(motion, strike_step - W, clip_start=seg_start, clip_end=seg_end)
            ) / (2.0 * W * dt)
            if not self.cfg.clean_reference_strike_velocity:
                raise RuntimeError(
                    "clean_reference_strike_velocity=false is no longer a valid racket contract: "
                    "motion body_lin_vel_w is a COM-point channel, not the URDF racket site. "
                    "Use the point-consistent centered difference."
                )
            lin = clean_lin

            pos_all[clip_id] = pos.detach().clone()
            vel_all[clip_id] = lin.detach().clone()
            nrm_all[clip_id] = (normal / (torch.norm(normal) + 1e-6)).detach().clone()
            # Raw (+Y/A-frame) twin — the question-bank A-frame guard compares against this one.
            nrm_raw_all[clip_id] = (normal_raw / (torch.norm(normal_raw) + 1e-6)).detach().clone()
            # Reference base (root) at the strike — root is articulation body index 0 (same order the
            # motion arrays use). The base->racket horizontal offset couples base_target to racket_target.
            base_all[clip_id] = self._reference_body_state(motion, strike_step, 0)[0].detach().clone()
            reach_all[clip_id] = (pos_all[clip_id, :2] - base_all[clip_id, :2]).detach().clone()

            cname = self._clip_names.get(clip_id, f"clip{clip_id}")
            p = pos_all[clip_id]
            v = vel_all[clip_id]
            nrm = nrm_all[clip_id]
            b = base_all[clip_id]
            off = reach_all[clip_id]
            report_lines.append(
                f"  {cname}: strike frame {strike_step}/{seg_end} (phase {phase:.3f}) "
                f"pos=({float(p[0]):.3f},{float(p[1]):.3f},{float(p[2]):.3f}) "
                f"vel=({float(v[0]):.3f},{float(v[1]):.3f},{float(v[2]):.3f}) "
                f"|v|={float(torch.norm(v)):.2f} "
                f"normal=({float(nrm[0]):.3f},{float(nrm[1]):.3f},{float(nrm[2]):.3f}) "
                f"baseXY=({float(b[0]):.3f},{float(b[1]):.3f}) "
                f"reachXY=({float(off[0]):.3f},{float(off[1]):.3f}) "
                f"raw_speed={float(torch.norm(raw_lin)):.3f} "
                f"clean_speed={float(torch.norm(clean_lin)):.3f} "
                f"raw_clean_diff={float(torch.norm(raw_lin - clean_lin)):.3f} "
                f"raw_fd_diff={float(torch.norm(raw_lin - fd1)):.3f}"
            )

        self._ref_racket_pos_rel_per_clip = pos_all
        self._ref_racket_vel_w_per_clip = vel_all
        self._ref_racket_normal_w_per_clip = nrm_all
        self._ref_racket_normal_raw_w_per_clip = nrm_raw_all
        self._ref_base_pos_rel_per_clip = base_all
        self._ref_reach_offset_xy_per_clip = reach_all
        # Legacy single-clip fields (no in-file consumers besides diagnostics): mirror clip 0.
        self._ref_racket_pos_rel = pos_all[0].detach().clone()
        self._ref_racket_vel_w = vel_all[0].detach().clone()
        self._ref_racket_normal_w = nrm_all[0].detach().clone()
        self._ref_base_pos_rel = base_all[0].detach().clone()
        self._ref_reach_offset_xy = reach_all[0].detach().clone()
        self._ref_strike_cached = True
        print(
            "[RacketTargetCommand] reference strike centers per clip "
            f"(clean_reference_strike_velocity={self.cfg.clean_reference_strike_velocity}, window=+-{W}):\n"
            + "\n".join(report_lines),
            flush=True,
        )
        # Table-clearance gate, reference half. It has to live HERE and not in __init__: the motion term
        # is unresolved at construction (see the _ref_strike_cached comment above), so this is the first
        # moment the reference strike point — the target CENTRE in reference_perturbed mode — exists.
        # Placed after the report so the operator sees every clip's numbers before the raise. The x is
        # the reference point's own — no station term: reference_perturbed derives base_target FROM the
        # racket target, so base_target_x_range moves the BASE, never the commanded contact point.
        for clip_id in range(nseg):
            self._assert_contact_clears_table(
                clip_id,
                float(pos_all[clip_id, 0]),
                float(pos_all[clip_id, 2]),
                "reference strike point",
            )
        # RETURN gate, same placement rationale: this is the first moment the bound strike state
        # exists, so it is the first moment we can ask whether it returns THIS CLIP'S OWN balls.
        # pos_all/vel_all are env-origin-relative / world; the scorer needs one consistent
        # env-local frame with z above the FLOOR, which is exactly what pos_all already is.
        # TIME LAW under a solved/banked target: the retiming scale is DEAD for the target (the
        # answer is solved at the ball, not read off a retimed clip) but it still divides the
        # time-to-strike the actor sees. Leaving it non-unity is therefore a silent inconsistency,
        # not a variation. Checked here because the motion term is unresolved at __init__.
        _solved = bool(str(getattr(self.cfg, "question_bank", "") or "").strip()) or \
            str(getattr(self.cfg, "target_mode", "")) == "solved"
        if _solved:
            # KNOWN-DEAD FOR BANK ARMS, DELIBERATELY NOT REPAIRED HERE. ``motion`` was bound above
            # as ``self._motion().motion`` — a MotionLoader, which stores no cfg — so this getattr
            # always yields None and the guard has never once fired. Repairing it would refuse
            # EVERY currently running arm at boot (all five run question_bank=... together with
            # motion.speed_scale_range=[0.6,1.0], and the same code is in the pin they execute
            # from). That is its own decision — "is the rule wrong or are the arms wrong?" — and it
            # does not belong in a wiring change mid-wave.
            # The NEW path does not get to ship with a guard that can never fire, so the continuous
            # source reads the real MotionCommand cfg. No continuous arm exists yet, so nothing
            # running can be refused by this scoping.
            _mcfg = getattr(motion, "cfg", None)
            if self._cq_enabled:
                _mcfg = getattr(self._motion(), "cfg", None)
            _rng = tuple(float(x) for x in (getattr(_mcfg, "speed_scale_range", (1.0, 1.0))
                                            or (1.0, 1.0)))
            if _rng != (1.0, 1.0):
                raise ValueError(
                    f"task.motion.speed_scale_range={_rng} with a solved/banked racket target: the "
                    f"retiming scale no longer shapes the TARGET (it is solved at the ball) but it "
                    f"still divides the actor's time-to-strike, so a non-unity range is a silent "
                    f"inconsistency between the command and the clock. Set "
                    f"motion.speed_scale_range: [1.0, 1.0]"
                )
            _spc = getattr(_mcfg, "speed_scale_per_clip", None)
            if _spc is not None and any(float(s) != 1.0 for s in _spc):
                raise ValueError(
                    f"task.motion.speed_scale_per_clip={tuple(float(s) for s in _spc)} with a "
                    f"solved/banked racket target — same inconsistency as speed_scale_range; set "
                    f"every entry to 1.0"
                )
        _signs = self._mount_signs_cfg(nseg)
        for clip_id in range(nseg):
            self._assert_reference_strike_can_return_its_own_regime(
                clip_id,
                pos_all[clip_id].detach().cpu().numpy(),
                vel_all[clip_id].detach().cpu().numpy(),
                nrm_all[clip_id].detach().cpu().numpy(),
                float(_signs[clip_id]) if _signs else float(self.cfg.mount_normal_sign),
            )

    def _reference_body_state(self, motion, step: int, body_index: int, require_ang_vel: bool = False):
        """Return reference body state from MotionLoader's full-articulation private arrays.

        This is the current, intentional coupling point to upstream ``MotionLoader`` internals. Public
        ``MotionCommand`` buffers expose only the tracking subset, while the racket FK needs the full
        articulation body order so it can read the racket body or wrist mount. Keep direct private-field
        access centralized here until MotionLoader grows a public full-body state API.
        """
        required = ["_body_pos_w", "_body_quat_w", "_body_lin_vel_w"]
        if require_ang_vel:
            required.append("_body_ang_vel_w")
        missing = [name for name in required if not hasattr(motion, name)]
        if missing:
            raise AttributeError(
                "RacketTargetCommand requires MotionLoader full-body reference arrays "
                f"{required}, but missing {missing}. This is the HOPE coupling point for "
                "reference_perturbed racket FK; update _reference_body_state if upstream MotionLoader changes."
            )

        pos = motion._body_pos_w[step, body_index]
        quat = motion._body_quat_w[step, body_index]
        lin = motion._body_lin_vel_w[step, body_index]
        ang = motion._body_ang_vel_w[step, body_index] if require_ang_vel else None
        return pos, quat, lin, ang

    def _perturb_scale(self) -> float:
        """Curriculum factor in [start, 1.0] that widens the reference perturbation over training.

        Success-gated mode (default): return the running ``_curr_perturb_scale``, which advances only
        when the policy demonstrates exact-strike success (see :meth:`_update_metrics`). Otherwise fall
        back to the legacy open-loop ramp keyed to ``env.common_step_counter``. The returned scale is
        clamped to ``[ref_perturb_curriculum_start, 1.0]``.
        """
        start = float(self.cfg.ref_perturb_curriculum_start)
        if self.cfg.target_mode == "reference_perturbed" and self.cfg.ref_perturb_success_gated:
            scale = self._curr_perturb_scale
        else:
            steps = float(getattr(self._env, "common_step_counter", 0))
            c = float(self.cfg.ref_perturb_curriculum_steps)
            frac = 1.0 if c <= 0.0 else min(1.0, steps / c)
            scale = start + (1.0 - start) * frac
        return min(1.0, max(start, scale))

    def _ensure_ref_normal_per_clip(self):
        """Cache the reference paddle face normal at each clip's strike frame ([num_segments, 3]).

        行数守卫(人话):这张表马上要被 motion.clip_id gather;行数和加载段数对不上就当场报
        人话错误,而不是让 GPU 越界变成一句没人看得懂的 CUDA device-side assert。
        """
        if self._ref_normal_per_clip is None:
            self._ensure_reference_strike_state()
            if self._ref_racket_normal_w_per_clip is None:
                raise RuntimeError("reference strike-state initialization produced no face normals")
            self._ref_normal_per_clip = self._ref_racket_normal_w_per_clip
        nseg = int(self._motion().motion.num_segments)
        rows = int(self._ref_normal_per_clip.shape[0])
        if rows != nseg:
            raise RuntimeError(
                f"per-clip reference face-normal cache has {rows} row(s) but the loaded motion "
                f"has {nseg} clip(s) — the cache is stale for this motion list; it must be "
                f"rebuilt with one row per loaded clip before any clip_id lookup"
            )

    def _sample_targets_uniform(self, env_ids: Sequence[int], origins: torch.Tensor, n: int):
        """Independent box sampling (legacy mode). Ranges are PLACEHOLDERS not tied to the swing."""
        pos = origins.clone()
        motion = self._motion()
        if self._pos_range_per_clip_t is not None and motion._multiseg:
            # PER-CLIP position box (unified policy): each env samples x/y/z from ITS clip's box (added to
            # the env origin). The y range is SIGNED per clip (the configured box is used directly, so a
            # near-center backhand box is valid and does not go through the shared +/-|y| fallback). This
            # replaces the shared x-range + |y|-sign + z-range logic below and lets each clip track its own
            # reference strike point.
            clip = motion.clip_id[env_ids]                      # (n,) long clip ids on the loaded list
            # 行数先对齐再 gather(spdmix v2 硬绑定四):两行家族框 + 六 clip 直接下标会越界成
            # 异步 CUDA assert;_per_clip_range_rows 先按族展开/当场报人话错误。
            rng_e = self._per_clip_range_rows(
                self._pos_range_per_clip_t, "racket_pos_range_per_clip"
            )[clip]                                             # (n, 3, 2): [env][x/y/z][lo/hi]
            lo = rng_e[..., 0]                                  # (n, 3)
            hi = rng_e[..., 1]                                  # (n, 3)
            pos[:, :3] += lo + (hi - lo) * torch.rand(n, 3, device=self.device)
        else:
            # Shared box (legacy / single-clip): identical sampling to before — backward compatible.
            pos[:, 0] += sample_uniform(*self.cfg.racket_pos_x_range, (n,), self.device)
            if motion._multiseg:
                # Unified policy: the target Y region is conditioned on the swing FAMILY so forehand and
                # backhand regions are non-overlapping (HITTER §IV). Sample |y| and set the sign per clip
                # via the family table (clip_family_per_clip; absent = the legacy (forehand, backhand)
                # 2-clip derivation, byte-identical to the old clips==0 hardcode): forehand-family clips
                # on -y if forehand_on_negative_y, backhand-family clips on the opposite side.
                clip = motion.clip_id[env_ids]
                ymag = sample_uniform(*self.cfg.racket_pos_y_abs_range, (n,), self.device)
                fh_sign = -1.0 if self.cfg.forehand_on_negative_y else 1.0
                sign = torch.where(self._clip_family_is_forehand()[clip], fh_sign, -fh_sign)
                pos[:, 1] = origins[:, 1] + sign * ymag
            else:
                pos[:, 1] += sample_uniform(*self.cfg.racket_pos_y_range, (n,), self.device)
            pos[:, 2] += sample_uniform(*self.cfg.racket_pos_z_range, (n,), self.device)
        self.racket_target_pos_w[env_ids] = pos

        if self._vel_range_per_clip_t is not None and motion._multiseg:
            # PER-CLIP velocity (unified policy): each env samples from ITS clip's box, so the slower
            # backhand gets a lower target speed than the forehand instead of one shared box that
            # overshoots the backhand. Vectorized: gather each env's clip range, then uniform-sample.
            clip = motion.clip_id[env_ids]                      # (n,) long clip ids on the loaded list
            rng_e = self._per_clip_range_rows(
                self._vel_range_per_clip_t, "racket_vel_range_per_clip"
            )[clip]                                             # (n, 3, 2): [env][x/y/z][lo/hi]
            lo = rng_e[..., 0]                                  # (n, 3)
            hi = rng_e[..., 1]                                  # (n, 3)
            vel = lo + (hi - lo) * torch.rand(n, 3, device=self.device)
        else:
            # Shared box (legacy / single-clip): identical sampling to before — backward compatible.
            vel = torch.empty(n, 3, device=self.device)
            vel[:, 0] = sample_uniform(*self.cfg.racket_vel_x_range, (n,), self.device)
            vel[:, 1] = sample_uniform(*self.cfg.racket_vel_y_range, (n,), self.device)
            vel[:, 2] = sample_uniform(*self.cfg.racket_vel_z_range, (n,), self.device)
        if motion.retiming_active:
            # R14: the vel boxes are centered on the clips' NATIVE strike velocities; a swing replayed
            # at speed s can only deliver ~s× that, so scale the demand (reference-target consistency).
            vel = vel * motion.speed_scale[env_ids].unsqueeze(-1)
        self.racket_target_vel_w[env_ids] = vel

        # --- HER-style achieved-target replay (mixture) --------------------------------------------
        # With prob achieved_target_mix_prob, overwrite the freshly box-sampled pos+vel with a jittered
        # PREVIOUSLY-ACHIEVED strike state from this clip's ring buffer (written at exact-strike frames
        # in _update_metrics). Replayed targets are reachable-by-demonstration, so the target
        # distribution stops asking for points the taught swing never passes through (Ace/HER, adapted
        # forward-looking for on-policy PPO — retroactive relabel would be obs-inconsistent). Clamped
        # into the per-clip box inflated by achieved_clamp_inflate so replay can neither collapse the
        # target support nor drift outside the deploy runner's hand-synced target clips. Non-replayed
        # envs keep the pure box sample; the per-clip reference normal below is shared by both paths.
        # Hard-gated on the question bank (belt to the init-time force-off braces): the bank
        # override below would clobber every replayed target, so the replay draw AND the
        # _resample_n_acc/_replay_n_acc accounting must never run — achieved_replay_frac would
        # otherwise report replays that no env ever trained on.
        # 判据是"目标是不是解出来的"。锁在 bank 对象上时,连续路径下这个块会真的跑、烧 RNG 和
        # 算力、推高 _resample_n_acc/_replay_n_acc,然后被后面的连续覆盖清掉 —— 正好复现报错
        # 文案里描述的"撒谎的 achieved_replay_frac"。
        if (self.cfg.achieved_target_mix_prob > 0.0 and not self._solved_targets_active
                and motion._multiseg):
            env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            clip_all = motion.clip_id[env_ids_t]
            # 按族分桶(spdmix v2 硬绑定四):缓存桶是正/反手两族;legacy 2-clip 下族行号==clip_id,
            # 逐字节不变;6-clip 变速档按族共享缓存,不再有 clip 2..5 永远选不中桶的哑弹。
            fam_all = self._metric_bucket_rows()[clip_all]
            replay = torch.rand(n, device=self.device) < float(self.cfg.achieved_target_mix_prob)
            self._resample_n_acc += float(n)
            infl = 1.0 + float(self.cfg.achieved_clamp_inflate)
            for c in self._clip_names:
                fill = self._ach_fill.get(c, 0)
                if fill < int(self.cfg.achieved_min_fill):
                    continue
                sel = replay & (fam_all == c)
                m = int(sel.sum())
                if m == 0:
                    continue
                rows = torch.randint(0, fill, (m,), device=self.device)
                rpos = self._ach_pos[c][rows] + (torch.rand(m, 3, device=self.device) * 2.0 - 1.0) * float(
                    self.cfg.achieved_jitter_pos
                )
                rvel_c = self._ach_vel[c][rows]
                if motion.retiming_active:
                    # R14: achieved velocities carry their SOURCE swing's playback speed; rescale to
                    # the replaying swing's own s so a slowed swing is not asked for a fast strike.
                    s_now = motion.speed_scale[env_ids_t[sel]]
                    rvel_c = rvel_c * (s_now / self._ach_spd[c][rows].clamp(min=1e-6)).unsqueeze(-1)
                rvel = rvel_c + (torch.rand(m, 3, device=self.device) * 2.0 - 1.0) * float(
                    self.cfg.achieved_jitter_vel
                )
                # 夹回各 env 自己 clip 的框(经 _per_clip_range_rows 对齐,行数保证够;
                # legacy 2-clip 下行 == 族桶行,数值逐字节不变)。
                clips_sel = clip_all[sel]
                if self._pos_range_per_clip_t is not None:
                    box = self._per_clip_range_rows(
                        self._pos_range_per_clip_t, "racket_pos_range_per_clip"
                    )[clips_sel]                                 # (m, 3, 2)
                    lo, hi = box[..., 0], box[..., 1]
                    ctr, half = (lo + hi) * 0.5, (hi - lo) * 0.5 * infl
                    rpos = torch.min(torch.max(rpos, ctr - half), ctr + half)
                if self._vel_range_per_clip_t is not None:
                    box = self._per_clip_range_rows(
                        self._vel_range_per_clip_t, "racket_vel_range_per_clip"
                    )[clips_sel]                                 # (m, 3, 2)
                    lo, hi = box[..., 0], box[..., 1]
                    ctr, half = (lo + hi) * 0.5, (hi - lo) * 0.5 * infl
                    if motion.retiming_active:
                        # R14: clamp replayed velocities into the box scaled by THIS swing's playback
                        # speed, so replay does not demand native-speed strikes from a slowed swing.
                        s_sel = motion.speed_scale[env_ids_t[sel]].unsqueeze(-1)
                        rvel = torch.min(torch.max(rvel, (ctr - half) * s_sel), (ctr + half) * s_sel)
                    else:
                        rvel = torch.min(torch.max(rvel, ctr - half), ctr + half)
                ids_sel = env_ids_t[sel]
                self.racket_target_pos_w[ids_sel] = self._env.scene.env_origins[ids_sel] + rpos
                self.racket_target_vel_w[ids_sel] = rvel
                self._replay_n_acc += float(m)

        if motion._multiseg:
            # Unified policy: the target paddle normal is the imitated swing's actual face normal at
            # strike (reachable by the imitation). The sampled racket velocity is the SWING-PATH direction,
            # which is ~18-110 deg off the +Y blade face, so normal_mode=velocity makes the normal goal
            # unsatisfiable (normal_pass=0 -> composite success stuck at 0).
            self._ensure_ref_normal_per_clip()
            clip = motion.clip_id[env_ids]
            normal = self._ref_normal_per_clip[clip]
        elif self.cfg.normal_mode == "velocity":
            normal = vel / (torch.norm(vel, dim=-1, keepdim=True) + 1e-6)
        else:  # "sampled"
            normal = torch.empty(n, 3, device=self.device)
            normal[:, 0] = sample_uniform(*self.cfg.racket_normal_x_range, (n,), self.device)
            normal[:, 1] = sample_uniform(*self.cfg.racket_normal_y_range, (n,), self.device)
            normal[:, 2] = sample_uniform(*self.cfg.racket_normal_z_range, (n,), self.device)
            normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)
        self.racket_target_normal_w[env_ids] = normal

        # Stage-1 question bank: overrides pos/vel + target_normal_cmd for these envs (no-op when off).
        self._apply_question_bank_targets(env_ids, origins, n)

    def _sample_targets_hitter_pure(
        self, env_ids: Sequence[int], origins: torch.Tensor, n: int, resample_base: bool = True
    ):
        """HITTER-faithful sampling (arXiv:2508.21043 §V-B-1 + §IV-C), 2026-07-07.

        Order and frames follow the paper exactly:

        1. BASE STATION first, sampled INDEPENDENTLY around the env origin (world frame) from
           ``base_target_x_range`` / ``base_target_y_range`` (which are the STATION BOX here, not a
           jitter — paper Fig. 4 evaluates initial station distances up to ±0.8 m).
        2. RACKET TARGET on a striking plane FIXED RELATIVE TO THE COMMANDED STATION ("0.4 m in
           front of the robot" on their G1; our A3 analog is the clips' blade reach x ≈ 0.70 m):
           the per-clip ``racket_pos_range_per_clip`` boxes are interpreted as STATION-RELATIVE
           x/y offsets (x degenerate = the fixed plane, y = the swing-side band) with z absolute
           above the ground. Forehand/backhand y-bands must be non-overlapping (paper §V-B-1).
        3. RACKET VELOCITY from the per-clip velocity boxes (world frame), then the target FACE
           NORMAL from ``normal_mode``: "velocity" = the paper's §IV-C impact model ("the racket
           plane is perpendicular to its velocity vector") — the policy must LEARN to orient the
           blade; do NOT fall back to the reference-clip normal here (that made the normal term
           trivially satisfied and is why deployed models could touch balls but not return them).

        No HER replay, no reference_reach coupling, no curriculum in this mode.

        ``resample_base=False`` (mid-swing refinement path): keep the CURRENT station and only
        re-draw the racket target/velocity around it — the paper's Fig. 3 refinement converges on
        WHERE the ball arrives; it never teleports the commanded stance mid-swing.
        """
        motion = self._motion()
        if self._pos_range_per_clip_t is None or self._vel_range_per_clip_t is None:
            raise RuntimeError(
                "target_mode='hitter_pure' requires racket_pos_range_per_clip AND "
                "racket_vel_range_per_clip (station-relative position boxes; see "
                "HOPEPingPongHitterPure.yaml)."
            )

        # 1) independent base station (world xy, around the env origin).
        if resample_base:
            base_xy = origins[:, :2].clone()
            base_xy[:, 0] += sample_uniform(*self.cfg.base_target_x_range, (n,), self.device)
            base_xy[:, 1] += sample_uniform(*self.cfg.base_target_y_range, (n,), self.device)
            self.base_target_pos_w[env_ids] = base_xy
        else:
            base_xy = self.base_target_pos_w[env_ids].clone()

        # 2) racket target: per-clip STATION-RELATIVE box (x = fixed plane, y = swing band), z above ground.
        if motion._multiseg:
            clip = motion.clip_id[env_ids]
            # 行数先对齐再 gather(spdmix v2 硬绑定四;单 clip 分支 clip 恒 0,历史上就允许
            # 两行表取第 0 行,不动)。
            pos_tab = self._per_clip_range_rows(
                self._pos_range_per_clip_t, "racket_pos_range_per_clip"
            )
            vel_tab = self._per_clip_range_rows(
                self._vel_range_per_clip_t, "racket_vel_range_per_clip"
            )
        else:
            clip = torch.zeros(n, dtype=torch.long, device=self.device)
            pos_tab = self._pos_range_per_clip_t
            vel_tab = self._vel_range_per_clip_t
        rng_e = pos_tab[clip]                                   # (n, 3, 2): [env][x/y/z][lo/hi]
        lo, hi = rng_e[..., 0], rng_e[..., 1]
        off = lo + (hi - lo) * torch.rand(n, 3, device=self.device)
        pos = origins.clone()
        pos[:, 0] = base_xy[:, 0] + off[:, 0]
        pos[:, 1] = base_xy[:, 1] + off[:, 1]
        pos[:, 2] = origins[:, 2] + off[:, 2]
        self.racket_target_pos_w[env_ids] = pos

        # 3) racket velocity (world) + face normal from normal_mode (paper: velocity direction).
        rng_v = vel_tab[clip]
        lo_v, hi_v = rng_v[..., 0], rng_v[..., 1]
        vel = lo_v + (hi_v - lo_v) * torch.rand(n, 3, device=self.device)
        self.racket_target_vel_w[env_ids] = vel

        if self.cfg.normal_mode == "sampled":
            normal = torch.empty(n, 3, device=self.device)
            normal[:, 0] = sample_uniform(*self.cfg.racket_normal_x_range, (n,), self.device)
            normal[:, 1] = sample_uniform(*self.cfg.racket_normal_y_range, (n,), self.device)
            normal[:, 2] = sample_uniform(*self.cfg.racket_normal_z_range, (n,), self.device)
        else:  # "velocity" (paper §IV-C impact model)
            normal = vel.clone()
        self.racket_target_normal_w[env_ids] = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)

    def _sample_targets_reference_perturbed(self, env_ids: Sequence[int], origins: torch.Tensor, n: int):
        """Target = this env's reference racket state @ strike + curriculum-scaled perturbation.

        The target center is selected by ``motion.clip_id`` for unified forehand+backhand training, so
        the policy no longer has to imitate one teacher strike while chasing a different sampled target.
        """
        self._ensure_reference_strike_state()
        if (self._ref_racket_pos_rel_per_clip is None
                or self._ref_racket_vel_w_per_clip is None
                or self._ref_racket_normal_w_per_clip is None):
            raise RuntimeError("reference strike-state initialization is incomplete")
        motion = self._motion()
        if motion._multiseg:
            clip = motion.clip_id[env_ids]
        else:
            clip = torch.zeros(n, dtype=torch.long, device=self.device)
        ref_pos = self._ref_racket_pos_rel_per_clip[clip]
        ref_vel = self._ref_racket_vel_w_per_clip[clip]
        ref_nrm = self._ref_racket_normal_w_per_clip[clip]

        scale = self._perturb_scale()
        dev = self.device
        pos_h = torch.tensor(self.cfg.ref_perturb_pos, device=dev) * scale
        vel_h = torch.tensor(self.cfg.ref_perturb_vel, device=dev) * scale
        nrm_h = float(self.cfg.ref_perturb_normal) * scale

        dpos = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * pos_h
        self.racket_target_pos_w[env_ids] = origins + ref_pos + dpos

        dvel = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * vel_h
        ref_vel_scaled = ref_vel * self.cfg.ref_vel_scale
        if motion.retiming_active:
            # R14: the cached reference strike velocity is native-speed; scale by this swing's playback speed.
            ref_vel_scaled = ref_vel_scaled * motion.speed_scale[env_ids].unsqueeze(-1)
        self.racket_target_vel_w[env_ids] = ref_vel_scaled + dvel

        dnrm = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * nrm_h
        normal = ref_nrm + dnrm
        self.racket_target_normal_w[env_ids] = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)

        self.metrics["ref_perturb_scale"][env_ids] = scale

        # Stage-1 question bank: overrides pos/vel + target_normal_cmd for these envs (no-op when off).
        self._apply_question_bank_targets(env_ids, origins, n)

    def _apply_question_bank_targets(self, env_ids: Sequence[int], origins: torch.Tensor, n: int):
        """Install a SOLVED question: bank row (discrete) or continuous buffer row. One seam.

        A question is the inverse-solved ANSWER to an incoming ball: target pos := the ball's
        arrival point (env-local -> world by adding the env origin), target vel := demanded racket
        velocity, target_normal_cmd := demanded face normal, plus the incoming ball itself. Two
        producers write it:

        * DISCRETE — a pre-solved bank row (scripts/gen_stage1_questions.py) at the clip's FIXED
          contact point. Exams and reproducibility.
        * CONTINUOUS — ``target_mode='solved'``: the ball is drawn from a continuous per-clip box
          and solved batched, handed out without replacement from a buffer (_cq_take). Training.

        Runs at the END of BOTH sampling paths so every resample route (reset / clip wrap) sees a
        solved target, and the downstream A1 delay/noise injectors act on it exactly like a
        box-sampled one. ``racket_target_normal_w`` storage remains the clip-reference lane for
        provenance, while the critic accessor, rewards and exact/composite metrics all use the
        shared face pair and see the demanded A-frame normal when cfg.face_command is on.
        """
        if self._question_bank is None and not self._cq_enabled:
            return
        motion = self._motion()
        if motion._multiseg:
            clip = motion.clip_id[env_ids]
        else:
            clip = torch.zeros(n, dtype=torch.long, device=self.device)
        if self._cq_enabled:
            # CONTINUOUS producer: same six values, drawn+solved instead of read off a row. The
            # buffer is addressed by LOADED clip (its boxes and its ref face are per loaded clip),
            # not by bank family. The +Y face guard has nothing to sweep here because the sign is
            # closed at SOLVE time (ref_normal = the runtime raw +Y face), so there is no second
            # copy of the face convention that could drift out of agreement.
            if not self._cq_boot_gate_done:
                self._cq_boot_gate()
            pos, incoming_vel, incoming_spin, vel, nrm, diff = self._cq_take(clip, n)
            self.metrics["question_exhausted_frac"][env_ids] = self._cq_last_exhausted_frac
        else:
            if not self._qb_face_frame_checked:
                self._check_question_bank_face_frame()
            # 族寻址:clip → 题库族行(clip_family_per_clip 缺席 = None = clip_id 直接当下标,
            # 现役行为逐字节不变;同族变速 clip 共享同一份题目/答案——烤入片段触球行逐位相同,
            # 复用合法,见 validate_runtime_motion_contract 的族级 SHA 对账)。
            family_table = self._qb_bank_family_table(motion, self._question_bank)
            bank_clip = clip if family_table is None else family_table[clip]
            pos, incoming_vel, incoming_spin, vel, nrm, diff = select_questions(
                self._question_bank, bank_clip, torch.rand(n, device=self.device)
            )
        # ---- BELOW THIS LINE THE TWO PRODUCERS SHARE ONE INSTALL, BYTE FOR BYTE ----------------
        # 人话:两条路必须走同一段安装代码。哪天这六行难写成一段共享代码了,就是两条路已经分叉了
        # ——tests/test_continuous_vs_bank_parity.py 的 seam 断言就是钉这件事的。
        self.racket_target_pos_w[env_ids] = origins + pos
        self.racket_target_vel_w[env_ids] = vel
        self.target_normal_cmd[env_ids] = nrm
        # The inverse-solved answer and its incoming ball are one atomic question. These writes
        # happen from the same row; the generic virtual-ball sampler below is disabled whenever a
        # SOLVED target is active (bank OR continuous — see _solved_targets_active) so it cannot
        # replace question A's ball with question B.
        self.vb_vel_in_w[env_ids] = incoming_vel
        self.vb_spin_in_w[env_ids] = incoming_spin
        # Held per env until its next resample; reset-mean reports the question difficulty mix.
        self.metrics["question_difficulty_deg"][env_ids] = diff

    def _install_event_training_questions(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        clip_ids: torch.Tensor,
        bank_rows: torch.Tensor,
    ) -> None:
        """Install T1 train-bank rows without resetting physical, action, history, or noise state."""

        if self._question_bank is None or not self._event_timing_bound:
            raise RuntimeError("post_strike_t1 question-bank contract is not bound")
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        clips = torch.as_tensor(clip_ids, dtype=torch.long, device=self.device).reshape(-1)
        rows = torch.as_tensor(bank_rows, dtype=torch.long, device=self.device).reshape(-1)
        if len(ids) == 0 or len(ids) != len(clips) or len(ids) != len(rows):
            raise ValueError("event question install requires equal, non-empty id/clip/row vectors")
        motion = self._motion()
        live_clips = motion.clip_id[ids] if motion._multiseg else torch.zeros_like(clips)
        if not torch.equal(live_clips, clips):
            raise RuntimeError("event motion and atomic train-bank question clips disagree")
        counts = self._question_bank.counts.to(device=self.device, dtype=torch.long)
        # 族寻址:题库行号 = 族表[clip](族表缺席 = clip 直接当行号,现役行为逐字节不变)。
        family_table = self._qb_bank_family_table(motion, self._question_bank)
        if family_table is None:
            bank_clips = clips
        else:
            if torch.any(clips < 0) or torch.any(clips >= int(family_table.shape[0])):
                raise ValueError("event question install references an invalid motion clip")
            bank_clips = family_table[clips]
        if (
            torch.any(bank_clips < 0)
            or torch.any(bank_clips >= len(counts))
            or torch.any(rows < 0)
            or torch.any(rows >= counts[bank_clips])
        ):
            raise ValueError("event question install references an invalid train-bank row")
        if not self._qb_face_frame_checked:
            self._check_question_bank_face_frame()

        origins = self._env.scene.env_origins[ids]
        bank = self._question_bank
        contact = bank.contact_pos[bank_clips]
        self.racket_target_pos_w[ids] = origins + contact
        self.racket_target_vel_w[ids] = bank.demanded_vel[bank_clips, rows]
        demanded_normal = bank.demanded_normal[bank_clips, rows]
        self.racket_target_normal_w[ids] = demanded_normal
        self.target_normal_cmd[ids] = demanded_normal
        self.vb_vel_in_w[ids] = bank.incoming_vel[bank_clips, rows]
        self.vb_spin_in_w[ids] = bank.incoming_spin[bank_clips, rows]
        self.metrics["question_difficulty_deg"][ids] = bank.difficulty_deg[bank_clips, rows]

        # Bank training uses a pinned ready anchor.  Do not draw a fresh base jitter here: the
        # immutable question row and its clip/deadline are installed as one deterministic message.
        self.base_target_pos_w[ids] = origins[:, :2] + self._qb_base_anchor_off_xy()[clips]
        self.station_anchor_pos_w[ids] = origins[:, :2] + torch.tensor(
            self.cfg.station_anchor_offset_xy, dtype=torch.float32, device=self.device
        )
        # Swing type by clip FAMILY (clip_family_per_clip; absent = the legacy 2-clip derivation,
        # byte-identical to the old clips==0 hardcode).
        self.swing_sign[ids] = torch.where(self._clip_family_is_forehand()[clips], 1.0, -1.0)

        # This is an intra-episode post-strike transition: close the previous swing's ledger and
        # start the new one, but do not classify it as a pre-strike fall and do not reset session
        # state.  _prev_clip_id still names the just-finished clip because MotionCommand publishes
        # the new clip earlier in the same manager step.
        if motion._multiseg:
            self._recover_from_clip[ids] = self._prev_clip_id[ids]
        self._resample_is_wrap = True
        try:
            self._count_swing_starts(ids, count_prestrike_falls=False)
        finally:
            self._resample_is_wrap = False
        self._hold_edge_pending[ids] = True
        self._previous_in_hold[ids] = False
        self._hold_start_yaw[ids] = 0.0
        self._exact_fired[ids] = False
        self._prev_motion_steps[ids] = motion.time_steps[ids]
        self._prev_racket_dist[ids] = torch.norm(
            self.racket_pos_w[ids] - self.racket_target_pos_w[ids], dim=-1
        ).detach()
        self.racket_progress[ids] = 0.0
        self._progress_reset_mask[ids] = True
        if hasattr(self, "time_left"):
            self.time_left[ids] = float("inf")
        if self._shadow is not None:
            self._shadow.on_resample(ids)
        if self._physical is not None:
            self._physical.on_resample(ids)
        # Deliberately no _reset_actor_target_state(): _push_actor_target() advances the existing
        # latency ring, AR(1) noise, dropout countdown and hold-last state below.

    def install_external_exam_questions(
        self,
        env_ids: Sequence[int],
        exam_bank,
        clip_ids: torch.Tensor,
        bank_rows: torch.Tensor,
    ) -> None:
        """Atomically install evaluator-owned exam rows without changing the training bank.

        ``cfg.question_bank`` and ``self._question_bank`` remain the saved *train* split.  A formal
        evaluator independently loads/validates the exam split, materializes an immutable
        content-addressed schedule, resets the environment to nominal stand, then calls this seam
        once before the first actor observation.  There is intentionally no automatic cursor or
        wrap behavior here: one environment answers one scheduled row, and any later resample is a
        protocol violation handled by the evaluator ledger.
        """

        raw_ids = torch.as_tensor(env_ids, device=self.device)
        raw_clips = torch.as_tensor(clip_ids, device=self.device)
        raw_rows = torch.as_tensor(bank_rows, device=self.device)
        for name, value in (("env_ids", raw_ids), ("clip_ids", raw_clips),
                            ("bank_rows", raw_rows)):
            if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
                raise ValueError(f"external exam {name} must use an integer dtype")
        ids = raw_ids.to(dtype=torch.long).reshape(-1)
        clips = raw_clips.to(dtype=torch.long).reshape(-1)
        rows = raw_rows.to(dtype=torch.long).reshape(-1)
        if len(ids) == 0 or len(ids) != len(clips) or len(ids) != len(rows):
            raise ValueError(
                "external exam questions require equal, non-empty env/clip/row vectors"
            )
        if len(torch.unique(ids)) != len(ids) or torch.any(ids < 0) or torch.any(ids >= self.num_envs):
            raise ValueError("external exam env ids must be unique and in range")
        if float(self.cfg.midswing_resample_prob) != 0.0:
            raise ValueError("external BankExam requires midswing_resample_prob=0")
        metadata = getattr(exam_bank, "metadata", None)
        if not isinstance(metadata, dict) or metadata.get("split") != "exam":
            raise ValueError("external BankExam requires a validated schema-v3 exam split")
        counts = exam_bank.counts.to(device=self.device, dtype=torch.long)
        motion = self._motion()
        # 族寻址(clip_family_per_clip 配了才有;None = 现役行为逐字节不变):考卷传进来的
        # clip_ids 是加载 clip 序号,题库行号要过族表折算,同族变速 clip 共享同一族考题。
        family_table = self._qb_bank_family_table(motion, exam_bank)
        if family_table is None:
            if torch.any(clips < 0) or torch.any(clips >= len(counts)):
                raise ValueError("external exam clip ids are outside the exam bank")
            bank_clips = clips
        else:
            if torch.any(clips < 0) or torch.any(clips >= int(family_table.shape[0])):
                raise ValueError("external exam clip ids are outside the loaded motion clips")
            bank_clips = family_table[clips]
            if torch.any(bank_clips >= len(counts)):
                raise ValueError("external exam clip family is outside the exam bank")
        if torch.any(rows < 0) or torch.any(rows >= counts[bank_clips]):
            raise ValueError("external exam row is outside its clip's validated question count")

        live_clips = motion.clip_id[ids] if motion._multiseg else torch.zeros_like(clips)
        if not torch.equal(live_clips, clips):
            raise ValueError(
                "external exam motion clip must be installed before its atomic racket question"
            )

        # The same +Y/A-frame guard used by training, now against the independently loaded exam
        # bank.  Checking all rows makes a bad regenerated bank fail before any score is emitted.
        # 族寻址下逐"加载 clip"核:每个 clip 用自己的参考面对照其族的全部考题法向。
        self._ensure_reference_strike_state()
        ref_raw = self._ref_racket_normal_raw_w_per_clip
        n_guard = len(counts) if family_table is None else int(family_table.shape[0])
        if ref_raw is None or ref_raw.shape[0] < n_guard:
            raise RuntimeError("external exam could not resolve per-clip raw face normals")
        demanded_all = exam_bank.demanded_normal.to(self.device)
        for clip in range(n_guard):
            fam = clip if family_table is None else int(family_table[clip])
            count = int(counts[fam])
            dots = torch.mv(demanded_all[fam, :count], ref_raw[clip])
            if float(dots.min()) <= 0.0:
                raise ValueError(
                    f"external exam clip {clip} contains a demanded normal opposite the +Y/A frame"
                )

        origins = self._env.scene.env_origins[ids]
        contact = exam_bank.contact_pos.to(self.device)[bank_clips]
        incoming_vel = exam_bank.incoming_vel.to(self.device)[bank_clips, rows]
        incoming_spin = exam_bank.incoming_spin.to(self.device)[bank_clips, rows]
        demanded_vel = exam_bank.demanded_vel.to(self.device)[bank_clips, rows]
        demanded_normal = demanded_all[bank_clips, rows]
        difficulty = exam_bank.difficulty_deg.to(self.device)[bank_clips, rows]

        self.racket_target_pos_w[ids] = origins + contact
        self.racket_target_vel_w[ids] = demanded_vel
        # Formal diagnostics grade the demanded exam face even for legacy actors that had no
        # face-command observation.  Face-command actors still read ``target_normal_cmd`` through
        # the audited +Y/A-frame lane below.
        self.racket_target_normal_w[ids] = demanded_normal
        self.target_normal_cmd[ids] = demanded_normal
        self.vb_vel_in_w[ids] = incoming_vel
        self.vb_spin_in_w[ids] = incoming_spin
        if "question_difficulty_deg" not in self.metrics:
            self.metrics["question_difficulty_deg"] = torch.zeros(
                self.num_envs, device=self.device
            )
        self.metrics["question_difficulty_deg"][ids] = difficulty

        # Preserve the saved training task's station semantics while removing reset-time jitter.
        if self.cfg.target_mode == "reference_perturbed":
            if self._ref_reach_offset_xy_per_clip is None:
                raise RuntimeError("external exam could not resolve reference reach offsets")
            base_off = contact[:, :2] - self._ref_reach_offset_xy_per_clip[clips]
        else:
            base_off = torch.zeros_like(contact[:, :2])
            blend = float(self.cfg.base_couple_blend)
            if blend > 0.0:
                base_off[:, 1] = (blend * contact[:, 1]).clamp(
                    -self.cfg.base_couple_max_offset, self.cfg.base_couple_max_offset
                )
        self.base_target_pos_w[ids] = origins[:, :2] + base_off
        self.station_anchor_pos_w[ids] = origins[:, :2] + torch.tensor(
            self.cfg.station_anchor_offset_xy, dtype=torch.float32, device=self.device
        )
        # Swing type by clip FAMILY (clip_family_per_clip; absent = the legacy 2-clip derivation,
        # byte-identical to the old clips==0 hardcode).
        self.swing_sign[ids] = torch.where(self._clip_family_is_forehand()[clips], 1.0, -1.0)

        self._exact_fired[ids] = False
        self._prev_motion_steps[ids] = motion.time_steps[ids]
        self.racket_progress[ids] = 0.0
        self._progress_reset_mask[ids] = True
        if hasattr(self, "time_left"):
            self.time_left[ids] = float("inf")
        self._prev_clip_id[ids] = clips
        self._recover_from_clip[ids] = -1
        self._rally_returned[ids] = False
        self._rally_pending_return[ids] = False
        # The reset sampled a training clip/question before the evaluator installed the exam item.
        # Refresh actual racket FK and the actor-visible clock without advancing physics or either
        # command clock; otherwise action 0 pairs the new question with zero/stale FK and TTS.
        self._compute_racket_state()
        self._compute_strike_timing()
        self._prev_racket_dist[ids] = torch.norm(
            self.racket_pos_w[ids] - self.racket_target_pos_w[ids], dim=-1
        ).detach()
        self._reset_actor_target_state(ids)

    def _check_question_bank_face_frame(self):
        """A-frame (+Y calibration) guard: every bank demanded normal must be same-side with the
        RAW clip reference face (2026-07-09 单翻病防复发卫兵).

        The face_command channel is +Y/A-frame end to end (bank rows, actor obs, reward measured
        side — see hope_rewards._face_pair). gen_stage1_questions.py sign-aligns each demanded
        normal to the raw +Y clip face, so ``dot(demanded, ref_raw) > 0`` holds for every valid
        row (shipped banks: min 0.86). A bank regenerated in the striking-face ("B") convention —
        the retired TIMELINE debt "题库按翻面重出", which must NEVER be completed — would flip the
        backhand rows and silently re-create the M3c disease; this guard turns that into a loud
        startup error instead. Runs once, lazily (the reference strike state resolves after
        __init__); comparing against the runtime FK reference also couples the check to the
        configured strike_phase_per_clip — a large phase drift shows up here as a shrinking
        margin, which is a feature, not a bug.
        """
        self._ensure_reference_strike_state()
        if self._ref_racket_normal_raw_w_per_clip is None:
            raise RuntimeError("reference strike-state initialization produced no raw face normals")
        bank = self._question_bank
        family_table = None
        if bank.metadata:
            motion = self._motion()
            # 族寻址(clip_family_per_clip 配了才有;None = 现役行为逐字节不变):运行时 motion
            # 对账按"每 clip 的 SHA ∈ 其族允许列表"核,下面的 +Y 卫兵也按族取题、逐 clip 对面。
            family_table = self._qb_bank_family_table(motion, bank)
            files = (
                [motion.cfg.motion_file]
                if isinstance(motion.cfg.motion_file, str)
                else list(motion.cfg.motion_file)
            )
            nseg = int(motion.motion.num_segments)
            phases_cfg = self._strike_phases_cfg(nseg)
            phases = (
                [float(value) for value in phases_cfg]
                if phases_cfg
                else [float(self.cfg.strike_phase)] * nseg
            )
            validate_runtime_motion_contract(
                bank.metadata,
                files,
                [int(value) for value in motion.motion.seg_len.tolist()],
                phases,
                clip_families=(family_table.tolist() if family_table is not None else None),
            )
        nseg = int(self._ref_racket_normal_raw_w_per_clip.shape[0])
        n_guard = (
            min(int(bank.counts.shape[0]), nseg)
            if family_table is None
            else min(int(family_table.shape[0]), nseg)
        )
        for c in range(n_guard):
            fam = c if family_table is None else int(family_table[c])
            q = int(bank.counts[fam])
            if q <= 0:
                continue
            rows = bank.demanded_normal[fam, :q]
            ref_raw = self._ref_racket_normal_raw_w_per_clip[c]
            d = torch.mv(rows, ref_raw)
            min_d = float(d.min())
            cname = getattr(self, "_family_names", {0: "forehand", 1: "backhand"}).get(
                fam, f"family{fam}")
            if family_table is not None:
                cname = f"{cname}[clip{c}]"
            if min_d <= 0.0:
                raise ValueError(
                    f"question bank {self.cfg.question_bank!r} clip {cname!r}: "
                    f"{int((d <= 0).sum())}/{q} demanded normals are OPPOSITE the raw +Y clip "
                    f"face (min dot = {min_d:.4f}). The face_command channel is +Y/A-frame by "
                    f"design (hope_rewards._face_pair) — this bank looks regenerated in the "
                    f"striking-face convention, which would silently re-create the M3c 单翻病. "
                    f"Use an A-frame bank (gen_stage1_questions.py output); NEVER re-emit banks "
                    f"in the flipped convention."
                )
            if min_d < 0.5:
                print(
                    f"[RacketTargetCommand] WARN question bank clip {cname}: min dot(demanded, "
                    f"ref_raw_normal) = {min_d:.4f} < 0.5 — unusually far from the calibration "
                    f"face; check strike_phase_per_clip vs the bank's anchor phase.",
                    flush=True,
                )
        self._qb_face_frame_checked = True

    def _qb_base_anchor_off_xy(self) -> torch.Tensor:
        """Per-clip PINNED ready-anchor XY offset from the env origin ((C, 2); cached after the
        first call — the reference reach offset is unresolved at __init__).

        S2a anti-cheat (stage_curriculum_v1): the base demand must NOT track the question. The
        per-question coupling below re-derives base_xy from racket_target_pos_w every resample, so
        once the question point starts varying (point_mode box+) it would leak each question's
        contact point into the base demand and reward stepping toward it — exactly what the
        stand-your-ground gate (root-XY excursion < 0.15 m, 0 steps) must rule out. Instead the
        SAME coupling is evaluated ONCE with the bank's FIXED contact point and reused verbatim.

        族寻址(clip_family_per_clip 配了才有)下表按"加载 clip"展开成 (nseg, 2):同族变速
        clip 共享该族的固定触球点,但 reference_perturbed 的 reach offset 是逐 clip 的,所以锚
        点也逐 clip 各算各的。两处调用端都拿运行时 clip_id 直接下标,两种模式行号语义一致。
        """
        if self._qb_base_anchor is not None:
            return self._qb_base_anchor
        if self._question_bank is not None:
            contact = self._question_bank.contact_pos  # (C, 3), tracking-env frame
            family_table = self._qb_bank_family_table(self._motion(), self._question_bank)
            if family_table is not None:
                contact = contact[family_table]  # (nseg, 3): one row per LOADED clip
        else:
            # CONTINUOUS: there is no fixed contact point by construction, so the per-clip CONSTANT
            # that plays its role is the CENTRE of the contact draw box — exactly the use the
            # dead-knob note above already records for racket_pos_range_per_clip under a solved
            # target. One constant per clip, evaluated once: the S2a argument is unchanged word for
            # word, and _assert_contact_clears_table already runs over this same table at __init__.
            box = self._per_clip_range_rows(
                self._pos_range_per_clip_t, "racket_pos_range_per_clip")     # (nseg, 3, 2)
            contact = (box[..., 0] + box[..., 1]) * 0.5                      # (nseg, 3)
        if self.cfg.target_mode == "reference_perturbed":
            self._ensure_reference_strike_state()
            if self._ref_reach_offset_xy_per_clip is None:
                raise RuntimeError("reference strike-state initialization produced no reach offsets")
            # Single-clip runs cache one reach offset row; clamp the gather like the samplers do.
            idx = torch.arange(contact.shape[0], device=self.device).clamp_(
                max=self._ref_reach_offset_xy_per_clip.shape[0] - 1
            )
            anchor = contact[:, :2] - self._ref_reach_offset_xy_per_clip[idx]
        else:
            # uniform coupling: origin + clamped Y blend toward the (fixed) contact point; X stays 0.
            anchor = torch.zeros_like(contact[:, :2])
            blend = float(self.cfg.base_couple_blend)
            if blend > 0.0:
                anchor[:, 1] = (blend * contact[:, 1]).clamp(
                    -self.cfg.base_couple_max_offset, self.cfg.base_couple_max_offset
                )
        # 行数守卫(人话):两处调用端拿运行时 clip_id 直接下标这张表;题库锚点行数少于加载
        # clip 数就当场报错(6-clip 变速列表必须配 clip_family_per_clip 族表把 2 族锚点展开),
        # 而不是让 GPU gather 越界变成异步 CUDA device-side assert。单 clip 臂(段数 1、
        # 表 2 行)历史上就取第 0 行,行数只要 >= 段数即合法,不动。
        nseg = int(self._motion().motion.num_segments)
        if int(anchor.shape[0]) < nseg:
            raise ValueError(
                f"question-bank ready-anchor table has {int(anchor.shape[0])} row(s) but the "
                f"loaded motion has {nseg} clip(s) — a >2-clip speed-variant list needs "
                f"task.motion.clip_family_per_clip so the two family anchors expand to one row "
                f"per loaded clip"
            )
        self._qb_base_anchor = anchor
        return anchor

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._post_strike_elapsed_s[env_ids_t] = 0.0
        self._post_strike_elapsed_valid[env_ids_t] = False
        # A reset/wrap may replace one held state with another without a boolean edge. Force
        # the next fresh metrics tick to stamp the new hold's yaw instead of reusing old state.
        self._hold_edge_pending[env_ids_t] = True
        self._previous_in_hold[env_ids_t] = False
        self._hold_start_yaw[env_ids_t] = 0.0
        origins = self._env.scene.env_origins[env_ids]
        motion = self._motion()
        # R14: re-arm the one-shot exact-strike latch for the new swing.
        self._exact_fired[env_ids] = False

        # UNCONDITIONAL swing accounting: every resample STARTS a new swing attempt. On the
        # true-reset path (not a wrap) it also ENDS the previous attempt — count a pre-strike
        # fall if the env terminated before reaching the strike frame.
        self._count_swing_starts(env_ids, count_prestrike_falls=not self._resample_is_wrap)

        # Desired racket pos/vel/normal — independent box sampling (legacy uniform), coupled to the
        # reference swing's strike state (reference_perturbed), or HITTER-faithful station-first
        # sampling (hitter_pure: base station independent, racket plane fixed relative to the station).
        if self.cfg.target_mode == "reference_perturbed":
            self._sample_targets_reference_perturbed(env_ids, origins, n)
        elif self.cfg.target_mode == "hitter_pure":
            self._sample_targets_hitter_pure(env_ids, origins, n)
        else:
            self._sample_targets_uniform(env_ids, origins, n)

        # Desired base XY (world): COUPLE it to the racket target so standing there keeps the racket
        # reachable by the imitated swing — base_target = racket_target_xy - (reference base->racket
        # offset). Independent sampling used to fight the arm's reach (the base_position reward pulled
        # the base away from where the racket needed it). base_target_*_range is now a SMALL JITTER
        # around the coupled point. Legacy "uniform" mode keeps the old origin-relative sampling.
        # hitter_pure sampled the station inside the sampler; question-bank tasks retain their
        # fixed per-clip ready anchor. The two modes are deliberately mutually exclusive here.
        if self.cfg.target_mode == "hitter_pure":
            pass
        elif self._solved_targets_active:
            # Stage-1/S2a BASE PIN: fixed per-clip ready anchor (the coupling evaluated ONCE with
            # the bank's fixed contact point / the continuous draw box's CENTRE — see
            # _qb_base_anchor_off_xy), never the per-question coupling below. Keying this on the
            # bank OBJECT would drop a continuous arm straight into the per-question coupling,
            # which re-derives base_xy from racket_target_pos_w every resample and therefore leaks
            # each question's contact point into the base demand — the exact S2a stand-your-ground
            # leak spelled out in _qb_base_anchor_off_xy. Continuous sampling makes the question
            # point vary BY CONSTRUCTION, so that is not hypothetical. Numerically identical while the question point is fixed (S1); becomes
            # load-bearing the moment the point varies (S2a box). base_target_*_range jitter still
            # applies after, unchanged.
            if motion._multiseg:
                clip = motion.clip_id[env_ids]
            else:
                clip = torch.zeros(n, dtype=torch.long, device=self.device)
            base_xy = origins[:, :2] + self._qb_base_anchor_off_xy()[clip]
        elif self.cfg.target_mode == "reference_perturbed":
            self._ensure_reference_strike_state()
            if self._ref_reach_offset_xy_per_clip is None:
                raise RuntimeError("reference strike-state initialization produced no reach offsets")
            if motion._multiseg:
                clip = motion.clip_id[env_ids]
            else:
                clip = torch.zeros(n, dtype=torch.long, device=self.device)
            base_xy = self.racket_target_pos_w[env_ids][:, :2] - self._ref_reach_offset_xy_per_clip[clip]
        elif self.cfg.base_couple_mode == "reference_reach":
            # uniform + HITTER separate-commands coupling (§V-B-1): base_target = racket_target_xy −
            # (reference base→racket strike offset). Same derivation as the reference_perturbed branch
            # above, but the racket target keeps the proven uniform box distribution (warm-start
            # friendly). Standing at the commanded station = racket target at the clip's reference
            # reach, so the striking plane is fixed RELATIVE TO THE COMMANDED BASE and the x-span of
            # the box moves the STATION, not the reach depth. The jitter below (base_target_*_range)
            # trains the policy to strike with the station deliberately offset — y-reach diversity.
            self._ensure_reference_strike_state()
            if self._ref_reach_offset_xy_per_clip is None:
                raise RuntimeError("reference strike-state initialization produced no reach offsets")
            if motion._multiseg:
                clip = motion.clip_id[env_ids]
            else:
                clip = torch.zeros(n, dtype=torch.long, device=self.device)
            base_xy = self.racket_target_pos_w[env_ids][:, :2] - self._ref_reach_offset_xy_per_clip[clip]
        else:
            # uniform: start at spawn, then WEAKLY couple the base toward the racket target's SIDEWAYS
            # offset (Y only; X is the fixed strike plane, so no forward repositioning). The base shifts a
            # fraction (base_couple_blend) of the target's Y offset, clamped to ±base_couple_max_offset, so
            # the robot leans/steps slightly toward far targets instead of stretching in place. blend=0 ->
            # the old spawn-only behaviour. This only moves a REWARD target — the racket target distribution
            # is unchanged. (No walking reference exists, so keep the blend small: it fights leg imitation.)
            base_xy = origins[:, :2].clone()
            blend = float(self.cfg.base_couple_blend)
            if blend > 0.0:
                racket_y_off = self.racket_target_pos_w[env_ids][:, 1] - origins[:, 1]
                base_xy[:, 1] += (blend * racket_y_off).clamp(
                    -self.cfg.base_couple_max_offset, self.cfg.base_couple_max_offset
                )
        if self.cfg.target_mode != "hitter_pure":
            # hitter_pure sampled the station inside the sampler; base_target_*_range was the
            # station box there, NOT a jitter — adding it again would double-sample.
            base_xy[:, 0] += sample_uniform(*self.cfg.base_target_x_range, (n,), self.device)
            base_xy[:, 1] += sample_uniform(*self.cfg.base_target_y_range, (n,), self.device)
            self.base_target_pos_w[env_ids] = base_xy

        # R10c 站位锚:常数 = env origin + 可配置偏移。每次 resample 重写同一常数(幂等,
        # 放这里只因 origins 在手);故意不带抖动、不跟 racket/base_target 耦合——它是"你该站在
        # 哪"的世界系锚,不是奖励目标。
        self.station_anchor_pos_w[env_ids] = origins[:, :2] + torch.tensor(
            self.cfg.station_anchor_offset_xy, dtype=torch.float32, device=self.device
        )

        # Swing type. Unified multi-clip: it IS the imitated clip's FAMILY (forehand family -> +1,
        # backhand family -> -1; clip_family_per_clip, absent = the legacy (forehand, backhand) 2-clip
        # derivation, byte-identical to the old clips==0 hardcode), matching the swing_type
        # observation. Single-clip legacy: infer from the target Y side.
        if motion._multiseg:
            clip = motion.clip_id[env_ids]
            self.swing_sign[env_ids] = torch.where(
                self._clip_family_is_forehand()[clip], 1.0, -1.0
            )
        else:
            base_y_nom = origins[:, 1] + self.cfg.base_nominal_offset[1]
            dy = self.racket_target_pos_w[env_ids][:, 1] - base_y_nom
            if self.cfg.forehand_on_negative_y:
                self.swing_sign[env_ids] = torch.where(dy <= 0.0, 1.0, -1.0)
            else:
                self.swing_sign[env_ids] = torch.where(dy >= 0.0, 1.0, -1.0)

        # Tier-1 virtual incoming ball: one (v_in, omega_in) per swing. The ball's position at the
        # strike time is the racket target BY CONSTRUCTION (the sampler defines the ball to arrive
        # there), so only velocity + spin are sampled. Boxes stay inside the venue-fit envelope.
        # ATOMICITY, enforcement half. The samplers (where the solve seam lives) run FIRST and this
        # runs AFTER; with the predicate keyed on the bank OBJECT a continuous arm would land here
        # and overwrite vb_vel_in_w / vb_spin_in_w with a fresh uniform draw — the policy would be
        # asked to answer a ball that is NOT the one its racket command was solved for. This single
        # line is the most dangerous silent disengagement in the whole change.
        if ((self.cfg.virtual_ball or self.cfg.vb_metrics_only)
                and not self._solved_targets_active):
            if self._vb_vel_range_per_clip_t is None:
                self.vb_vel_in_w[env_ids, 0] = sample_uniform(*self.cfg.vb_vel_x_range, (n,), self.device)
                self.vb_vel_in_w[env_ids, 1] = sample_uniform(*self.cfg.vb_vel_y_range, (n,), self.device)
                self.vb_vel_in_w[env_ids, 2] = sample_uniform(*self.cfg.vb_vel_z_range, (n,), self.device)
            else:
                # PER-CLIP incoming-ball regime: a block is posed fast balls and a loop slow ones in
                # the SAME run. Same gather discipline as the racket boxes (_per_clip_range_rows
                # validates the row count BEFORE any GPU index, so a mis-sized table is a human-
                # readable ValueError, not an async CUDA device-side assert).
                _clip_e = self._motion().clip_id[
                    torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
                ]
                _vb_rng = self._per_clip_range_rows(
                    self._vb_vel_range_per_clip_t, "vb_vel_range_per_clip"
                )[_clip_e]                                        # (n, 3, 2)
                _u = torch.rand(n, 3, device=self.device)
                self.vb_vel_in_w[env_ids] = (
                    _vb_rng[:, :, 0] + (_vb_rng[:, :, 1] - _vb_rng[:, :, 0]) * _u
                )
            if self._vb_spin_abs_max_per_clip_t is None:
                _s = float(self.cfg.vb_spin_abs_max)
                self.vb_spin_in_w[env_ids] = sample_uniform(-_s, _s, (n, 3), self.device)
            else:
                _clip_e = self._motion().clip_id[
                    torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
                ]
                _s_e = self._vb_spin_abs_max_per_clip_t[_clip_e].unsqueeze(-1)      # (n, 1)
                self.vb_spin_in_w[env_ids] = (
                    (torch.rand(n, 3, device=self.device) * 2.0 - 1.0) * _s_e
                )

        # Rally drift accounting: base->NEW-station error at swing start (the recovery debt the
        # previous swing left). Wrap path only — at true resets base_pos_w still caches the
        # pre-teleport pose (the lazy-stamp rationale in __init__), which would book the reset
        # teleport as recovery debt. Denominator: _drift_n_acc (same wrap-only event count).
        if self._resample_is_wrap and n > 0:
            _ids_so = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            _off = torch.norm(self.base_pos_w[_ids_so, :2] - self.base_target_pos_w[_ids_so], dim=-1)
            self._station_offset_start_sum_acc += float(_off.sum())

        # Stamp the motion phase baseline for these envs so the per-swing wrap detector in
        # _update_command does not immediately re-trigger after this (e.g. reset-time) resample.
        self._prev_motion_steps[env_ids] = self._motion().time_steps[env_ids]
        self._prev_racket_dist[env_ids] = torch.norm(
            self.racket_pos_w[env_ids] - self.racket_target_pos_w[env_ids], dim=-1
        ).detach()
        self.racket_progress[env_ids] = 0.0
        self._progress_reset_mask[env_ids] = True

        # Install the task only after every truth field (bank row, contact, demanded velocity/
        # signed normal and incoming ball) has been sampled.  Revisions below update a separate
        # actor tuple and can never mutate these buffers.
        self._begin_same_ball_planner_task(env_ids)

        # A1 target latency: a TRUE reset (not an intra-episode wrap) starts a fresh "deploy
        # session".  Backfill the complete atomic planner message and clear every stateful sensor
        # defect; otherwise a new episode can inherit the previous episode's held frame, dropout
        # countdown, AR(1) error or per-swing bias.  Intra-episode wraps deliberately keep those
        # dynamics: the next command reaching the actor late is the deployment effect being modeled.
        if not self._resample_is_wrap:
            self._reset_actor_target_state(env_ids)

        # SHADOW ball lifecycle: a resample (reset or wrap) starts these envs' next question —
        # back to the kinematic incoming path (metrics-only; no reward/obs effect).
        if self._shadow is not None:
            self._shadow.on_resample(env_ids)

        # PHYSICAL ball lifecycle: the new question's serve is scheduled from here — the ball
        # parks until time_to_strike enters the serve horizon (metrics-only; no reward/obs effect).
        if self._physical is not None:
            self._physical.on_resample(env_ids)

    def _racket_fk(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure current articulation FK.

        Returns ``(position, quaternion, link-point linear velocity, raw +Y normal, signed
        striking-face normal)`` without rebinding any reward/observation-visible command buffer.
        The Phase-B physics callback consumes these locals so enabling a metrics-only ball cannot
        leak substep-fresh racket state into the policy or reward stream.
        """
        data = self.robot.data
        # Isaac Lab 2.1's legacy ``body_*`` state deliberately mixes frames:
        # ``body_pos_w`` / ``body_quat_w`` are LINK/actor-frame quantities, but
        # ``body_lin_vel_w`` is the COM-point velocity.  Racket position and
        # velocity must describe the SAME rigid point.  Use the explicit link
        # velocity buffers before applying the fixed wrist->site offset;
        # otherwise the old formula adds omega x (link-origin->site) to a COM
        # velocity and over-counts omega x (link-origin->COM), a measured
        # 0.40/0.60 m/s error at the hopex forehand/backhand strike frames.
        # The fallback keeps dependency-light unit stubs and pre-2.1 shims
        # importable; the pinned Isaac Lab 2.1 runtime always exposes the link
        # properties (formal runtime verification covers that contract).
        if self._racket_mode == "body":
            idx = self._racket_body_index
            pos = data.body_pos_w[:, idx]
            quat = data.body_quat_w[:, idx]
            if not hasattr(data, "body_link_lin_vel_w"):
                raise RuntimeError(
                    "RacketTargetCommand requires Isaac Lab body_link_lin_vel_w: body_pos_w is a "
                    "link-origin position but body_lin_vel_w is a COM-point velocity. Falling back "
                    "would silently corrupt racket speed. Use the pinned Isaac Lab 2.1 runtime."
                )
            lin_vel = data.body_link_lin_vel_w[:, idx]
        else:
            widx = self._wrist_body_index
            wpos = data.body_pos_w[:, widx]
            wquat = data.body_quat_w[:, widx]
            if not hasattr(data, "body_link_lin_vel_w") or not hasattr(data, "body_link_ang_vel_w"):
                raise RuntimeError(
                    "RacketTargetCommand wrist FK requires body_link_{lin,ang}_vel_w; legacy "
                    "body_lin_vel_w is a different rigid point and is not a safe fallback."
                )
            wlin = data.body_link_lin_vel_w[:, widx]
            wang = data.body_link_ang_vel_w[:, widx]
            offset_w = quat_apply(wquat, self._mount_offset)
            pos = wpos + offset_w
            lin_vel = wlin + torch.cross(wang, offset_w, dim=-1)
            quat = quat_mul(wquat, self._mount_quat)
        # Face normal = chosen local axis of the racket frame, mapped to world, times the striking-FACE
        # sign. 人话(franco 2026-07-09 拍板"哪面拍子超前就是哪面"):统一正反手策略里两个挥拍用的是
        # 拍子相反的两面(正手=红面/+Y,反手=黑面/−Y),所以开了 mount_normal_sign_per_clip 时符号按
        # 每个 env 的 clip_id 取;表为空(默认)走标量 mount_normal_sign,现役行为逐位不变(此时连
        # _motion() 都不碰)。racket_normal 奖励(hope_rewards._normal_kernel_raw)和训练内拍面误差
        # 指标(racket_normal_error_deg,_update_metrics)都读 self.racket_normal_w,一处修两处好。
        # Asset audit 2026-07-10 confirms local +Y is the red outer-face normal;
        # see docs/interfaces/racket_contact_geometry.md.
        axis_w = matrix_from_quat(quat)[:, :, self.cfg.mount_normal_axis]
        if self.cfg.mount_normal_sign_per_clip:
            motion = self._motion()
            if self._mount_sign_per_clip_t is None:
                # 懒构建 + fail-loud:表长和加载 clip 数对不上当场报错(照 _strike_phases_cfg 先例)。
                mns = self._mount_signs_cfg(int(motion.motion.num_segments))
                self._mount_sign_per_clip_t = torch.tensor(
                    [float(s) for s in mns], dtype=torch.float32, device=self.device
                )
            if motion._multiseg:
                sign = self._mount_sign_per_clip_t[motion.clip_id].unsqueeze(-1)  # (num_envs, 1)
            else:
                sign = self._mount_sign_per_clip_t[0]  # 单 clip:表长已校验 = 1
        else:
            sign = self.cfg.mount_normal_sign
        # RAW (+Y calibration frame, "A" convention) normal, kept alongside the striking-face-signed
        # one: the face_command reward channel pairs against +Y-frame question-bank targets and must
        # read THIS buffer (hope_rewards._face_pair; 2026-07-09 单翻病定案). Sign table empty =>
        # racket_normal_w == raw * 1.0, bitwise identical.
        return pos, quat, lin_vel, axis_w, axis_w * sign

    def _compute_racket_state(self):
        # This remains the only writer of the command's racket-state buffers. Phase-B substep
        # contact scans call _racket_fk() and keep its results local.
        (
            self.racket_pos_w,
            self.racket_quat_w,
            self.racket_lin_vel_w,
            self.racket_normal_raw_w,
            self.racket_normal_w,
        ) = self._racket_fk()

    def _strike_steps_for_envs(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return absolute float clip indices for the configured contact phase."""

        motion = self._motion()
        ml = motion.motion
        ids = env_ids.to(device=self.device, dtype=torch.long)
        if motion._multiseg:
            if self._strike_phase_per_clip_t is None:
                phases = self._strike_phases_cfg(int(ml.num_segments))
                self._strike_phase_per_clip_t = (
                    torch.tensor([float(value) for value in phases], device=self.device)
                    if phases
                    else torch.full(
                        (int(ml.num_segments),),
                        float(self.cfg.strike_phase),
                        device=self.device,
                    )
                )
            clips = motion.clip_id[ids]
            starts = ml.seg_start[clips].float()
            lengths = ml.seg_len[clips].float()
            return starts + torch.round(
                self._strike_phase_per_clip_t[clips] * (lengths - 1.0)
            )
        total = max(int(ml.time_step_total), 1)
        return torch.full(
            (len(ids),),
            float(round(self.cfg.strike_phase * (total - 1))),
            device=self.device,
        )

    def _advance_post_strike_elapsed(self, exact_strike: torch.Tensor) -> None:
        """Advance an exact, reset-aware control-time clock for the current attempt.

        The planner phase governor can change ``motion.speed_scale`` every tick.  Consequently,
        dividing a cumulative reference-frame gap by the *current* speed is not elapsed wall time.
        This latch starts from the phase-aligned exact-strike opportunity (independent of hit
        success), advances once per unique 50-Hz control tick, and closes at reset/wrap/T1 reveal.
        """

        token = getattr(self._env, "common_step_counter", None)
        if type(token) is not int:
            raise RuntimeError(
                "post-strike elapsed clock requires integer env.common_step_counter"
            )
        if self._post_strike_elapsed_last_step == token:
            return
        self._post_strike_elapsed_last_step = token

        dt = float(self._env.step_dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise RuntimeError("post-strike elapsed clock requires finite positive env.step_dt")
        exact = exact_strike.detach().bool()
        if tuple(exact.shape) != (self.num_envs,):
            raise RuntimeError("exact_strike must be a per-environment boolean mask")

        motion = self._motion()
        wrapped = getattr(motion, "just_resampled", None)
        if wrapped is None:
            wrapped = torch.zeros_like(exact)
        revealed = getattr(motion, "event_just_installed", None)
        if revealed is None:
            revealed = torch.zeros_like(exact)
        wrapped = wrapped.detach().bool()
        revealed = revealed.detach().bool()
        if tuple(wrapped.shape) != (self.num_envs,) or tuple(revealed.shape) != (
            self.num_envs,
        ):
            raise RuntimeError("wrap/reveal masks must be per-environment")

        # The exact opportunity can lie on the small positive-TTS side of the +/-dt/2 detector.
        # In that case ``pre_strike`` is still true, but this is nevertheless the unique origin
        # for the attempt.  Let exact win over pre-strike; wrap/reveal always win over exact.
        close = (self.pre_strike.detach().bool() & ~exact) | wrapped | revealed
        continuing = self._post_strike_elapsed_valid & ~close
        self._post_strike_elapsed_s[continuing] += dt
        origin = exact & ~wrapped & ~revealed
        self._post_strike_elapsed_s[origin] = 0.0
        self._post_strike_elapsed_valid[origin] = True
        self._post_strike_elapsed_s[close] = 0.0
        self._post_strike_elapsed_valid[close] = False

    def post_strike_age_and_same_attempt(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the control-time recovery age and current-attempt validity mask."""

        return self._post_strike_elapsed_s, self._post_strike_elapsed_valid

    @staticmethod
    def _planner_initial_tts_bucket_ids(initial_tts: torch.Tensor) -> torch.Tensor:
        """Classify task-entry preparation time into four disjoint deployment buckets."""

        if initial_tts.ndim != 1:
            raise ValueError("planner initial TTS bucket input must be one-dimensional")
        if not bool(torch.isfinite(initial_tts).all()):
            raise ValueError("planner initial TTS bucket input must be finite")
        buckets = torch.full_like(initial_tts, -1, dtype=torch.long)
        buckets[initial_tts < 0.5] = 0
        buckets[initial_tts == 0.5] = 1
        buckets[(initial_tts > 0.5) & (initial_tts <= 0.9)] = 2
        buckets[initial_tts > 0.9] = 3
        if bool((buckets < 0).any()):
            raise RuntimeError("planner initial TTS bucket partition is incomplete")
        return buckets

    def _ensure_exact_timing_bucket_state(self) -> None:
        """Lazy allocation for unit-test construction and older restored command objects."""

        if not hasattr(self, "_exact_attempt_initial_tts_bucket"):
            self._exact_attempt_initial_tts_bucket = torch.full(
                (self.num_envs,), -1, dtype=torch.long, device=self.device
            )
        if not hasattr(self, "_exact_pending_timing_bucket_events"):
            self._exact_pending_timing_bucket_events = {
                name: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                for name in _TIMING_BUCKET_SPARSE_EVENTS
            }

    def _assign_exact_attempt_initial_tts(
        self, env_ids: torch.Tensor, initial_tts: torch.Tensor
    ) -> None:
        """Latch a new attempt's bucket and flush any same-wrap sparse outcomes into it."""

        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        tts = torch.as_tensor(initial_tts, dtype=torch.float32, device=self.device).reshape(-1)
        if len(ids) != len(tts):
            raise ValueError("planner initial TTS bucket assignment length mismatch")
        self._ensure_exact_timing_bucket_state()
        bucket_ids = self._planner_initial_tts_bucket_ids(tts)
        self._exact_attempt_initial_tts_bucket[ids] = bucket_ids

        ledger = self._ensure_exact_behavior_decision_counters()
        with torch.inference_mode():
            for event_name, pending in self._exact_pending_timing_bucket_events.items():
                selected_pending = pending[ids]
                for bucket_id, bucket_name in enumerate(_PLANNER_INITIAL_TTS_BUCKETS):
                    ledger[f"planner_initial_tts_{bucket_name}_{event_name}"].add_(
                        (selected_pending & (bucket_ids == bucket_id)).sum(dtype=torch.long)
                    )
                pending[ids] = False

    def _book_exact_timing_bucket_sparse_events(
        self, masks: dict[str, torch.Tensor]
    ) -> None:
        """Book strike/capture/return masks against the attempt's latched initial TTS."""

        if not getattr(self, "planner_revision_enabled", False):
            return
        self._ensure_exact_timing_bucket_state()
        ledger = self._ensure_exact_behavior_decision_counters()
        motion = self._motion()
        wrapped = getattr(motion, "just_resampled", None)
        if wrapped is None:
            wrapped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            wrapped = wrapped.detach().to(device=self.device)
            if wrapped.dtype != torch.bool or wrapped.shape != (self.num_envs,):
                raise ValueError("motion.just_resampled must be a per-env boolean mask")

        bucket_ids = self._exact_attempt_initial_tts_bucket
        for event_name in _TIMING_BUCKET_SPARSE_EVENTS:
            mask = masks.get(event_name)
            if mask is None:
                continue
            mask = mask.detach().to(device=self.device)
            if mask.dtype != torch.bool or mask.shape != (self.num_envs,):
                raise ValueError(f"{event_name} must be a per-env boolean mask")
            pending = mask & wrapped
            self._exact_pending_timing_bucket_events[event_name] |= pending
            current = mask & ~wrapped
            for bucket_id, bucket_name in enumerate(_PLANNER_INITIAL_TTS_BUCKETS):
                ledger[f"planner_initial_tts_{bucket_name}_{event_name}"].add_(
                    (current & (bucket_ids == bucket_id)).sum(dtype=torch.long)
                )

    def _sample_planner_initial_tts(self, count: int) -> torch.Tensor:
        """Draw the checkpoint-bound preparation-time mixture exactly once per new task."""

        if count <= 0:
            return torch.empty(0, device=self.device)
        mixture = self._planner_initial_tts_mixture
        if mixture is None:
            raise RuntimeError("planner initial-TTS mixture is unavailable")
        component_ids = torch.multinomial(
            self._planner_initial_tts_component_weight,
            count,
            replacement=True,
        )
        lo = self._planner_initial_tts_component_lo[component_ids]
        hi = self._planner_initial_tts_component_hi[component_ids]
        # Point-mass rows (notably the explicit 0.5 s deployment baseline) remain exact because
        # (hi - lo) is bitwise zero; sub-0.5 s rows are a stress stratum, not a support floor.
        samples = lo + (hi - lo) * torch.rand(count, device=self.device)
        ledger = self._ensure_exact_behavior_decision_counters()
        counts = torch.bincount(
            component_ids,
            minlength=len(mixture.components),
        )
        ledger["planner_initial_tts_sample_count"].add_(count)
        for index, component_count in enumerate(counts):
            ledger[f"planner_initial_tts_component_{index}_count"].add_(
                component_count
            )
        ledger["planner_initial_tts_sub_0p5_count"].add_(
            (samples < 0.5).sum(dtype=torch.long)
        )
        ledger["planner_initial_tts_exact_0p5_count"].add_(
            (samples == 0.5).sum(dtype=torch.long)
        )
        ledger["planner_initial_tts_above_0p5_count"].add_(
            (samples > 0.5).sum(dtype=torch.long)
        )
        return samples

    def _begin_same_ball_planner_task(self, env_ids: Sequence[int]) -> None:
        """Create one new task identity without changing the already-sampled physical truth."""

        if not self.planner_revision_enabled or len(env_ids) == 0:
            return
        motion = self._motion()
        if not motion.planner_revision_enabled:
            raise RuntimeError(
                "half-configured planner revisions: racket command enabled but motion governor off"
            )
        profile = self._planner_revision_profile
        motion_profile = motion._planner_revision_profile
        if profile is None or motion_profile is None or (
            profile.profile_sha256 != motion_profile.profile_sha256
        ):
            raise RuntimeError(
                "half-configured planner revisions: racket/motion profile SHA mismatch"
            )
        motion_mixture = motion._planner_initial_tts_mixture
        if (
            motion_mixture is None
            or self._planner_initial_tts_mixture is None
            or motion_mixture.document()
            != self._planner_initial_tts_mixture.document()
        ):
            raise RuntimeError(
                "half-configured planner revisions: racket/motion initial-TTS mixture mismatch"
            )
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if getattr(self, "_coupled_transport", False):
            # 任务安装即时生效(任务安装不是 mocap 流,BEGIN 元组不过在途环)。上一颗球还
            # 在途的修订全部作废:中继丢弃已结束任务的消息,绝不能把旧球的元组提交进新任务。
            self._pend_valid[:, ids] = False
        if self._resample_is_wrap:
            self._planner_task_id[ids] += 1
        else:
            self._planner_control_epoch[ids] += 1
            self._planner_task_id[ids] = 1
        self._planner_task_revision[ids] = 1
        initial_tts = self._sample_planner_initial_tts(len(ids))
        self._assign_exact_attempt_initial_tts(ids, initial_tts)
        strike_step = self._strike_steps_for_envs(ids)
        normal = self.target_normal_cmd[ids]
        normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp(min=1.0e-12)
        motion.begin_planner_task(
            ids,
            control_epoch=self._planner_control_epoch[ids],
            task_id=self._planner_task_id[ids],
            strike_step=strike_step,
            initial_tts=initial_tts,
            target_position=self.racket_target_pos_w[ids],
            target_velocity=self.racket_target_vel_w[ids],
            target_normal=normal,
        )
        self.time_to_strike[ids] = initial_tts
        self._planner_visible_pos[ids] = self.racket_target_pos_w[ids]
        self._planner_visible_vel[ids] = self.racket_target_vel_w[ids]
        self._planner_visible_normal[ids] = normal
        self._planner_visible_tts[ids] = initial_tts
        self._planner_visible_last_precontact[ids] = False
        self.metrics["planner_task_revision"][ids] = 1.0

    def _revise_same_ball_actor_tuple(self) -> None:
        """Generate and atomically submit one bounded estimate revision for each pre-strike task."""

        if not self.planner_revision_enabled:
            return
        motion = self._motion()
        profile = self._planner_revision_profile
        if profile is None:
            raise RuntimeError("planner revision profile is unavailable")
        coupled = bool(getattr(self, "_coupled_transport", False))
        if coupled and getattr(motion, "event_timing_enabled", False):
            # 未定义交互,fail-loud:event timing 的 deadline 属于不可变考卷排程(不在 planner
            # 元组里),把提交压 d 步无法对它保持一致语义(排程该不该也晚 d 步?)。想同时用,
            # 先给事件时钟定义耦合传输合同再解锁。
            raise RuntimeError(
                "coupled transport delay + event timing is undefined: the event scheduler owns "
                "deadlines outside the planner tuple, so delaying revision submission by d steps "
                "has no coherent meaning for scheduled rows. Disable target_delay_steps or "
                "event timing."
            )
        # The runner can advance an accepted source timestamp locally between messages.  Mirror
        # that here so the actor clock never freezes inside the profile's no-new-revision cutoff.
        active = motion._planner_active
        self._planner_visible_tts[active] = (
            self._planner_visible_tts[active] - profile.policy_dt_s
        ).clamp(min=0.0)
        self._planner_visible_tts[active] = motion._planner_canonicalize_tts(
            self._planner_visible_tts[active], profile
        )
        eligible = motion._planner_active & (
            motion._planner_truth_tts + profile.early_deadline_tolerance_s
            >= profile.min_tts_s
        )
        ids = torch.where(eligible)[0]
        if len(ids) == 0 and not coupled:
            # 耦合模式不许在这早退:本步就算没新生成,也必须弹环提交 d 步前的在途修订。
            return
        # Estimate noise converges toward contact.  Every proposal is clamped against the
        # immutable task-begin tuple.  This is essential for latest-value transport: the runner may
        # observe revision N+2 without ever seeing N+1, yet both remain valid members of one task.
        convergence = motion._planner_truth_tts[ids].clamp(0.0, 1.0).unsqueeze(-1)
        begin_pos = self.racket_target_pos_w[ids]
        pos = begin_pos
        pos_std = float(self.cfg.planner_revision_position_std_m)
        if pos_std > 0.0:
            proposal = pos + torch.randn_like(pos) * (pos_std * convergence)
        else:
            proposal = pos
        delta = proposal - begin_pos
        delta_norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp(min=1.0e-12)
        pos = begin_pos + delta * torch.minimum(
            torch.ones_like(delta_norm), profile.max_position_revision_delta_m / delta_norm
        )

        begin_vel = self.racket_target_vel_w[ids]
        vel = begin_vel
        vel_std = float(self.cfg.planner_revision_velocity_std_mps)
        if vel_std > 0.0:
            proposal = vel + torch.randn_like(vel) * (vel_std * convergence)
        else:
            proposal = vel
        delta = proposal - begin_vel
        delta_norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp(min=1.0e-12)
        vel = begin_vel + delta * torch.minimum(
            torch.ones_like(delta_norm), profile.max_velocity_revision_delta_mps / delta_norm
        )

        truth_normal = self.target_normal_cmd[ids]
        normal_std = float(self.cfg.planner_revision_normal_std_rad)
        normal = truth_normal
        if normal_std > 0.0:
            tangent = torch.randn_like(normal)
            tangent -= (tangent * truth_normal).sum(dim=-1, keepdim=True) * truth_normal
            tangent /= torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp(min=1.0e-12)
            angle = torch.randn(len(ids), 1, device=self.device) * (
                normal_std * convergence
            )
            normal = truth_normal * torch.cos(angle) + tangent * torch.sin(angle)
        dot = (truth_normal * normal).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        angle = torch.acos(dot)
        ratio = torch.minimum(
            torch.ones_like(angle),
            profile.max_normal_revision_delta_rad / angle.clamp(min=1.0e-12),
        )
        normal = truth_normal + ratio * (normal - truth_normal)
        normal /= torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp(min=1.0e-12)

        raw_truth_tts = motion._planner_truth_tts[ids]
        truth_tts = motion._planner_canonicalize_tts(raw_truth_tts, profile)
        tts = truth_tts
        tts_std = float(self.cfg.planner_revision_tts_std_s)
        if tts_std > 0.0:
            deadline_jitter = torch.randn_like(tts) * (
                tts_std * convergence.squeeze(-1)
            )
            deadline_jitter = deadline_jitter.clamp(
                min=-profile.max_deadline_revision_delta_s,
                max=profile.max_deadline_revision_delta_s,
            )
            tts = tts + deadline_jitter
        tts = tts.clamp(profile.min_tts_s, profile.max_tts_s)
        if coupled:
            # 耦合传输:本步生成的元组只入在途环;真正提交的是 d 步前生成的那一份(原子出队,
            # tts 已按源时间戳补偿/或按 uncompensated 原样)。记账与 last_precontact 改用
            # "提交时刻"的 truth 时钟——与消息到达中继的真实时刻一致。
            ids, pos, vel, normal, tts = self._exchange_pending_planner_revision(
                ids, pos, vel, normal, tts
            )
            if len(ids) == 0:
                self.metrics["planner_task_revision"] = (
                    self._planner_task_revision.clamp(min=0).float()
                )
                self.metrics["planner_same_task_revision_active"] = eligible.float()
                return
            truth_tts = motion._planner_canonicalize_tts(
                motion._planner_truth_tts[ids], profile
            )
        revision = self._planner_task_revision[ids] + 1
        accepted = motion.submit_planner_revision(
            ids,
            control_epoch=self._planner_control_epoch[ids],
            task_id=self._planner_task_id[ids],
            task_revision=revision,
            desired_tts=tts,
            target_position=pos,
            target_velocity=vel,
            target_normal=normal,
        )
        self._book_planner_revision_decisions(
            attempted_tts=truth_tts,
            accepted=accepted,
        )
        last_precontact = torch.isclose(
            truth_tts,
            torch.full_like(truth_tts, profile.min_tts_s),
            rtol=0.0,
            atol=profile.early_deadline_tolerance_s,
        )
        accepted_ids = ids[accepted]
        if len(accepted_ids) > 0:
            self._planner_task_revision[accepted_ids] = revision[accepted]
            self._planner_visible_pos[accepted_ids] = pos[accepted]
            self._planner_visible_vel[accepted_ids] = vel[accepted]
            self._planner_visible_normal[accepted_ids] = normal[accepted]
            self._planner_visible_tts[accepted_ids] = tts[accepted]
            self._planner_visible_last_precontact[accepted_ids] = last_precontact[accepted]
        self.metrics["planner_task_revision"] = self._planner_task_revision.clamp(min=0).float()
        self.metrics["planner_same_task_revision_active"] = eligible.float()

    def _exchange_pending_planner_revision(
        self,
        ids: torch.Tensor,
        pos: torch.Tensor,
        vel: torch.Tensor,
        normal: torch.Tensor,
        tts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """耦合传输在途环的一次"压入本步 / 弹出 d 步前"交换(每控制步恰好调用一次)。

        单指针环(长度 d):先读槽 p(恰好是 d 步前写入的元组),再把本步生成的元组写进
        同一个槽,指针前进——生成于 t 的修订恰在 t+d 提交,d=2 时效果整整晚 2 步。
        出队的 tts 处理遵循 target_delay_tts_mode:
        * source_timestamp_compensated——减去在途时长 d*dt(runner 的源时间戳补偿),
          提交进调度器的 deadline ≈ 提交时刻的真实剩余时间,actor 时钟连续;
        * uncompensated——原样提交(显式陈旧 tts 阴性对照;调度器可能因 deadline 超过
          begin 包络而拒收,这正是该对照要暴露的行为)。
        无效槽位不提交:当步没生成(env 不在 eligible 集),或新任务安装时被
        _begin_same_ball_planner_task 作废(旧球消息不得进新任务)。补偿后过期的元组
        交给 submit_planner_revision 的接受判据拒收(fail-safe 拒绝并记账,而非静默丢弃)。
        """
        if tts.dim() != 1 or pos.shape != (len(ids), 3) or vel.shape != (len(ids), 3) \
                or normal.shape != (len(ids), 3) or len(tts) != len(ids):
            raise ValueError(
                "coupled transport exchange expects one atomic (pos, vel, normal, tts) tuple "
                "per generating env"
            )
        p = self._pend_ptr
        out_valid = self._pend_valid[p].clone()
        out_pos = self._pend_pos[p].clone()
        out_vel = self._pend_vel[p].clone()
        out_normal = self._pend_normal[p].clone()
        out_tts = self._pend_tts[p].clone()
        self._pend_valid[p] = False
        self._pend_valid[p, ids] = True
        self._pend_pos[p, ids] = pos
        self._pend_vel[p, ids] = vel
        self._pend_normal[p, ids] = normal
        self._pend_tts[p, ids] = tts
        self._pend_ptr = (p + 1) % int(self._pend_valid.shape[0])
        sub_ids = torch.where(out_valid)[0]
        sub_tts = out_tts[sub_ids]
        if self._delay_tts_mode == "source_timestamp_compensated":
            sub_tts = sub_tts - self._delay_steps * float(self._env.step_dt)
        return sub_ids, out_pos[sub_ids], out_vel[sub_ids], out_normal[sub_ids], sub_tts

    def _book_planner_revision_decisions(
        self,
        *,
        attempted_tts: torch.Tensor,
        accepted: torch.Tensor,
    ) -> None:
        """Book exact actor-revision activation, including the final pre-contact tick."""

        if not self.planner_revision_enabled or self._planner_revision_profile is None:
            raise RuntimeError("planner revision decisions require an enabled validated profile")
        tts = attempted_tts.detach().reshape(-1)
        decisions = accepted.detach().reshape(-1)
        if decisions.dtype != torch.bool:
            raise TypeError("planner revision accepted mask must have boolean dtype")
        if tts.shape != decisions.shape:
            raise ValueError("planner revision TTS and decision masks must have identical shape")
        if not bool(torch.isfinite(tts).all()):
            raise ValueError("planner revision attempted TTS must be finite")
        ledger = self._ensure_exact_behavior_decision_counters()
        last_tick = torch.isclose(
            tts,
            torch.full_like(tts, self._planner_revision_profile.min_tts_s),
            rtol=0.0,
            atol=self._planner_revision_profile.early_deadline_tolerance_s,
        )
        ledger["planner_revision_attempt_count"].add_(tts.numel())
        ledger["planner_revision_accepted_count"].add_(
            decisions.sum(dtype=torch.long)
        )
        ledger["planner_revision_rejected_count"].add_(
            (~decisions).sum(dtype=torch.long)
        )
        ledger["planner_revision_last_precontact_attempt_count"].add_(
            last_tick.sum(dtype=torch.long)
        )
        ledger["planner_revision_last_precontact_accepted_count"].add_(
            (last_tick & decisions).sum(dtype=torch.long)
        )

    def _compute_strike_timing(self):
        """Refresh time_to_strike / pre_strike / strike_window from the CURRENT motion phase.

        The ``motion`` command term computes before this one (it is a parent-class field, registered
        first), so by now ``motion.time_steps`` is already advanced for this control step. Computing the
        timing here — at the top of _update_metrics, alongside the fresh racket FK — keeps the strike
        masks ALIGNED with the racket pose they gate. Previously the masks were only set in
        _update_command (which runs AFTER _update_metrics), so _update_metrics read a 1-step-stale
        time_to_strike: ``exact_strike`` fired one control frame LATE, after the paddle had flown ~one
        step past the strike (~12 cm at a 6 m/s swing) — which collapsed the measured position accuracy
        (exact-strike pos<7.5cm read ~11% instead of the true ~68%) while barely moving velocity (flat
        near the peak). That made strike_composite_success_exact ~6x pessimistic vs the honest probe.
        """
        motion = self._motion()
        ml = motion.motion
        if motion._multiseg:
            # Per-clip strike frame on the concatenated time axis: the contact phase differs per swing
            # (v2 blade re-plane: forehand 0.47, backhand 0.333), so resolve strike_step per env from its clip.
            if self._strike_phase_per_clip_t is None:
                sp = self._strike_phases_cfg(int(ml.num_segments))
                if sp:
                    self._strike_phase_per_clip_t = torch.tensor([float(x) for x in sp], device=self.device)
                else:
                    self._strike_phase_per_clip_t = torch.full(
                        (int(ml.num_segments),), float(self.cfg.strike_phase), device=self.device
                    )
            clip = motion.clip_id
            seg_start = ml.seg_start[clip]
            seg_len = ml.seg_len[clip]
            phase = self._strike_phase_per_clip_t[clip]
            strike_step = seg_start + (phase * (seg_len - 1).float()).round().long()
            if motion.retiming_active:
                # R14: at playback speed s the clock covers (strike_step - t) frames in
                # (strike_step - t)/s control steps. Use the FLOAT clock so tts still decreases by
                # exactly step_dt per unheld step and the exact-strike detector fires once per swing.
                self.time_to_strike = (
                    (strike_step.float() - motion.time_steps_f) * self._env.step_dt / motion.speed_scale
                )
            else:
                self.time_to_strike = (strike_step - motion.time_steps).float() * self._env.step_dt
        else:
            total = max(int(ml.time_step_total), 1)
            strike_step = round(self.cfg.strike_phase * (total - 1))
            if motion.retiming_active:
                self.time_to_strike = (
                    (float(strike_step) - motion.time_steps_f) * self._env.step_dt / motion.speed_scale
                )
            else:
                self.time_to_strike = (strike_step - motion.time_steps).float() * self._env.step_dt
        # 窗掩码时钟:默认 = 上面算好的 legacy 有符号 clip 时钟(planner 关时天然 ±窗)。
        tts_for_window = self.time_to_strike
        if self.planner_revision_enabled:
            if not motion.planner_revision_enabled:
                raise RuntimeError(
                    "half-configured planner revisions: motion governor disabled at runtime"
                )
            # Physical-ball/reward/critic truth is the immutable task deadline, independent of
            # the actor's noisy revised estimate and independent of the reference phase chosen to
            # meet it.  This breaks the old circular definition (TTS derived from clip phase).
            # Schema-4 owns a task deadline: it reaches zero at contact and
            # remains zero through follow-through.  Legacy clip-derived clocks
            # keep their historical signed post-strike values above.
            self.time_to_strike = motion._planner_truth_tts.clamp(min=0.0)
            # 击球窗掩码改读带符号孪生时钟(2026-07-25):deadline 的钉 0 曾让
            # |tts|<=0.12 的窗覆盖整个随挥段(~50-100 步)——position/normal 触球后
            # 停拍可薅钱、站稳包/face 税全程计费、模仿在恢复段被 0.25x 捂嘴。
            # 窗回到设计的 ±0.12 s;time_to_strike/pre_strike/exact_strike(自带
            # 一拍锁存)语义不变,obs/critic 仍读钉 0 的任务期限。
            tts_for_window = motion._planner_truth_tts_signed
        if getattr(motion, "event_timing_enabled", False):
            # For a successfully revealed T1 row, WHEN is owned by the immutable schedule rather
            # than by the clip phase.  The native clip/hold pair is only the feasible trajectory
            # used to arrive there; it may never move the deadline.  Pending, unavailable and
            # infeasible rows keep their old clip timing but are masked out of exact grading below.
            event_mask = motion.event_installed
            scheduled_tts = (
                motion.event_deadline_ticks_remaining.float() * self._env.step_dt
            )
            self.time_to_strike = torch.where(event_mask, scheduled_tts, self.time_to_strike)
            # event timing 现役全 disabled;若启用,T1 行的 deadline ticks 亦是截断时钟,
            # 其窗关闭语义待事件时钟带符号化时再接(掩码暂沿用 tts_for_window)。
        self.pre_strike = self.time_to_strike > 0.0
        _tts_abs = tts_for_window.abs()
        self.strike_window = _tts_abs <= self.cfg.strike_window_s
        # 1c split windows: POSITION gets the tight window, NORMAL/VELOCITY the wide one; None
        # (default) falls back to strike_window_s for that channel — numerically identical to the
        # legacy single window. The stability penalties / hit-rate metric keep strike_window.
        _pos_s = self.cfg.strike_window_pos_s
        _wide_s = self.cfg.strike_window_wide_s
        self.strike_window_pos = self.strike_window if _pos_s is None else (_tts_abs <= float(_pos_s))
        self.strike_window_wide = self.strike_window if _wide_s is None else (_tts_abs <= float(_wide_s))

    def _update_command(self):
        motion = self._motion()
        # Timing is refreshed in _update_metrics (aligned with the FK); recompute here too so a direct
        # _update_command call outside the compute() path stays correct. Idempotent within a step
        # (motion.time_steps is unchanged between the two calls).
        self._compute_strike_timing()

        # Re-sample the target at each new swing. Use the motion command's robust just_resampled signal
        # (set this same step when it wrapped a swing) instead of a time_steps<prev heuristic — the latter
        # fails for the unified policy when a wrap jumps the index to a HIGHER concatenated segment start
        # (forehand->backhand). Targets for fresh episodes are sampled by the manager's reset.
        event_ids = (
            torch.where(motion.event_just_installed)[0]
            if getattr(motion, "event_timing_enabled", False)
            else torch.empty(0, dtype=torch.long, device=self.device)
        )
        if len(event_ids) > 0:
            self._install_event_training_questions(
                event_ids,
                motion.event_current_clip_id[event_ids],
                motion.event_current_bank_row[event_ids],
            )
        wrapped = torch.where(motion.just_resampled)[0] if hasattr(motion, "just_resampled") else \
            torch.where(motion.time_steps < self._prev_motion_steps)[0]
        if len(wrapped) > 0:
            # Wrap path: a wrapped env passed its strike frame alive (strike < seg end), so no
            # pre-strike fall is counted — the flag only gates fall accounting inside
            # _resample_command's _count_swing_starts hook.
            if motion._multiseg:
                # Latch the clip that JUST finished (before _prev_clip_id is re-snapshotted below):
                # while the post-wrap hold lasts, a fall belongs to THIS swing's recovery.
                self._recover_from_clip[wrapped] = self._prev_clip_id[wrapped]
            self._resample_is_wrap = True
            try:
                self._resample_command(wrapped)
            finally:
                self._resample_is_wrap = False
        # The recovery window ends when the post-wrap hold expires (the new swing's clock starts
        # advancing) — from then on falls are genuinely pre-strike of the new clip. in_hold is this
        # step's post-decrement hold state, so a zero-length hold clears the latch immediately.
        if motion._multiseg and hasattr(motion, "in_hold"):
            self._recover_from_clip[~motion.in_hold] = -1
        self._prev_motion_steps = motion.time_steps.clone()
        # Snapshot the clip each env is swinging THIS step: at the next true reset the motion command
        # will already have resampled clip_id, so fall attribution reads this snapshot instead.
        if motion._multiseg:
            self._prev_clip_id = motion.clip_id.clone()

        # --- A1 mid-swing target refinement (the planner refines WHERE, not WHEN) -------------------
        # Each step, envs still approaching the strike (pre_strike AND time_to_strike > tts floor)
        # re-draw their target with per-step prob p, exactly as the deploy planner refines its ball
        # prediction mid-swing. ONLY the target sampling runs (position/velocity/normal through the
        # existing uniform / per-clip-box / reference-perturbed path, including the HER achieved-target
        # mixture inside it):
        #   * strike timing untouched — same strike step, the swing clock keeps running;
        #   * NO _count_swing_starts — a refinement is not a new swing attempt (metrics denominators
        #     would otherwise be inflated);
        #   * base target / swing type / _prev_motion_steps untouched;
        #   * the racket-progress baseline is reset via _progress_reset_mask (same mechanism as the
        #     resample path) so the target jump creates no fake progress reward;
        #   * the achieved-target replay WRITE is unaffected (it stores the LIVE target state at
        #     exact-strike frames, and tts floor > 0 keeps refinement away from the strike frame).
        # prob==0 (default) short-circuits before any RNG draw — byte-identical baseline.
        _ms_prob = float(self.cfg.midswing_resample_prob)
        if _ms_prob > 0.0:
            eligible = self.pre_strike & (self.time_to_strike > float(self.cfg.midswing_resample_tts_floor))
            redraw = eligible & (torch.rand(self.num_envs, device=self.device) < _ms_prob)
            ids = torch.where(redraw)[0]
            if len(ids) > 0:
                origins = self._env.scene.env_origins[ids]
                if self.cfg.target_mode == "reference_perturbed":
                    self._sample_targets_reference_perturbed(ids, origins, len(ids))
                elif self.cfg.target_mode == "hitter_pure":
                    # Refinement re-draws WHERE around the UNCHANGED station (paper Fig. 3
                    # convergence; the commanded stance never teleports mid-swing).
                    self._sample_targets_hitter_pure(ids, origins, len(ids), resample_base=False)
                else:
                    self._sample_targets_uniform(ids, origins, len(ids))
                self._prev_racket_dist[ids] = torch.norm(
                    self.racket_pos_w[ids] - self.racket_target_pos_w[ids], dim=-1
                ).detach()
                self.racket_progress[ids] = 0.0
                self._progress_reset_mask[ids] = True
            # Per-env 0/1 indicator; the wandb reset-mean = per-step refinement fraction (~ prob *
            # eligible fraction). Written every step while the feature is on so zero-redraw steps count.
            self.metrics["midswing_resample_count"] = redraw.float()

        # Same-ball planner: refine the complete estimate (including WHEN) without changing truth.
        self._revise_same_ball_actor_tuple()

        # A1 target latency/jitter: refresh the ACTOR-visible target view once per step (no-op alias
        # when the knobs are off). Runs LAST so it sees this step's wrap/refinement target updates.
        self._push_actor_target()
        # CommandTerm.compute() ran exact-strike metrics before this update.  Consume every due
        # opportunity now regardless of whether the policy hit it, so a miss cannot pause or shift
        # the immutable sequence.  MotionCommand will fail closed next step if this handoff is ever
        # skipped, preventing a stale due row from being silently extended.
        if getattr(motion, "event_timing_enabled", False):
            motion.finalize_event_deadlines()

    def _push_actor_target(self):
        """A1: refresh the ACTOR-visible target view once per control step (latency + jitter).

        Applied on PUSH (not on read) so the jitter is drawn ONCE per step and every actor obs term
        reads the same tensor within the step (determinism). The jitter std decays with the time to
        strike (SMASH Eq. 14 — the mocap ball prediction converges as the strike approaches):
        per-step std = knob * clamp(time_to_strike, 0, 1). The ring buffer stores the jittered
        values, so a delayed read reproduces the prediction noise AS OF push time (what the mocap
        link actually emitted then). The TRUE live target is untouched — rewards, metrics, the
        privileged critic, and the achieved-target-replay write keep reading racket_target_pos_w /
        racket_target_vel_w / target_normal_cmd / swing_sign / time_to_strike.  In the explicit
        non-live TTS modes, all five actor-visible fields are emitted atomically: delay and dropout
        can never mix two question rows. ``source_timestamp_compensated`` advances the delayed
        source TTS by the configured transport age; ``uncompensated`` deliberately exposes the
        stale source value as a negative control.
        """
        if not self._actor_view_active:
            self.metrics["actor_time_to_strike_s"][:] = self.time_to_strike
            return  # default path: delayed_* alias the live tensors — nothing to compute, no RNG
        pos = (
            self._planner_visible_pos
            if self.planner_revision_enabled
            else self.racket_target_pos_w
        )
        vel = (
            self._planner_visible_vel
            if self.planner_revision_enabled
            else self.racket_target_vel_w
        )
        normal = (
            self._planner_visible_normal
            if self.planner_revision_enabled
            else self.target_normal_cmd
        )
        sign = self.swing_sign
        tts = (
            self._planner_visible_tts
            if self.planner_revision_enabled
            else self.time_to_strike
        )
        if self.planner_revision_enabled:
            tts = tts.clamp(min=0.0)
            planner_epoch = self._planner_control_epoch
            planner_task = self._planner_task_id
            planner_revision = self._planner_task_revision
            planner_last_precontact = self._planner_visible_last_precontact
        if self._jitter_pos > 0.0 or self._jitter_vel > 0.0:
            scale = self.time_to_strike.clamp(0.0, 1.0).unsqueeze(-1)
            if self._jitter_pos > 0.0:
                pos = pos + torch.randn_like(pos) * (self._jitter_pos * scale)
            if self._jitter_vel > 0.0:
                vel = vel + torch.randn_like(vel) * (self._jitter_vel * scale)
        if self._mnoise_ar1_sigma > 0.0:
            rho = self._mnoise_ar1_rho
            self._mnoise_ar1_state.mul_(rho).add_(
                torch.randn_like(self._mnoise_ar1_state), alpha=self._mnoise_ar1_sigma * (1.0 - rho * rho) ** 0.5
            )
            pos = pos + self._mnoise_ar1_state
        if self._mnoise_white > 0.0:
            pos = pos + torch.randn_like(pos) * self._mnoise_white
        if self._a1v2_active:
            # (c) per-swing systematic bias: resample at each strike moment (pre_strike falling edge
            # = sensor re-lock after the contact), constant until the next strike.
            struck = self._prev_pre_strike & ~self.pre_strike
            if self._bias_per_swing > 0.0 and struck.any():
                self._swing_bias[struck] = torch.randn(int(struck.sum()), 3, device=self.device) * self._bias_per_swing
            # (b) forced hold-last window right after the strike (sensor loses the target at contact)
            if self._post_strike_drop_steps > 0 and struck.any():
                self._drop_cd[struck] = self._post_strike_drop_steps
            self._prev_pre_strike.copy_(self.pre_strike)
            pos = pos + self._swing_bias
            # (a) random frame loss + (b) countdown: actor view HOLDS the last emitted value
            drop = self._drop_cd > 0
            if self._drop_prob > 0.0:
                drop = drop | (torch.rand(self.num_envs, device=self.device) < self._drop_prob)
            self._drop_cd = (self._drop_cd - 1).clamp_(min=0)
            d3 = drop.unsqueeze(-1)
            pos = torch.where(d3, self._held_pos, pos)
            vel = torch.where(d3, self._held_vel, vel)
            normal = torch.where(d3, self._held_normal, normal)
            sign = torch.where(drop, self._held_sign, sign)
            if self._atomic_tts_active:
                tts = torch.where(drop, self._held_tts, tts)
            if self.planner_revision_enabled:
                planner_epoch = torch.where(drop, self._held_planner_epoch, planner_epoch)
                planner_task = torch.where(drop, self._held_planner_task, planner_task)
                planner_revision = torch.where(
                    drop, self._held_planner_revision, planner_revision
                )
                planner_last_precontact = torch.where(
                    drop,
                    self._held_planner_last_precontact,
                    planner_last_precontact,
                )
            self._held_pos.copy_(pos)
            self._held_vel.copy_(vel)
            self._held_normal.copy_(normal)
            self._held_sign.copy_(sign)
            if self._atomic_tts_active:
                self._held_tts.copy_(tts)
            if self.planner_revision_enabled:
                self._held_planner_epoch.copy_(planner_epoch)
                self._held_planner_task.copy_(planner_task)
                self._held_planner_revision.copy_(planner_revision)
                self._held_planner_last_precontact.copy_(planner_last_precontact)
        if self._actor_ring_steps > 0:
            # Write this step's (jittered) target into slot `w`; the next slot in the length-
            # (delay+1) ring was written exactly `delay` pushes ago — that is the actor's view.
            w = self._delay_ptr
            self._delay_buf_pos[w].copy_(pos)
            self._delay_buf_vel[w].copy_(vel)
            self._delay_buf_normal[w].copy_(normal)
            self._delay_buf_sign[w].copy_(sign)
            if self._atomic_tts_active:
                self._delay_buf_tts[w].copy_(tts)
            if self.planner_revision_enabled:
                self._delay_buf_planner_epoch[w].copy_(planner_epoch)
                self._delay_buf_planner_task[w].copy_(planner_task)
                self._delay_buf_planner_revision[w].copy_(planner_revision)
                self._delay_buf_planner_last_precontact[w].copy_(
                    planner_last_precontact
                )
            r = (w + 1) % (self._actor_ring_steps + 1)
            self._delay_ptr = r
            self.delayed_racket_target_pos_w.copy_(self._delay_buf_pos[r])
            self.delayed_racket_target_vel_w.copy_(self._delay_buf_vel[r])
            self.delayed_target_normal_cmd.copy_(self._delay_buf_normal[r])
            self.delayed_swing_sign.copy_(self._delay_buf_sign[r])
            if self._atomic_tts_active:
                delayed_tts = self._delay_buf_tts[r]
                if self._delay_tts_mode == "source_timestamp_compensated":
                    delayed_tts = delayed_tts - self._actor_ring_steps * float(self._env.step_dt)
                if self.planner_revision_enabled:
                    delayed_tts = delayed_tts.clamp(min=0.0)
                self.delayed_time_to_strike.copy_(delayed_tts)
            if self.planner_revision_enabled:
                self._book_planner_revision_actor_delivery(
                    self._delay_buf_planner_epoch[r],
                    self._delay_buf_planner_task[r],
                    self._delay_buf_planner_revision[r],
                    self._delay_buf_planner_last_precontact[r],
                )
        else:
            # Jitter-only (delay==0): the actor view is live + this step's noise, no latency.
            self.delayed_racket_target_pos_w.copy_(pos)
            self.delayed_racket_target_vel_w.copy_(vel)
            self.delayed_target_normal_cmd.copy_(normal)
            self.delayed_swing_sign.copy_(sign)
            if self._atomic_tts_active:
                # Zero transport delay is explicitly equivalent to live TTS in both atomic modes.
                self.delayed_time_to_strike.copy_(tts)
            if self.planner_revision_enabled:
                self._book_planner_revision_actor_delivery(
                    planner_epoch,
                    planner_task,
                    planner_revision,
                    planner_last_precontact,
                )
        self.metrics["actor_time_to_strike_s"][:] = self.actor_time_to_strike()

    def _book_planner_revision_actor_delivery(
        self,
        control_epoch: torch.Tensor,
        task_id: torch.Tensor,
        task_revision: torch.Tensor,
        last_precontact: torch.Tensor,
    ) -> None:
        """Count a revision only when its complete atomic tuple reaches actor observations."""

        epoch = control_epoch.detach().reshape(-1)
        task = task_id.detach().reshape(-1)
        revision = task_revision.detach().reshape(-1)
        last = last_precontact.detach().reshape(-1)
        expected = self._planner_actor_task_revision.shape
        if (
            epoch.shape != expected
            or task.shape != expected
            or revision.shape != expected
            or last.shape != expected
            or last.dtype != torch.bool
        ):
            raise ValueError("actor-visible planner identity must be one complete per-env tuple")
        changed = (
            (epoch > 0)
            & (task > 0)
            & (
                (epoch != self._planner_actor_control_epoch)
                | (task != self._planner_actor_task_id)
                | (revision != self._planner_actor_task_revision)
            )
        )
        delivered_revision = changed & (revision > 1)
        ledger = self._ensure_exact_behavior_decision_counters()
        ledger["planner_revision_actor_visible_count"].add_(
            delivered_revision.sum(dtype=torch.long)
        )
        ledger["planner_revision_last_precontact_actor_visible_count"].add_(
            (delivered_revision & last & self.pre_strike.detach().bool()).sum(
                dtype=torch.long
            )
        )
        self._planner_actor_control_epoch.copy_(
            torch.where(changed, epoch, self._planner_actor_control_epoch)
        )
        self._planner_actor_task_id.copy_(
            torch.where(changed, task, self._planner_actor_task_id)
        )
        self._planner_actor_task_revision.copy_(
            torch.where(changed, revision, self._planner_actor_task_revision)
        )

    def _reset_actor_target_state(self, env_ids: Sequence[int]) -> None:
        """Start a true episode with a fresh, internally consistent A1 sensor state.

        The runner receives a complete planner command before its first policy step.  Mirror that
        contract by replacing every delayed/held field with the newly sampled command and clearing
        stochastic state that must not cross an episode boundary.  This helper is intentionally not
        used on clip wraps, which are the within-session latency/dropout transitions A1 trains.
        """
        if not self._actor_view_active or len(env_ids) == 0:
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        source_pos = self._planner_visible_pos if self.planner_revision_enabled else self.racket_target_pos_w
        source_vel = self._planner_visible_vel if self.planner_revision_enabled else self.racket_target_vel_w
        source_normal = self._planner_visible_normal if self.planner_revision_enabled else self.target_normal_cmd
        source_tts = self._planner_visible_tts if self.planner_revision_enabled else self.time_to_strike
        if self.planner_revision_enabled:
            source_tts = source_tts.clamp(min=0.0)
        self.delayed_racket_target_pos_w[ids] = source_pos[ids]
        self.delayed_racket_target_vel_w[ids] = source_vel[ids]
        self.delayed_target_normal_cmd[ids] = source_normal[ids]
        self.delayed_swing_sign[ids] = self.swing_sign[ids]
        if self._atomic_tts_active:
            self.delayed_time_to_strike[ids] = source_tts[ids]
        if self._mnoise_ar1_sigma > 0.0:
            self._mnoise_ar1_state[ids] = 0.0
        if self._a1v2_active:
            self._swing_bias[ids] = 0.0
            self._drop_cd[ids] = 0
            self._prev_pre_strike[ids] = True
            self._held_pos[ids] = source_pos[ids]
            self._held_vel[ids] = source_vel[ids]
            self._held_normal[ids] = source_normal[ids]
            self._held_sign[ids] = self.swing_sign[ids]
            if self._atomic_tts_active:
                self._held_tts[ids] = source_tts[ids]
            if self.planner_revision_enabled:
                self._held_planner_epoch[ids] = self._planner_control_epoch[ids]
                self._held_planner_task[ids] = self._planner_task_id[ids]
                self._held_planner_revision[ids] = self._planner_task_revision[ids]
                self._held_planner_last_precontact[ids] = (
                    self._planner_visible_last_precontact[ids]
                )
        if self._actor_ring_steps > 0:
            self._delay_buf_pos[:, ids] = source_pos[ids].unsqueeze(0)
            self._delay_buf_vel[:, ids] = source_vel[ids].unsqueeze(0)
            self._delay_buf_normal[:, ids] = source_normal[ids].unsqueeze(0)
            self._delay_buf_sign[:, ids] = self.swing_sign[ids].unsqueeze(0)
            if self._atomic_tts_active:
                self._delay_buf_tts[:, ids] = source_tts[ids].unsqueeze(0)
            if self.planner_revision_enabled:
                self._delay_buf_planner_epoch[:, ids] = (
                    self._planner_control_epoch[ids].unsqueeze(0)
                )
                self._delay_buf_planner_task[:, ids] = (
                    self._planner_task_id[ids].unsqueeze(0)
                )
                self._delay_buf_planner_revision[:, ids] = (
                    self._planner_task_revision[ids].unsqueeze(0)
                )
                self._delay_buf_planner_last_precontact[:, ids] = (
                    self._planner_visible_last_precontact[ids].unsqueeze(0)
                )
        if self.planner_revision_enabled:
            # A true reset hands the initial revision to the actor immediately;
            # seed the high-water mark without booking it as a same-task revision.
            self._planner_actor_control_epoch[ids] = self._planner_control_epoch[ids]
            self._planner_actor_task_id[ids] = self._planner_task_id[ids]
            self._planner_actor_task_revision[ids] = self._planner_task_revision[ids]

    def _count_swing_starts(self, env_ids, count_prestrike_falls: bool) -> None:
        """UNCONDITIONAL swing accounting (Phase A wandb fix). Increment-only here; the decay is
        applied once per step in _update_metrics next to the exact accumulators, so
        swing_completion_rate = exact_n_acc / swing_starts_acc shares one EMA timescale.
        NOTE: an episode TIMEOUT mid-swing counts as an uncompleted start (slight deflation,
        ~one boundary swing per 10 s episode) but never as a fall (terminated excludes timeouts)."""
        n = int(len(env_ids))
        if n == 0:
            return
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        exact_ledger = self._ensure_exact_behavior_decision_counters()
        exact_ledger["swing_start_count"].add_(n)
        self._swing_starts_acc += float(n)
        motion = self._motion()
        if motion._multiseg:
            # per-族累计(spdmix v2 硬绑定四):桶是正/反手两族。legacy 2-clip 族行号==clip_id
            # 逐字节不变;6-clip 下正手 1.0/1.2 变体不再被记进"backhand"桶(clip==1 的老病)。
            fams = self._metric_bucket_rows()[motion.clip_id[env_ids]]
            for c in self._clip_names:
                self._swing_starts_acc_c[c] += float((fams == c).sum())
        # --- per-swing SAME-LEDGER rally booking (metric-sync fix; see the rally block in __init__) --
        # The attempt that ENDS at this resample books its start and its returned-latch TOGETHER
        # (same call, same ledger, decayed together in _update_metrics) — returns can never outrun
        # starts. An env's very first resample books nothing (_rally_active False: no attempt
        # existed yet). The ended attempt is attributed to _prev_clip_id, the clip it was actually
        # swinging (motion has already resampled clip_id for the NEW attempt by this point).
        ended = self._rally_active[env_ids_t]
        returned = self._rally_returned[env_ids_t] & ended
        self._rally_starts_acc += float(ended.sum())
        self._rally_returns_acc += float(returned.sum())
        if motion._multiseg:
            ended_fams = self._metric_bucket_rows()[self._prev_clip_id[env_ids_t]]
            for c in self._clip_names:
                _csel = ended_fams == c
                self._rally_starts_acc_c[c] += float((ended & _csel).sum())
                self._rally_returns_acc_c[c] += float((returned & _csel).sum())
        self._rally_active[env_ids_t] = True
        # The NEW attempt starts with the parked wrap-boundary latch, not blank: a strike that
        # fired on this very wrap step belongs to the attempt beginning here (see the guard in
        # _vb_book_strike_step). One-shot — consumed on transfer.
        self._rally_returned[env_ids_t] = self._rally_pending_return[env_ids_t]
        self._rally_pending_return[env_ids_t] = False
        self._close_exact_swing_attempts(env_ids_t)

        # Rally drift close-out: a WRAP means the previous swing ran to completion — book its base
        # displacement (norm + forward component) from the swing-start stamp to the current base.
        # True resets never close out (the swing was aborted/fallen; the teleport is not drift).
        _ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if not count_prestrike_falls:  # wrap path (see _resample_is_wrap)
            _stamped = ~self._swing_start_pending[_ids_t]  # only swings whose start was stamped
            if bool(_stamped.any()):
                _d = self.base_pos_w[_ids_t, :2] - self._swing_start_base_xy[_ids_t]
                self._drift_sum_acc += float((torch.norm(_d, dim=-1) * _stamped.float()).sum())
                self._drift_fwd_sum_acc += float((_d[:, 0] * _stamped.float()).sum())
                self._drift_n_acc += float(_stamped.sum())
        # Stamp the NEW swing's start lazily (first _update_metrics after this resample): at reset
        # time base_pos_w still caches the PRE-teleport pose.
        self._swing_start_pending[_ids_t] = True
        if count_prestrike_falls:
            term = self._env.termination_manager.terminated[env_ids]
            pre = self.pre_strike[env_ids]
            # POST-strike fall = terminated at/after the strike frame (tts <= 0, follow-through) OR
            # during the post-wrap hold (_recover_from_clip latch >= 0: the previous swing's recovery,
            # even though the wrap already flipped pre_strike=True for the NEXT swing). Both are the
            # "hit, then fall while recovering" failure that completion + pre-strike metrics miss.
            rec = self._recover_from_clip[env_ids]
            recovering = rec >= 0
            true_pre = term & pre & ~recovering
            post = term & (~pre | recovering)
            self._book_exact_behavior_terminal_reset(
                env_ids_t, pre_strike=pre, recovering=recovering
            )
            self._prestrike_fall_acc += float(true_pre.sum())
            self._poststrike_fall_acc += float(post.sum())
            if motion._multiseg:
                # Attribute the fall to the clip the env was ON when it fell: pre-strike falls to the
                # _prev_clip_id snapshot (motion already resampled clip_id for the new episode);
                # post-wrap-hold falls to the latched clip whose swing caused the recovery.
                fall_clips = torch.where(recovering, rec, self._prev_clip_id[env_ids])
                fall_fams = self._metric_bucket_rows()[fall_clips]
                for c in self._clip_names:
                    csel = fall_fams == c
                    self._prestrike_fall_acc_c[c] += float((true_pre & csel).sum())
                    self._poststrike_fall_acc_c[c] += float((post & csel).sum())
            # True reset: the new episode starts fresh (its stand-start/reset hold is genuine
            # pre-strike preparation, not recovery), so clear the latch for these envs.
            self._recover_from_clip[env_ids_t] = -1

    def _close_exact_swing_attempts(self, env_ids: torch.Tensor) -> None:
        """Close old attempts and start new ones with paired decision-window accounting.

        ``swing_start_count`` remains a useful mechanism/throughput counter, while pruning uses
        ``swing_completion_count / swing_outcome_count``.  The paired counters are written in the
        same call, so a PPO-window boundary cannot create the old false ``strike > start``
        invariant failure.
        """

        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        if len(ids) == 0:
            return
        ledger = self._ensure_exact_behavior_decision_counters()
        ended = self._exact_attempt_active[ids]
        completed = ended & self._exact_attempt_completed[ids]
        ledger["swing_outcome_count"].add_(ended.sum(dtype=torch.long))
        ledger["swing_completion_count"].add_(completed.sum(dtype=torch.long))
        if getattr(self, "planner_revision_enabled", False):
            self._ensure_exact_timing_bucket_state()
            timing_bucket = self._exact_attempt_initial_tts_bucket[ids]
            for bucket_id, bucket_name in enumerate(_PLANNER_INITIAL_TTS_BUCKETS):
                in_bucket = timing_bucket == bucket_id
                ledger[f"planner_initial_tts_{bucket_name}_swing_outcome_count"].add_(
                    (ended & in_bucket).sum(dtype=torch.long)
                )
                ledger[f"planner_initial_tts_{bucket_name}_swing_completion_count"].add_(
                    (completed & in_bucket).sum(dtype=torch.long)
                )
            # The new attempt exists after this call but receives its bucket later in the same
            # resample, when _begin_same_ball_planner_task samples its initial deadline.
            self._exact_attempt_initial_tts_bucket[ids] = -1
        self._exact_attempt_active[ids] = True
        self._exact_attempt_completed[ids] = self._exact_pending_completion[ids]
        self._exact_pending_completion[ids] = False

    def _latch_exact_swing_completion(self, exact_strike: torch.Tensor) -> None:
        """Latch an exact strike onto the attempt that owns the current motion frame."""

        exact = exact_strike.detach().bool()
        wrapped = getattr(self._motion(), "just_resampled", None)
        if wrapped is None:
            self._exact_attempt_completed |= exact
            return
        wrapped = wrapped.detach().bool()
        self._exact_attempt_completed |= exact & ~wrapped
        self._exact_pending_completion |= exact & wrapped

    def _update_footwork_signals(self, racket_dist: torch.Tensor) -> None:
        """Base-FREE footwork-to-strike signals (reward/metric only; NEVER observed). The legs are driven
        to move by racket PROGRESS (reducing the racket->target distance), not by any base target. All
        guards degrade to 0 if a body/sensor cannot resolve, so this can never crash training."""
        data = self.robot.data
        # --- racket-target distance + dense progress (the base-free movement driver) ---
        self.racket_target_distance = racket_dist
        # progress = previous - current distance. Resample/reset steps are not learnable progress:
        # the target and/or reference clip jumped, so reset the baseline and emit exactly zero.
        motion = self._motion()
        reset_progress = self._progress_reset_mask.clone()
        if hasattr(motion, "just_resampled"):
            reset_progress |= motion.just_resampled
        progress = (self._prev_racket_dist - racket_dist).clamp(-0.15, 0.15)
        self.racket_progress = torch.where(reset_progress, torch.zeros_like(progress), progress)
        self._prev_racket_dist = racket_dist.detach()
        self._progress_reset_mask.zero_()
        self.metrics["racket_target_distance"] = racket_dist
        self.metrics["racket_progress"] = self.racket_progress
        self.metrics["racket_progress_prestrike"] = torch.where(
            self.pre_strike, self.racket_progress, torch.zeros_like(self.racket_progress)
        )
        # --- base stability components (training-only) ---
        pg = getattr(data, "projected_gravity_b", None)
        if pg is not None:
            self.proj_grav_xy = torch.norm(pg[:, :2], dim=-1)
        self.base_ang_vel_xy_norm = torch.norm(data.root_ang_vel_b[:, :2], dim=-1)
        self.vertical_speed = torch.abs(data.root_lin_vel_b[:, 2])
        self.metrics["proj_grav_xy"] = self.proj_grav_xy
        self.metrics["base_ang_vel_xy"] = self.base_ang_vel_xy_norm
        self.metrics["base_vertical_speed"] = self.vertical_speed
        # P2.4 (PACE smooth deceleration): mean planar base speed during the approach — the quantity
        # the base_decel_tracking reward shapes toward v_des = clamp(v_gain*dist_xy, 0, v_max).
        # Watch it fall on far targets when the term is enabled (task.rewards.base_decel_weight>0).
        base_speed_xy = torch.norm(data.root_lin_vel_w[:, :2], dim=-1)
        self.metrics["base_speed_xy_prestrike"] = torch.where(
            self.pre_strike, base_speed_xy, torch.zeros_like(base_speed_xy)
        )
        # Rally: planar base speed through the FOLLOW-THROUGH (strike-window exit -> wrap) — the
        # braking window the post_strike_brake reward shapes. Held-write (carries the last
        # in-window value between swings) so the tail-mean reads the typical post-strike speed.
        _brake_win = (~self.pre_strike) & (~self.strike_window)
        self.metrics["post_strike_base_speed_xy"] = torch.where(
            _brake_win, base_speed_xy, self.metrics["post_strike_base_speed_xy"]
        )
        # Rally: cumulative displacement from the env origin (the P7 forward-drift accumulator).
        self.metrics["base_dist_from_origin"] = torch.norm(
            self.base_pos_w[:, :2] - self._env.scene.env_origins[:, :2], dim=-1
        )
        self._update_hold_recovery_metrics(getattr(self._motion(), "in_hold", None))
        # Rally: lazy swing-start stamp (fresh base_pos_w — see __init__ rationale).
        if bool(self._swing_start_pending.any()):
            _pend = self._swing_start_pending
            self._swing_start_base_xy[_pend] = self.base_pos_w[_pend, :2]
            self._swing_start_pending.zero_()
        # --- foot footwork signals (slip² / velocity / drag); feet may STEP, so this is PENALTY-only ---
        if self._foot_idx_robot and self._contact_sensor is not None and self._foot_idx_contact:
            f_force = torch.norm(self._contact_sensor.data.net_forces_w[:, self._foot_idx_contact, :], dim=-1)
            in_contact = (f_force > 10.0).float()  # (E,2)
            f_vel = data.body_lin_vel_w[:, self._foot_idx_robot, :]  # (E,2,3)
            f_xy_speed = torch.norm(f_vel[..., :2], dim=-1)  # (E,2)
            f_speed = torch.norm(f_vel, dim=-1)  # (E,2)
            f_height = data.body_pos_w[:, self._foot_idx_robot, 2]  # (E,2)
            self.foot_slip_sq = (in_contact * f_xy_speed.square()).sum(dim=-1)  # contact * ||v_xy||²
            self.foot_vel_sq = f_speed.square().sum(dim=-1)  # excessive/violent foot motion
            # Phase A fix: the old height gate (f_height < 0.10 m) sat BELOW the planted ankle
            # origin (~0.07 m), so EVERY low step counted as "dragging" — stepping itself was
            # taxed, one of the reasons the policy never learned to move left/right. "Drag" now
            # means lateral speed while the foot is LOADED (in contact) = sliding under load;
            # airborne swing-leg motion is free (foot_vel_sq still bounds violent motion).
            self.foot_drag = (in_contact * f_xy_speed).sum(dim=-1)
            self.metrics["foot_slip_sq"] = self.foot_slip_sq
            self.metrics["foot_vel_mean"] = f_speed.mean(dim=-1)
            self.metrics["foot_lift_rate"] = (1.0 - in_contact).mean(dim=-1)  # 0 = both planted, 1 = airborne
            self.metrics["foot_vel_at_strike"] = torch.where(
                self.strike_window, f_speed.mean(dim=-1), torch.zeros(self.num_envs, device=self.device)
            )
        # --- anti-arm-only: ARM joints near a limit + arm joint velocity (resolve arm joint idx once) ---
        if not getattr(self, "_arm_resolved", False):
            self._arm_resolved = True
            self._arm_joint_idx, self._leg_joint_idx, self._waist_twist_idx = [], [], []
            try:
                self._arm_joint_idx = list(self.robot.find_joints([".*shoulder.*", ".*elbow.*", ".*wrist.*"])[0])
            except Exception:
                pass
            try:
                self._leg_joint_idx = list(self.robot.find_joints([".*hip.*", ".*knee.*", ".*ankle.*"])[0])
            except Exception:
                pass
            # waist YAW+ROLL: the "twist/lean instead of step" DOFs the policy uses to face a lateral
            # target without moving its feet. Penalized (pre-strike) so reaching a far target needs footwork.
            # waist_pitch is EXCLUDED (it is the swing wind-up / natural lean, not a lateral-reach cheat).
            try:
                self._waist_twist_idx = list(self.robot.find_joints(["waist_yaw_joint", "waist_roll_joint"])[0])
            except Exception:
                pass
        limits = getattr(data, "soft_joint_pos_limits", None)
        if limits is None:
            limits = getattr(data, "joint_pos_limits", None)
        if self._arm_joint_idx and limits is not None:
            ai = self._arm_joint_idx
            half = ((limits[:, ai, 1] - limits[:, ai, 0]) * 0.5).clamp(min=1e-6)
            d = torch.minimum(
                data.joint_pos[:, ai] - limits[:, ai, 0], limits[:, ai, 1] - data.joint_pos[:, ai]
            ).clamp(min=0.0)
            self.arm_overreach_frac = ((d / half) < 0.1).float().mean(dim=-1)  # within 10% of a limit
            self.metrics["arm_overreach_frac"] = self.arm_overreach_frac
            self.metrics["arm_joint_vel_max"] = torch.max(torch.abs(data.joint_vel[:, ai]), dim=-1).values
        # --- diagnostic: do the LEGS actually move before the strike? (footwork is happening) ---
        if self._leg_joint_idx:
            leg_vel = torch.max(torch.abs(data.joint_vel[:, self._leg_joint_idx]), dim=-1).values
            self.metrics["leg_joint_vel_max"] = leg_vel
            self.metrics["leg_moving_prestrike"] = torch.where(
                self.pre_strike, (leg_vel > 0.2).float(), torch.zeros(self.num_envs, device=self.device)
            )
        # --- anti twist-instead-of-step: |waist_yaw|+|waist_roll| deviation from the default (neutral,
        #     facing-forward) pose. This is the magnitude the prestrike_waist_twist reward penalizes, so
        #     reaching a lateral target by turning the torso (feet planted) becomes costly -> step instead. ---
        if self._waist_twist_idx:
            wi = self._waist_twist_idx
            self.waist_twist = torch.abs(data.joint_pos[:, wi] - data.default_joint_pos[:, wi]).sum(dim=-1)
            self.metrics["waist_twist_abs"] = self.waist_twist
            self.metrics["waist_twist_prestrike"] = torch.where(
                self.pre_strike, self.waist_twist, torch.zeros(self.num_envs, device=self.device)
            )

    def _update_hold_recovery_metrics(self, in_hold) -> None:
        """Book heading recovery on true hold start/expiry edges.

        A resample can replace a held state with another held state without a boolean transition;
        ``_hold_edge_pending`` turns that into a fresh start while suppressing a false expiry. A
        pending zero-length hold records neither edge. This helper is isolated for CPU boundary
        tests because these ledger semantics are easy to regress inside the larger metrics pass.
        """
        if in_hold is None:
            return
        in_hold = in_hold.bool()
        pending = self._hold_edge_pending
        expired = self._previous_in_hold & ~in_hold & ~pending
        if bool(expired.any()):
            quat = self.base_quat_w[expired]
            forward_x = 1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2)
            forward_y = 2.0 * (quat[:, 1] * quat[:, 2] + quat[:, 0] * quat[:, 3])
            expiry_yaw = torch.atan2(forward_y, forward_x).abs()
            self._heading_expiry_sum_acc += float(expiry_yaw.sum())
            self._heading_expiry_n_acc += float(expired.sum())

            spawn_yaw = self._hold_start_yaw[expired]
            conditioned = spawn_yaw > _RECOVERY_START_YAW_THRESHOLD
            if bool(conditioned.any()):
                self._recovery_spawn_sum_acc += float(spawn_yaw[conditioned].sum())
                self._recovery_expiry_sum_acc += float(expiry_yaw[conditioned].sum())
                self._recovery_n_acc += float(conditioned.sum())

        started = ((~self._previous_in_hold) | pending) & in_hold
        if bool(started.any()):
            quat = self.base_quat_w[started]
            forward_x = 1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2)
            forward_y = 2.0 * (quat[:, 1] * quat[:, 2] + quat[:, 0] * quat[:, 3])
            self._hold_start_yaw[started] = torch.atan2(forward_y, forward_x).abs()

        self._previous_in_hold = in_hold.clone()
        self._hold_edge_pending.zero_()

    def _vb_evaluate(self, exact_strike: torch.Tensor, pos_err: torch.Tensor):
        """Tier-1 at-strike virtual-ball evaluation (rewardDesign.md).

        Runs once per control step from ``_update_metrics`` (rewards/obs read the same fresh
        buffers after ``command_manager.compute()``). Whenever ANY env sits at its exact-strike
        frame, the FULL batch goes through contact + coarse rollout — the cost is kernel-launch
        bound and batch-size independent, so gathering the ~30 striking envs saves nothing
        (verify_tier1 (b)); ``vb_fired`` masks consumption. On strike-free steps the one-shot
        mask is cleared and nothing is computed.
        """
        from whole_body_tracking.tasks.tracking.mdp import virtual_ball as _vb

        if not bool(exact_strike.any()):
            self.vb_fired.zero_()
            return
        if self._vb_params is None:
            self._vb_params = _vb.load_venue_params()
            print(
                f"[RacketTargetCommand] virtual ball ON: venue constants from "
                f"{self._vb_params.source_path} (k_d={self._vb_params.k_d}, "
                f"k_m={self._vb_params.k_m}, e(u_n)={self._vb_params.paddle_e_g1}"
                f"*exp({self._vb_params.paddle_e_g2}*u_n), a_t={self._vb_params.paddle_a_t})",
                flush=True,
            )
        prm = self._vb_params

        v_in, w_in = self.vb_vel_in_w, self.vb_spin_in_w
        v_r, n_face = self.racket_lin_vel_w, self.racket_normal_w
        # FACE-IDENTITY GATE: compare the convention-matched signed pair BEFORE orient_normal.
        # Under the active face-command contract this is achieved raw mount +Y/A versus demanded
        # raw A.  ``orient_normal`` is sign-invariant and may orient the contact plane for impulse
        # physics, but it must never turn the opposite rubber face into a scored hit.
        face_achieved, face_target = face_tracking_pair(self)
        if self.cfg.face_command:
            # A-frame commands become the external physical-B demand through the exact per-env
            # mount sign already materialized by _compute_racket_state.  Recover that +/-1 without
            # indexing the motion a second time; invalid/degenerate rows fail inside the gate.
            mount_sign = torch.where(
                torch.sum(self.racket_normal_w * self.racket_normal_raw_w, dim=-1, keepdim=True)
                >= 0.0,
                torch.ones_like(self.racket_normal_w[:, :1]),
                -torch.ones_like(self.racket_normal_w[:, :1]),
            )
            target_physical_b = self.target_normal_cmd * mount_sign
        else:
            target_physical_b = self.racket_target_normal_w
        signed_face_ok, _signed_face_dot = _vb.signed_face_hemisphere(
            face_achieved,
            face_target,
            achieved_physical_b=self.racket_normal_w,
            target_physical_b=target_physical_b,
        )
        # CAPTURE GATE: correct signed face, close enough at the strike frame, AND paddle moving
        # INTO the ball along the oriented contact normal (a stationary/retreating wall-block scores
        # nothing — verify (c)3).
        n_or = _vb.orient_normal(n_face, v_in, v_r)
        approach = torch.sum(v_r * n_or, dim=-1)
        gate = (
            exact_strike
            & signed_face_ok
            & (pos_err < float(self.cfg.vb_capture_radius))
            & (approach > float(self.cfg.vb_min_approach_speed))
        )

        # Achieved-state contact (venue paddle model, e(u_n)) + coarse landing rollout.
        # The rollout must start in the ENV-LOCAL frame: the virtual table landmarks
        # (vb_table_near_x / net / far end) and vb_target_xy are per-env offsets from the env
        # origin, while racket_pos_w is TRUE world frame (env grids span tens of meters at 4096
        # envs — using it raw put every landing ~|env_origin| away from the target; caught in the
        # first vb_warmE14k run: virtual_land_err_m ~62 m). Env grids are pure translations, so
        # velocities, normals, and spins need no correction.
        v_plus, w_plus = _vb.predict_paddle_contact(v_in, v_r, n_face, w_in, prm)
        land = _vb.coarse_landing(
            self.racket_pos_w - self._env.scene.env_origins,
            v_plus,
            w_plus,
            prm,
            # Physical table contact = ball CENTER crossing surface + R (oracle / C++ /
            # landing.py / shadow-ball convention; venue landings were extracted at this
            # plane). Bare surface read the landing ~24 mm long (measured on ball-physics-
            # unify 2026-07-05, ported here so vb and shadow metrics share one convention).
            surface_z=float(self.cfg.vb_table_surface_z) + prm.ball_radius,
            net_x=self._vb_net_x,
            h=float(self.cfg.vb_rollout_h),
            n_steps=int(self.cfg.vb_rollout_steps),
        )
        lx, ly = land["land_xy"][:, 0], land["land_xy"][:, 1]
        on_opp = (
            land["land_valid"] & (lx > self._vb_net_x) & (lx <= self._vb_far_x) & (ly.abs() <= self._vb_half_w)
        )
        depth_ok = lx > (self._vb_net_x + float(self.cfg.vb_min_landing_depth))
        net_clear = land["net_valid"] & (land["net_z"] > self._vb_net_top_z + self._vb_ball_r)
        # Outgoing topspin component about t_hat = z_hat x d_hat of the outgoing horizontal
        # direction (Ace-style): omega . t_hat = -w_x*d_y + w_y*d_x.
        d_xy = v_plus[:, :2]
        d_hat = d_xy / (torch.linalg.norm(d_xy, dim=-1, keepdim=True) + 1e-9)
        topspin = -w_plus[:, 0] * d_hat[:, 1] + w_plus[:, 1] * d_hat[:, 0]

        # One-shot caches consumed by hope_rewards.virtual_* THIS step.
        self.vb_fired = gate
        self.vb_landing_xy = land["land_xy"]
        self.vb_landing_valid = land["land_valid"]
        self.vb_on_opponent = on_opp
        self.vb_depth_ok = depth_ok
        self.vb_net_z = land["net_z"]
        self.vb_net_clear = net_clear
        self.vb_net_crossed = land["net_valid"]
        self.vb_topspin = topspin
        self.vb_spin_out_norm = torch.linalg.norm(w_plus, dim=-1)

        # Sample-weighted EMA rates (hit rate over exact-strike samples; outcome rates over captured
        # hits). NOTE: accumulators only decay on strike-carrying steps — exact at 4096 envs (a
        # strike happens ~every step), slightly stale at small env counts (diagnostics only).
        decay = float(self.cfg.exact_success_decay)
        _legal = gate & net_clear & on_opp
        self._vb_book_strike_step(decay, exact_strike, gate, net_clear, land["land_valid"], _legal)
        enough_e = self._vb_exact_acc >= float(self.cfg.exact_success_min_count)
        enough_h = self._vb_hit_acc >= 1.0
        self.metrics["virtual_hit_rate"][:] = (self._vb_hit_acc / max(self._vb_exact_acc, 1e-6)) if enough_e else 0.0
        self.metrics["virtual_net_clear_rate"][:] = (self._vb_net_acc / max(self._vb_hit_acc, 1e-6)) if enough_h else 0.0
        self.metrics["virtual_land_valid_rate"][:] = (
            (self._vb_land_valid_acc / max(self._vb_hit_acc, 1e-6)) if enough_h else 0.0
        )
        self.metrics["virtual_land_inbounds_rate"][:] = (
            (self._vb_inb_acc / max(self._vb_hit_acc, 1e-6)) if enough_h else 0.0
        )
        # PRIMARY in-training curve (franco 2026-07-06): 上台率 per strike OPPORTUNITY — legal
        # returns (captured hit & net cleared & landed on the opponent half) over EXACT-STRIKE
        # samples. Composes hit-rate x land-rate, so "hit rarely but land those" cannot inflate
        # it the way virtual_land_inbounds_rate (hit-denominator) can. 击球率 = virtual_hit_rate
        # stays the auxiliary; strike_composite is diagnostics. Canonical bookkeeping = MuJoCo.
        self.metrics["virtual_return_rate"][:] = (
            (self._vb_inb_acc / max(self._vb_exact_acc, 1e-6)) if enough_e else 0.0
        )
        # CONTINUOUS-RALLY return rate (franco 2026-07-08 长期追踪): legal returns per swing —
        # falls and never-reached-strike swings count as failures. The trusted curve
        # (virtual_return_rate_rally*) is now computed in _update_metrics from the per-swing
        # SAME-LEDGER counters (metric-sync fix; see the rally block in __init__ and
        # _rally_report). The OLD mixed-ledger readout survives one transition period as *_legacy
        # (written after the per-clip accumulator updates below — the old write points, strike-
        # carrying steps only, frozen between strikes — so it reproduces the old readings exactly
        # for new/old comparison). Its known disease: >1 spikes under synchronized reset queues.
        # Per-side (forehand/backhand) return rate — 反手先行 judging needs the per-side number.
        _motion = self._motion()
        _is_multiseg = getattr(_motion, "_multiseg", False)
        if _is_multiseg:
            _fam = self._metric_bucket_rows()[_motion.clip_id]
            for _c, _cn in self._clip_names.items():
                _sel = exact_strike & (_fam == _c)
                self._vb_exact_acc_c[_c] = decay * self._vb_exact_acc_c[_c] + float(_sel.sum())
                self._vb_inb_acc_c[_c] = decay * self._vb_inb_acc_c[_c] + float((_legal & _sel).sum())
                self._vb_hit_acc_c[_c] = decay * self._vb_hit_acc_c[_c] + float((gate & _sel).sum())
                _n = self._vb_exact_acc_c[_c]
                _scale = (1.0 / max(_n, 1e-6)) if _n >= float(self.cfg.exact_success_min_count) else 0.0
                self.metrics[f"virtual_return_rate_{_cn}"][:] = self._vb_inb_acc_c[_c] * _scale
                self.metrics[f"virtual_hit_rate_{_cn}"][:] = self._vb_hit_acc_c[_c] * _scale
        if self.cfg.rally_legacy_metrics:
            _lg_global, _lg_per_clip = self._rally_legacy_values()
            self.metrics["virtual_return_rate_rally_legacy"][:] = _lg_global
            if _is_multiseg:
                for _cn in self._clip_names.values():
                    self.metrics[f"virtual_return_rate_rally_{_cn}_legacy"][:] = _lg_per_clip[_cn]
        self.metrics["virtual_approach_speed"] = torch.where(
            exact_strike, approach, self.metrics["virtual_approach_speed"]
        )
        fired_valid = gate & land["land_valid"]
        if bool(fired_valid.any()):
            derr = torch.linalg.norm(land["land_xy"] - self._vb_target_xy.unsqueeze(0), dim=-1)
            self.metrics["virtual_land_err_m"][:] = derr[fired_valid].mean()
            self.metrics["virtual_topspin_revs"][:] = (topspin[fired_valid] / (2.0 * math.pi)).mean()

    def _vb_book_strike_step(
        self,
        decay: float,
        exact_strike: torch.Tensor,
        gate: torch.Tensor,
        net_clear: torch.Tensor,
        land_valid: torch.Tensor,
        legal: torch.Tensor,
    ) -> None:
        """EMA booking for THIS strike-carrying step's virtual-ball outcomes + the rally latch.

        Only reached on strike-carrying steps (_vb_evaluate returns early otherwise), so these
        accumulators skip decay on strike-free steps — that schedule asymmetry vs the every-step
        _decay_swing_accounting is one half of the legacy rally disease (the other half is the
        ~1-swing booking phase lag vs _swing_starts_acc). The rally latch on the last line is the
        metric-sync FIX's booking point: it only marks "this swing produced a legal return";
        the actual ledger entry happens at the swing's end in _count_swing_starts.
        """
        self._vb_exact_acc = decay * self._vb_exact_acc + float(exact_strike.sum())
        self._vb_hit_acc = decay * self._vb_hit_acc + float(gate.sum())
        self._vb_net_acc = decay * self._vb_net_acc + float((gate & net_clear).sum())
        self._vb_land_valid_acc = decay * self._vb_land_valid_acc + float((gate & land_valid).sum())
        self._vb_inb_acc = decay * self._vb_inb_acc + float(legal.sum())
        self._book_sparse_reward_eligibility(
            exact_strike=exact_strike,
            capture=gate,
            net_clear=gate & net_clear,
            landing_valid=gate & land_valid,
            legal_return=legal,
            book_strike_opportunity=False,
        )
        # Rally latch with a wrap-boundary guard: on the step a clip WRAPS, the motion term has
        # already advanced to the NEW clip before this metrics pass, so a strike frame sitting at
        # the swing's entry (strike phase ~0, or rsi_skip_settle_frames landing on the strike
        # offset) fires exact_strike for the NEW attempt while the OLD attempt is still unbooked
        # (_count_swing_starts only runs later, inside _update_command). Latching such a return
        # into _rally_returned would credit it to the OLD rally (cross-rally leakage) AND lose it
        # for the new one. Park it in _rally_pending_return instead; _count_swing_starts books the
        # ended attempt first, then hands the parked latch to the attempt that owns it. Current
        # clips strike mid-swing (phase 0.28-0.50) and never fire at the wrap step — defensive.
        wrapped = getattr(self._motion(), "just_resampled", None)
        if wrapped is None:
            self._rally_returned = self._rally_returned | legal
        else:
            self._rally_returned = self._rally_returned | (legal & ~wrapped)
            self._rally_pending_return = self._rally_pending_return | (legal & wrapped)

    def _book_sparse_reward_eligibility(
        self,
        *,
        exact_strike: torch.Tensor,
        capture: torch.Tensor,
        net_clear: torch.Tensor,
        landing_valid: torch.Tensor,
        legal_return: torch.Tensor,
        book_strike_opportunity: bool = True,
    ) -> None:
        """Book exact, non-decayed sparse-reward counters for one simulator step."""

        masks = {
            "virtual_capture_count": capture,
            "virtual_net_clear_count": net_clear,
            "virtual_landing_valid_count": landing_valid,
            "virtual_legal_return_count": legal_return,
        }
        if book_strike_opportunity:
            masks = {"strike_opportunity_count": exact_strike, **masks}
        ledger = getattr(self, "_sparse_reward_eligibility_counters", None)
        if ledger is None:
            names = (
                "strike_opportunity_count",
                "virtual_capture_count",
                "virtual_net_clear_count",
                "virtual_landing_valid_count",
                "virtual_legal_return_count",
            )
            ledger = {name: torch.zeros((), dtype=torch.long, device=self.device) for name in names}
            for family in self._clip_names.values():
                for name in names:
                    ledger[f"{name}_{family}"] = torch.zeros(
                        (), dtype=torch.long, device=self.device
                    )
            self._sparse_reward_eligibility_counters = ledger
        for name, mask in masks.items():
            ledger[name].add_(mask.detach().sum(dtype=torch.long))

        self._book_exact_timing_bucket_sparse_events(masks)

        motion = self._motion()
        if getattr(motion, "_multiseg", False):
            fam_id = self._metric_bucket_rows()[motion.clip_id]
            for family_row, family in self._clip_names.items():
                selected = fam_id == family_row
                for name, mask in masks.items():
                    ledger[f"{name}_{family}"].add_(
                        (mask.detach() & selected).sum(dtype=torch.long)
                    )

    def consume_sparse_reward_eligibility_counters(self) -> dict[str, torch.Tensor]:
        """Snapshot and reset one PPO update's sparse outcome ledger exactly once."""

        if not hasattr(self, "_sparse_reward_eligibility_counters"):
            empty = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._book_sparse_reward_eligibility(
                exact_strike=empty,
                capture=empty,
                net_clear=empty,
                landing_valid=empty,
                legal_return=empty,
            )

        snapshot = {
            name: value.detach().clone()
            for name, value in self._sparse_reward_eligibility_counters.items()
        }
        # Reward/command updates can allocate these under inference mode.  Reset under the same
        # mode so the logger never mutates an inference tensor from normal mode.
        with torch.inference_mode():
            for value in self._sparse_reward_eligibility_counters.values():
                value.zero_()
        return snapshot

    def _ensure_exact_behavior_decision_counters(self) -> dict[str, torch.Tensor]:
        """Create the fixed behavior counters and any configured termination-reason counters."""

        ledger = getattr(self, "_exact_behavior_decision_counters", None)
        if ledger is None:
            ledger = {}
            self._exact_behavior_decision_counters = ledger
        for name in (
            "swing_start_count",
            "swing_outcome_count",
            "swing_completion_count",
            "terminal_reset_count",
            "timeout_reset_count",
            "physical_fall_count",
            "pre_strike_physical_fall_count",
            "post_strike_physical_fall_count",
            "non_physical_terminal_reset_count",
            "ready_tilt_eligible_sample_count",
            "ready_base_speed_eligible_sample_count",
            "ready_station_offset_eligible_sample_count",
            "ready_foot_contact_eligible_sample_count",
            "ready_foot_slip_eligible_sample_count",
            "ready_phase_sample_count",
            "ready_planner_task_entry_sample_count",
            "ready_planner_legacy_hold_violation_count",
            "ready_foot_sensor_unavailable_sample_count",
            "ready_nonfinite_value_count",
        ):
            if name not in ledger:
                ledger[name] = torch.zeros((), dtype=torch.long, device=self.device)
        if self._cq_enabled:  # 类级默认 False 兜底,不必再写 getattr
            # CONTINUOUS question accounting. Goes into the SAME integer ledger the hourly monitor
            # already parses (one HOPE_EXACT_BEHAVIOR_UPDATE_JSON line per PPO update), so a target
            # drought is visible without touching a single line of the monitor.
            # continuous_question_exhausted_rate > 0 is THE anomaly to catch.
            for name in (
                "continuous_question_draw_count",
                "continuous_question_admitted_count",
                "continuous_question_install_count",
                "continuous_question_exhausted_count",
                "continuous_question_refill_count",
                "continuous_question_redraw_round_sum",
                "continuous_question_resid_um_sum",
                "continuous_question_rows_discarded_count",
                "continuous_question_pool_dry_count",
                "continuous_question_pool_underflow_count",
                "continuous_question_reject_face_deg_over_cap_count",
                "continuous_question_reject_exam_holdout_count",
                "continuous_question_closed_loop_um_sum",
                "continuous_question_closed_loop_row_count",
                "continuous_question_closed_loop_fail_count",
            ):
                if name not in ledger:
                    ledger[name] = torch.zeros((), dtype=torch.long, device=self.device)
            # Reason names are stroke_adapt_torch.REASONS verbatim, so a training histogram and a
            # deploy stroke_adapt log can be compared directly.
            from .stroke_adapt_torch import REASONS as _CQ_REASONS

            for _reason in _CQ_REASONS:
                name = f"continuous_question_reject_{_reason}_count"
                if name not in ledger:
                    ledger[name] = torch.zeros((), dtype=torch.long, device=self.device)
            _nb = max(1, int(self.cfg.cq_accept_buckets))
            _nc = len(getattr(self, "_clip_names", {})) or 2
            for _c in range(max(_nc, 8)):
                for _b in range(_nb):
                    for _kind in ("asked", "kept"):
                        name = f"continuous_question_bucket{_c}_{_b}_{_kind}_count"
                        if name not in ledger:
                            ledger[name] = torch.zeros((), dtype=torch.long, device=self.device)
        if getattr(self, "planner_revision_enabled", False):
            for name in (
                "planner_initial_tts_sample_count",
                "planner_initial_tts_sub_0p5_count",
                "planner_initial_tts_exact_0p5_count",
                "planner_initial_tts_above_0p5_count",
                "planner_revision_attempt_count",
                "planner_revision_accepted_count",
                "planner_revision_rejected_count",
                "planner_revision_last_precontact_attempt_count",
                "planner_revision_last_precontact_accepted_count",
                "planner_revision_actor_visible_count",
                "planner_revision_last_precontact_actor_visible_count",
            ):
                if name not in ledger:
                    ledger[name] = torch.zeros((), dtype=torch.long, device=self.device)
            for bucket_name in _PLANNER_INITIAL_TTS_BUCKETS:
                for event_name in (
                    "swing_outcome_count",
                    "swing_completion_count",
                    *_TIMING_BUCKET_SPARSE_EVENTS,
                ):
                    name = f"planner_initial_tts_{bucket_name}_{event_name}"
                    if name not in ledger:
                        ledger[name] = torch.zeros(
                            (), dtype=torch.long, device=self.device
                        )
            mixture = getattr(self, "_planner_initial_tts_mixture", None)
            if mixture is not None:
                for index, _component in enumerate(mixture.components):
                    name = f"planner_initial_tts_component_{index}_count"
                    if name not in ledger:
                        ledger[name] = torch.zeros(
                            (), dtype=torch.long, device=self.device
                        )
        for name in (
            "ready_tilt_rad_sum",
            "ready_base_speed_xy_mps_sum",
            "ready_station_offset_m_sum",
            "ready_foot_contact_fraction_sum",
            "ready_foot_slip_speed_mps_sum",
        ):
            if name not in ledger:
                ledger[name] = torch.zeros((), dtype=torch.float64, device=self.device)

        termination_manager = getattr(self._env, "termination_manager", None)
        for term_name in tuple(getattr(termination_manager, "active_terms", ())):
            key = f"termination_reason_{term_name}_count"
            if key not in ledger:
                ledger[key] = torch.zeros((), dtype=torch.long, device=self.device)
        return ledger

    @staticmethod
    def _selected_bool(value, env_ids: torch.Tensor) -> torch.Tensor:
        """Select one manager mask without accepting numeric/non-boolean reason tensors."""

        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=env_ids.device)
        value = value.to(device=env_ids.device)
        if value.dtype != torch.bool:
            raise TypeError("termination reason masks must have boolean dtype")
        if value.ndim == 0:
            return value.expand(len(env_ids))
        if value.ndim != 1:
            raise ValueError("termination reason masks must be scalar or one-dimensional")
        return value[env_ids]

    def _book_exact_behavior_terminal_reset(
        self,
        env_ids: torch.Tensor,
        *,
        pre_strike: torch.Tensor,
        recovering: torch.Tensor,
    ) -> None:
        """Attribute true-reset reasons; only absolute balance guards are physical falls.

        ``pre_strike`` is the ending attempt's timing latch.  ``recovering`` wins over that latch:
        a fall during a post-wrap hold belongs to the completed swing's post-strike recovery even
        though the next swing already reports pre-strike timing.
        """

        ledger = self._ensure_exact_behavior_decision_counters()
        tm = getattr(self._env, "termination_manager", None)
        if tm is None or len(env_ids) == 0:
            return
        terminated = self._selected_bool(
            getattr(tm, "terminated", torch.zeros((), dtype=torch.bool, device=self.device)),
            env_ids,
        )
        timed_out = self._selected_bool(
            getattr(tm, "time_outs", torch.zeros((), dtype=torch.bool, device=self.device)),
            env_ids,
        )
        selected_phase_masks = {}
        for name, value in (("pre_strike", pre_strike), ("recovering", recovering)):
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value, device=self.device)
            value = value.to(device=self.device)
            if value.dtype != torch.bool:
                raise TypeError(f"{name} attribution mask must have boolean dtype")
            if value.shape != terminated.shape:
                raise ValueError(
                    f"{name} attribution mask shape {tuple(value.shape)} does not match "
                    f"selected terminations {tuple(terminated.shape)}"
                )
            selected_phase_masks[name] = value
        pre_strike = selected_phase_masks["pre_strike"]
        recovering = selected_phase_masks["recovering"]
        reason_masks: dict[str, torch.Tensor] = {}
        get_term = getattr(tm, "get_term", None)
        if callable(get_term):
            for term_name in tuple(getattr(tm, "active_terms", ())):
                mask = self._selected_bool(get_term(term_name), env_ids)
                reason_masks[str(term_name)] = mask

        # Commit only after every dynamic reason mask has passed strict dtype/shape selection.
        # A malformed late-listed term must not leave a partially updated decision transaction.
        ledger["terminal_reset_count"].add_(terminated.sum(dtype=torch.long))
        ledger["timeout_reset_count"].add_(timed_out.sum(dtype=torch.long))
        for term_name, mask in reason_masks.items():
            ledger[f"termination_reason_{term_name}_count"].add_(
                mask.sum(dtype=torch.long)
            )

        physical = torch.zeros_like(terminated)
        for term_name in _PHYSICAL_FALL_TERMINATION_TERMS:
            physical |= reason_masks.get(term_name, torch.zeros_like(terminated))
        physical &= terminated
        true_pre = physical & pre_strike & ~recovering
        post = physical & (~pre_strike | recovering)
        ledger["physical_fall_count"].add_(physical.sum(dtype=torch.long))
        ledger["pre_strike_physical_fall_count"].add_(true_pre.sum(dtype=torch.long))
        ledger["post_strike_physical_fall_count"].add_(post.sum(dtype=torch.long))
        ledger["non_physical_terminal_reset_count"].add_(
            (terminated & ~physical).sum(dtype=torch.long)
        )

    def _book_exact_ready_behavior_samples(self) -> None:
        """Accumulate ready-state balance sums with an explicit denominator per quantity.

        Legacy playback exposes readiness as every hold/recovery step.  Planner-revision playback
        intentionally sets all legacy holds to zero because its initial TTS owns preparation time;
        for that mode, readiness is sampled exactly once on the first metrics tick after each new
        ``(control_epoch, task_id)`` is installed.  This is deliberately named a task-entry sample
        in the ledger, but it is not an in-function instantaneous snapshot of installation.  The
        identity latch is instrumentation-only and never affects observations, rewards, resets,
        sampling, or the phase governor.
        """

        motion = self._motion()
        in_hold = getattr(motion, "in_hold", None)
        if in_hold is None:
            if not getattr(self, "planner_revision_enabled", False):
                return
            eligible = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        else:
            eligible = in_hold.detach().bool()

        planner_entry = torch.zeros_like(eligible)
        planner_legacy_hold_violation = torch.zeros_like(eligible)
        if getattr(self, "planner_revision_enabled", False):
            if not getattr(motion, "planner_revision_enabled", False):
                raise RuntimeError(
                    "planner ready instrumentation requires the planner-owned motion clock"
                )
            # MotionCommand's constructor already rejects every non-zero legacy hold clock in
            # planner mode.  Keep this device-side runtime witness in case a later state-machine
            # regression nevertheless raises ``in_hold``.  Such a hold must never enter ready
            # denominators: the planner deadline is the sole preparation clock.
            planner_legacy_hold_violation = eligible
            # Unit-test construction may bypass __init__; lazy allocation preserves the same
            # zero/zero sentinel used by the real constructor without weakening live validation.
            if not hasattr(self, "_exact_ready_sampled_control_epoch"):
                self._exact_ready_sampled_control_epoch = torch.zeros(
                    self.num_envs, dtype=torch.long, device=self.device
                )
                self._exact_ready_sampled_task_id = torch.zeros(
                    self.num_envs, dtype=torch.long, device=self.device
                )
            epoch = self._planner_control_epoch.detach()
            task = self._planner_task_id.detach()
            active = motion._planner_active.detach().bool()
            planner_entry = (
                active
                & (epoch > 0)
                & (task > 0)
                & (
                    (epoch != self._exact_ready_sampled_control_epoch)
                    | (task != self._exact_ready_sampled_task_id)
                )
            )
            eligible = planner_entry

        ledger = self._ensure_exact_behavior_decision_counters()
        ledger["ready_phase_sample_count"].add_(eligible.sum(dtype=torch.long))
        ledger["ready_planner_task_entry_sample_count"].add_(
            planner_entry.sum(dtype=torch.long)
        )
        ledger["ready_planner_legacy_hold_violation_count"].add_(
            planner_legacy_hold_violation.sum(dtype=torch.long)
        )
        data = self.robot.data
        tilt = torch.acos(self.metrics["base_upright"].detach().clamp(-1.0, 1.0))
        base_speed = torch.linalg.vector_norm(data.root_lin_vel_w[:, :2].detach(), dim=-1)
        station_offset = torch.linalg.vector_norm(
            (self.base_pos_w[:, :2] - self.base_target_pos_w).detach(), dim=-1
        )

        def book(count_key: str, sum_key: str, values: torch.Tensor) -> None:
            finite = eligible & torch.isfinite(values)
            ledger[count_key].add_(finite.sum(dtype=torch.long))
            ledger[sum_key].add_(
                torch.where(finite, values, torch.zeros_like(values)).sum(dtype=torch.float64)
            )
            ledger["ready_nonfinite_value_count"].add_(
                (eligible & ~torch.isfinite(values)).sum(dtype=torch.long)
            )

        book("ready_tilt_eligible_sample_count", "ready_tilt_rad_sum", tilt)
        book(
            "ready_base_speed_eligible_sample_count",
            "ready_base_speed_xy_mps_sum",
            base_speed,
        )
        book(
            "ready_station_offset_eligible_sample_count",
            "ready_station_offset_m_sum",
            station_offset,
        )

        sensor_ready = bool(
            getattr(self, "_foot_idx_robot", ())
            and getattr(self, "_foot_idx_contact", ())
            and getattr(self, "_contact_sensor", None) is not None
        )
        if sensor_ready:
            contact = self.metrics["foot_contact_frac"].detach()
            slip = self.metrics["foot_slip_speed"].detach()
            book(
                "ready_foot_contact_eligible_sample_count",
                "ready_foot_contact_fraction_sum",
                contact,
            )
            book(
                "ready_foot_slip_eligible_sample_count",
                "ready_foot_slip_speed_mps_sum",
                slip,
            )
        else:
            # A zero foot denominator is ambiguous unless the ledger says whether the phase was
            # absent or the contact sensor was unavailable.  Count the latter explicitly; never
            # fabricate zero contact/slip observations.
            ledger["ready_foot_sensor_unavailable_sample_count"].add_(
                eligible.sum(dtype=torch.long)
            )

        if getattr(self, "planner_revision_enabled", False):
            self._exact_ready_sampled_control_epoch[planner_entry] = (
                self._planner_control_epoch[planner_entry]
            )
            self._exact_ready_sampled_task_id[planner_entry] = (
                self._planner_task_id[planner_entry]
            )

    def consume_exact_behavior_decision_counters(self) -> dict[str, torch.Tensor]:
        """Consume one update's behavior and existing sparse-outcome ledgers as one transaction."""

        ledger = self._ensure_exact_behavior_decision_counters()
        snapshot = {name: value.detach().clone() for name, value in ledger.items()}
        with torch.inference_mode():
            for value in ledger.values():
                value.zero_()
        sparse = self.consume_sparse_reward_eligibility_counters()
        overlap = snapshot.keys() & sparse.keys()
        if overlap:
            raise RuntimeError(f"exact behavior ledger has duplicate sparse keys: {sorted(overlap)}")
        snapshot.update(sparse)
        return snapshot

    def _rally_legacy_values(self) -> tuple[float, dict]:
        """OLD mixed-ledger rally readout (transition-period *_legacy curves + unit tests).

        numerator _vb_inb_acc: books at exact-strike frames, decays only on strike-carrying steps;
        denominator _swing_starts_acc: books at swing starts, decays every step. Different decay
        schedules + booking phase lag => a synchronized reset queue drives the ratio through 1
        (0.31->1.48 oscillation, 2026-07-08 取证). Kept verbatim for new/old comparison only —
        judge with virtual_return_rate_rally (per-swing same-ledger counters, _rally_report).
        """
        enough_e = self._vb_exact_acc >= float(self.cfg.exact_success_min_count)
        _g = (self._vb_inb_acc / max(self._swing_starts_acc, 1e-6)) if enough_e else 0.0
        _per = {}
        for _c, _cn in self._clip_names.items():
            _per[_cn] = (
                (self._vb_inb_acc_c[_c] / max(self._swing_starts_acc_c[_c], 1e-6)) if enough_e else 0.0
            )
        return _g, _per

    def _decay_swing_accounting(self, decay: float) -> None:
        """Once-per-step decay of ALL swing-denominated ledgers (increments live elsewhere:
        _count_swing_starts for starts/falls/rally pairs, _resample_command for HER replay).
        The per-swing rally pair decays HERE — numerator and denominator on one schedule, which
        together with the paired booking in _count_swing_starts is the metric-sync fix."""
        self._swing_starts_acc = decay * self._swing_starts_acc
        self._prestrike_fall_acc = decay * self._prestrike_fall_acc
        self._poststrike_fall_acc = decay * self._poststrike_fall_acc
        self._resample_n_acc = decay * self._resample_n_acc
        self._replay_n_acc = decay * self._replay_n_acc
        self._rally_starts_acc = decay * self._rally_starts_acc
        self._rally_returns_acc = decay * self._rally_returns_acc
        self._drift_n_acc = decay * self._drift_n_acc
        self._drift_sum_acc = decay * self._drift_sum_acc
        self._drift_fwd_sum_acc = decay * self._drift_fwd_sum_acc
        self._station_offset_start_sum_acc = decay * self._station_offset_start_sum_acc
        self._heading_expiry_sum_acc = decay * self._heading_expiry_sum_acc
        self._heading_expiry_n_acc = decay * self._heading_expiry_n_acc
        self._recovery_spawn_sum_acc = decay * self._recovery_spawn_sum_acc
        self._recovery_expiry_sum_acc = decay * self._recovery_expiry_sum_acc
        self._recovery_n_acc = decay * self._recovery_n_acc
        for _c in self._clip_names:
            self._swing_starts_acc_c[_c] = decay * self._swing_starts_acc_c[_c]
            self._prestrike_fall_acc_c[_c] = decay * self._prestrike_fall_acc_c[_c]
            self._poststrike_fall_acc_c[_c] = decay * self._poststrike_fall_acc_c[_c]
            self._rally_starts_acc_c[_c] = decay * self._rally_starts_acc_c[_c]
            self._rally_returns_acc_c[_c] = decay * self._rally_returns_acc_c[_c]

    def _rally_report(self) -> None:
        """virtual_return_rate_rally* from the per-swing SAME-LEDGER counters (metric-sync fix).

        Start and returned-flag of every ended swing are booked TOGETHER (_count_swing_starts)
        and decayed TOGETHER (_decay_swing_accounting), so returns/starts is a decay-weighted
        average of per-swing 0/1 outcomes: <=1 by construction and equal to the true per-swing
        return rate under ANY reset synchronization. 人话:新算法=每拍一票,回了球记 1 没回记 0,
        比值就是真实上台率,永远不会超过 1。The legacy mixed-ledger curve stays available as
        *_legacy during the transition (see _vb_evaluate / _rally_legacy_values)."""
        _min_n = float(self.cfg.exact_success_min_count)
        _enough = self._rally_starts_acc >= _min_n
        self.metrics["virtual_return_rate_rally"][:] = (
            (self._rally_returns_acc / max(self._rally_starts_acc, 1e-6)) if _enough else 0.0
        )
        if getattr(self._motion(), "_multiseg", False):
            for _c, _cn in self._clip_names.items():
                _cs = self._rally_starts_acc_c[_c]
                self.metrics[f"virtual_return_rate_rally_{_cn}"][:] = (
                    (self._rally_returns_acc_c[_c] / max(_cs, 1e-6)) if _cs >= _min_n else 0.0
                )

    def _update_adaptive_sigma(self, enough: bool, denom: float) -> None:
        """P2.3 SMASH-style ADAPTIVE TRACKING SIGMA (coarse-to-fine) 的唯一落地处。

        every sigma_update_every steps, set the racket position/velocity reward stds to the
        clamped decayed MEAN exact-strike error, so the kernel always brackets the current
        operating band instead of a hand-tuned constant (SMASH Table IV: removing this collapses
        success 86.4 -> 22.6). Mutates the LIVE reward-term params in place (read per compute()
        call); also keeps racket_strike_success's own std_pos/std_vel in lockstep so the
        multiplicative bonus agrees with the additive terms.

        第三通道(adaptive_sigma_normal,默认关):SMASH 是 pos/ori/vel 三路一起收紧的;我们
        原先只收 pos/vel,拍面奖励的相对权重随 pos 收紧被静默弱化最多 ~7x。开启后
        racket_normal.std 与 racket_strike_success.std_normal 按 exact-strike 面角误差
        (弧度)的同一路衰减 EMA 锁步更新:clamp(sigma_ema_scale * mean_err_rad,
        sigma_normal_min, sigma_normal_max)。锁步的理由与 pos/vel 相同——加法项和乘法
        成功奖励必须在同一宽度上打分,否则两处梯度对不上。
        """
        if (
            self.cfg.adaptive_sigma
            and enough
            and self._env.common_step_counter % int(self.cfg.sigma_update_every) == 0
        ):
            pos_mean = self._exact_pos_err_sum / denom
            vel_mean = self._exact_vel_err_sum / denom
            sigma_pos = min(max(float(self.cfg.sigma_ema_scale) * pos_mean, float(self.cfg.sigma_pos_min)),
                            float(self.cfg.sigma_pos_max))
            sigma_vel = min(max(float(self.cfg.sigma_ema_scale) * vel_mean, float(self.cfg.sigma_vel_min)),
                            float(self.cfg.sigma_vel_max))
            _normal_on = bool(getattr(self.cfg, "adaptive_sigma_normal", False))
            if _normal_on:
                nrm_mean = self._exact_nrm_err_sum / denom
                sigma_normal = min(
                    max(float(self.cfg.sigma_ema_scale) * nrm_mean, float(self.cfg.sigma_normal_min)),
                    float(self.cfg.sigma_normal_max),
                )
            rm = self._env.reward_manager
            try:
                rm.get_term_cfg("racket_position").params["std"] = sigma_pos
                rm.get_term_cfg("racket_velocity").params["std"] = sigma_vel
                succ = rm.get_term_cfg("racket_strike_success").params
                succ["std_pos"] = sigma_pos
                succ["std_vel"] = sigma_vel
                if _normal_on:
                    # 锁步落地:两处必须同一个值(见 docstring),缺一处都算合同漂移。
                    rm.get_term_cfg("racket_normal").params["std"] = sigma_normal
                    succ["std_normal"] = sigma_normal
            except ValueError:
                pass  # a variant task without these terms: adaptive sigma is a no-op there
            self._adaptive_sigma_pos = sigma_pos
            self._adaptive_sigma_vel = sigma_vel
            if _normal_on:
                self._adaptive_sigma_normal = sigma_normal
        if self.cfg.adaptive_sigma:
            self.metrics["adaptive_sigma_pos"][:] = self._adaptive_sigma_pos
            self.metrics["adaptive_sigma_vel"][:] = self._adaptive_sigma_vel
            if getattr(self.cfg, "adaptive_sigma_normal", False):
                self.metrics["adaptive_sigma_normal"][:] = self._adaptive_sigma_normal

    def _update_metrics(self):
        # CommandTerm.compute() runs _update_metrics() BEFORE _update_command(), so refresh the
        # actual racket FK AND the strike timing here (once per step) — metrics, rewards, and
        # observations then all read the same fresh, phase-aligned buffers (rewards/obs read them
        # after the full command_manager.compute()). _compute_strike_timing must run before any
        # exact_strike / strike_window gating below (see its docstring: the old stale read measured
        # the strike one control frame too late).
        self._compute_racket_state()
        self._compute_strike_timing()
        origins = self._env.scene.env_origins
        # commanded base repositioning = distance of the base target from spawn (0 if coupling disabled).
        self.metrics["base_target_offset_norm"] = torch.norm(self.base_target_pos_w - origins[:, :2], dim=-1)
        pos_err = torch.norm(self.racket_pos_w - self.racket_target_pos_w, dim=-1)
        vel_err = torch.norm(self.racket_lin_vel_w - self.racket_target_vel_w, dim=-1)
        measured_normal, target_normal = face_tracking_pair(self)
        cos_ang = torch.sum(measured_normal * target_normal, dim=-1).clamp(-1.0, 1.0)
        # 弧度值先算(adaptive sigma 第三通道的驱动量纲),度数只是它的展示换算——数值逐字节同旧式。
        normal_err_rad = torch.acos(cos_ang)
        normal_err_deg = normal_err_rad * (180.0 / math.pi)
        base_err = torch.norm(self.base_pos_w[:, :2] - self.base_target_pos_w, dim=-1)
        base_pos_rel = self.base_pos_w[:, :2] - origins[:, :2]
        base_err_xy = self.base_pos_w[:, :2] - self.base_target_pos_w
        racket_pos_err_vec = self.racket_pos_w - self.racket_target_pos_w
        racket_vel_err_vec = self.racket_lin_vel_w - self.racket_target_vel_w

        # Episode-wide (instantaneous) errors.
        self.metrics["racket_pos_error"] = pos_err
        self.metrics["racket_vel_error"] = vel_err
        self.metrics["racket_normal_error_deg"] = normal_err_deg
        if self.cfg.face_command:
            # Backward-compatible explicit name for dashboards introduced with the face-frame fix.
            self.metrics["face_cmd_normal_error_deg"] = normal_err_deg
        self.metrics["base_pos_error"] = base_err
        self.metrics["time_to_strike_s"] = self.time_to_strike
        self.metrics["pre_strike_flag"] = self.pre_strike.float()
        self.metrics["strike_window_hit_rate"] = self.strike_window.float()
        if self.cfg.target_mode == "reference_perturbed":
            self.metrics["ref_perturb_scale"] = torch.full_like(pos_err, self._perturb_scale())
        else:
            self.metrics["ref_perturb_scale"].zero_()
        # Per-axis ERROR components only (which direction is the miss?). The per-axis actual/target
        # state and the speed/normal-cos scalars were dropped as redundant wandb clutter.
        for axis_idx, axis in enumerate(("x", "y")):
            self.metrics[f"base_pos_{axis}"] = base_pos_rel[:, axis_idx]
            self.metrics[f"base_pos_error_{axis}"] = base_err_xy[:, axis_idx]
        for axis_idx, axis in enumerate(("x", "y", "z")):
            self.metrics[f"racket_pos_error_{axis}"] = racket_pos_err_vec[:, axis_idx]
            self.metrics[f"racket_vel_error_{axis}"] = racket_vel_err_vec[:, axis_idx]

        # Strike-window-gated: hold the value sampled during the most recent strike window. The gating
        # masks come from the previous _update_command (<=1-step / 20 ms lag at 50 Hz — negligible vs
        # the ±strike_window_s window). Between strikes the held value carries to the next reset.
        in_win = self.strike_window
        exact_strike = torch.abs(self.time_to_strike) <= (0.5 * self._env.step_dt + 1e-6)
        motion = self._motion()
        if motion.retiming_active:
            # R14: float32 clock drift can (~1e-4/swing) land two consecutive tts values inside the
            # ±dt/2 window; latch so one-shot consumers (vb rewards, exact EMAs, achieved-buffer
            # writes) fire once per swing. The latch re-arms at every target resample.
            exact_strike = exact_strike & ~self._exact_fired
            self._exact_fired = self._exact_fired | exact_strike
        if getattr(motion, "event_timing_enabled", False):
            # Unavailable/infeasible rows remain on the scheduler denominator but must not grade a
            # stale previous target merely because its old clip happens to cross zero.  Before the
            # first origin, the normal exact clip strike is accepted and is the sole arming event.
            exact_strike = exact_strike & motion.event_exact_strike_allowed
            motion.record_event_exact_strike(torch.where(exact_strike)[0])
        self._advance_post_strike_elapsed(exact_strike)
        self._latch_exact_swing_completion(exact_strike)
        self.metrics["exact_strike_hit_rate"] = exact_strike.float()

        # --- DEBUG: swing-through sign verification (cfg.debug_reward_logging) -----------------------
        # err_minus = ||racket_pos - (target - vel*t_to_strike)||  (the CURRENT form used by the reward)
        # err_plus  = ||racket_pos - (target + vel*t_to_strike)||  (the FLIPPED form the user suspected)
        # Held over the strike window / exact strike so the reset-mean reports the in-window value. Expect
        # err_minus_win < err_plus_win (sign correct) and err_minus_exact ~= err_plus_exact (t~0 collapse).
        if self.cfg.debug_reward_logging:
            _ttf = self.time_to_strike.unsqueeze(-1)
            _tp_minus = self.racket_target_pos_w - self.racket_target_vel_w * _ttf
            _tp_plus = self.racket_target_pos_w + self.racket_target_vel_w * _ttf
            _err_minus = torch.norm(self.racket_pos_w - _tp_minus, dim=-1)
            _err_plus = torch.norm(self.racket_pos_w - _tp_plus, dim=-1)
            self.metrics["dbg_err_minus_win"] = torch.where(in_win, _err_minus, self.metrics["dbg_err_minus_win"])
            self.metrics["dbg_err_plus_win"] = torch.where(in_win, _err_plus, self.metrics["dbg_err_plus_win"])
            self.metrics["dbg_err_minus_exact"] = torch.where(
                exact_strike, _err_minus, self.metrics["dbg_err_minus_exact"]
            )
            self.metrics["dbg_err_plus_exact"] = torch.where(
                exact_strike, _err_plus, self.metrics["dbg_err_plus_exact"]
            )
        self.metrics["racket_pos_error_exact_strike"] = torch.where(
            exact_strike, pos_err, self.metrics["racket_pos_error_exact_strike"]
        )
        self.metrics["racket_vel_error_exact_strike"] = torch.where(
            exact_strike, vel_err, self.metrics["racket_vel_error_exact_strike"]
        )
        self.metrics["racket_normal_error_deg_exact_strike"] = torch.where(
            exact_strike, normal_err_deg, self.metrics["racket_normal_error_deg_exact_strike"]
        )
        # --- CONDITIONAL exact-strike success (the trustworthy, undiluted metric) -------------------
        # Old bug: strike_composite_success_exact was a per-env HELD value (last exact-strike result,
        # else init 0). CommandTerm.reset() logs mean(metric[env_ids]) over the RESETTING envs then zeros
        # them, so every env that reset without ever registering an exact-strike frame contributed 0 ->
        # the logged value was ~10x diluted vs the true conditional pass rate (raw probe ~0.32 logged
        # ~0.03), and the success-gated curriculum never advanced off ref_perturb_curriculum_start.
        # Fix: report the fraction of *exact-strike samples* that pass each threshold as a sample-weighted
        # EMA, broadcast to every env, so the reset-mean, the curriculum's .mean(), and the per-env value
        # all equal the conditional rate. pos/vel/normal are also logged separately.
        pass_pos = (pos_err < self.cfg.strike_success_pos_thresh) & exact_strike
        pass_vel = (vel_err < self.cfg.strike_success_vel_thresh) & exact_strike
        pass_normal = (normal_err_deg < self.cfg.strike_success_normal_thresh_deg) & exact_strike
        pass_comp = pass_pos & pass_vel & pass_normal
        decay = float(self.cfg.exact_success_decay)
        self._exact_n_acc = decay * self._exact_n_acc + float(exact_strike.sum())
        self._exact_pass_comp_acc = decay * self._exact_pass_comp_acc + float(pass_comp.sum())
        self._exact_pass_pos_acc = decay * self._exact_pass_pos_acc + float(pass_pos.sum())
        self._exact_pass_vel_acc = decay * self._exact_pass_vel_acc + float(pass_vel.sum())
        # 5/10 cm position buckets on the exact-strike sample (NOT the window-exit frame).
        _pass_5cm = (pos_err < 0.05) & exact_strike
        _pass_10cm = (pos_err < 0.10) & exact_strike
        self._exact_pass_5cm_acc = decay * self._exact_pass_5cm_acc + float(_pass_5cm.sum())
        self._exact_pass_10cm_acc = decay * self._exact_pass_10cm_acc + float(_pass_10cm.sum())
        self._exact_pass_normal_acc = decay * self._exact_pass_normal_acc + float(pass_normal.sum())
        # Reuse the existing exact sparse strike denominator for unconditional swing completion.
        # Virtual-ball evaluation below adds only its downstream outcomes, so a strike is booked
        # exactly once whether virtual rewards are enabled or not.
        _no_sparse_outcome = torch.zeros_like(exact_strike)
        self._book_sparse_reward_eligibility(
            exact_strike=exact_strike,
            capture=_no_sparse_outcome,
            net_clear=_no_sparse_outcome,
            landing_valid=_no_sparse_outcome,
            legal_return=_no_sparse_outcome,
        )
        # Tier-1 virtual-ball at-strike evaluation (one-shot buffers consumed by the virtual_*
        # reward terms after this compute()); no-op (and vb_fired stays False) when disabled.
        # vb_metrics_only runs the same evaluation purely for the virtual_*_rate curves (the
        # reward stack of such tasks has no virtual_* terms, so the caches go unread).
        if self.cfg.virtual_ball or self.cfg.vb_metrics_only:
            self._vb_evaluate(exact_strike, pos_err)
        # SHADOW physical ball (metrics-only): runs AFTER _vb_evaluate so this step's capture gate
        # (vb_fired) and the fresh per-env analytic landing prediction can be consumed/snapshotted.
        if self._shadow is not None:
            self._shadow.update(exact_strike)
        # PHYSICAL ball truth instrument (metrics-only): same seam/ordering as the shadow driver —
        # serve scheduling, exact-strike serve-accuracy measurement, park drive. Off = no-op branch.
        if self._physical is not None:
            self._physical.update(exact_strike)
        # GLOBAL error-magnitude EMAs (P2.3 adaptive sigma driver) — same decay/mask as the pass
        # counters above; per-clip variants exist further down but sigma needs one global signal.
        self._exact_pos_err_sum = decay * self._exact_pos_err_sum + float((pos_err * exact_strike).sum())
        self._exact_vel_err_sum = decay * self._exact_vel_err_sum + float((vel_err * exact_strike).sum())
        # 拍面通道:同一 decay/掩码,累加 exact-strike 面角误差(弧度)。见 adaptive_sigma_normal。
        self._exact_nrm_err_sum = decay * self._exact_nrm_err_sum + float((normal_err_rad * exact_strike).sum())
        # UNCONDITIONAL swing accounting: decay the start/fall accumulators at the SAME per-step
        # rate as the exact accumulators (increments happen in _count_swing_starts), then report
        #   swing_completion_rate = exact-strike arrivals / swing starts   (falls count against it)
        #   pre_strike_fall_rate  = pre-strike terminations / swing starts
        # These are the honest companions to the CONDITIONAL composite below, whose denominator
        # only contains exact-strike samples (pre-strike falls are invisible to it).
        self._decay_swing_accounting(decay)
        # Per-swing SAME-LEDGER rally rate (metric-sync fix): reported here, every step, right
        # after the paired counters decayed together. See _rally_report for the invariants.
        if self.cfg.virtual_ball or self.cfg.vb_metrics_only:
            self._rally_report()
        _s_denom = max(self._swing_starts_acc, 1e-6)
        _s_enough = self._swing_starts_acc >= float(self.cfg.exact_success_min_count)
        self.metrics["swing_completion_rate"][:] = min(self._exact_n_acc / _s_denom, 1.0) if _s_enough else 0.0
        self.metrics["pre_strike_fall_rate"][:] = min(self._prestrike_fall_acc / _s_denom, 1.0) if _s_enough else 0.0
        self.metrics["post_strike_fall_rate"][:] = (
            min(self._poststrike_fall_acc / _s_denom, 1.0) if _s_enough else 0.0
        )
        # Rally drift ratios: denominator = completed (wrapped) swings, NOT all starts.
        _d_denom = max(self._drift_n_acc, 1e-6)
        _d_enough = self._drift_n_acc >= float(self.cfg.exact_success_min_count)
        self.metrics["base_drift_per_swing"][:] = (self._drift_sum_acc / _d_denom) if _d_enough else 0.0
        self.metrics["base_drift_fwd_per_swing"][:] = (self._drift_fwd_sum_acc / _d_denom) if _d_enough else 0.0
        self.metrics["base_station_offset_at_swing_start"][:] = (
            (self._station_offset_start_sum_acc / _d_denom) if _d_enough else 0.0
        )
        heading_denom = max(self._heading_expiry_n_acc, 1e-6)
        heading_enough = self._heading_expiry_n_acc >= float(self.cfg.exact_success_min_count)
        self.metrics["base_heading_abs_at_swing_start"][:] = (
            self._heading_expiry_sum_acc / heading_denom if heading_enough else 0.0
        )
        self.metrics["base_heading_hold_expiry_count"][:] = self._heading_expiry_n_acc

        recovery_denom = max(self._recovery_n_acc, 1e-6)
        recovery_enough = self._recovery_n_acc >= float(self.cfg.exact_success_min_count)
        self.metrics["heading_recovery_spawn_yaw"][:] = (
            self._recovery_spawn_sum_acc / recovery_denom if recovery_enough else 0.0
        )
        self.metrics["heading_recovery_expiry_yaw"][:] = (
            self._recovery_expiry_sum_acc / recovery_denom if recovery_enough else 0.0
        )
        # Consumers must gate on this count before interpreting a zero error; zero samples is
        # "not measured", not perfect recovery.
        self.metrics["heading_recovery_count"][:] = self._recovery_n_acc
        # HER replay diagnostics: fraction of resampled targets drawn from the achieved buffer
        # (~achieved_target_mix_prob once the per-clip buffers pass achieved_min_fill).
        self.metrics["achieved_replay_frac"][:] = (
            self._replay_n_acc / max(self._resample_n_acc, 1e-6)
            if self._resample_n_acc >= float(self.cfg.exact_success_min_count)
            else 0.0
        )
        for _c, _cn in self._clip_names.items():
            _cd = max(self._swing_starts_acc_c[_c], 1e-6)
            _ce = self._swing_starts_acc_c[_c] >= float(self.cfg.exact_success_min_count)
            self.metrics[f"swing_completion_rate_{_cn}"][:] = (
                min(self._exact_n_acc_c[_c] / _cd, 1.0) if _ce else 0.0
            )
            # Fall attribution uses _prev_clip_id (the clip during the fall) while starts use the NEW
            # clip; with uniform clip resampling the denominators match in expectation.
            self.metrics[f"pre_strike_fall_rate_{_cn}"][:] = (
                min(self._prestrike_fall_acc_c[_c] / _cd, 1.0) if _ce else 0.0
            )
            self.metrics[f"post_strike_fall_rate_{_cn}"][:] = (
                min(self._poststrike_fall_acc_c[_c] / _cd, 1.0) if _ce else 0.0
            )

        # --- R-b envelope-violation accounting (cfg.track_envelope_violation; default OFF) --------
        # The envelope no longer terminates under envelope_as_penalty, so terminated-based fall
        # metrics go blind to it — count it here instead. Same z-only expressions as the removed
        # terminations (bad_anchor_pos_z_only / bad_motion_body_pos_z_only), same EMA
        # timescale/denominator as pre_strike_fall_rate: tracking_loss_rate = violation RISING
        # EDGES per swing start; envelope_violated_frac = per-step violation fraction (挂机 monitor).
        if self._envelope_track:
            _em = self._motion()
            if self._envelope_body_idx is None:
                _names = list(self.cfg.envelope_body_names)
                _missing = [n for n in _names if n not in _em.cfg.body_names]
                if _missing:
                    raise ValueError(
                        f"RacketTargetCommand.track_envelope_violation: envelope body name(s) "
                        f"{_missing} not in motion.cfg.body_names {_em.cfg.body_names} — the "
                        "tracking_loss accounting would silently watch the wrong bodies.")
                self._envelope_body_idx = [i for i, n in enumerate(_em.cfg.body_names) if n in _names]
            _eth = float(self.cfg.envelope_threshold)
            _anchor_viol = (_em.anchor_pos_w[:, -1] - _em.robot_anchor_pos_w[:, -1]).abs() > _eth
            _bidx = self._envelope_body_idx
            _body_viol = torch.any(
                (_em.body_pos_relative_w[:, _bidx, -1] - _em.robot_body_pos_w[:, _bidx, -1]).abs() > _eth,
                dim=-1,
            )
            _viol = (_anchor_viol | _body_viol) & ~_em.in_hold
            _rising = _viol & ~self._prev_envelope_viol
            self._prev_envelope_viol = _viol
            self._tracking_loss_acc = decay * self._tracking_loss_acc + float(_rising.sum())
            self.metrics["envelope_violated_frac"] = _viol.float()
            self.metrics["tracking_loss_rate"][:] = (
                min(self._tracking_loss_acc / _s_denom, 1.0) if _s_enough else 0.0
            )
            if getattr(_em, "_multiseg", False):
                _rfam = self._metric_bucket_rows()[_em.clip_id]
                for _c, _cn in self._clip_names.items():
                    self._tracking_loss_acc_c[_c] = decay * self._tracking_loss_acc_c[_c] + float(
                        (_rising & (_rfam == _c)).sum()
                    )
                    _cd = max(self._swing_starts_acc_c[_c], 1e-6)
                    _ce = self._swing_starts_acc_c[_c] >= float(self.cfg.exact_success_min_count)
                    self.metrics[f"tracking_loss_rate_{_cn}"][:] = (
                        min(self._tracking_loss_acc_c[_c] / _cd, 1.0) if _ce else 0.0
                    )

        enough = self._exact_n_acc >= float(self.cfg.exact_success_min_count)
        denom = max(self._exact_n_acc, 1e-6)
        self._exact_composite_rate = (self._exact_pass_comp_acc / denom) if enough else 0.0
        # Broadcast in place so the entries reset() zeros are refreshed before the next reset logs them.
        self.metrics["strike_composite_success_exact"][:] = self._exact_composite_rate
        self.metrics["strike_pos_pass_exact"][:] = (self._exact_pass_pos_acc / denom) if enough else 0.0
        self.metrics["strike_vel_pass_exact"][:] = (self._exact_pass_vel_acc / denom) if enough else 0.0
        self.metrics["strike_normal_pass_exact"][:] = (self._exact_pass_normal_acc / denom) if enough else 0.0
        # Exact-strike position accuracy buckets (comparable with composite: same mask + EMA denominator).
        self.metrics["exact_strike_pos_success_5cm"][:] = (self._exact_pass_5cm_acc / denom) if enough else 0.0
        self.metrics["exact_strike_pos_success_10cm"][:] = (self._exact_pass_10cm_acc / denom) if enough else 0.0
        # Distribution of position error over THIS step's exact-strike samples (p90 + mean), broadcast.
        _ex_errs = pos_err[exact_strike]
        if _ex_errs.numel() > 0:
            self.metrics["exact_strike_pos_err_mean"][:] = _ex_errs.mean()
            self.metrics["exact_strike_pos_err_p90"][:] = torch.quantile(_ex_errs, 0.90)
        self.metrics["exact_strike_sample_count_decayed"][:] = self._exact_n_acc
        # --- per-clip (forehand/backhand) breakdown of the exact-strike pass rates + errors -----------
        # Same sample-weighted EMA as the global block above, selected by the motion command's clip_id so
        # wandb shows each swing separately. pass_pos/vel/normal already include `& exact_strike`. Multiseg
        # (unified forehand+backhand) only; single-clip leaves these at 0.
        _motion = self._motion()
        if getattr(_motion, "_multiseg", False):
            # 按族分桶(legacy 2-clip 族行号==clip_id 逐字节不变;6-clip 下"forehand"桶=正手
            # 三档合计,"backhand"桶=反手三档合计,不再把正手 1.0 档记成反手)。
            _fam = self._metric_bucket_rows()[_motion.clip_id]
            for _c, _cn in self._clip_names.items():
                _sel = exact_strike & (_fam == _c)
                _self_f = _sel.float()
                self._exact_n_acc_c[_c] = decay * self._exact_n_acc_c[_c] + float(_sel.sum())
                self._exact_pass_pos_acc_c[_c] = decay * self._exact_pass_pos_acc_c[_c] + float((pass_pos & _sel).sum())
                self._exact_pass_vel_acc_c[_c] = decay * self._exact_pass_vel_acc_c[_c] + float((pass_vel & _sel).sum())
                self._exact_pass_normal_acc_c[_c] = decay * self._exact_pass_normal_acc_c[_c] + float((pass_normal & _sel).sum())
                self._exact_pass_comp_acc_c[_c] = decay * self._exact_pass_comp_acc_c[_c] + float((pass_comp & _sel).sum())
                self._exact_pos_err_sum_c[_c] = decay * self._exact_pos_err_sum_c[_c] + float((pos_err * _self_f).sum())
                self._exact_vel_err_sum_c[_c] = decay * self._exact_vel_err_sum_c[_c] + float((vel_err * _self_f).sum())
                self._exact_nrm_err_sum_c[_c] = decay * self._exact_nrm_err_sum_c[_c] + float((normal_err_deg * _self_f).sum())
                _n = self._exact_n_acc_c[_c]
                # rate = acc / n once enough decayed samples accumulated (else 0). errors = decayed mean
                # error over THIS clip's exact-strike samples. _scale folds in the "enough" gate.
                _scale = (1.0 / max(_n, 1e-6)) if _n >= float(self.cfg.exact_success_min_count) else 0.0
                self.metrics[f"strike_pos_pass_exact_{_cn}"][:] = self._exact_pass_pos_acc_c[_c] * _scale
                self.metrics[f"strike_vel_pass_exact_{_cn}"][:] = self._exact_pass_vel_acc_c[_c] * _scale
                self.metrics[f"strike_normal_pass_exact_{_cn}"][:] = self._exact_pass_normal_acc_c[_c] * _scale
                self.metrics[f"strike_composite_success_exact_{_cn}"][:] = self._exact_pass_comp_acc_c[_c] * _scale
                self.metrics[f"racket_pos_error_exact_strike_{_cn}"][:] = self._exact_pos_err_sum_c[_c] * _scale
                self.metrics[f"racket_vel_error_exact_strike_{_cn}"][:] = self._exact_vel_err_sum_c[_c] * _scale
                self.metrics[f"racket_normal_error_deg_exact_strike_{_cn}"][:] = self._exact_nrm_err_sum_c[_c] * _scale
            # --- HER achieved-target buffer WRITE ---------------------------------------------------
            # Record the racket state the policy ACTUALLY produced at this step's exact-strike frames
            # (pos env-origin-relative, vel world). Alive envs only by construction: terminated envs
            # were reset before the command computes, so their state never lands here. Gated on the mix
            # prob so the buffers cost nothing when replay is off.
            if self.cfg.achieved_target_mix_prob > 0.0:
                for _c in self._clip_names:
                    _bidx = torch.where(exact_strike & (_fam == _c))[0]
                    _m = int(_bidx.numel())
                    if _m == 0:
                        continue
                    _size = self._ach_pos[_c].shape[0]
                    _rows = (self._ach_ptr[_c] + torch.arange(_m, device=self.device)) % _size
                    self._ach_pos[_c][_rows] = self.racket_pos_w[_bidx] - origins[_bidx]
                    self._ach_vel[_c][_rows] = self.racket_lin_vel_w[_bidx]
                    self._ach_spd[_c][_rows] = _motion.speed_scale[_bidx]
                    self._ach_ptr[_c] = int((self._ach_ptr[_c] + _m) % _size)
                    self._ach_fill[_c] = min(self._ach_fill[_c] + _m, _size)
            for _c, _cn in self._clip_names.items():
                self.metrics[f"achieved_buffer_fill_{_cn}"][:] = float(self._ach_fill[_c])
        # Per-axis position error AT the exact strike frame (which axis is the miss?). The position-only
        # strike_success_exact was dropped — strike_pos_pass_exact above is the same signal, undiluted.
        _axis_err_exact = torch.abs(self.racket_pos_w - self.racket_target_pos_w)
        for _ai, _ax in enumerate(("x", "y", "z")):
            self.metrics[f"racket_pos_error_{_ax}_exact_strike"] = torch.where(
                exact_strike, _axis_err_exact[:, _ai], self.metrics[f"racket_pos_error_{_ax}_exact_strike"]
            )
        # P2.3 SMASH-style ADAPTIVE TRACKING SIGMA — 摘到 _update_adaptive_sigma(可 host 单测
        # 直接驱动),此处仅按原节拍调用;行为与旧内联块逐字节一致。
        self._update_adaptive_sigma(enough, denom)
        # A1 target latency diagnostic: constant broadcast, refreshed every step because
        # CommandTerm.reset() zeros metric entries of resetting envs before logging them.
        # (midswing_resample_count is written per step in _update_command while the feature is on.)
        self.metrics["target_delay_steps_in_effect"][:] = float(self._delay_steps)

        # Success-gated curriculum: widen the perturbation only once the smoothed CONDITIONAL exact-strike
        # composite success (fraction of exact-strike samples passing all three thresholds) clears the bar.
        if self.cfg.target_mode == "reference_perturbed" and self.cfg.ref_perturb_success_gated:
            if (
                self._curr_perturb_scale < 1.0
                and enough
                and self._exact_composite_rate > self.cfg.ref_perturb_advance_threshold
            ):
                self._curr_perturb_scale = min(
                    1.0, self._curr_perturb_scale + float(self.cfg.ref_perturb_advance_rate)
                )
        self.metrics["racket_pos_error_at_strike"] = torch.where(
            in_win, pos_err, self.metrics["racket_pos_error_at_strike"]
        )
        self.metrics["racket_vel_error_at_strike"] = torch.where(
            in_win, vel_err, self.metrics["racket_vel_error_at_strike"]
        )
        self.metrics["racket_normal_error_deg_at_strike"] = torch.where(
            in_win, normal_err_deg, self.metrics["racket_normal_error_deg_at_strike"]
        )
        self.metrics["strike_success"] = torch.where(
            in_win, (pos_err < self.cfg.strike_success_pos_thresh).float(), self.metrics["strike_success"]
        )
        # Base target is tracked before the strike, so log that error during the pre-strike phase.
        self.metrics["base_pos_error_pre_strike"] = torch.where(
            self.pre_strike, base_err, self.metrics["base_pos_error_pre_strike"]
        )

        # Swing-quality detail held at the most recent strike: actual/target paddle speed and the
        # per-axis position error (which direction is the miss?).
        racket_speed = torch.norm(self.racket_lin_vel_w, dim=-1)
        target_speed = torch.norm(self.racket_target_vel_w, dim=-1)
        axis_err = torch.abs(self.racket_pos_w - self.racket_target_pos_w)
        self.metrics["racket_speed_at_strike"] = torch.where(
            in_win, racket_speed, self.metrics["racket_speed_at_strike"]
        )
        self.metrics["racket_target_speed_at_strike"] = torch.where(
            in_win, target_speed, self.metrics["racket_target_speed_at_strike"]
        )
        self.metrics["racket_pos_error_x_at_strike"] = torch.where(
            in_win, axis_err[:, 0], self.metrics["racket_pos_error_x_at_strike"]
        )
        self.metrics["racket_pos_error_y_at_strike"] = torch.where(
            in_win, axis_err[:, 1], self.metrics["racket_pos_error_y_at_strike"]
        )
        self.metrics["racket_pos_error_z_at_strike"] = torch.where(
            in_win, axis_err[:, 2], self.metrics["racket_pos_error_z_at_strike"]
        )
        # Window-exit-held (kept for continuity; the trustworthy contact-frame version is
        # exact_strike_pos_success_5cm/10cm, computed on the exact-strike mask above).
        self.metrics["strike_success_5cm_window_exit"] = torch.where(
            in_win, (pos_err < 0.05).float(), self.metrics["strike_success_5cm_window_exit"]
        )
        self.metrics["strike_success_10cm_window_exit"] = torch.where(
            in_win, (pos_err < 0.10).float(), self.metrics["strike_success_10cm_window_exit"]
        )

        # Robot-health diagnostics (episode-wide, instantaneous).
        data = self.robot.data
        self.metrics["base_height"] = data.root_pos_w[:, 2]
        self.metrics["base_upright"] = matrix_from_quat(self.base_quat_w)[:, 2, 2]  # 1.0 = perfectly upright
        self.metrics["joint_vel_abs_max"] = torch.max(torch.abs(data.joint_vel), dim=-1).values
        # --- stability diagnostics: absolute base roll/pitch + foot contact/slip --------------------
        # Resolve foot body indices + the contact sensor once (robust to USD Link/link casing; degrades
        # gracefully to 0 if absent so it can never crash training).
        if not getattr(self, "_stab_resolved", False):
            self._stab_resolved = True
            self._foot_idx_robot, self._foot_idx_contact, self._contact_sensor = [], [], None
            try:
                self._foot_idx_robot = list(self.robot.find_bodies([".*ankle_roll.*"])[0])
            except Exception:
                pass
            try:
                cs = self._env.scene.sensors["contact_forces"]
                self._contact_sensor = cs
                self._foot_idx_contact = list(cs.find_bodies([".*ankle_roll.*"])[0])
            except Exception:
                pass
        _roll, _pitch, _ = euler_xyz_from_quat(self.base_quat_w)
        # wrap to (-180, 180] so a level base reads ~0 (euler_xyz_from_quat can return [0, 2pi))
        self.metrics["base_roll_deg"] = torch.rad2deg(torch.atan2(torch.sin(_roll), torch.cos(_roll)))
        self.metrics["base_pitch_deg"] = torch.rad2deg(torch.atan2(torch.sin(_pitch), torch.cos(_pitch)))
        if self._foot_idx_robot and self._contact_sensor is not None and self._foot_idx_contact:
            f_force = torch.norm(self._contact_sensor.data.net_forces_w[:, self._foot_idx_contact, :], dim=-1)  # (E,2)
            in_contact = (f_force > 10.0).float()  # 10 N = the sensor's force_threshold
            self.metrics["foot_contact_frac"] = in_contact.mean(dim=-1)  # 1.0 = both feet planted
            self.feet_contact_frac = self.metrics["foot_contact_frac"]  # clean attr for feet_contact reward
            f_speed = torch.norm(data.body_lin_vel_w[:, self._foot_idx_robot, :2], dim=-1)  # horizontal (E,2)
            _slip_sum = (f_speed * in_contact).sum(dim=-1)  # sum over feet of horizontal speed while in contact
            # metric = MEAN slip over the contacting feet (m/s; planted foot should be ~0).
            self.metrics["foot_slip_speed"] = _slip_sum / in_contact.sum(dim=-1).clamp(min=1.0)
            # reward signal = SUM over feet (so both slipping feet are penalized); pre_strike-gated in the reward.
            self.foot_slip_in_contact = _slip_sum
        # footwork-to-strike signals (racket progress, foot slip²/vel/drag, arm overreach, strike stability)
        self._update_footwork_signals(pos_err)
        if self._has_jpos_limits:
            limits = getattr(data, "soft_joint_pos_limits", None)
            if limits is None:
                limits = data.joint_pos_limits
            half_span = ((limits[..., 1] - limits[..., 0]) * 0.5).clamp(min=1e-6)
            dist = torch.minimum(data.joint_pos - limits[..., 0], limits[..., 1] - data.joint_pos).clamp(min=0.0)
            self.metrics["joint_pos_near_limit_frac"] = ((dist / half_span) < 0.1).float().mean(dim=-1)
        if self._has_torque:
            tau_abs = torch.abs(data.applied_torque)
            self.metrics["joint_torque_abs_mean"] = torch.mean(tau_abs, dim=-1)
            self.metrics["joint_torque_abs_max"] = torch.max(tau_abs, dim=-1).values
        act = getattr(self._env.action_manager, "action", None)
        if act is not None:
            a_abs = torch.abs(act)
            self.metrics["action_abs_mean"] = torch.mean(a_abs, dim=-1)
            self.metrics["action_abs_max"] = torch.max(a_abs, dim=-1).values
            prev_act = getattr(self._env.action_manager, "prev_action", None)
            if prev_act is not None:
                delta_abs = torch.abs(act - prev_act)
                self.metrics["action_delta_abs_mean"] = torch.mean(delta_abs, dim=-1)
                self.metrics["action_delta_abs_max"] = torch.max(delta_abs, dim=-1).values
            else:
                self.metrics["action_delta_abs_mean"].zero_()
                self.metrics["action_delta_abs_max"].zero_()

        # Instrumentation only: hold/recovery phase sums and denominators for this simulator step.
        # These tensors are never read by observations, rewards, resets, curricula or sampling.
        self._book_exact_ready_behavior_samples()

    # ------------------------------------------------------------------ #
    # Observation helpers (base-relative quantities)
    # ------------------------------------------------------------------ #
    def racket_target_pos_b(self) -> torch.Tensor:
        """Desired racket position relative to the base (yaw-heading frame). HITTER actor obs.

        PRIVILEGED: uses ``base_pos_w`` (world base position). Mocap streams the base pose at 300 Hz
        during play, but that link is not bridged into the deploy front-end, so this term is fabricated
        at deploy; base-position-freedom is a deliberate robustness choice. Used by the `full` obs mode;
        the deploy-parity mode (legacy task name: `real_sensor_only`) replaces it with
        :meth:`racket_target_pos_b_rel`.
        """
        return quat_rotate_inverse(yaw_quat(self.base_quat_w), self.racket_target_pos_w - self.base_pos_w)

    def racket_target_pos_b_rel(self) -> torch.Tensor:
        """Desired racket position relative to the CURRENT racket (FK), in the yaw-heading frame.

        DEPLOY-HONEST (no world base position): expanding the rotation, the base position cancels::

            R_yaw^T (target_w - racket_w) = R_yaw^T(target_w - base_w) - R_yaw^T(racket_w - base_w)
                                          = (target in base frame) - (racket FK in base frame)

        Both terms are computable on the real robot from the planner's target + racket forward
        kinematics (joint encoders), WITHOUT a fabricated base pose. Replaces
        :meth:`racket_target_pos_b` in the deploy-parity observation mode (legacy task name:
        ``real_sensor_only``).

        A1: reads the ACTOR-visible target view (delayed/jittered when the A1 knobs are on;
        the live tensor itself otherwise — byte-identical default). This method backs the
        deploy-parity ACTOR obs only; the critic's :meth:`racket_target_pos_b` stays live.
        """
        return quat_rotate_inverse(yaw_quat(self.base_quat_w), self.actor_racket_target_pos_w() - self.racket_pos_w)

    # --- A1 ACTOR-visible target accessors (delayed/jittered view; live aliases when off) ------- #
    def actor_racket_target_pos_w(self) -> torch.Tensor:
        """ACTOR-visible desired racket position (world): the A1 delayed/jittered view when target
        latency/jitter is enabled, else the live tensor itself (zero-overhead alias). Rewards,
        metrics, and the privileged critic keep reading the TRUE live ``racket_target_pos_w``."""
        return self.delayed_racket_target_pos_w

    def actor_racket_target_vel_w(self) -> torch.Tensor:
        """ACTOR-visible desired racket velocity (world). See :meth:`actor_racket_target_pos_w`."""
        return self.delayed_racket_target_vel_w

    def actor_swing_sign(self) -> torch.Tensor:
        """ACTOR-visible swing sign (forehand +1 / backhand -1), delayed with the target when A1
        latency is on (the swing-type flag rides the same planner->runner message as the target)."""
        return self.delayed_swing_sign

    def actor_target_normal_cmd(self) -> torch.Tensor:
        """Actor-visible demanded face normal from the same atomic A1 message as pos/vel/sign."""
        return self.delayed_target_normal_cmd

    def actor_time_to_strike(self) -> torch.Tensor:
        """Actor-visible TTS from the configured planner-tuple timing convention.

        The default ``live`` branch returns the live tensor itself (not a copy), preserving the
        historical observation path.  Atomic delayed modes return the materialized ring output;
        rewards, exact-strike gates, physics and the privileged critic never call this accessor.
        """
        if self._delay_tts_mode == "live" and not self.planner_revision_enabled:
            return self.time_to_strike
        return self.delayed_time_to_strike

    def _target_xy_err_b(self, target_xy_w: torch.Tensor) -> torch.Tensor:
        """(world XY target − current base XY) rotated into the yaw-heading base frame — the shared
        math behind both station-style obs terms (Hitter base_target_pos_b / R10c station anchor)."""
        delta_xy = target_xy_w - self.base_pos_w[:, :2]
        delta = torch.cat([delta_xy, torch.zeros(self.num_envs, 1, device=self.device)], dim=-1)
        return quat_rotate_inverse(yaw_quat(self.base_quat_w), delta)[:, :2]

    def base_target_pos_b(self) -> torch.Tensor:
        """Desired base XY position relative to the current base (yaw-heading frame). HITTER actor obs."""
        return self._target_xy_err_b(self.base_target_pos_w)

    def station_anchor_err_b(self) -> torch.Tensor:
        """R10c 站位锚误差(2 维,actor obs,station_obs 旗标):世界系常数锚点 − 当前 base XY,
        旋进 yaw-heading base 系。与 Hitter 的 base_target_pos_b 同一套数学(见 _target_xy_err_b),
        区别只在目标:这里是 reset 常数(env origin + 偏移 = 出生点),不是采样出来的奖励站位。
        部署侧同 Hitter:mocap base 位置可算相对 Δ,掉 mocap 喂 Δ=0 优雅退化成"没漂"。"""
        return self._target_xy_err_b(self.station_anchor_pos_w)

    # --- HITTER Table-I exact accessors (hitter_pure contract, 2026-07-07) ----------------------- #
    # The paper expresses target vectors in the WORLD frame and gives the actor the base forward
    # vector e_base,x separately (instead of pre-rotating into the heading frame). Deploy sources:
    # position differences = planner target − mocap base position (both in the mocap/table world
    # frame, no rotation needed); e_base,x = IMU orientation after the runner's yaw-align-at-engage.
    def base_forward_xy(self) -> torch.Tensor:
        """Base forward unit vector e_base,x, world-frame xy (HITTER Table I)."""
        fwd = quat_apply(
            self.base_quat_w,
            torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3),
        )[:, :2]
        return fwd / (torch.norm(fwd, dim=-1, keepdim=True) + 1e-6)

    def base_target_delta_xy_w(self) -> torch.Tensor:
        """Target base position p̂_base,xy − p_base,xy, WORLD frame (HITTER Table I)."""
        return self.base_target_pos_w - self.base_pos_w[:, :2]

    def racket_target_rel_base_w(self) -> torch.Tensor:
        """Target racket position relative to the base, WORLD frame (HITTER Table I / §V-B-1:
        "the racket position relative to the base ... expressed in the world frame").
        A1: reads the ACTOR-visible target view (delayed/jittered when the knobs are on)."""
        return self.actor_racket_target_pos_w() - self.base_pos_w

    # ------------------------------------------------------------------ #
    # Debug visualization (no-op stubs; targets are world-frame buffers).
    # ------------------------------------------------------------------ #
    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


@configclass
class RacketTargetCommandCfg(CommandTermCfg):
    """Configuration for :class:`RacketTargetCommand`."""

    class_type: type = RacketTargetCommand

    asset_name: str = MISSING
    motion_command_name: str = "motion"

    # The target is re-sampled per swing (on clip wrap / reset), not on a fixed time schedule,
    # so disable the base CommandTerm time-based resampling.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    # --- racket mount FK ---
    racket_body_name: str = "pingpang_red_Link"
    wrist_body_name: str = "right_wrist_yaw_Link"
    # Exact official pingpang_red_joint / MJCF right_racket-site transform.
    # The old value came from pingbang_ball_joint and was 1.49 um away; tiny in
    # magnitude, but it prevented a literal single-point Isaac/MuJoCo/C++ contract.
    mount_offset: tuple[float, float, float] = (0.21021, 0.032078, 0.032036)
    # Fixed wrist->racket rotation (w, x, y, z); only used in the wrist_offset FK fallback. Identity
    # for the A3 ping-pong URDF (all mount joints are rpy=0). Set non-identity if the mount tilts the
    # paddle relative to the wrist frame.
    mount_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    mount_normal_axis: int = 1  # racket-local +Y is the face normal (red/hitting face; confirmed in Step 11)
    mount_normal_sign: float = 1.0  # +1 = red/forehand face; -1 = black/backhand face
    # 每 clip 一个击球面符号(franco 2026-07-09 拍板:"哪面拍子超前就是哪面"——正反手各用固定的一面)。
    # 病根(M3b 判死取证,= jiayi origin/hitter b7b7dfc 同病):统一正反手策略里两个挥拍用的是拍子
    # 相反的两面(正手=红面/+Y,反手=黑面/−Y),单一标量符号让反手的拍面目标(normal_mode="velocity")
    # 永不可达——+Y 面在反手挥拍里永远不超前,拍面误差被钉在 ~115-137°,综合成功清零(pos/vel 其实
    # 都过),考卷 CF 换拍面=1.000 就是这个签名。设 (正手符号, 反手符号) 例 (1.0, -1.0),顺序 =
    # motion_file 的 clip 顺序(同 strike_phase_per_clip)。空 () = 全部 clip 用标量 mount_normal_sign
    # (默认,现役行为逐位不变)。表长和加载 clip 数不一致、或出现 ±1 以外的值,当场报错(fail-loud,
    # 见 _mount_signs_cfg)。每个 clip 的建议符号用 scripts/suggest_face_sign.py 离线算(触球帧 n·v)。
    mount_normal_sign_per_clip: tuple = ()

    # --- strike timing (fraction of the reference clip where the paddle meets the ball) ---
    strike_phase: float = 0.46  # HITTER clip: strike at frame 43/94 ≈ 0.46
    # Unified multi-clip (HITTER forehand+backhand single policy): per-clip strike phase, aligned with the
    # MotionLoader segment order (i.e. the order of motion files: forehand, backhand). Empty -> use the
    # scalar strike_phase for every clip. e.g. (0.36, 0.74) for forehand_new + backhand_new.
    strike_phase_per_clip: tuple = ()
    strike_window_s: float = 0.1  # half-window; goal-racket reward active within ±strike_window_s
    # 1c SPLIT strike windows (reward_staged_design 2026-07-08 §② C1; defaults None = both fall
    # back to strike_window_s, byte-identical single-window behavior). When set:
    #   strike_window_pos_s  — TIGHT half-window for racket_position (+ the position factor of
    #                          racket_strike_success): contact must be precise (SMASH 0.02 s).
    #   strike_window_wide_s — WIDE half-window for racket_normal / racket_velocity (SMASH ±0.1 s:
    #                          face+velocity get slack, damping wrist-accel spikes / sim2real gap).
    # The legacy strike_window (strike_window_s) keeps gating the strike-stability penalties and
    # the strike_window_hit_rate metric. 人话:触点要准(紧窗),挥向挥速给余量(宽窗)。
    strike_window_pos_s: float | None = None
    strike_window_wide_s: float | None = None
    strike_success_pos_thresh: float = 0.075  # m; "strike_success" metric = fraction of strikes with racket pos error below this
    strike_success_vel_thresh: float = 0.5  # m/s; exact-strike racket velocity acceptance threshold
    strike_success_normal_thresh_deg: float = 15.0  # deg; exact-strike face-normal acceptance threshold

    # --- nominal stance (offset of the base from the env origin) ---
    base_nominal_offset: tuple[float, float, float] = (0.0, 0.0, 0.93)

    # --- target generation mode ---
    # "uniform": independent box sampling from the *_range fields below (legacy; the boxes are
    #   PLACEHOLDERS not tied to the swing, so the imitated swing's racket may never pass through them).
    # "reference_perturbed": target = the reference swing's racket state AT the strike frame (pos/vel/
    #   normal, computed by the same FK as the actual racket) + a curriculum-scaled uniform perturbation.
    #   Reachable by construction (a perfect imitator scores exactly); the *_range fields are ignored.
    # "hitter_pure": HITTER-faithful (arXiv:2508.21043 §V-B-1 + §IV-C, 2026-07-07): base station sampled
    #   INDEPENDENTLY from base_target_*_range (a STATION BOX, not jitter); racket target on a striking
    #   plane FIXED RELATIVE to the commanded station (racket_pos_range_per_clip = STATION-RELATIVE x/y
    #   offsets, z absolute); face-normal target from normal_mode ("velocity" = paper impact model) —
    #   NEVER the reference-clip normal. No HER, no reference_reach coupling, no curriculum.
    target_mode: str = "reference_perturbed"

    # reference_perturbed perturbation (final half-extents; scaled 0->1 by the curriculum below).
    ref_perturb_pos: tuple[float, float, float] = (0.15, 0.20, 0.15)  # m, per-axis half-range
    ref_perturb_vel: tuple[float, float, float] = (1.0, 1.0, 0.8)  # m/s, per-axis half-range
    ref_perturb_normal: float = 0.30  # face-normal jitter magnitude (added then renormalized)
    # Curriculum: perturbation half-extents ramp from `start`*final to 1.0*final.
    # Success-gated mode (default): `_curr_perturb_scale` starts at `ref_perturb_curriculum_start` and
    # only advances (by `ref_perturb_advance_rate` per control step) once the smoothed exact-strike
    # composite success exceeds `ref_perturb_advance_threshold` — keeps the strike error inside the
    # racket reward kernel's responsive band until the policy demonstrably hits, then widens.
    # Open-loop fallback (success_gated=False): ramp over `ref_perturb_curriculum_steps` control steps
    # (env.common_step_counter); set steps<=0 to disable the ramp (always full).
    ref_perturb_curriculum_steps: int = 30000
    ref_perturb_curriculum_start: float = 0.05
    ref_perturb_success_gated: bool = True
    ref_perturb_advance_threshold: float = 0.30  # widen once smoothed exact-strike composite success > this
    ref_perturb_advance_rate: float = 1.0e-5  # scale increment per control step while above threshold

    # --- reference velocity scaling (stage slow -> fast hitting) ---
    # The sampled racket-velocity target is `ref_vel_scale * reference_vel + perturbation`. The reference
    # forehand clip strikes at ~6 m/s; set <1.0 to train a slower, more controllable hit FIRST (e.g. 0.6
    # -> ~3.6 m/s) and ramp back to 1.0 once the slow strike is accurate. NOTE: at scale!=1.0 the target
    # velocity no longer equals the imitated swing's velocity, so a perfect imitator no longer matches it
    # exactly (the "reachable by construction" guarantee holds for position/normal, not scaled velocity).
    ref_vel_scale: float = 1.0

    # --- clean reference strike velocity (denoise the target velocity) ---
    # The motion's stored body_lin_vel_w is a finite-difference (torch.gradient) of the 30->50 fps
    # interpolated joint trajectory propagated through FK (see scripts/csv_to_npz.py). At the fast,
    # high-jerk racket tip those FD/interpolation errors accumulate to ~1 m/s and are INCONSISTENT with
    # the position trajectory (stored-vel vs central-diff-of-pos differ ~1.1 m/s near the strike). Since
    # the racket-velocity reward target is essentially the reference velocity, that ~1 m/s noise is the
    # floor on racket_vel_error_exact_strike (velEx parked ~0.74 regardless of reward tuning).
    # When True, the cached strike target velocity is recomputed from the FINAL racket FK position
    # (body_pos_w, the same FK as the actual racket) by a centered finite difference over +-window frames,
    # which is consistent with the position the policy actually tracks and rejects single-frame jitter.
    # The legacy False path mixed a COM velocity with a link/site position and
    # is now rejected at runtime; the field remains only to fail old configs
    # loudly instead of silently changing their meaning.
    clean_reference_strike_velocity: bool = True
    clean_strike_vel_window: int = 2  # half-window (frames) for the centered finite difference (try 2 or 3)

    # --- debug logging (sign verification + raw/gated reward kernels) ---
    # When True, RacketTargetCommand logs dbg_err_{minus,plus}_{win,exact} (swing-through sign check) and
    # the reward terms log dbg_{racket_pos,racket_vel,racket_normal,base}_{raw,gated}. Pure logging; no
    # behaviour change. Turn off for production runs (extra wandb scalars).
    debug_reward_logging: bool = False

    # --- R-b envelope-violation accounting (reward_staged_design 2026-07-08 §⑥ R-b细则) ---------
    # True (set by train.py's terminations.envelope_as_penalty translation) -> the command term
    # counts tracking-envelope violations that no longer terminate: tracking_loss_rate
    # (violation rising edges / swing starts, same EMA timescale as pre_strike_fall_rate; per-clip
    # variants too) + envelope_violated_frac (per-step violation fraction — the挂机/loafing
    # monitor). Threshold/bodies mirror TerminationsCfg.anchor_pos/ee_body_pos exactly. Default
    # False = no new metrics, byte-identical logging.
    track_envelope_violation: bool = False
    envelope_threshold: float = 0.25  # m; z-only error, same value the removed terminations used
    # A3 names (mixed casing is INTENTIONAL — matches the URDF; see robots/agibot_a3.py) = the
    # exact list AgibotA3FlatEnvCfg.__post_init__ pins into terminations.ee_body_pos
    # (A3_FEET_BODIES + A3_HAND_BODIES). Resolved against motion.cfg.body_names with a loud
    # error on a missing name.
    envelope_body_names: tuple = (
        "left_ankle_roll_Link", "right_ankle_roll_Link", "left_wrist_yaw_Link", "right_wrist_yaw_Link",
    )

    # --- conditional exact-strike success metric (logging + curriculum gating) ---
    # The logged strike_*_pass_exact / strike_composite_success_exact are a sample-weighted EMA of the
    # exact-strike pass rate: acc = decay*acc + this-step-count each control step. decay ~0.99 gives a
    # ~100-step (~2 s @ 50 Hz) memory; higher = smoother but slower to reflect the current policy. The
    # rate (and the curriculum) only trust it once `exact_success_min_count` decayed samples accumulate.
    exact_success_decay: float = 0.99
    exact_success_min_count: float = 50.0
    # --- metric-sync fix (2026-07-09, correctness — no ablation arm) ---------------------------
    # virtual_return_rate_rally* now uses per-swing SAME-LEDGER counting: at each swing's END the
    # attempt's start and its "returned legally" flag book together and decay together, so the
    # rate is <=1 by construction and immune to synchronized reset queues (the 0.31->1.48
    # oscillation, 2026-07-08 取证定案). This flag additionally keeps the OLD mixed-ledger curves
    # alive under the same names + `_legacy` suffix for one transition period of new/old
    # comparison; flip to False to stop emitting them once the transition ends.
    # 人话:上台率曲线换了记账法(恒<=1);这个开关只控制"旧算法对照曲线(_legacy)还要不要发"。
    rally_legacy_metrics: bool = True

    # --- P2.3 SMASH-style adaptive tracking sigma (coarse-to-fine reward kernel widths) ---
    # When on, every `sigma_update_every` control steps the racket_position/racket_velocity reward
    # stds (and racket_strike_success's std_pos/std_vel) are set to
    #   clamp(sigma_ema_scale * decayed_mean_exact_strike_error, sigma_min, sigma_max)
    # so the exp kernel always brackets the CURRENT operating error band. Replaces the hand-run
    # 1.8 -> 1.0 -> 0.8 -> 0.5 velocity-std curriculum. sigma_*_max should match the task YAML's
    # starting stds; sigma_*_min should sit at the acceptance thresholds (0.075 m / 0.5 m/s).
    adaptive_sigma: bool = False
    sigma_update_every: int = 500
    sigma_ema_scale: float = 1.0
    sigma_pos_min: float = 0.075
    sigma_pos_max: float = 0.20
    sigma_vel_min: float = 0.5
    sigma_vel_max: float = 1.0
    # 第三通道:拍面法向 sigma(弧度)。SMASH 是 pos/ori/vel 三路一起自适应收紧的;只收
    # pos/vel 会让拍面奖励随 pos 收紧被静默弱化最多 ~7x(reward_redesign_20260725 §2)。
    # True 时 racket_normal.std 与 racket_strike_success.std_normal 一起按 exact-strike
    # 面角误差(弧度)的同一路衰减 EMA 锁步更新:clamp(sigma_ema_scale * mean_err_rad,
    # min, max)。min=0.262(15° 验收线),max=0.52(~2x 验收 = 起步宽度)。默认 False =
    # 逐字节不变;单开(adaptive_sigma=False)在构造期 fail-loud(_validate_adaptive_sigma_cfg)。
    adaptive_sigma_normal: bool = False
    sigma_normal_min: float = 0.262
    sigma_normal_max: float = 0.52

    # --- A1 target latency & time-variance (mocap->planner->runner realism; roadmap A1) -------------
    # MOTIVATION: training otherwise hands the actor a PERFECT, instantly-updated target, while the
    # real loop (mocap -> planner -> runner) delivers it LATE (transport + planning latency), NOISY
    # (ball-prediction error that shrinks as the strike approaches — SMASH Eq. 14), and REFINED
    # mid-swing (the planner re-plans WHERE, not WHEN). PACE injects sensor delays for the same
    # reason. Without this, the mocap-closed-loop deployment faces out-of-distribution target
    # dynamics. Scope: ONLY the ACTOR-visible planner tuple is degraded; rewards, true timing,
    # metrics/gates, the privileged critic, and the achieved-target-replay write use live values.
    # ALL defaults OFF => byte-identical baseline (delay==0 aliases the live target tensors;
    # target_delay_tts_mode="live" returns the live TTS tensor; jitter/prob==0 draw no RNG).
    # actor sees atomic pos/vel/face-command/swing_sign this many 50 Hz steps late。
    # planner_revision_enabled 时语义升级为"耦合传输"(2026-07-25,取代旧 NO-LAUNCH 守卫):
    # 延迟改发生在修订"提交"侧——生成于 t 的修订元组(pos/vel/normal/tts,原子)t+d 才提交给
    # 相位调度器,接受记账 / 调度器 desired_tts / actor 元组同吃一条延迟流(mocap→relay 语义);
    # actor 端不再叠观测延迟环。BEGIN(任务安装)保持即时。要求 target_delay_tts_mode 为
    # source_timestamp_compensated(或显式阴性对照 uncompensated);'live' fail-loud。
    target_delay_steps: int = 0
    # Actor-visible TTS convention:
    #   live                         historical behavior; target fields can be delayed but TTS is live.
    #   source_timestamp_compensated TTS rides the same delayed tuple, then the runner-equivalent
    #                                source timestamp correction subtracts delay_steps * step_dt.
    #   uncompensated                TTS rides the tuple without correction (explicit stale-TTS
    #                                negative control; diagnoses "planner says wait" failures).
    # With delay_steps=0 both atomic modes are numerically equivalent to live.  Reward/critic/strike
    # masks always use ``time_to_strike`` and are unaffected by this actor-only switch.
    target_delay_tts_mode: str = "live"
    # SMASH-style tts-decaying gaussian noise on the ACTOR-visible target, drawn ONCE per step on the
    # ring-buffer push (determinism within a step): per-step std = knob * clamp(time_to_strike, 0, 1),
    # i.e. the knob is the std at time_to_strike >= 1 s, decaying to 0 at the strike (prediction
    # convergence). Units: m (pos) / m/s (vel).
    target_jitter_pos_per_s: float = 0.0
    # Calibrated mocap MEASUREMENT noise on the actor-visible target position (m). Venue fit
    # 2026-07-03 (`capture.position_noise`): white 0.0019, ar1 marginal 0.0052, rho/frame 0.946
    # @300 Hz -> 0.946**6 = 0.717 per 50 Hz policy step. Defaults OFF.
    target_noise_white: float = 0.0
    target_noise_ar1_sigma: float = 0.0
    target_noise_ar1_rho: float = 0.717
    # A1 v2 — the three Ace-style sensor defects the mocap link actually has (venue capture fit:
    # occlusion gaps concentrate at contacts, gap_p50 10 ms / racket occlusion ~30 ms; re-lock
    # after a contact carries a fresh systematic bias). All default OFF.
    target_dropout_prob: float = 0.0        # per-step P(frame lost) -> actor view holds last value
    target_post_strike_dropout_s: float = 0.0  # forced hold-last window right after each strike (s)
    target_bias_per_swing: float = 0.0      # m: constant bias per swing, resampled at swing start
    target_jitter_vel_per_s: float = 0.0
    # Mid-swing target refinement: each control step, envs with pre_strike AND time_to_strike >
    # midswing_resample_tts_floor re-draw their target (position/velocity/normal via the existing
    # sampling path) with this per-step probability. Strike timing is untouched (same strike step),
    # no swing start is counted, and the racket-progress baseline is reset so the target jump creates
    # no fake progress.
    midswing_resample_prob: float = 0.0
    midswing_resample_tts_floor: float = 0.3  # s; no refinement inside the last `floor` seconds before the strike

    # Replacement for truth-redrawing midswing_resample: one physical ball keeps one immutable
    # task identity while the actor receives bounded, atomically revised planner estimates every
    # policy step.  All fields are installed from one task.planner_revision block by train.py;
    # manually enabling only one command term fails closed at runtime.
    planner_revision_enabled: bool = False
    planner_revision_profile: dict | None = None
    planner_revision_initial_tts_range_s: tuple[float, float] = (0.5, 1.5)
    planner_revision_initial_tts_mixture: dict | None = None
    planner_revision_position_std_m: float = 0.0
    planner_revision_velocity_std_mps: float = 0.0
    planner_revision_normal_std_rad: float = 0.0
    planner_revision_tts_std_s: float = 0.0

    # --- Tier-1 VIRTUAL INCOMING BALL + at-strike landing evaluation (rewardDesign.md) -----------
    # Per swing, a virtual incoming ball (v_in, omega_in) is sampled that BY CONSTRUCTION arrives at
    # the racket target point at the strike time. On the exact-strike frame, the achieved racket FK
    # state is pushed through the venue-fitted paddle contact model (virtual_ball.predict_paddle_
    # contact, e(u_n) restitution) and a coarse RK4 landing rollout; the cached outcome buffers feed
    # the one-shot virtual_* reward terms in hope_rewards.py. No obs change; 175-D contract untouched.
    virtual_ball: bool = False
    # METRICS-ONLY virtual ball (franco 2026-07-06 "训练的时候以上台率为准"): run the same
    # per-swing ball sampling + at-strike contact/landing evaluation PURELY for the
    # virtual_*_rate metrics (in-training 上台率/击球率 curves) on tasks whose reward stack has
    # no virtual_* terms (DeployParity / Hitter). No reward change; the vb_* one-shot caches are
    # simply never consumed. Default OFF (the per-swing sampling consumes RNG, so byte-exact
    # reproducibility of old runs is preserved); pinned ON in the jiayi-lineage task YAMLs.
    vb_metrics_only: bool = False
    # Incoming-ball velocity box (world/env frame, m/s; -x = toward the robot). Kept inside the venue
    # fit's validity envelope (ball speed 1-7 m/s); vertical component ~near-apex-to-descending.
    vb_vel_x_range: tuple[float, float] = (-4.5, -2.0)
    vb_vel_y_range: tuple[float, float] = (-0.6, 0.6)
    vb_vel_z_range: tuple[float, float] = (-1.0, 0.5)
    # OPTIONAL PER-CLIP incoming-ball velocity boxes, shaped exactly like racket_vel_range_per_clip:
    # a tuple indexed by clip_id of ((x_lo,x_hi),(y_lo,y_hi),(z_lo,z_hi)). None -> the shared
    # vb_vel_*_range box for every clip (BACKWARD COMPATIBLE, byte-identical to today).
    # 人话:这是"给挡球喂快球、给拉球喂慢球"的开关。拍速和来球速度大约 1:1 互相替代(实测相关
    # 系数 -0.61),所以每个动作有自己的最佳来球速度;一个全局球箱会把某些动作的峰值排除在外
    # ——bh_loop_c 在自己速度下峰值 5.12 m/s、而球箱是 2.0-4.6,分数 0.150,就是这么来的。
    # Length is checked against the loaded clip count at construction (fail-loud, like
    # strike_phase_per_clip); two rows expand through clip_family_per_clip like the racket boxes.
    vb_vel_range_per_clip: tuple | None = None
    # Incoming spin: per-axis uniform (rad/s). 50 rad/s ~ 8 rev/s per axis keeps |omega| inside the
    # quaternion-validated 0-15 rev/s envelope.
    vb_spin_abs_max: float = 50.0
    # OPTIONAL PER-CLIP |spin| ceiling (rad/s), one per loaded clip. None -> the scalar above for
    # every clip. A block answering heavy topspin and a loop answering a float serve are different
    # regimes, and the ball box is what decides which one a stroke ever meets.
    vb_spin_abs_max_per_clip: tuple | None = None
    # REFERENCE-STRIKE RETURN GATE. > 0 -> at construction, each bound strike frame is scored with
    # the repo's OWN NumPy return scorer (scripts/virtual_return_scorer.py, the same contact +
    # landing contract the in-training metric uses) against balls drawn from THAT CLIP'S OWN
    # incoming regime; a clip whose legal-return fraction falls below this raises, naming the clip
    # and the rate. 0.0 = gate off (the default, byte-identical to today).
    reference_return_gate_min_rate: float = 0.0
    reference_return_gate_samples: int = 256
    reference_return_gate_seed: int = 0
    # Escape hatch for the "landing/net rewards need a solved answer" construction gate.
    allow_unbanked_landing_rewards: bool = False
    # virtual_spin reward semantics: "topspin" = Ace-style outgoing-topspin generation (ball
    # quality); "minimize" = stage-1 placement-first mode (franco 2026-07-04) — reward CANCELING
    # the incoming spin, kernel exp(-|omega_out|^2 / vb_spin_min_sigma^2) on the outgoing spin
    # magnitude, same legal-landing gate. Sigma in rad/s (10 ~ 1.6 rev/s residual).
    vb_spin_mode: str = "topspin"
    vb_spin_min_sigma: float = 10.0
    # CAPTURE GATE: the virtual contact only evaluates when (a) the racket center is within this
    # distance of the ball (= racket 0.075 + ball 0.020, the v0 real-hit radius) at the exact-strike
    # frame, and (b) the paddle is actively moving INTO the ball along the oriented contact normal
    # faster than vb_min_approach_speed (kills the phantom-block / retreating-racket exploit,
    # verify_tier1 (c)3 — a stationary wall-block scores nothing).
    vb_capture_radius: float = 0.095
    vb_min_approach_speed: float = 0.3
    # Virtual table placement in the env frame. The _hopex clips are HOPE +X aligned with the root at
    # the env origin, so the HOPE convention (robot ~0.5 m behind its table end, centered on the
    # width) puts the near table edge at x = +0.5 and the surface at z = +0.76 above the env origin.
    # Net/far-end/half-width follow from the ITTF table (geometry.py): net at near_x + 1.37 etc.
    vb_table_near_x: float = 0.5
    vb_table_surface_z: float = 0.76
    # Landing target on the opponent half (env frame). Default = P2 half center (near_x + 2.055, 0).
    vb_target_x: float = 2.555
    vb_target_y: float = 0.0
    # Reward shaping constants (read by hope_rewards.virtual_*).
    vb_landing_sigma: float = 0.3     # m — Gaussian width on ||landing_xy - target_xy|| (v0 parity)
    vb_net_margin: float = 0.12      # m — target clearance above the net top (v0 pass_net parity)
    vb_net_sigma: float = 0.10       # m — Gaussian width on the net-clearance error
    vb_spin_ref: float = 250.0       # rad/s (~40 rev/s) — full-credit outgoing topspin (Ace-style)
    vb_min_landing_depth: float = 0.3  # m past the net for the in-bounds bonus (dink guard, verify (c)1)
    # Coarse rollout resolution (verify_tier1 (b): h=10 ms, 1.0 s horizon covers 1-7 m/s shots).
    vb_rollout_h: float = 0.01
    vb_rollout_steps: int = 100

    # --- SHADOW physical ball + table (flag-gated, METRICS-ONLY; defaults OFF = byte-identical) --
    # shadow_ball=True spawns one real PhysX ball per env (scene entity "shadow_ball", attached by
    # hope_env_cfg.attach_shadow_ball_scene / the train.py override translation; sphere R/mass from
    # configs/ball_physics_venue.yaml) driven by shadow_ball.ShadowBallDriver: kinematic linear
    # incoming flight to the question contact point, the SAME venue paddle-contact model as the
    # reward path at a captured strike, then dynamic PhysX flight with the venue aero wrench per
    # physics substep and engine-integrated landing metrics (shadow_land_x/y, shadow_hit_count,
    # shadow_miss_count, shadow_vs_virtual_land_err vs the analytic vb prediction). PURE
    # MEASUREMENT: never read by rewards/observations/bank-target logic; requires virtual_ball=True
    # (loud error otherwise). Honesty notes + mechanisms: shadow_ball.py module docstring.
    shadow_ball: bool = False
    # shadow_table=True additionally (a) enables the shadow ball's collider and (b) places the
    # table_tennis static table collider (+ visual USD mesh) at the tracking task's virtual-table
    # pose (near edge x=vb_table_near_x, surface z=vb_table_surface_z, centered on y=0), with the
    # same multiplicative-restitution materials table_tennis uses — so the shadow ball physically
    # bounces where the virtual table is. No net collider (the vb model gates the net
    # analytically). Requires shadow_ball=True.
    shadow_table: bool = False

    # --- PHYSICAL ball + table — Phase A TRUTH INSTRUMENT (flag-gated, METRICS-ONLY; default ----
    # OFF = byte-identical). physical_ball=True spawns one real PhysX ball per env (scene entity
    # "pb_ball") + a real static table collider ("pb_table", + visual USD), attached by
    # hope_env_cfg.attach_physical_ball_scene / the env-cfg physical_ball flag / the train.py
    # task.physical_ball translation, and driven by physical_ball.PhysicalBallManager: each
    # swing's question incoming ball is realized physically — reverse-integrated venue-model
    # launch (arrives at the question contact point with the question incoming velocity exactly
    # at the strike frame), PhysX flight + per-substep venue aero wrench, CODE-DRIVEN fitted
    # table bounce (venue contact.table params), robot pass-through unless the evaluator-only,
    # content-addressed Phase-B rider sets ``physical_ball_impulse`` dynamically. Metrics:
    # pb_serve_err_m /
    # pb_serve_vel_err at the exact-strike frame + serve/bounce/landing counts. PURE MEASUREMENT:
    # never read by rewards/observations/bank-target logic; consumes NO RNG; requires
    # virtual_ball=True (loud error otherwise). Full honesty notes: physical_ball.py docstring.
    physical_ball: bool = False

    # --- reachable racket-target workspace (offsets from the env origin, world frame, meters) ---
    # Used only by target_mode="uniform". PLACEHOLDER ranges (not the reference strike point).
    racket_pos_x_range: tuple[float, float] = (0.25, 0.55)
    racket_pos_y_range: tuple[float, float] = (-0.45, 0.45)
    # Unified multi-clip: |y| sampling range; the SIGN is set per clip (forehand on -y, backhand on +y,
    # per forehand_on_negative_y) so forehand/backhand target regions are non-overlapping (HITTER §IV).
    racket_pos_y_abs_range: tuple[float, float] = (0.05, 0.45)
    racket_pos_z_range: tuple[float, float] = (0.70, 1.15)

    # --- desired racket velocity (world frame, m/s) ---
    racket_vel_x_range: tuple[float, float] = (1.5, 4.0)
    racket_vel_y_range: tuple[float, float] = (-1.0, 1.0)
    racket_vel_z_range: tuple[float, float] = (0.0, 1.5)
    # Optional PER-CLIP velocity boxes (uniform mode + unified multi-clip only). None -> use the SHARED
    # racket_vel_*_range above for every clip (BACKWARD COMPATIBLE: old behavior, nothing changes). When
    # set, it is a tuple indexed by clip_id (0=forehand, 1=backhand — same order as strike_phase_per_clip /
    # the command's _clip_names), each entry ((x_lo,x_hi),(y_lo,y_hi),(z_lo,z_hi)). Reason: the forehand and
    # backhand reference clips have DIFFERENT natural strike speeds (~2.6 vs ~2.0 m/s at the racket), so a
    # single shared box overshoots the slower backhand and its strike can never satisfy the velocity gate.
    # Confirmed by the MuJoCo per-clip eval probe: lowering only the backhand target box raised backhand
    # composite 0.32->0.79 (deterministic) / 0.39->0.77 (dither) with forehand byte-identical.
    racket_vel_range_per_clip: tuple | None = None
    # Escape hatch for the construction-time gate that requires every per-clip velocity box to have
    # x_lo > 0 (+x is toward the opponent, so a non-positive floor lets the sampler command a return
    # that never crosses the net). True = the box is deliberately non-forward (e.g. a block/drop study).
    allow_non_forward_target_velocity: bool = False

    # OPTIONAL per-clip racket target-POSITION boxes (uniform mode, unified multi-clip policy). None ->
    # use the shared racket_pos_x_range + |y|-sign + racket_pos_z_range box for every clip (BACKWARD
    # COMPATIBLE: old behavior, nothing changes). When set, it is a tuple indexed by clip_id (0=forehand,
    # 1=backhand — same order as strike_phase_per_clip), each entry ((x_lo,x_hi),(y_lo,y_hi),(z_lo,z_hi))
    # added to the env origin. NOTE the y range is SIGNED here and used directly, so it REPLACES the
    # shared |y|-sign logic. Reason: each clip's strike frame can sit at a different height/depth/lateral
    # offset, so a shared box can make one clip's strike-frame position unreachable. Per-clip boxes let
    # each clip's target track its own reference strike point.
    racket_pos_range_per_clip: tuple | None = None

    # ORDERED per-clip NAMES (clip_id -> human name), from train.py's ``racket.clip_names``. Empty
    # -> the legacy two family names, and per-clip metric buckets stay FAMILY buckets, byte-identical.
    # 人话:动作库不再只有"正手/反手"两个名字。给了这张表,每个 clip 就有自己的指标桶
    # (virtual_return_rate_<name>),同族的两个动作(拉/挡)不会再共用一个桶——正手回球率
    # 0.0000 藏在 45% 的合计里,就是共用桶造成的。
    clip_names_per_clip: tuple = ()

    # --- HER-style achieved-target replay (uniform mode + unified multi-clip only) -------------------
    # On-policy-compatible hindsight relabeling (Ace/HER, adapted for PPO): true retroactive relabeling is
    # observation-inconsistent here (the target is in the actor obs every step), so instead the NEXT
    # swing's target is drawn, with probability `achieved_target_mix_prob`, from a per-clip ring buffer of
    # racket states the policy ACTUALLY produced at previous exact-strike frames (pos env-origin-relative,
    # vel world). Every replayed target is reachable-by-demonstration — it kills the "the box asks for a
    # point the taught swing never passes through" mismatch without moving the box. Mixture (not pure
    # replay) + jitter + clamping into the per-clip box inflated by `achieved_clamp_inflate` (clamp
    # applies only when per-clip boxes are configured — the unified task always sets them) prevent the
    # target distribution from collapsing onto what the policy already does or drifting far from the
    # training workspace. NOTE the deploy-side target clips must be re-synced to the training boxes
    # whenever the boxes change (they are hand-maintained in pp_policy.hpp / imitate_presets.py).
    # 0.0 = OFF (backward compatible: pure box sampling). TRAIN-ONLY: eval entry points force this to
    # 0.0 so checkpoints are always scored on the pure box distribution. The buffer only fills at
    # exact-strike frames of envs that are still alive, so fallen approaches never contribute targets.
    achieved_target_mix_prob: float = 0.0
    achieved_buffer_size: int = 4096  # per-clip ring buffer capacity (entries)
    achieved_min_fill: int = 256  # replay only once a clip's buffer holds at least this many entries
    achieved_jitter_pos: float = 0.03  # m, uniform per-axis jitter added to a replayed position
    achieved_jitter_vel: float = 0.15  # m/s, uniform per-axis jitter added to a replayed velocity
    achieved_clamp_inflate: float = 0.20  # clamp replayed targets into the per-clip box inflated by this fraction

    # --- desired racket face normal ---
    normal_mode: str = "velocity"  # "velocity" (n = v/|v|) or "sampled"
    racket_normal_x_range: tuple[float, float] = (0.5, 1.0)
    racket_normal_y_range: tuple[float, float] = (-0.3, 0.3)
    racket_normal_z_range: tuple[float, float] = (-0.3, 0.3)

    # --- Stage-1 question bank + face-command channel (defaults OFF; byte-identical baseline) -------
    # Path to an offline bank npz (scripts/gen_stage1_questions.py): per clip a FIXED contact point
    # plus an atomic incoming-ball + inverse-solved answer tuple. Non-empty -> every target
    # resample (reset / wrap / mid-swing) OVERRIDES the sampled incoming velocity/spin and target
    # pos/vel/normal from one bank row; the A1 delay/noise injectors act
    # downstream unchanged. Bank positions are tracking-env-frame (world = env_origin + pos).
    # Empty (default) = OFF: the sampling paths above run untouched, target_normal_cmd stays zeros.
    question_bank: str = ""
    # Explicit reproducibility escape hatch for pre-schema-v2 banks. New research runs must
    # keep this false so missing incoming-spin/provenance cannot be silently inferred.
    question_bank_allow_legacy: bool = False
    # Re-anchor the racket_normal reward (and racket_strike_success through it) onto the demanded
    # target_normal_cmd instead of the clip-locked racket_target_normal_w (the clip-locked face is
    # the 0%-return root cause — eval mode B 2026-07-05). Exact/composite metrics use the same
    # pairing as the reward; the critic reference lane remains unchanged. False = old path.
    face_command: bool = False
    # Reward/critic/metric face pairing. ``shared_plus_y`` is the production A-frame convention;
    # ``legacy_signed_vs_A`` is an explicit diagnostic continuation of the pre-fix mismatch.
    face_command_pairing: str = "shared_plus_y"

    # --- CONTINUOUS question production (owner ruling: 离散题库是考试用的,训练必须连续采样) ----
    # ONE switch: ``target_mode: "solved"``. Everything else below is derived or fail-closed, so
    # there is no second thing an agent has to remember to turn on. The producer is a
    # PRODUCER-CONSUMER BUFFER, not a pool and not a small bank: one batched solve fills it, a
    # cursor hands rows out WITHOUT REPLACEMENT, and a spent row is discarded forever. Statistically
    # identical to solving at every reset; ~1.5% of iteration time instead of ~160% (the solver is
    # kernel-launch bound, so the fix is to make the call RARE, not small).
    cq_buffer_rows: int = 8192          # rows kept per clip; one refill measured at ~258 ms @8192
    cq_overdraw: float = 1.45           # first-pass overdraw (a 2nd call costs full price; this is free)
    cq_n_iters: int = 12                # 100% solve at 12; 4 halves the wall at a visible resid cost
    cq_tol_m: float = 0.02
    cq_speed_budget: float = 3.4        # pp_policy.hpp:234 gate_speed_max - margin (deploy's own number)
    cq_max_redraw_rounds: int = 3
    cq_max_face_deg: float = 35.0       # face-vs-clip-face cap (shipped banks: p50 7.8-9.1, max 20.3)
    # DEFAULT 0.0, NOT the producer's own 50.0: every schema-v3 bank hard-asserts
    # incoming_spin_mode=='zero', so a non-zero default would smuggle a spin regime no shipped
    # calibration has ever covered in on a plumbing change. Opening spin is its own arm.
    cq_spin_abs_max: float = 0.0
    # Aim is a POINT, not a range. The landing reward (hope_rewards.virtual_landing), the
    # virtual_land_err_m metric and the physical ball's return-flight target all read ONE fixed
    # _vb_target_xy; a per-env aim would have the ball solved for A and graded at B with nothing
    # asserting. None -> (vb_target_x, vb_target_y). Unlock by making _vb_target_xy per-env FIRST.
    cq_aim_xy: tuple | None = None
    # Incoming-ball velocity box, per LOADED clip, (num_clips, 3, 2). Required under "solved".
    cq_vel_range_per_clip: tuple | None = None
    # Hold the exam split OUT of training: reject drawn balls whose question_split(v_in) says
    # "exam" (stage1_question_bank.EXAM_FRAC). Costs ~20% more draws and keeps the repo invariant
    # that no trained question IS an exam question — without it a paired exam comparison can grade
    # a policy on balls it trained on.
    cq_exam_holdout: bool = True
    # Dedicated RNG stream for the question draw, so it is reproducible independently of every
    # other sampler's consumption pattern (generate() threads a torch.Generator all the way down).
    cq_seed: int = 0x51501234
    cq_accept_buckets: int = 4          # |v_in_x| bins for the per-regime accept ledger
    cq_max_exhausted_frac: float = 0.10     # above this a refill WARNs with the full histogram
    cq_abort_exhausted_frac: float = 0.50   # above this it raises
    cq_min_accept_rate: float = 0.05    # a bucket below this with enough asks is a config error
    cq_closed_loop_rows: int = 256      # rows re-rolled through the SCORER's own call per refill
    cq_closed_loop_max_err_m: float = 0.10  # the schema's own bound (stage1_question_bank.py:402)
    # CONTRACT ANCHOR: one same-family schema-v3 train bank, loaded and validated but NEVER trained
    # on. It restores what a continuous arm otherwise loses — the physics contract SHA, the runtime
    # motion contract (motion SHA / seg_len / strike_phase), clip-order authority — and it is the
    # boot parity gate's data (K of its stored balls replayed through the live solver).
    cq_anchor_bank: str = ""
    # Exam paper for scripts/judge.sh. Read FIRST, before the legacy question_bank string-replace,
    # so a continuous arm needs no operator memory and no --exam-bank.
    exam_bank: str = ""

    # --- desired base XY target (offsets from the env origin, world frame, meters) ---
    base_target_x_range: tuple[float, float] = (-0.10, 0.10)
    base_target_y_range: tuple[float, float] = (-0.35, 0.35)
    # Weak base->racket coupling (UNIFORM mode only). base_couple_blend = fraction of the racket target's
    # sideways (Y) offset that the base target shifts toward; clamped to ±base_couple_max_offset meters.
    # 0.0 = disabled (spawn-only). Conservative because no walking reference exists (it fights leg imitation).
    base_couple_blend: float = 0.0
    base_couple_max_offset: float = 0.20
    # UNIFORM-mode base-target derivation (HITTER §V-B-1 alignment, 2026-07-05):
    #   "blend"           — legacy: spawn + weak Y blend above (BASE-FREE tasks leave this the default).
    #   "reference_reach" — HITTER separate-commands scheme: base_target = racket_target_xy −
    #                       (reference base→racket strike offset, per clip). Standing AT the commanded
    #                       station puts the racket target at the clip's reference reach — the striking
    #                       plane is fixed RELATIVE TO THE COMMANDED BASE (HITTER's "0.4 m in front"),
    #                       and footwork (mostly lateral) is driven by the base channel, not by
    #                       stretching at a deep world point. base_target_*_range then acts as a JITTER
    #                       around the coupled station (widen y to train y-reach diversity).
    # Sim2real: the paired actor obs (base_target_pos_b) is a RELATIVE Δxy in the yaw-heading frame —
    # deployable from mocap base position (300 Hz, position-only) without any absolute world frame; if
    # mocap drops, feeding Δ=0 degrades gracefully to "already at station" (today's BASE-FREE behavior).
    base_couple_mode: str = "blend"

    # --- R10c 站位锚(station_obs 旗标的数据源;franco 2026-07-09) ---------------------------------
    # 世界系常数锚点相对 env origin 的 XY 偏移(米)。默认 (0,0) = 出生点本身(固定点阶段 clip 已
    # 重落地,frame-0 root xy ≈ origin)。想把"该站的位置"挪开出生点时用它覆盖(如 S2b 身补课程)。
    # 只喂观测 station_anchor_err_b,不进奖励;env 侧开关见 hope_env_cfg.station_obs / train.py
    # racket.station_obs。
    station_anchor_offset_xy: tuple[float, float] = (0.0, 0.0)

    # --- swing-type convention ---
    forehand_on_negative_y: bool = True  # right arm holds the paddle: target on -Y side -> forehand (+1)
