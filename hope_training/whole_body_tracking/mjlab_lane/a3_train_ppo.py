#!/usr/bin/env python3
"""PPO (rsl-rl) on the A3 court scene -- the "is it actually learning?" run.

WHAT THIS IS (plain language)
-----------------------------
The plant step proved we can carry the vendor's robot physics into MuJoCo Warp
field by field.  The scene step put a calibrated table, net and ball around it.
This step closes the loop: a real policy, real PPO updates, on GPU, with the
same plant -- so we can say "it trains", not just "it steps".

It deliberately does **not** try to match the Isaac A211/C211 observation and
reward columns.  That is the next step (cross-engine parity).  What is claimed
here is narrower and checkable:

  * our robot + our table + our plant, batched at nworld = 4096,
  * driven by rsl-rl 5.4.0 PPO through mjlab's own runner class,
  * with a reward that a human can read off a curve: stay in the split-ready
    stance, stay upright, and get the racket to the ball.

WHAT IS *NOT* mjlab-DEFAULT HERE (and why)
------------------------------------------
mjlab's ManagerBasedRlEnv is not used.  Its managers would re-open exactly the
three doors the plant step nailed shut: ``ActionCfg`` installs mjlab actuators
over the vendor's 31 pure-torque motors, ``EventCfg``/``CollisionCfg`` rewrite
geoms, and the observation manager assumes an mjlab ``Entity``.  So this file
implements the rsl-rl ``VecEnv`` interface *directly* on the ``A3PlantEnv``
handle that ``a3_court_env.build_court_env`` returns.  Everything above the
env boundary (PPO, storage, logger, runner, checkpointing) is stock rsl-rl /
mjlab -- ``MjlabOnPolicyRunner``, not a hand-rolled loop.

CONTROL ABI
-----------
The vendor's actuators are pure torque motors and the deployed controller
computes the PD itself.  We do the same, once per *physics* step (1 kHz):

    q_raw = action_offset + action_scale * a                    # 50 Hz
    q_des = shared_soft_hard_state_guard(q_raw, q, qd)          # 50 Hz
    tau   = clamp(kp*(q_des - q) - kd*qd, ctrlrange)            # 1 kHz

so the policy's action is a residual joint-position target around the pinned
Isaac ``runtime_plant.default_joint_pos_rad``.  That affine offset is distinct
from the split-ready physical reset pose.  There is no raw-policy clip: Isaac's
active runner also has ``clip_actions=null``.  FullMDP calls the same pure
tensor q_des guard as Isaac: finite proposals are projected into the soft/hard
inner envelope, while NaN/Inf or a measured/predicted hard-inner crossing is
braked before physics and terminates after that safe transition.  The legacy
non-FullMDP lane retains its historical hard clamp.

HOW A RUN FROM THIS FILE MAY BE REPORTED
----------------------------------------
The headline is the **binary per-episode racket-ball contact rate** measured
against the **zero policy on the same scene**, over **at least two runs**, with
the run-to-run band shown.  ``reach_term_weighted`` / ``touch_term_weighted``
are weighted reward terms with ceilings 2.0 / 4.0 -- they are not
probabilities, they are not contact rates, and quoting them as such once turned
"0.12% -> 49.2%/97.8%" into "touch 4e-5 -> 0.21".  ``--report`` enforces this
and refuses (exit 2) anything weaker.  See the "How this run is allowed to be
reported" section below.

Usage
-----
  python a3_train_ppo.py --smoke                       # 64 worlds, 3 iters
  python a3_train_ppo.py --nworld 4096 --iterations 60 --seed 0 --tag s0
  python a3_train_ppo.py --analyze RUN_s0.jsonl RUN_s1.jsonl --out BAND.json
  python a3_train_ppo.py --report TRAIN_s0.json TRAIN_s1.json \
      --report-zero-policy EVAL_zero.json \
      --report-eval EVAL_ckpt_s0.json EVAL_ckpt_s1.json --out REPORT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Lock what CUDA_VISIBLE_DEVICES actually means, before anything can initialise
# CUDA.  PLAIN LANGUAGE: `CUDA_VISIBLE_DEVICES=2` picks "the third card in
# CUDA_DEVICE_ORDER", and that variable defaults to FASTEST_FIRST, which has no
# contract to agree with the PCI order nvidia-smi prints.  On this pod all
# three cards are the same model so the two orders happened to agree -- that is
# an after-the-fact coincidence, not a guarantee, and on a mixed-model box `=2`
# could land on a card another lane is using.  PCI_BUS_ID is the order humans
# mean.  `setdefault` so an explicit choice by the caller still wins.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_MDP = (
  _HERE.parent / "source" / "whole_body_tracking" / "whole_body_tracking"
  / "tasks" / "tracking" / "mdp"
)
if str(_MDP) not in sys.path:
  sys.path.insert(0, str(_MDP))

import action_ball_qdes_guard as shared_qdes_guard  # noqa: E402

ACTION_JOINT_ORDER_CONTRACT_ID = "a3-gmr-dof-pos-to-runtime-articulation-v1"
ACTION_JOINT_ORDER_CONTRACT_SHA256 = (
  "b09987ff7a1bfa624b566cc8884d16672ba73c1acc3f92efb8a4faa99d314815"
)
ACTION_OFFSET_SOURCE = "runtime_plant.default_joint_pos_rad"
ACTION_OFFSET_FLOAT32_SHA256 = (
  "1b638d7b2e1ac7e552aace2ac8c2b00980dd9daf691f930b5fe775cebc84af78"
)
EXECUTABLE_QDES_GUARD = "action_ball_shared_soft_hard_state_guard_v1"
_ACTION_JOINT_ORDER_CONTRACT = (
  _HERE.parents[2] / "configs/a3_joint_order_bijection_v1.json"
)

# a3_court_env pulls in a3_plant_env, calibrate_restitution and geometry.
import a3_court_env as court  # noqa: E402

plant = court.plant
geom = court.geom


def _joint_order_names(path: Path) -> list[str]:
  return [row.strip() for row in path.read_text(encoding="utf-8").splitlines()
          if row.strip() and not row.lstrip().startswith("#")]


def _joint_order_names_sha256(names) -> str:
  payload = json.dumps(
    list(names), ensure_ascii=True, separators=(",", ":")
  ).encode("utf-8")
  return hashlib.sha256(payload).hexdigest()


def _runtime_action_wiring(model, prefix: str):
  """Return the one actuator(GMR)->policy(runtime) column authority.

  Policy actions, observations, ready pose and measured teacher are always in
  the schema-2 runtime articulation order.  MuJoCo's motors remain in vendor
  GMR order; only the final ``ctrl`` write converts back.
  """
  import mujoco

  raw = _ACTION_JOINT_ORDER_CONTRACT.read_bytes()
  if hashlib.sha256(raw).hexdigest() != ACTION_JOINT_ORDER_CONTRACT_SHA256:
    raise RuntimeError("A3 action joint-order contract bytes differ")
  contract = json.loads(raw)
  required = {
    "schema_version", "contract_id", "expected_joint_count", "source_order",
    "target_order", "target_from_source_indices",
    "source_from_target_indices", "legacy_mirrors",
    "runtime_metadata_contract", "status",
  }
  if (set(contract) != required or contract["schema_version"] != 1
      or contract["contract_id"] != ACTION_JOINT_ORDER_CONTRACT_ID
      or contract["expected_joint_count"] != 31):
    raise RuntimeError("A3 action joint-order contract schema differs")
  repo = _HERE.parents[2]
  source_path = repo / contract["source_order"]["path"]
  target_path = repo / contract["target_order"]["path"]
  source_raw, target_raw = source_path.read_bytes(), target_path.read_bytes()
  if (hashlib.sha256(source_raw).hexdigest()
      != contract["source_order"]["file_sha256"]
      or hashlib.sha256(target_raw).hexdigest()
      != contract["target_order"]["file_sha256"]):
    raise RuntimeError("A3 action joint-order name bytes differ")
  source, target = _joint_order_names(source_path), _joint_order_names(target_path)
  runtime_from_actuator = contract["target_from_source_indices"]
  actuator_from_runtime = contract["source_from_target_indices"]
  expected = list(range(31))
  if (len(source) != 31 or len(target) != 31
      or len(set(source)) != 31 or len(set(target)) != 31
      or _joint_order_names_sha256(source)
      != contract["source_order"]["names_sha256"]
      or _joint_order_names_sha256(target)
      != contract["target_order"]["names_sha256"]
      or any(type(index) is not int for index in runtime_from_actuator)
      or any(type(index) is not int for index in actuator_from_runtime)
      or sorted(runtime_from_actuator) != expected
      or sorted(actuator_from_runtime) != expected
      or target != [source[index] for index in runtime_from_actuator]
      or any(actuator_from_runtime[source_index] != runtime_index
             for runtime_index, source_index in enumerate(runtime_from_actuator))):
    raise RuntimeError("A3 action joint-order permutation differs")
  if int(model.nu) != 31:
    raise RuntimeError("MuJoCo actuator count differs from the 31-DOF contract")
  actuator_names = []
  for actuator_index in range(model.nu):
    joint_id = int(model.actuator_trnid[actuator_index, 0])
    qualified_name = mujoco.mj_id2name(
      model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
    ) or ""
    if prefix and not qualified_name.startswith(prefix):
      raise RuntimeError("MuJoCo actuator joint is outside the robot namespace")
    actuator_names.append(qualified_name[len(prefix):] if prefix
                          else qualified_name)
  if actuator_names != source:
    raise RuntimeError("MuJoCo actuator order differs from the GMR authority")
  runtime_from_actuator_array = np.asarray(
    runtime_from_actuator, dtype=np.int64)
  actuator_from_runtime_array = np.asarray(
    actuator_from_runtime, dtype=np.int64)
  runtime_from_actuator_array.setflags(write=False)
  actuator_from_runtime_array.setflags(write=False)
  return (runtime_from_actuator_array, actuator_from_runtime_array,
          tuple(target))


def _runtime_action_offset(ready_pose_payload: bytes, runtime_joint_names):
  """Load the active Isaac affine origin from the same pinned ready artifact."""
  try:
    document = json.loads(ready_pose_payload)
    runtime_plant = document["runtime_plant"]
    values = runtime_plant["default_joint_pos_rad"]
  except (KeyError, TypeError, ValueError) as exc:
    raise RuntimeError(
      "ready artifact lacks runtime_plant.default_joint_pos_rad") from exc
  expected_names = list(runtime_joint_names)
  if (type(runtime_plant) is not dict
      or runtime_plant.get("joint_names") != expected_names
      or runtime_plant.get("articulation_joint_names") != expected_names
      or runtime_plant.get("action_joint_ids") != list(range(31))
      or type(values) is not list or len(values) != 31
      or any(type(value) not in (int, float) for value in values)):
    raise RuntimeError("ready artifact runtime action-offset ABI differs")
  offset_le = np.asarray(values, dtype="<f4")
  digest = hashlib.sha256(offset_le.tobytes(order="C")).hexdigest()
  if (offset_le.shape != (31,) or not np.isfinite(offset_le).all()
      or digest != ACTION_OFFSET_FLOAT32_SHA256):
    raise RuntimeError("ready artifact runtime action offset differs")
  return np.asarray(offset_le, dtype=np.float32), digest


# ==========================================================================
# Task configuration.  Everything a human would want to argue about is here.
# ==========================================================================


@dataclass
class TaskCfg:
  """The reward/termination recipe.  Deliberately small and readable."""

  # --- episode -----------------------------------------------------------
  episode_length_s: float = 3.0
  action_scale: float = 0.25          # rad of joint-target residual per unit action
  action_scale_mode: str = "flat"
  """``flat`` = one number for all 31 joints (this file's own default).

  ``vendor`` = the Isaac/deploy decoder scale, per joint
  ``0.25 * effort_limit / kp`` (``robots/agibot_a3.py::AGIBOT_A3_ACTION_SCALE``),
  i.e. ``a = 1`` asks for a quarter of that joint's torque budget at zero
  velocity.  Ranges from 0.0375 rad (head and wrist pitch/yaw, 6/40 and 6/20)
  to 0.6875 rad (hip yaw and hip pitch, 220/80), so it is NOT a rescaling of
  ``flat`` -- it re-weights which joints the policy can move.  Provided because
  A211 parity will need it.

  Those two endpoints are not hand-written trivia: ``isaac_alignment.py``
  re-derives the whole 31-joint table from the live Isaac actuator literals and
  the vendor MJCF ``ctrlrange`` on every call, and the ``action_decoder`` row
  records whether ``vendor`` mode reproduces it joint by joint.  (An earlier
  version of this docstring said 0.647 / waist yaw; the live read is what
  corrected it.)
  """
  # Active Isaac uses ``clip_actions=null``.  The old MuJoCo-only +/-4 raw
  # clip made pinned measured-teacher targets mathematically unreachable.
  # Keep the field so the live alignment ledger can classify this semantic;
  # any non-None value is an unsupported, cross-engine-divergent setting.
  action_clip: object = None

  # --- observation scaling ----------------------------------------------
  obs_scale_lin_vel: float = 0.5
  obs_scale_ang_vel: float = 0.25
  obs_scale_joint_vel: float = 0.05
  obs_scale_ball_vel: float = 0.1
  obs_clip: float = 100.0

  # --- reward weights ----------------------------------------------------
  # NOTE on `w_alive`: a large constant alive bonus is a *reporting* hazard, not
  # a learning one -- it adds the same number to every step of every policy and
  # so compresses the visible dynamic range of the reward curve.  The pilot run
  # (w_alive = 1.0) sat at r/step = 2.99 -> 3.03 while the racket->ball distance
  # improved 0.41 -> 0.33 m: real learning, invisible curve.  Kept small.
  w_alive: float = 0.25
  w_pose: float = 1.0                 # split-ready joint tracking
  w_upright: float = 0.5
  w_height: float = 0.5
  w_reach: float = 2.0                # smooth racket->ball distance shaping
  w_touch: float = 4.0                # sharp bonus for actually being on the ball
  w_action_rate: float = -0.02
  w_joint_vel: float = -0.002
  w_torque: float = -0.05             # on the *normalized* torque
  r_termination: float = -5.0

  # reward kernels
  k_pose: float = 2.0                 # exp(-k * mean square joint error)
  k_upright: float = 4.0
  k_height: float = 40.0
  reach_len_m: float = 0.8            # exp(-d / reach_len)
  touch_sigma_m: float = 0.15         # exp(-(d/sigma)^2)

  # --- termination -------------------------------------------------------
  min_pelvis_z: float = 0.70          # ready pelvis sits at 1.0684
  max_tilt_proj_g: float = -0.5       # projected gravity z; -1 = perfectly upright

  # --- reset randomization ----------------------------------------------
  reset_joint_noise_rad: float = 0.05
  reset_joint_vel_noise: float = 0.0
  reset_root_xy_noise_m: float = 0.02
  reset_root_yaw_noise_rad: float = 0.05

  # --- ball ---------------------------------------------------------------
  ball_reserve_after_s: float = 2.0   # re-serve if the rally has clearly died
  ball_dead_z_hope: float = -0.35     # below the table plane == dead
  ball_dead_x_lo_hope: float = -1.2   # past the robot
  ball_dead_x_hi_hope: float = 3.4


@dataclass
class SimCfg:
  """Plant/scene knobs.  Defaults are the shipped, calibrated values."""

  nworld: int = 4096
  cone: str = "elliptic"              # calibrated tangential channel
  add_pairs: bool = True
  njmax: int = 572                    # measured, not the warp heuristic (64)
  nconmax: int = 128
  ball_spawn_hope: tuple = (2.0, -0.7625, 0.68)


# ==========================================================================
# Small math helpers (kept local so this file has no mjlab-internal deps).
# ==========================================================================


def quat_rotate_inverse(q, v):
  """Rotate ``v`` (world) into the frame of quaternion ``q`` (w, x, y, z)."""
  import torch

  q_w = q[:, 0:1]
  q_vec = q[:, 1:4]
  a = v * (2.0 * q_w * q_w - 1.0)
  b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
  c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0
  return a - b + c


# ==========================================================================
# The capacity gate.
# ==========================================================================
#
# PLAIN LANGUAGE.  MuJoCo Warp allocates every array it will ever need before
# the first step: constraint rows (`njmax` per world), contacts and broadphase
# candidate pairs (`naconmax` across all worlds).  When one step needs more
# than was allocated the engine does NOT stop.  It throws the surplus away and
# keeps going, so the reward curve stays pretty while the physics underneath it
# is wrong.  What the engine *does* do is record the fact: `d.overflow` is one
# integer per world, one bit per kind of overflow, set by the engine itself and
# sticky until that world is reset.  This gate reads that integer.  Any bit
# set, anywhere, stops the run.
#
# WHY WE STOPPED COUNTING `nefc`/`nacon` OURSELVES (this file used to).
# `naconmax` is simultaneously the ceiling for three different arrays:
# narrowphase contacts (`nacon`), broadphase candidate pairs (`ncollision`) and
# `collision_pair`.  When the broadphase array overflows, the surplus candidate
# pairs are dropped *before* narrowphase ever runs -- so `nacon` can never
# reach its own ceiling, and the deeper the overflow, the healthier the number
# we were watching looks.  Measured on 2026-08-06 with `--nconmax 10`: the
# engine printed 1134 "broadphase overflow" lines while the receipt said
# `PASS_NO_OVERFLOW` with `naconmax_headroom_x = 1.42` and the process exited 0.
# `d.overflow` has no such blind spot: it covers all nine kinds at once,
# including the three the engine never prints.
#
# The `nefc` / `nacon` / `ncollision` peaks are still tracked -- but only to
# compute headroom numbers for the receipt.  They no longer decide pass/fail.

OVERFLOW_FLAG_MEANING = {
  "NEFC": "a world needed more constraint rows than njmax; surplus rows dropped "
          "(raise --njmax)",
  "NJMAX_NNZ": "the sparse constraint Jacobian had more non-zeros than allocated "
               "(raise --njmax)",
  "BROADPHASE": "more candidate collision pairs than naconmax; pairs dropped "
                "BEFORE narrowphase, which is why nacon looks healthy "
                "(raise --nconmax)",
  "NARROWPHASE": "more actual contacts than naconmax; contacts dropped "
                 "(raise --nconmax)",
  "CCD": "the convex-collision (CCD) work buffer filled up; contacts dropped",
  "HFIELD": "height-field collision buffer filled up",
  "CONTACT_MATCH": "a contact-match sensor's match buffer filled up",
  "NVMAX": "the constraint-island active-DOF buffer filled up; engine says "
           "'behavior undefined'",
  "EPA_HORIZON": "the EPA horizon buffer inside convex collision filled up",
}

# Read against the mujoco-warp 3.10.0.3 sources on 2026-08-06: ALL NINE kinds
# do print something (`opt.warn_overflow` is hardwired True at io.py:436).  The
# catch is subtler than "four are silent", which is what we used to believe:
#
#   * EPA_HORIZON prints "Warning: EPA horizon = N isn't large enough."
#     (collision_gjk.py:1392/1411) -- no "overflow" anywhere in the string, so
#     every grep anyone has ever run for this misses it completely.
#   * BROADPHASE and NARROWPHASE print from world 0 only (forward.py:263/270),
#     so the line count is not a count of affected worlds.
#
# Both are reasons the stdout channel is a cross-check and `d.overflow` is the
# gate, not the other way round.
OVERFLOW_PRINTF_WITHOUT_THE_WORD_OVERFLOW = ("EPA_HORIZON",)


def _overflow_flag_names() -> tuple:
  """Bit order of mujoco-warp's ``OverflowType``, read from the engine itself."""
  try:
    from mujoco_warp._src.types import OverflowType

    return tuple(f.name for f in sorted(OverflowType, key=lambda f: f.value))
  except Exception:  # pragma: no cover - only if the engine moves the enum
    return ("NEFC", "NJMAX_NNZ", "BROADPHASE", "NARROWPHASE", "CCD",
            "HFIELD", "CONTACT_MATCH", "NVMAX", "EPA_HORIZON")


OVERFLOW_FLAGS = _overflow_flag_names()
OVERFLOW_BIT = {name: 1 << i for i, name in enumerate(OVERFLOW_FLAGS)}


def decode_overflow_mask(mask: int) -> list:
  """Turn the engine's bitmask into the names a human can act on."""
  return [n for i, n in enumerate(OVERFLOW_FLAGS) if int(mask) & (1 << i)]


class CapacityOverflow(RuntimeError):
  """The engine reported an overflow.  Never softened, never downgraded."""

  def __init__(self, mask: int, where: str, detail: str = "") -> None:
    self.mask = int(mask)
    self.flags = decode_overflow_mask(self.mask)
    self.where = where
    named = "|".join(self.flags) or "<unknown>"
    why = "  ".join(f"{f}: {OVERFLOW_FLAG_MEANING.get(f, 'unknown kind')}."
                    for f in self.flags)
    super().__init__(
      f"CAPACITY_OVERFLOW at {where}: mujoco-warp set d.overflow = {self.mask} "
      f"= {named}. {why} {detail} "
      f"The physics in the affected worlds is wrong from this point on, so "
      f"every number after it is suspect.  Re-size with --njmax/--nconmax and "
      f"re-run.  Do not soften this check.")


# The exact strings mujoco-warp 3.10.0.3 prints on stdout when it drops
# something.  Used by --warn-scan-log to pull the engine's own shouting out of
# a 5000-line training log and into the run summary (it was always there; it
# was just never read).
WARP_OVERFLOW_PRINTF_MARKERS = (
  "nefc overflow",
  "njmax_nnz overflow",
  "broadphase overflow",
  "narrowphase overflow",
  "CCD overflow",
  "Collision buffer overflow",
  "height field collision overflow",
  "contact match overflow",
  "nvmax overflow",
  # EPA_HORIZON's line does not contain the word "overflow" at all.
  "EPA horizon",
)


def scan_warp_overflow_warnings(log_path) -> dict:
  """Count the engine's own overflow lines in a log file this run is teed into.

  PLAIN LANGUAGE: a second, independent channel.  The gate reads `d.overflow`
  off the GPU; this reads what the engine shouted on stdout.  If the two ever
  disagree, the run fails -- the whole point of the 2026-08-06 audit was that
  1134 engine warnings sat unread in a log while the receipt said PASS.
  """
  out = {"log_path": str(log_path), "scanned": False, "lines": 0,
         "by_marker": {}, "examples": []}
  try:
    p = Path(log_path)
    if not p.is_file():
      out["error"] = "log file not found"
      return out
    counts = {m: 0 for m in WARP_OVERFLOW_PRINTF_MARKERS}
    examples: list = []
    with p.open("r", errors="replace") as fh:
      for line in fh:
        for m in WARP_OVERFLOW_PRINTF_MARKERS:
          if m in line:
            counts[m] += 1
            out["lines"] += 1
            if len(examples) < 3:
              examples.append(line.rstrip()[:300])
            break
    out["scanned"] = True
    out["by_marker"] = {k: v for k, v in counts.items() if v}
    out["examples"] = examples
  except Exception as exc:  # pragma: no cover
    out["error"] = repr(exc)
  return out


def _device_identity(requested_device, smi: dict | None = None) -> dict:
  """What GPU did this process ACTUALLY get, as opposed to what we asked for?

  PLAIN LANGUAGE: ``--device cuda:0`` and ``CUDA_VISIBLE_DEVICES=2`` are both
  *intentions*.  They record what we asked the driver for, not what it handed
  back -- and ``CUDA_VISIBLE_DEVICES`` indexes in ``CUDA_DEVICE_ORDER``, which
  defaults to FASTEST_FIRST and has no contract to agree with nvidia-smi's PCI
  order.  These fields come from inside the process: the card's own UUID, its
  PCI bus id, and how many cards this process can see at all.  The UUID can be
  matched afterwards against nvidia-smi's compute-process list, which is the
  only way a finished run can prove which card it ran on.
  """
  import torch

  out: dict[str, Any] = {
    "requested_device": str(requested_device),
    "cuda_visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "cuda_device_order_env": os.environ.get("CUDA_DEVICE_ORDER", ""),
    "cuda_device_order_locked_to_pci_bus_id": (
      os.environ.get("CUDA_DEVICE_ORDER", "") == "PCI_BUS_ID"),
    "pid": os.getpid(),
  }
  uuid_s = None
  try:
    idx = int(torch.cuda.current_device())
    props = torch.cuda.get_device_properties(idx)
    raw = getattr(props, "uuid", None)
    if raw is not None:
      uuid_s = str(raw)
      if not uuid_s.startswith("GPU-"):
        uuid_s = "GPU-" + uuid_s
    count = int(torch.cuda.device_count())
    out.update({
      "torch_cuda_device_count": count,
      "torch_current_device_index": idx,
      "device_name": str(props.name),
      "device_uuid": uuid_s,
      "pci_bus_id": getattr(props, "pci_bus_id", None),
      "total_memory_MiB": float(props.total_memory) / 2**20,
      "exactly_one_visible_device": count == 1,
    })
  except Exception as exc:  # pragma: no cover
    out["error"] = repr(exc)
  out.update(_match_uuid_against_smi(uuid_s, out["pid"], smi))
  return out


def capacity_fields(*, njmax: int, naconmax: int, nefc_peak: int,
                    nacon_peak: int, ncollision_peak: int, worlds_flagged: int,
                    overflow_mask: int, samples_step: int, samples_forward: int,
                    ncollision_known: bool = True) -> dict:
  """Turn raw engine counters into the receipt's capacity block.  Pure.

  Split out of :meth:`A3ReadyBallVecEnv.capacity_snapshot` so the two pieces of
  arithmetic that were previously wrong can be unit-tested without a GPU:
  which counter the contact headroom divides by (T2), and whether
  ``nefc == njmax`` counts as an overflow (P10 -- it does not; the engine drops
  a row only when ``nefc > njmax``).
  """
  # T2: `naconmax` caps narrowphase contacts AND broadphase candidate pairs,
  # and the candidates are always the larger of the two (measured 1.7--2.5x).
  # Dividing by `nacon` alone -- what this file used to do -- overstates the
  # headroom by exactly that factor.
  binding = max(nacon_peak, ncollision_peak) if ncollision_known else nacon_peak
  measured = samples_step > 0
  return {
    "probe_enabled": True,
    "gate": "mujoco_warp_d_overflow",
    "capacity_samples_stepped": samples_step,
    "capacity_samples_forward": samples_forward,
    "overflow_mask": overflow_mask,
    "overflow_flags": decode_overflow_mask(overflow_mask),
    "worlds_with_any_overflow_flag": worlds_flagged,
    "njmax_allocated_per_world": njmax,
    "naconmax_allocated_all_worlds": naconmax,
    "nefc_peak_per_world_running": nefc_peak if measured else None,
    "nacon_peak_all_worlds_running": nacon_peak if measured else None,
    "ncollision_peak_all_worlds_running": (ncollision_peak
                                           if (ncollision_known and measured)
                                           else None),
    "naconmax_binding_peak_all_worlds": binding if measured else None,
    # Headroom is None, not a huge number, when nothing was measured.  A zero
    # peak used to be indistinguishable from "measured, and really zero", which
    # is how a zero-length run published `njmax_headroom_x: 572.0`.
    "njmax_headroom_x": (njmax / nefc_peak) if (measured and nefc_peak > 0)
                        else None,
    "naconmax_headroom_x": ((naconmax / binding) if (measured and binding > 0)
                            else None),
    "nefc_over_njmax": bool(measured and nefc_peak > njmax),
    "nefc_exactly_fills_njmax": bool(measured and nefc_peak == njmax),
    "naconmax_binding_over": bool(measured and binding > naconmax),
  }


def _match_uuid_against_smi(uuid_s, pid, smi: dict | None) -> dict:
  """Cross-check the in-process UUID against nvidia-smi's compute-process list."""
  procs = list((smi or {}).get("compute_procs") or [])
  mine = [ln for ln in procs if ln.strip().split(",")[0].strip() == str(pid)]
  foreign = [ln for ln in mine if uuid_s and uuid_s not in ln]
  return {
    "nvidia_smi_own_compute_proc_lines": mine,
    "device_uuid_matches_nvidia_smi": bool(uuid_s and mine and not foreign),
    "nvidia_smi_lines_for_this_pid_on_another_uuid": foreign,
  }


# ==========================================================================
# How this run is allowed to be reported.
# ==========================================================================
#
# PLAIN LANGUAGE.  On 2026-08-06 an audit found that this lane's own headline
# number was wrong in a way that made a decent result look like a bad one.  The
# curve everyone was quoting, `touch: 4e-5 -> 0.21`, is not a contact rate and
# never was: `touch` is a *weighted reward term*, `w_touch * exp(-(d/sigma)^2)`,
# whose ceiling is 4.0, so 0.21 is a kernel mean of 5.25% -- and even that is
# not "5.25% of the time the racket was on the ball", it is the average of a
# smooth bump function of distance.  The one number with physical meaning is
# binary: did the racket and the ball actually touch during this episode, yes
# or no.  Measured that way the same policies go from 0.12% (do-nothing
# baseline) to 49.2% and 97.8% -- 400-800x, not "0.21".  Reproduced on
# 2026-08-06 with two fresh 4096x300 runs: `touch_term_weighted` 0.003 -> 0.252
# and 0.004 -> 0.189, i.e. the old headline; binary contact rate 0.14% -> 80.7%
# and 56.0%, i.e. the same two runs told honestly.
#
# Two more things the audit found, both of which this section encodes:
#   * the binary counter existed but was wired only into `--eval`, so no
#     training curve could ever show it;
#   * one seed, one point is not a result on this engine -- the same config run
#     four times gave touch = 0.21 / 0.46 / 0.59 / 0.61, and 0.21 (the number
#     that got reported) was the worst of the four.
#
# So: the receipt names its own units, the binary rate rides the training
# curve, and `--report` refuses to print a headline that is not
# "zero-policy baseline vs binary contact rate, over at least two runs".

# ==========================================================================
# The lane's own ABI, declared once so an auditor can read it without a GPU.
# ==========================================================================
#
# PLAIN LANGUAGE.  These three tuples are what this lane's policy sees, what it
# is paid for, and what ends its episode.  They used to exist only as an
# anonymous `torch.cat([...])` inside `_compute_obs` and two string tuples
# buried in `__init__`, which meant the only way to answer "is this the same
# question Isaac is asking?" was to read the source and copy it out by hand --
# and a hand copy is what `isaac_alignment.py` exists to stop.  `_compute_obs`
# now BUILDS from `OBS_LAYOUT`, so the names below are load-bearing: reorder
# them and the observation really is reordered.

#: Ordered (name, width) actor rows.  This lane is symmetric -- the critic is
#: fed the same group (`obs_groups` in `build_agent_cfg`), which is itself a
#: named divergence from Isaac's asymmetric 211/319 pair.
OBS_LAYOUT = (
  ("base_lin_vel_body", 3),
  ("base_ang_vel_body", 3),
  ("projected_gravity", 3),
  ("joint_pos_rel_ready", 31),
  ("joint_vel_scaled", 31),
  ("actions", 31),
  ("ball_minus_racket_body", 3),
  ("ball_lin_vel_body_scaled", 3),
  ("ball_minus_pelvis_body", 3),
  ("racket_minus_pelvis_body", 3),
)
OBS_WIDTH = sum(width for _name, width in OBS_LAYOUT)   # 114

#: reward term -> which of the five Isaac ActionBall reward groups it belongs
#: to, or ``None`` when this lane has no term in that group at all.  The groups
#: are Isaac's own vocabulary (balance / mimic / strike / target / outcome);
#: writing this lane's terms in that vocabulary is what lets the alignment
#: ledger say "mimic coverage = 0 terms" as a machine fact rather than prose.
REWARD_TERM_GROUP = {
  "alive": "balance",
  "pose": "balance",
  "upright": "balance",
  "height": "balance",
  "reach_term_weighted": "strike_guidance",
  "touch_term_weighted": "strike_guidance",
  "action_rate": "regularizer",
  "joint_vel": "regularizer",
  "torque": "regularizer",
  "termination": "safety",
}
REWARD_TERMS = tuple(REWARD_TERM_GROUP)

#: Terminal reasons this lane implements, in the order `_terminate` resolves
#: them.  Truncation (`timeout_truncation`) is separate and is NOT in here.
TERMINATION_TERMS = ("fall_height", "fall_tilt", "nonfinite_state")

# The reward terms whose value is a *weight times a shaping kernel in [0, 1]*.
# Mapping: receipt key -> (kernel key, weight attribute on TaskCfg).
KERNEL_REWARD_TERMS = {
  "reach_term_weighted": ("reach_kernel_mean", "w_reach"),
  "touch_term_weighted": ("touch_kernel_mean", "w_touch"),
}

# The only contact metric with a physical meaning, and where it lives in a
# per-iteration record / an eval receipt's stats block.
BINARY_CONTACT_KEY = "contact.fraction_of_episodes_with_a_racket_touch"

_NOT_A_PROBABILITY = (
  "reward_terms_mean values are weighted per-step reward, NOT probabilities "
  "and NOT contact rates; divide by reward_terms_max_possible to get the "
  "shaping kernel, and use "
  + BINARY_CONTACT_KEY
  + " for anything that claims the racket touched the ball")


def reward_term_ceilings(cfg) -> dict:
  """Per-step ceiling of every reward term, straight off the config.

  Pure, so the receipt can say what "0.21" is 0.21 *of* without anyone having
  to go read the weights out of the source.  Penalty terms (action_rate,
  joint_vel, torque, termination) are <= 0 by construction, so their ceiling is
  0.0 and they have no lower bound worth printing.
  """
  return {
    "alive": float(cfg.w_alive),
    "pose": float(cfg.w_pose),
    "upright": float(cfg.w_upright),
    "height": float(cfg.w_height),
    "reach_term_weighted": float(cfg.w_reach),
    "touch_term_weighted": float(cfg.w_touch),
    "action_rate": 0.0,
    "joint_vel": 0.0,
    "torque": 0.0,
    "termination": 0.0,
  }


def _assert_reward_registry_agrees() -> None:
  """`REWARD_TERM_GROUP` must name exactly the terms the receipt prices.

  Import-time on purpose.  Adding a reward term and forgetting to say which
  Isaac group it belongs to is the exact failure the alignment ledger cannot
  see from the outside: the ledger would keep reporting "mimic coverage = 0"
  while a mimic term quietly existed.
  """
  priced = set(reward_term_ceilings(TaskCfg()))
  declared = set(REWARD_TERM_GROUP)
  if priced != declared:
    raise RuntimeError(
      "REWARD_TERM_GROUP and reward_term_ceilings disagree: "
      f"only_priced={sorted(priced - declared)} "
      f"only_declared={sorted(declared - priced)}")


_assert_reward_registry_agrees()


def reward_term_report(weighted_means: dict, cfg) -> dict:
  """Receipt block for the reward terms: value, ceiling, and kernel mean.

  Pure.  ``weighted_means`` is what the accumulators produce (weight already
  applied).  The kernel means are the same numbers divided by their weight, so
  a reader who wants "how close to the ceiling is this term" does not have to
  know the weight -- and a reader who mistakes either one for a contact rate is
  told in the receipt itself that it is not one.
  """
  ceilings = reward_term_ceilings(cfg)
  kernels = {}
  for term, (kernel_key, weight_attr) in KERNEL_REWARD_TERMS.items():
    w = float(getattr(cfg, weight_attr, 0.0))
    v = weighted_means.get(term)
    kernels[kernel_key] = (float(v) / w) if (v is not None and w) else None
  return {
    "reward_terms_mean": {k: float(v) for k, v in weighted_means.items()},
    "reward_terms_max_possible": {k: ceilings[k] for k in weighted_means
                                  if k in ceilings},
    "reward_kernel_mean": kernels,
    "reward_terms_are_weighted_not_probabilities": True,
    "reward_terms_note": _NOT_A_PROBABILITY,
  }


def binary_contact_fields(*, probe_on: bool, touched_episodes: float,
                          episodes_finished: float,
                          racket_substeps: float = 0.0,
                          table_substeps: float = 0.0) -> dict:
  """The binary "did the racket touch the ball this episode" block.  Pure.

  Two ways this can be un-measured, and neither may look like a measured zero:

  * the probe was switched off -> ``probe: "OFF"``, fraction ``None``;
  * the probe was on but no episode finished inside this window (short eval,
    tiny iteration) -> ``probe: "ON"``, fraction ``None``, reason
    ``NO_EPISODES_FINISHED``.

  The old code divided by ``max(episodes, 1)``, so "nothing to divide" printed
  as ``0.0`` -- a perfect zero contact rate, indistinguishable from a policy
  that genuinely never touched the ball.  Same defect class as the capacity
  gate's zero-sample PASS.
  """
  if not probe_on:
    return {
      "probe": "OFF",
      "fraction_of_episodes_with_a_racket_touch": None,
      "measured": False,
      "reason": "CONTACT_PROBE_OFF",
      "note": "--no-contact-probe was passed: this run cannot support any "
              "claim about whether the racket touched the ball",
    }
  eps = float(episodes_finished)
  if eps <= 0:
    return {
      "probe": "ON",
      "ball_racket_contact_substeps": float(racket_substeps),
      "ball_table_contact_substeps": float(table_substeps),
      "episodes_with_a_racket_touch": float(touched_episodes),
      "episodes_finished": 0.0,
      "fraction_of_episodes_with_a_racket_touch": None,
      "measured": False,
      "reason": "NO_EPISODES_FINISHED",
      "note": "no episode ended inside this window, so the per-episode rate "
              "has no denominator; this is not a contact rate of zero",
    }
  return {
    "probe": "ON",
    "ball_racket_contact_substeps": float(racket_substeps),
    "ball_table_contact_substeps": float(table_substeps),
    "episodes_with_a_racket_touch": float(touched_episodes),
    "episodes_finished": eps,
    "fraction_of_episodes_with_a_racket_touch": float(touched_episodes) / eps,
    "measured": True,
    "denominator": "episodes that ENDED inside this window",
  }


#: Isaac's ActionBall run ends the episode on robot-vs-table contact
#: (`robot_hit_table`, charged in the hard-safety union).  This lane has the
#: table but no such guard.  Until it does, a run in which the robot touched
#: the table is not a clean learning result -- it may have been paid for
#: balance it bought by leaning on furniture that Isaac calls fatal.
ROBOT_TABLE_CONTACT_KEY = "robot_table.fraction_of_episodes_with_a_robot_table_contact"


def robot_table_contact_fields(*, probe_on: bool, touched_episodes: float,
                               episodes_finished: float, substeps: float = 0.0,
                               n_robot_geoms: int = 0) -> dict:
  """The per-episode robot-vs-table contact block.  Pure.

  Same "not measured is not zero" discipline as the ball block: an unmeasured
  channel reports ``null``, never ``0.0``.  A missing denominator is the state
  in which somebody reads "0" as "the robot never touched the table".
  """
  if not probe_on:
    return {"probe": "OFF", "measured": False,
            "fraction_of_episodes_with_a_robot_table_contact": None,
            "reason": "no table geom in this scene, or the contact probe is off",
            "isaac_twin": "robot_hit_table (terminal in the Isaac ActionBall run)"}
  eps = float(episodes_finished)
  if eps <= 0.0:
    return {"probe": "ON", "measured": False,
            "fraction_of_episodes_with_a_robot_table_contact": None,
            "robot_table_contact_substeps": float(substeps),
            "n_robot_collision_geoms": int(n_robot_geoms),
            "reason": "no episode ended inside this window, so the rate has no "
                      "denominator; this is not a contact rate of zero",
            "isaac_twin": "robot_hit_table (terminal in the Isaac ActionBall run)"}
  return {
    "probe": "ON",
    "measured": True,
    "fraction_of_episodes_with_a_robot_table_contact": float(touched_episodes) / eps,
    "episodes_with_a_robot_table_contact": float(touched_episodes),
    "robot_table_contact_substeps": float(substeps),
    "episodes_finished": eps,
    "n_robot_collision_geoms": int(n_robot_geoms),
    "terminal_here": False,
    "isaac_twin": "robot_hit_table (terminal in the Isaac ActionBall run)",
    "note": "本车道不因此终止,Isaac 会。非零就意味着这条曲线里含有 Isaac 判死的行为,"
            "--report 会拒绝把它当学习结果。",
  }


def alignment_receipt_block() -> dict:
  """The compact `isaac_alignment` block every receipt carries.

  PLAIN LANGUAGE.  A curve produced by this lane is a statement about THIS
  lane.  Whether it is also a statement about the Isaac A211/C211 run depends
  on whether the two are asking the same question, and that is not something a
  reader should have to reconstruct from two source trees at 2am.  So every
  receipt carries the verdict, computed live at write time from both sides:
  observation ABI, action decoder, termination union, reward groups, episode
  shape, question distribution, and the rest (`isaac_alignment.py`).

  Never raises.  A receipt that cannot resolve the ledger says so in the
  receipt -- `available: false` plus the reason -- because a missing block and
  a clean block must not look the same.
  """
  try:
    import isaac_alignment as align

    ledger = align.build_ledger()
    return {
      "available": True,
      "kind": ledger["kind"],
      "ledger_sha256": ledger["ledger_sha256"],
      "isaac_repo_root": ledger["isaac_repo_root"],
      "verdict_counts": ledger["verdict_counts"],
      "blocking_axes": ledger["blocking_axes"],
      "cross_engine_comparable": ledger["cross_engine_comparable"],
      "bitwise_parity_is_never_a_valid_acceptance": True,
      "rows": {k: {"declared": r["declared"], "observed": r["observed"],
                   "human": r["human"]}
               for k, r in ledger["rows"].items()},
      "scope_sentence": (
        "这条 run 的数字是**本车道内部**的陈述。只要 blocking_axes 非空,"
        "它就不是 Isaac A211/C211 的结果,也不能拿去跟 Isaac 的曲线并排读。"),
    }
  except BaseException as exc:  # noqa: BLE001 -- receipts must survive this
    return {"available": False, "error": repr(exc),
            "cross_engine_comparable": False,
            "scope_sentence": (
              "对齐台账没算出来,所以这条 run 与 Isaac 的关系是**未知**,"
              "不是'对齐'。")}


def _dig(obj, dotted: str):
  """Follow a dotted path through nested dicts; ``None`` if anything is missing."""
  cur = obj
  for part in dotted.split("."):
    if not isinstance(cur, dict) or part not in cur:
      return None
    cur = cur[part]
  return cur


def _binary_contact_curve(records) -> list:
  """Per-iteration binary contact rate, ``None`` where it was not measured."""
  return [_dig(r, BINARY_CONTACT_KEY) for r in records]


def _band(values) -> dict:
  """min / mean / max over runs, ignoring the ones that measured nothing."""
  seen = [float(v) for v in values if v is not None]
  if not seen:
    return {"n": 0, "lo": None, "mean": None, "hi": None, "spread": None}
  return {"n": len(seen), "lo": min(seen), "mean": float(np.mean(seen)),
          "hi": max(seen), "spread": max(seen) - min(seen)}


def _report_alignment_scope(runs: list, evals: list | None = None) -> dict:
  """Per-receipt cross-engine scope, from each receipt's own ledger.  Pure."""
  per = {}
  for name, r in list(runs) + list(evals or []):
    block = r.get("isaac_alignment")
    if not isinstance(block, dict) or not block:
      per[name] = {"ledger": None,
                   "note": "receipt predates the alignment ledger; its "
                           "relationship to the Isaac run is UNRECORDED, "
                           "which is not the same as aligned"}
      continue
    per[name] = {"ledger": block.get("ledger_sha256"),
                 "available": block.get("available"),
                 "cross_engine_comparable": block.get("cross_engine_comparable"),
                 "blocking_axes": block.get("blocking_axes")}
  any_comparable = any(v.get("cross_engine_comparable") is True
                       for v in per.values())
  return {
    "per_receipt": per,
    "every_receipt_is_cross_engine_comparable": (
      bool(per) and all(v.get("cross_engine_comparable") is True
                        for v in per.values())),
    "any_receipt_claims_comparability": any_comparable,
    "sentence": (
      "本报告是**本车道内部**的陈述:'接触率从 X 涨到 Y'说的是这条 mjlab "
      "车道的球拍碰没碰到球。只要 per_receipt 里还有 blocking_axes,它就不是 "
      "Isaac A211/C211 的结果。跨引擎对拍在任何情况下都只能是统计口径 —— "
      "mujoco-warp 无 CPU 回退且实测非确定性,逐位一致不是更严的标准,是错的标准。"),
  }


class ReportRefused(RuntimeError):
  """``--report`` was asked to print a headline the evidence does not support."""


def report_refusals(runs: list, baseline, evals: list | None = None) -> list:
  """Every reason this set of receipts may NOT be reported.  Pure, fail-closed.

  Each reason is ``(CODE, plain-language sentence)``.  The codes exist so a
  mutation test can assert *which* rule fired, not merely that something did.

  ``evals`` is optional and holds one ``--eval ckpt`` receipt per run: the
  deterministic-policy measurement, which is what the 0.12% -> 49.2%/97.8%
  headline actually came from.  Given, it is checked as strictly as the rest;
  omitted, the report falls back to the on-policy training curve and says so.
  """
  out = []
  if len(runs) < 2:
    out.append((
      "SINGLE_SEED_NOT_EVIDENCE",
      f"only {len(runs)} run given.  mujoco-warp is non-deterministic and this "
      "config measurably swings ~3x between identical runs (touch 0.21 / 0.46 "
      "/ 0.59 / 0.61 on 2026-08-06), so one run is a sample, not a result.  "
      "Give at least two."))
  for name, r in runs:
    if r.get("status") != "completed":
      out.append(("RUN_DID_NOT_COMPLETE",
                  f"{name}: status={r.get('status')!r}; a run that stopped "
                  "early -- or a pre-2026-08-06 receipt that does not say "
                  "whether it did -- cannot be quoted as a learning result"))
    verdict = _dig(r, "capacity.verdict")
    if verdict != "PASS_NO_OVERFLOW":
      out.append(("RUN_HAS_NO_CAPACITY_PASS",
                  f"{name}: capacity verdict={verdict!r}.  The physics behind "
                  "the curve is only trustworthy when the capacity gate saw "
                  "samples and every overflow bit stayed clear"))
    if _dig(r, "learning.binary_contact_rate.measured") is not True:
      out.append(("NO_BINARY_CONTACT_RATE",
                  f"{name}: no binary contact rate on the training curve "
                  "(pre-2026-08-06 receipt, or --no-contact-probe).  The "
                  "weighted `touch` term is not a substitute -- that swap is "
                  "the whole reason this gate exists"))
  # The robot-vs-table channel.  Isaac's ActionBall run treats robot-table
  # contact as terminal, in the same class as falling over; this lane has the
  # table and no such guard, so a curve in which the robot touched it contains
  # behaviour the reference run would have killed.  Installing the termination
  # is a launch decision.  Refusing to report a contaminated curve is not.
  for name, r in runs:
    peak = _dig(r, "learning.robot_table_contact.peak_fraction_of_episodes")
    if peak is None:
      out.append((
        "ROBOT_TABLE_CONTACT_NOT_MEASURED",
        f"{name}: no robot-vs-table contact rate (pre-2026-08-06 receipt, "
        "--no-contact-probe, or a scene with no table).  Isaac terminates on "
        "this; unmeasured is not the same as zero"))
    elif float(peak) > 0.0:
      out.append((
        "ROBOT_LEANED_ON_THE_TABLE",
        f"{name}: the robot touched the table in up to {100.0 * float(peak):.2f}% "
        "of episodes.  Isaac's ActionBall run ends the episode on exactly that "
        "contact, so this curve contains behaviour the reference run forbids "
        "and cannot be quoted as a learning result until the guard is installed "
        "or the run is re-declared"))

  # Cross-engine authority.  This does NOT refuse a within-lane report -- the
  # contact rate versus the zero policy is a perfectly good statement about
  # this lane and predates the ledger, so demanding a ledger from every receipt
  # would break honest old evidence for no gain.  What it refuses is a receipt
  # that ASSERTS it is comparable to the Isaac A211/C211 run while its own
  # alignment ledger lists open blocking axes.  That combination is not a
  # judgement call; it is a receipt contradicting itself.
  for name, r in list(runs) + list(evals or []):
    block = r.get("isaac_alignment")
    if not isinstance(block, dict):
      continue
    blocking = block.get("blocking_axes") or []
    if block.get("cross_engine_comparable") and blocking:
      out.append((
        "CLAIMS_ISAAC_COMPARABILITY_WITHOUT_EARNING_IT",
        f"{name}: the receipt says cross_engine_comparable=true while its own "
        f"alignment ledger still lists blocking axes {blocking}.  A number "
        "from this lane is a statement about this lane until every one of "
        "those is closed"))
  for name, e in (evals or []):
    if e.get("mode") != "ckpt":
      out.append(("EVAL_IS_NOT_A_CHECKPOINT_RUN",
                  f"{name}: mode={e.get('mode')!r}, not 'ckpt'; this slot is "
                  "for the trained policy's deterministic evaluation"))
    if e.get("status") != "completed" or (
        _dig(e, "capacity.verdict") != "PASS_NO_OVERFLOW"):
      out.append(("EVAL_DID_NOT_COMPLETE_OR_PASS",
                  f"{name}: status={e.get('status')!r}, capacity verdict="
                  f"{_dig(e, 'capacity.verdict')!r}"))
    if _dig(e, "stats.contact.fraction_of_episodes_with_a_racket_touch") is None:
      out.append(("EVAL_HAS_NO_BINARY_CONTACT_RATE",
                  f"{name}: no measured binary contact rate in the eval "
                  "receipt (--no-contact-probe, or too short a window for any "
                  "episode to end)"))
  if evals and len(evals) != len(runs):
    out.append(("EVAL_COUNT_DOES_NOT_MATCH_RUNS",
                f"{len(evals)} eval receipt(s) for {len(runs)} run(s).  Pair "
                "them one-to-one or leave --report-eval off entirely; a "
                "partial pairing is how one good seed ends up standing in for "
                "the band"))
  if baseline is None:
    out.append((
      "NO_ZERO_POLICY_BASELINE",
      "no --report-zero-policy receipt.  'the contact rate went up' means "
      "nothing without the do-nothing policy measured on the same scene: it "
      "is 0.12% here, so 49.2% is 400x, not noise"))
    return out
  bname, b = baseline
  if b.get("mode") != "zero":
    out.append(("BASELINE_IS_NOT_A_ZERO_POLICY_RUN",
                f"{bname}: mode={b.get('mode')!r}, not 'zero'.  A trained "
                "checkpoint cannot be its own baseline"))
  if b.get("status") != "completed" or (
      _dig(b, "capacity.verdict") != "PASS_NO_OVERFLOW"):
    out.append(("BASELINE_DID_NOT_COMPLETE_OR_PASS",
                f"{bname}: status={b.get('status')!r}, capacity verdict="
                f"{_dig(b, 'capacity.verdict')!r}.  The baseline is held to "
                "the same standard as the runs it anchors"))
  if _dig(b, "stats.contact.fraction_of_episodes_with_a_racket_touch") is None:
    out.append(("BASELINE_HAS_NO_BINARY_CONTACT_RATE",
                f"{bname}: the baseline receipt has no measured binary contact "
                "rate, so there is nothing to compare against"))
  return out


# ==========================================================================
# The environment.
# ==========================================================================


class A3ReadyBallVecEnv:
  """rsl-rl ``VecEnv`` over the A3 court scene.

  Registered as a duck-typed VecEnv rather than a subclass so that importing
  this module never depends on rsl_rl being importable (the --analyze path
  runs anywhere).
  """

  def __init__(self, sim_cfg: SimCfg, task_cfg: TaskCfg, device: str,
               xml_path: Path | None = None,
               ready_pose_path: Path | None = None,
               seed: int = 0, count_contacts: bool = False,
               capacity_probe: bool = True,
               ready_pose_payload: bytes | None = None,
               ready_pose_source: str | None = None) -> None:
    import mujoco
    import torch

    if task_cfg.action_clip is not None:
      raise ValueError(
        "raw policy action clipping is unsupported: active Isaac uses "
        "clip_actions=null; constrain only the decoded q_des envelope")
    if (ready_pose_payload is None) != (ready_pose_source is None) or (
        ready_pose_payload is not None and ready_pose_path is not None):
      raise ValueError("ready-pose bytes require one exclusive source pair")
    if ready_pose_payload is not None:
      pose_payload = ready_pose_payload
      pose_source = ready_pose_source
    elif ready_pose_path is not None:
      rp = Path(ready_pose_path)
      if not rp.is_file():
        raise FileNotFoundError(f"explicit ready pose does not exist: {rp}")
      pose_payload = rp.read_bytes()
      pose_source = str(rp)
    else:
      rp = _HERE / "ready_pose.json"
      if not rp.is_file():
        rp = Path("/workspace/mjlab_lane/ready_pose.json")
      pose_payload = rp.read_bytes()
      pose_source = str(rp)
    pose = court.load_ready_pose_bytes(pose_payload, pose_source)

    self.cfg = task_cfg
    self.sim_cfg = sim_cfg
    self.device = torch.device(device)
    self._torch = torch

    xml = xml_path or plant.default_xml()
    self.env = court.build_court_env(
      xml_path=xml,
      num_envs=sim_cfg.nworld,
      device=device,
      ball_pos_hope=sim_cfg.ball_spawn_hope,
      cone=sim_cfg.cone,
      add_pairs=sim_cfg.add_pairs,
      njmax=sim_cfg.njmax,
      nconmax=sim_cfg.nconmax,
    )
    self.sim = self.env.sim
    m = self.env.mj_model
    self.mj_model = m
    self.num_envs = int(self.sim.num_envs)
    self.decimation = int(self.env.decimation)
    self.step_dt = float(self.env.step_dt)
    self.physics_dt = float(self.env.physics_dt)
    self.max_episode_length = int(round(task_cfg.episode_length_s / self.step_dt))

    # ---- ready pose --------------------------------------------------
    self.pose = pose
    qpos0, qvel0, idx = court.ready_qpos(self.env, self.pose)
    self.root_qadr = int(idx["root_qadr"])
    self.row_map_agrees = bool(idx["consistency"]["agree"])

    # root dof address: the free joint's dof address.
    root_jid = int(np.argmin(np.where(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE,
                                      np.arange(m.njnt), m.njnt)))
    self.root_vadr = int(m.jnt_dofadr[root_jid])

    # ---- actuator wiring (vendor PD, computed outside the plant) ------
    kp_actuator, kd_actuator, q_adr_actuator, v_adr_actuator = (
      plant._pd_wiring(self.env)
    )
    runtime_from_actuator, actuator_from_runtime, action_joint_names = (
      _runtime_action_wiring(m, self.env.entity_prefix)
    )
    if list(pose["joint_names"]) != list(action_joint_names):
      raise RuntimeError("ready pose is not in the runtime action joint order")
    action_offset_np, action_offset_sha256 = _runtime_action_offset(
      pose_payload, action_joint_names)
    actuator_rows = tuple(np.asarray(row) for row in (
      kp_actuator, kd_actuator, q_adr_actuator, v_adr_actuator))
    if any(row.shape != (31,) for row in actuator_rows):
      raise RuntimeError("MuJoCo PD wiring is not one scalar per actuator")
    kp_actuator, kd_actuator, q_adr_actuator, v_adr_actuator = actuator_rows
    kp_np = kp_actuator[runtime_from_actuator]
    kd_np = kd_actuator[runtime_from_actuator]
    q_adr_act = q_adr_actuator[runtime_from_actuator]
    v_adr_act = v_adr_actuator[runtime_from_actuator]
    self.num_actions = int(m.nu)
    jnt_of_act = m.actuator_trnid[:, 0].astype(int)[runtime_from_actuator]
    jrange = m.jnt_range[jnt_of_act]

    T = lambda x, dt=torch.float32: torch.as_tensor(  # noqa: E731
      np.asarray(x), dtype=dt, device=self.device)
    self.kp = T(kp_np)
    self.kd = T(kd_np)
    self.q_adr_act = T(q_adr_act, torch.long)
    self.v_adr_act = T(v_adr_act, torch.long)
    self.actuator_from_runtime = T(actuator_from_runtime, torch.long)
    self._action_joint_names = action_joint_names
    self._action_offset_sha256 = action_offset_sha256
    # Contiguity lets us use slices instead of gathers in the 1 kHz inner loop.
    self._q_slice = (slice(int(q_adr_act[0]), int(q_adr_act[0]) + len(q_adr_act))
                     if np.all(np.diff(q_adr_act) == 1) else None)
    self._v_slice = (slice(int(v_adr_act[0]), int(v_adr_act[0]) + len(v_adr_act))
                     if np.all(np.diff(v_adr_act) == 1) else None)
    self.tau_lo = T(m.actuator_ctrlrange[runtime_from_actuator, 0])
    self.tau_hi = T(m.actuator_ctrlrange[runtime_from_actuator, 1])
    self.tau_scale = torch.maximum(self.tau_hi.abs(), self.tau_lo.abs())
    self.jnt_lo = T(jrange[:, 0])
    self.jnt_hi = T(jrange[:, 1])
    self.q_ready = T(qpos0[q_adr_act])
    self.action_offset = T(action_offset_np)
    if bool(((self.action_offset < self.jnt_lo)
             | (self.action_offset > self.jnt_hi)).any()):
      raise RuntimeError("runtime action offset is outside MuJoCo hard limits")
    if task_cfg.action_scale_mode == "vendor":
      # AGIBOT_A3_ACTION_SCALE = 0.25 * effort_limit_sim / stiffness, per joint.
      # effort_limit_sim is bit-identical to the MJCF ctrlrange (verified in the
      # plant step), and the kp here is the same VENDOR_KP table, so this is the
      # Isaac decoder reproduced from the compiled model rather than copied.
      self.act_scale = 0.25 * self.tau_hi / self.kp
    else:
      self.act_scale = torch.full_like(self.kp, float(task_cfg.action_scale))

    # ---- ball wiring ---------------------------------------------------
    bj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, court.BALL_JOINT)
    assert bj >= 0, "ball free joint missing from the court model"
    self.b_q = int(m.jnt_qposadr[bj])
    self.b_v = int(m.jnt_dofadr[bj])
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "robot/right_racket")
    if sid < 0:
      sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_racket")
    assert sid >= 0, "right_racket site missing from the court model"
    self.racket_sid = int(sid)

    # ---- base reset state ----------------------------------------------
    self.qpos_init = T(qpos0)               # (nq,)
    self.qvel_init = T(qvel0)
    self.hope_to_scene = T(np.asarray(court.HOPE_TO_SCENE))
    self.ready_pelvis_z = float(qpos0[self.root_qadr + 2])

    # ---- serve recipe (validated narrow returner serve) -----------------
    sc = geom.ServeConfig.reachable_returner()
    self.serve_cfg = sc
    self.serve_pos_lo = T([sc.pos_x_range[0], sc.pos_y_range[0], sc.pos_z_range[0]])
    self.serve_pos_hi = T([sc.pos_x_range[1], sc.pos_y_range[1], sc.pos_z_range[1]])
    self.serve_vel_lo = T([sc.vel_x_range[0], sc.vel_y_range[0], sc.vel_z_range[0]])
    self.serve_vel_hi = T([sc.vel_x_range[1], sc.vel_y_range[1], sc.vel_z_range[1]])

    # ---- buffers ---------------------------------------------------------
    N = self.num_envs
    self.episode_length_buf = torch.zeros(N, dtype=torch.long, device=self.device)
    self.ball_age_buf = torch.zeros(N, dtype=torch.long, device=self.device)
    self.actions = torch.zeros(N, self.num_actions, device=self.device)
    self.last_actions = torch.zeros_like(self.actions)
    self.action_nonfinite_buf = torch.zeros(
      N, dtype=torch.bool, device=self.device)
    self._qdes_previous_executable = self.action_offset.unsqueeze(0).expand(
      N, -1).clone()
    self._qdes_previous_executable_valid = torch.ones(
      N, dtype=torch.bool, device=self.device)
    self._qdes_guard_terminal = torch.zeros(
      N, dtype=torch.bool, device=self.device)
    self.gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(N, 1)
    self.common_step_counter = 0
    self._ball_reserve_steps = int(round(task_cfg.ball_reserve_after_s / self.step_dt))

    # statistics accumulators (GPU-side; one sync per iteration)
    self._acc = {k: torch.zeros((), device=self.device) for k in (
      "ep_ret_sum", "ep_len_sum", "ep_cnt", "ep_min_d_sum",
      "term_fall_h", "term_tilt", "term_nonfinite", "term_timeout",
      "steps", "rew_sum", "reserves",
    )}
    # The two shaping terms carry `_term_weighted` in their names on purpose.
    # They used to be called `reach` and `touch`, and a receipt that says
    # `touch: 0.21` next to `reach: 0.98` reads like two percentages -- it is
    # in fact `4.0 * exp(-(d/0.15)^2)` and `2.0 * exp(-d/0.8)`, ceilings 4.0
    # and 2.0.  The name now says which one it is.
    self._rew_terms = REWARD_TERMS
    for k in self._rew_terms:
      self._acc["r_" + k] = torch.zeros((), device=self.device)
    self._cur_ret = torch.zeros(N, device=self.device)
    self._cur_min_d = torch.full((N,), 1e3, device=self.device)

    self.generator = torch.Generator(device=self.device)
    self.generator.manual_seed(int(seed))

    # Per-substep contact probe.  ON by default everywhere now, training
    # included -- it used to be wired only into `--eval`, which is why the
    # training curve had no honest contact metric on it and the weighted
    # `touch` reward term got quoted as one instead.  `--no-contact-probe`
    # switches it off, and a run that does so records
    # `fraction_of_episodes_with_a_racket_touch: null` rather than 0.
    self.count_contacts = bool(count_contacts)
    self._contact_ok = False
    self._robot_table_ok = False
    if self.count_contacts:
      self._setup_contact_probe(mujoco)

    # Constraint/contact capacity gate.  ON by default, including during the
    # timed runs.  When a world needs more constraint rows than `njmax`,
    # mujoco-warp drops the surplus rows and carries on (the a3_plant_env.py
    # header records what that costs: the warp heuristic njmax=64 put ~96% of
    # 4096 worlds non-finite), and an over-full broadphase array overruns into
    # a CUDA illegal access.  The standalone census sized both caps under a
    # zero-ctrl sprawl of the bare-plus-court scene; nothing re-checked them
    # once a *learning* policy, per-env reset randomisation and live ball
    # serves were driving the scene.  So the check rides inside the training
    # loop.  What it reads is the engine's own `d.overflow` bitmask -- see the
    # "capacity gate" section at the top of this file for why we no longer
    # re-derive the answer from `nefc`/`nacon` ourselves.
    self.capacity_probe = bool(capacity_probe)
    self._cap_ok = False
    self.njmax_alloc = int(getattr(self.sim.wp_data, "njmax", -1))
    self.naconmax_alloc = int(getattr(self.sim.wp_data, "naconmax", -1))
    if self.capacity_probe:
      self._setup_capacity_probe()

    # first full reset
    self.reset()
    self.num_obs = int(self._obs_buf.shape[1])

  # ---- rsl-rl VecEnv surface ------------------------------------------

  @property
  def action_contract_identity(self) -> dict:
    """Return one immutable-by-copy action identity for completion evidence."""
    return {
      "action_joint_order_contract_id": ACTION_JOINT_ORDER_CONTRACT_ID,
      "action_joint_order_contract_sha256": ACTION_JOINT_ORDER_CONTRACT_SHA256,
      "action_offset_source": ACTION_OFFSET_SOURCE,
      "action_offset_sha256": self._action_offset_sha256,
      "raw_action_clip": None,
      "executable_qdes_guard": EXECUTABLE_QDES_GUARD,
      "transfer_authority": False,
      "matched_cross_backend_authority": False,
    }

  @property
  def unwrapped(self):
    return self

  def get_observations(self):
    from tensordict import TensorDict

    return TensorDict({"policy": self._obs_buf}, batch_size=[self.num_envs])

  def close(self):
    return None

  # ---- internals --------------------------------------------------------

  def _setup_contact_probe(self, mujoco):
    """Wire a sync-free ball<->racket contact counter over mjwarp's contact array.

    Fail-closed: if the caller asked for the contact metric and we cannot wire
    it, the run stops.  It used to print one line and carry on with
    ``_contact_ok = False``, which produced a receipt with no contact block at
    all -- and "no contact block" is exactly the state in which somebody
    reaches for the weighted ``touch`` term instead.
    """
    torch = self._torch
    m = self.mj_model
    d = self.sim.data
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
    ball = gid(court.BALL_GEOM)
    rackets = [gid(n) for n in court.RACKET_GEOMS]
    table = gid("court_table_top")
    if ball < 0 or any(r < 0 for r in rackets):
      raise RuntimeError(
        "CONTACT_PROBE_UNAVAILABLE: the ball/racket geoms are not in this "
        f"model (ball={ball}, rackets={rackets}), so the only physically "
        "meaningful contact metric cannot be measured.  Refusing to run a "
        "job that would silently report no contact data.  Pass "
        "--no-contact-probe if you deliberately want a run that records "
        "fraction_of_episodes_with_a_racket_touch = null.")
    try:
      contact = d.contact
      geom = contact.geom
      worldid = contact.worldid
      nacon = getattr(d, "nacon")
      self._con_geom, self._con_world = geom, worldid
      self._nacon = nacon
      self._naconmax = int(geom.shape[0])
      self._con_idx = torch.arange(self._naconmax, device=self.device)
    except Exception as exc:
      raise RuntimeError(
        "CONTACT_PROBE_UNAVAILABLE: mujoco-warp did not expose the contact "
        f"array this build needs ({exc!r}).  Refusing to run unmeasured; "
        "pass --no-contact-probe to record that on purpose.") from exc
    self._ball_gid = int(ball)
    # One small lookup table over geom ids (0 = don't care, 1 = racket,
    # 2 = table top) instead of comparing every contact row against every
    # racket geom.  Fewer kernels, same predicate.
    #
    # HONEST COST NOTE (paired 4096x12 runs, same card, 2026-08-06): this probe
    # costs ~13% throughput and the table did NOT change that -- 38,816 vs
    # 38,904 env-step/s before and after, against 44,987 / 44,735 with the
    # probe off.  The cost is inherent: it is a pass over the whole
    # pre-allocated contact array (524,288 rows at 4096 worlds x nconmax 128)
    # once per PHYSICS substep, and it has to be per-substep because a
    # racket-ball contact only lasts one or two of them.  We pay it because a
    # training curve without the binary contact rate is a curve nobody can read
    # honestly -- that is what T11 is about.  `--no-contact-probe` exists for
    # timing runs and makes the receipt say so.
    ngeom = int(m.ngeom)
    cls = torch.zeros(ngeom, dtype=torch.int8, device=self.device)
    for r in rackets:
      cls[int(r)] = 1
    if table >= 0:
      cls[int(table)] = 2
    self._geom_class = cls
    self._contact_ok = True
    self._acc["contact_ball_racket_substeps"] = torch.zeros((), device=self.device)
    self._acc["contact_ball_table_substeps"] = torch.zeros((), device=self.device)
    self._acc["ep_touched_racket"] = torch.zeros((), device=self.device)
    self._cur_touched = torch.zeros(self.num_envs, device=self.device)

    # ---- the ROBOT-vs-table channel -----------------------------------
    #
    # PLAIN LANGUAGE.  Isaac's ActionBall run ends the episode when the robot
    # touches the table (`robot_hit_table`, same class as falling over).  This
    # lane has a table in the scene, lets the robot collide with it, and has no
    # such guard -- so a policy here may lean on the table and be paid for the
    # balance it buys.  Installing a hard termination is a launch decision, not
    # a review edit; measuring it is not.  So this counts it, and
    # `report_refusals` refuses to report a run whose robot ever did it.  That
    # keeps the record and the block in the same change: a counter with no
    # consequence is a counter nobody reads.
    self._table_gid = int(table) if table >= 0 else -1
    is_robot = torch.zeros(ngeom, dtype=torch.bool, device=self.device)
    prefix = self.env.entity_prefix
    floor = gid(court.FLOOR_GEOM)
    n_robot_geoms = 0
    for g_id in range(ngeom):
      name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g_id) or ""
      if name.startswith(prefix) and g_id != floor:
        is_robot[g_id] = True
        n_robot_geoms += 1
    self._is_robot_geom = is_robot
    self._n_robot_geoms = n_robot_geoms
    self._robot_table_ok = (self._table_gid >= 0 and n_robot_geoms > 0)
    if self._robot_table_ok:
      self._acc["contact_robot_table_substeps"] = torch.zeros((), device=self.device)
      self._acc["ep_robot_touched_table"] = torch.zeros((), device=self.device)
      self._cur_robot_table = torch.zeros(self.num_envs, device=self.device)

  def _setup_capacity_probe(self) -> None:
    """Wire the engine's own overflow flags (the gate) plus reporting peaks.

    The gate is `d.overflow`: one integer per world, one bit per overflow kind,
    written by mujoco-warp itself, sticky until that world is reset.  The
    `nefc` / `nacon` / `ncollision` running maxima that used to *be* the gate
    are kept, but they are reporting only now -- they feed the headroom numbers
    in the receipt and nothing else.
    """
    torch = self._torch
    d = self.sim.data
    self._overflow_arr = getattr(d, "overflow", None)
    self._nefc_arr = getattr(d, "nefc", None)
    self._nacon_arr = getattr(d, "nacon", None)
    self._ncollision_arr = getattr(d, "ncollision", None)
    if self._overflow_arr is None:
      raise RuntimeError(
        "CAPACITY_GATE_UNAVAILABLE: this mujoco-warp build does not expose "
        "d.overflow, so the capacity gate cannot be wired at all.  Refusing "
        "to run unguarded.  Pass --no-capacity-probe if you deliberately want "
        "a run that records verdict=NOT_MEASURED and cannot claim the gate held.")
    if self._nefc_arr is None or self._nacon_arr is None:
      raise RuntimeError(
        "CAPACITY_GATE_UNAVAILABLE: d.nefc / d.nacon are not exposed, so the "
        "headroom half of the capacity receipt cannot be filled in.")
    if self.njmax_alloc <= 0 or self.naconmax_alloc <= 0:
      raise RuntimeError(
        "CAPACITY_GATE_UNAVAILABLE: njmax/naconmax allocation is not readable "
        f"(njmax={self.njmax_alloc}, naconmax={self.naconmax_alloc}).")
    dev = self.device
    N = self.num_envs
    # Sticky per-world OR of every overflow bit the engine ever set, kept for
    # the whole run.  Deliberately NOT cleared on reset: mjwarp.reset_data
    # zeroes the engine's own copy, and losing the evidence with it is exactly
    # the failure mode this is here to prevent.
    self._ov_acc = torch.zeros(N, dtype=torch.int32, device=dev)
    self._ov_shifts = torch.arange(len(OVERFLOW_FLAGS), dtype=torch.int32,
                                   device=dev)
    self._peak_nefc = torch.zeros(N, dtype=torch.int32, device=dev)
    self._peak_nacon = torch.zeros(1, dtype=torch.int32, device=dev)
    self._peak_ncollision = torch.zeros(1, dtype=torch.int32, device=dev)
    self._njmax_t = torch.tensor(self.njmax_alloc, dtype=torch.int32, device=dev)
    self._naconmax_t = torch.tensor(self.naconmax_alloc, dtype=torch.int32,
                                    device=dev)
    # Two counters, because "the probe ran" and "the physics actually moved"
    # are different claims.  Only stepped samples can support a PASS.
    self._cap_samples_step = 0
    self._cap_samples_forward = 0
    self._cap_flags_seen = 0
    self._cap_ok = True

  def _probe_capacity(self, kind: str = "step"):
    """One elementwise pass over the engine's counters.  No host sync at all.

    `kind="forward"` additionally re-derives the three big overflow bits by
    hand.  That is not belt-and-braces, it is necessary: mujoco-warp only runs
    its own nefc/broadphase/narrowphase overflow check inside `_advance`
    (`forward.py:222-272`), which `mjwarp.step` reaches and `mjwarp.forward`
    does not.  So a `sim.forward()` that overflows would set no bit at all.
    The predicates below are the engine's own, copied literally, `>` included.
    """
    torch = self._torch
    ov = self._overflow_arr[:]
    nefc = self._nefc_arr[:]
    nacon = self._nacon_arr[:]
    torch.bitwise_or(self._ov_acc, ov, out=self._ov_acc)
    torch.maximum(self._peak_nefc, nefc, out=self._peak_nefc)
    torch.maximum(self._peak_nacon, nacon, out=self._peak_nacon)
    if self._ncollision_arr is not None:
      torch.maximum(self._peak_ncollision, self._ncollision_arr[:],
                    out=self._peak_ncollision)
    if kind == "forward":
      synth = (nefc > self._njmax_t).to(torch.int32) * OVERFLOW_BIT["NEFC"]
      synth = synth + ((nacon > self._naconmax_t).to(torch.int32)
                       * OVERFLOW_BIT["NARROWPHASE"])
      if self._ncollision_arr is not None:
        synth = synth + ((self._ncollision_arr[:] > self._naconmax_t)
                         .to(torch.int32) * OVERFLOW_BIT["BROADPHASE"])
      torch.bitwise_or(self._ov_acc, synth, out=self._ov_acc)
      self._cap_samples_forward += 1
    else:
      self._cap_samples_step += 1

  def _capacity_flags(self) -> int:
    """OR the per-world sticky masks down to one bitmask.  One host sync."""
    torch = self._torch
    bits = torch.bitwise_and(
      torch.bitwise_right_shift(self._ov_acc.unsqueeze(-1), self._ov_shifts), 1)
    mask = int(torch.bitwise_left_shift(bits.amax(dim=0),
                                        self._ov_shifts).sum().item())
    self._cap_flags_seen |= mask
    return mask

  def _capacity_detail(self) -> str:
    try:
      nefc = int(self._peak_nefc.max().item())
      nacon = int(self._peak_nacon[0].item())
      ncol = (int(self._peak_ncollision[0].item())
              if self._ncollision_arr is not None else -1)
      flagged = int((self._ov_acc != 0).sum().item())
      return (f"Peaks so far: nefc {nefc} rows/world vs njmax "
              f"{self.njmax_alloc}; nacon {nacon} and ncollision {ncol} vs "
              f"naconmax {self.naconmax_alloc} across all worlds; "
              f"{flagged} of {self.num_envs} worlds carry a flag.")
    except Exception as exc:  # pragma: no cover
      return f"(peak read-back failed: {exc!r})"

  def _capacity_gate(self, where: str) -> None:
    """Fail closed the instant the engine reports any overflow bit.

    Called once per env step (20 physics substeps), not once per PPO iteration
    (480 substeps).  The broadphase axis goes overflow -> out-of-bounds write
    -> CUDA illegal access faster than one iteration, which is how the old
    per-iteration check kept arriving after the process was already dead.
    """
    if not self._cap_ok:
      return
    mask = self._capacity_flags()
    if mask:
      raise CapacityOverflow(mask, where, detail=self._capacity_detail())

  def _probe_contacts(self):
    """Per-substep, sync-free: which worlds had the racket on the ball.

    Runs inside the 1 kHz loop over the whole pre-allocated contact array, so
    every extra pass over it is throughput.  Kept to: one validity compare, two
    ball compares, one select, one gather through the geom-class table, three
    masks, two reductions and one scatter.  No host sync, no `.item()`.
    """
    torch = self._torch
    g = self._con_geom[:]
    valid = self._con_idx < self._nacon[0]
    g0, g1 = g[:, 0], g[:, 1]
    is0 = g0 == self._ball_gid
    is1 = g1 == self._ball_gid
    other = torch.where(is0, g1, g0).long()
    kind = self._geom_class[other]            # 1 = racket, 2 = table top
    ball_row = valid & (is0 | is1)
    hit = ball_row & (kind == 1)
    tab = ball_row & (kind == 2)
    self._acc["contact_ball_racket_substeps"] += hit.sum()
    self._acc["contact_ball_table_substeps"] += tab.sum()
    w = self._con_world[:].long().clamp_(0, self.num_envs - 1)
    self._cur_touched.scatter_add_(0, w, hit.float())
    if self._robot_table_ok:
      # Same pass, same arrays: which rows are table-vs-robot rather than
      # table-vs-ball.  Four more elementwise ops, no extra sync.
      t0 = g0 == self._table_gid
      t1 = g1 == self._table_gid
      partner = torch.where(t0, g1, g0).long()
      rt = valid & (t0 | t1) & self._is_robot_geom[partner]
      self._acc["contact_robot_table_substeps"] += rt.sum()
      self._cur_robot_table.scatter_add_(0, w, rt.float())

  def _rand(self, *shape, lo=0.0, hi=1.0):
    torch = self._torch
    r = torch.rand(*shape, device=self.device, generator=self.generator)
    return lo + (hi - lo) * r

  def _qpos_act(self):
    d = self.sim.data
    if self._q_slice is not None:
      return d.qpos[:, self._q_slice]
    return d.qpos[:, self.q_adr_act]

  def _qvel_act(self):
    d = self.sim.data
    if self._v_slice is not None:
      return d.qvel[:, self._v_slice]
    return d.qvel[:, self.v_adr_act]

  def _serve(self, ids):
    """Write a fresh serve into the ball's free joint for ``ids``."""
    torch = self._torch
    n = int(ids.numel()) if hasattr(ids, "numel") else len(ids)
    if n == 0:
      return
    u = torch.rand(n, 3, device=self.device, generator=self.generator)
    pos_hope = self.serve_pos_lo + (self.serve_pos_hi - self.serve_pos_lo) * u
    u2 = torch.rand(n, 3, device=self.device, generator=self.generator)
    vel = self.serve_vel_lo + (self.serve_vel_hi - self.serve_vel_lo) * u2
    pos = pos_hope + self.hope_to_scene
    d = self.sim.data
    bq, bv = self.b_q, self.b_v
    d.qpos[ids, bq:bq + 3] = pos
    d.qpos[ids, bq + 3:bq + 7] = torch.tensor(
      [1.0, 0.0, 0.0, 0.0], device=self.device).expand(n, 4)
    d.qvel[ids, bv:bv + 3] = vel
    d.qvel[ids, bv + 3:bv + 6] = 0.0
    self.ball_age_buf[ids] = 0

  def _reset_idx(self, ids):
    torch = self._torch
    n = int(ids.numel())
    if n == 0:
      return
    # mjwarp's own masked reset first: clears qacc, warmstart, contacts, act.
    self.sim.reset(ids)
    d = self.sim.data
    cfg = self.cfg

    qpos = self.qpos_init.unsqueeze(0).repeat(n, 1)
    qvel = self.qvel_init.unsqueeze(0).repeat(n, 1)
    if cfg.reset_joint_noise_rad > 0:
      noise = self._rand(n, self.num_actions,
                         lo=-cfg.reset_joint_noise_rad,
                         hi=cfg.reset_joint_noise_rad)
      q = torch.clamp(self.q_ready.unsqueeze(0) + noise, self.jnt_lo, self.jnt_hi)
      qpos[:, self.q_adr_act] = q
    if cfg.reset_joint_vel_noise > 0:
      qvel[:, self.v_adr_act] = self._rand(
        n, self.num_actions, lo=-cfg.reset_joint_vel_noise,
        hi=cfg.reset_joint_vel_noise)
    rq = self.root_qadr
    if cfg.reset_root_xy_noise_m > 0:
      qpos[:, rq:rq + 2] += self._rand(n, 2, lo=-cfg.reset_root_xy_noise_m,
                                       hi=cfg.reset_root_xy_noise_m)
    if cfg.reset_root_yaw_noise_rad > 0:
      yaw = self._rand(n, lo=-cfg.reset_root_yaw_noise_rad,
                       hi=cfg.reset_root_yaw_noise_rad)
      # compose a yaw-only quaternion onto the (near-identity) ready quat
      cy, sy = torch.cos(0.5 * yaw), torch.sin(0.5 * yaw)
      q0 = qpos[:, rq + 3:rq + 7]
      w0, x0, y0, z0 = q0[:, 0], q0[:, 1], q0[:, 2], q0[:, 3]
      qpos[:, rq + 3] = cy * w0 - sy * z0
      qpos[:, rq + 4] = cy * x0 - sy * y0
      qpos[:, rq + 5] = cy * y0 + sy * x0
      qpos[:, rq + 6] = cy * z0 + sy * w0

    d.qpos[ids] = qpos
    d.qvel[ids] = qvel
    d.ctrl[ids] = 0.0
    self._serve(ids)
    self.episode_length_buf[ids] = 0
    self.actions[ids] = 0.0
    self.last_actions[ids] = 0.0
    self.action_nonfinite_buf[ids] = False
    self._qdes_previous_executable[ids] = self.action_offset
    self._qdes_previous_executable_valid[ids] = True
    self._qdes_guard_terminal[ids] = False
    self._cur_ret[ids] = 0.0
    self._cur_min_d[ids] = 1e3

  def reset(self):
    torch = self._torch
    ids = torch.arange(self.num_envs, device=self.device)
    self._reset_idx(ids)
    self.sim.forward()
    if self._cap_ok:
      # `forward()` rebuilds collisions and constraints exactly like `step()`
      # does, so it can overflow exactly like `step()` does.  It used to be a
      # blind spot: nothing sampled here, and the next `step()` overwrote the
      # counters before anyone looked.
      self._probe_capacity("forward")
    self._compute_obs()
    if self._cap_ok:
      self._capacity_gate("reset")
    return self.get_observations(), {}

  # ---- observation ------------------------------------------------------

  def _state(self):
    torch = self._torch
    d = self.sim.data
    rq, rv = self.root_qadr, self.root_vadr
    qpos, qvel = d.qpos, d.qvel
    base_quat = qpos[:, rq + 3:rq + 7]
    base_pos = qpos[:, rq:rq + 3]
    base_lin_w = qvel[:, rv:rv + 3]
    base_ang_b = qvel[:, rv + 3:rv + 6]          # free joint: already body-local
    proj_g = quat_rotate_inverse(base_quat, self.gravity_w)
    ball_pos = qpos[:, self.b_q:self.b_q + 3]
    ball_vel = qvel[:, self.b_v:self.b_v + 3]
    racket = d.site_xpos[:, self.racket_sid]
    return dict(qpos=qpos, qvel=qvel, base_quat=base_quat, base_pos=base_pos,
                base_lin_w=base_lin_w, base_ang_b=base_ang_b, proj_g=proj_g,
                ball_pos=ball_pos, ball_vel=ball_vel, racket=racket)

  def _compute_obs(self, st=None):
    torch = self._torch
    cfg = self.cfg
    st = st or self._state()
    q = st["base_quat"]
    d_ball_racket = st["ball_pos"] - st["racket"]
    d_ball_pelvis = st["ball_pos"] - st["base_pos"]
    d_racket_pelvis = st["racket"] - st["base_pos"]
    # Built from OBS_LAYOUT rather than an anonymous cat: the declared names
    # ARE the observation order, so an auditor that reads OBS_LAYOUT is reading
    # the live layout and not a hand copy of it.
    rows = {
      "base_lin_vel_body": quat_rotate_inverse(q, st["base_lin_w"])
                           * cfg.obs_scale_lin_vel,
      "base_ang_vel_body": st["base_ang_b"] * cfg.obs_scale_ang_vel,
      "projected_gravity": st["proj_g"],
      "joint_pos_rel_ready": self._qpos_act() - self.q_ready.unsqueeze(0),
      "joint_vel_scaled": self._qvel_act() * cfg.obs_scale_joint_vel,
      "actions": self.actions,
      "ball_minus_racket_body": quat_rotate_inverse(q, d_ball_racket),
      "ball_lin_vel_body_scaled": quat_rotate_inverse(q, st["ball_vel"])
                                  * cfg.obs_scale_ball_vel,
      "ball_minus_pelvis_body": quat_rotate_inverse(q, d_ball_pelvis),
      "racket_minus_pelvis_body": quat_rotate_inverse(q, d_racket_pelvis),
    }
    missing = [n for n, _w in OBS_LAYOUT if n not in rows]
    extra = [n for n in rows if n not in {a for a, _ in OBS_LAYOUT}]
    if missing or extra:
      raise RuntimeError(
        "OBS_LAYOUT and the observation producers disagree: "
        f"missing={missing} extra={extra}")
    parts = []
    for name, width in OBS_LAYOUT:
      row = rows[name]
      if int(row.shape[-1]) != int(width):
        raise RuntimeError(
          f"observation row {name!r} is {int(row.shape[-1])} wide, "
          f"OBS_LAYOUT declares {int(width)}")
      parts.append(row)
    obs = torch.cat(parts, dim=-1)
    if int(obs.shape[-1]) != OBS_WIDTH:
      raise RuntimeError(
        f"observation width {int(obs.shape[-1])} != declared {OBS_WIDTH}")
    obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    self._obs_buf = torch.clamp(obs, -cfg.obs_clip, cfg.obs_clip)
    return self._obs_buf

  # ---- step -------------------------------------------------------------

  def _after_physics_substep(self, substep_index):
    """Optional device-only observer; the base plant has no extra predicate."""

  def _advance_plant(self, actions):
    """Apply one policy action through the one real 20-substep plant loop."""
    torch = self._torch
    d = self.sim.data

    incoming = actions.to(self.device)
    if tuple(incoming.shape) != tuple(self.actions.shape):
      raise ValueError(
        "policy action shape differs from the runtime action buffer: "
        f"{tuple(incoming.shape)} != {tuple(self.actions.shape)}")
    pre_clamp_qdes = self.action_offset.unsqueeze(0) + self.act_scale * incoming
    finite_qdes = torch.isfinite(pre_clamp_qdes)
    safe_actions = torch.where(finite_qdes, incoming, self.actions)
    self.last_actions = self.actions
    self.actions = safe_actions
    if getattr(self, "full_a_mode", False):
      hard_span = self.jnt_hi - self.jnt_lo
      soft_lower = self.jnt_lo + 0.05 * hard_span
      soft_upper = self.jnt_hi - 0.05 * hard_span
      guard = shared_qdes_guard.action_ball_qdes_guard(
        pre_clamp_qdes=pre_clamp_qdes,
        previous_executable_qdes=self._qdes_previous_executable,
        previous_executable_valid=self._qdes_previous_executable_valid,
        default_qdes=self.action_offset.unsqueeze(0).expand_as(pre_clamp_qdes),
        soft_lower=soft_lower.unsqueeze(0).expand_as(pre_clamp_qdes),
        soft_upper=soft_upper.unsqueeze(0).expand_as(pre_clamp_qdes),
        hard_lower=self.jnt_lo.unsqueeze(0).expand_as(pre_clamp_qdes),
        hard_upper=self.jnt_hi.unsqueeze(0).expand_as(pre_clamp_qdes),
        joint_pos=self._qpos_act(),
        joint_vel=self._qvel_act(),
        policy_dt_s=0.02,
        hard_margin_rad=0.0,
        hard_margin_fraction=0.05,
        project_finite_without_termination=True,
        projection_soft_inset_fraction=0.05,
      )
      q_des = guard.executable_qdes
      self.action_nonfinite_buf = guard.qdes_nonfinite.any(dim=1)
      self._qdes_guard_terminal.copy_(guard.hard_violation_env)
      self._qdes_previous_executable.copy_(q_des)
      self._qdes_previous_executable_valid.fill_(True)
    else:
      # Keep the historical base-lane ABI outside FullMDP.
      self.action_nonfinite_buf = ~finite_qdes.all(dim=1)
      q_des = torch.clamp(
        self.action_offset.unsqueeze(0) + self.act_scale * safe_actions,
        self.jnt_lo, self.jnt_hi)
      self._qdes_guard_terminal.copy_(self.action_nonfinite_buf)

    tau_sq = torch.zeros(self.num_envs, device=self.device)
    for substep_index in range(self.decimation):
      tau = self.kp * (q_des - self._qpos_act()) - self.kd * self._qvel_act()
      tau = torch.clamp(tau, self.tau_lo, self.tau_hi)
      d.ctrl[:] = tau[:, self.actuator_from_runtime]
      tau_n = tau / self.tau_scale
      tau_sq += (tau_n * tau_n).mean(dim=-1)
      self.sim.step()
      self._after_physics_substep(substep_index)
      if self._contact_ok:
        self._probe_contacts()
      if self._cap_ok:
        # This sample MUST stay inside the decimation loop.  `_reset_idx` at
        # the bottom of this method calls `sim.reset(ids)`, and
        # `mjwarp.reset_data` zeroes `d.overflow` for every world it resets
        # (io.py:2483).  A read taken after that point would lose precisely
        # the evidence the gate exists to catch.
        self._probe_capacity("step")
    tau_sq /= self.decimation

    self.episode_length_buf += 1
    self.ball_age_buf += 1
    self.common_step_counter += 1

    st = self._state()
    return st, tau_sq, pre_clamp_qdes

  def step(self, actions):
    torch = self._torch
    cfg = self.cfg
    st, tau_sq, _pre_clamp_qdes = self._advance_plant(actions)

    rew, terms = self._reward(st, tau_sq)
    terminated, reasons = self._terminate(st)
    truncated = self.episode_length_buf >= self.max_episode_length
    truncated = truncated & (~terminated)
    rew = rew + cfg.r_termination * terminated.float()
    terms["termination"] = cfg.r_termination * terminated.float()
    dones = terminated | truncated

    # -- ball housekeeping: re-serve a dead rally without ending the episode
    ball_hope = st["ball_pos"] - self.hope_to_scene
    dead = ((ball_hope[:, 2] < cfg.ball_dead_z_hope)
            | (ball_hope[:, 0] < cfg.ball_dead_x_lo_hope)
            | (ball_hope[:, 0] > cfg.ball_dead_x_hi_hope)
            | (self.ball_age_buf > self._ball_reserve_steps)
            | (~torch.isfinite(ball_hope).all(dim=1)))
    dead = dead & (~dones)
    reserve_ids = dead.nonzero(as_tuple=False).squeeze(-1)

    self._accumulate(rew, terms, dones, terminated, truncated, reasons,
                     st, reserve_ids.numel())

    reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
    if reset_ids.numel() > 0:
      self._reset_idx(reset_ids)
    if reserve_ids.numel() > 0:
      self._serve(reserve_ids)
    if reset_ids.numel() > 0 or reserve_ids.numel() > 0:
      self.sim.forward()
      if self._cap_ok:
        self._probe_capacity("forward")

    self._compute_obs()
    if self._cap_ok:
      self._capacity_gate(f"env step {self.common_step_counter}")
    extras: dict[str, Any] = {"time_outs": truncated}
    return (self.get_observations(), rew, dones.long(), extras)

  # ---- reward / termination --------------------------------------------

  def _reward(self, st, tau_sq):
    torch = self._torch
    cfg = self.cfg
    qerr = self._qpos_act() - self.q_ready.unsqueeze(0)
    pose = torch.exp(-cfg.k_pose * (qerr * qerr).mean(dim=-1))
    upright = torch.exp(-cfg.k_upright * (1.0 + st["proj_g"][:, 2]) ** 2)
    dz = st["base_pos"][:, 2] - self.ready_pelvis_z
    height = torch.exp(-cfg.k_height * dz * dz)
    d = torch.linalg.norm(st["ball_pos"] - st["racket"], dim=-1)
    self._last_d = d
    reach = torch.exp(-d / cfg.reach_len_m)
    touch = torch.exp(-(d / cfg.touch_sigma_m) ** 2)
    da = self.actions - self.last_actions
    action_rate = (da * da).mean(dim=-1)
    qd = self._qvel_act()
    joint_vel = (qd * qd).mean(dim=-1)
    terms = {
      "alive": torch.full_like(pose, cfg.w_alive),
      "pose": cfg.w_pose * pose,
      "upright": cfg.w_upright * upright,
      "height": cfg.w_height * height,
      "reach_term_weighted": cfg.w_reach * reach,
      "touch_term_weighted": cfg.w_touch * touch,
      "action_rate": cfg.w_action_rate * action_rate,
      "joint_vel": cfg.w_joint_vel * joint_vel,
      "torque": cfg.w_torque * tau_sq,
    }
    rew = sum(terms.values())
    rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
    return rew, terms

  def _terminate(self, st):
    torch = self._torch
    cfg = self.cfg
    fall_h = st["base_pos"][:, 2] < cfg.min_pelvis_z
    tilt = st["proj_g"][:, 2] > cfg.max_tilt_proj_g
    finite = (torch.isfinite(st["qpos"]).all(dim=1)
              & torch.isfinite(st["qvel"]).all(dim=1)
              & (~self.action_nonfinite_buf))
    nonfinite = ~finite
    terminated = fall_h | tilt | nonfinite
    return terminated, {"fall_h": fall_h & finite, "tilt": tilt & finite & (~fall_h),
                        "nonfinite": nonfinite}

  # ---- statistics --------------------------------------------------------

  def _accumulate(self, rew, terms, dones, terminated, truncated, reasons, st,
                  n_reserve):
    torch = self._torch
    a = self._acc
    a["steps"] += self.num_envs
    a["rew_sum"] += rew.sum()
    for k in terms:
      a["r_" + k] += terms[k].sum()
    a["term_fall_h"] += reasons["fall_h"].sum()
    a["term_tilt"] += reasons["tilt"].sum()
    a["term_nonfinite"] += reasons["nonfinite"].sum()
    a["term_timeout"] += truncated.sum()
    a["reserves"] += n_reserve
    self._cur_ret += rew
    self._cur_min_d = torch.minimum(self._cur_min_d, self._last_d)
    df = dones.float()
    a["ep_ret_sum"] += (self._cur_ret * df).sum()
    a["ep_len_sum"] += (self.episode_length_buf.float() * df).sum()
    a["ep_min_d_sum"] += (torch.clamp(self._cur_min_d, max=10.0) * df).sum()
    a["ep_cnt"] += df.sum()
    if self._contact_ok:
      a["ep_touched_racket"] += ((self._cur_touched > 0).float() * df).sum()
      self._cur_touched *= (1.0 - df)
    if self._robot_table_ok:
      a["ep_robot_touched_table"] += ((self._cur_robot_table > 0).float() * df).sum()
      self._cur_robot_table *= (1.0 - df)

  def pop_stats(self) -> dict:
    torch = self._torch
    keys = list(self._acc.keys())
    vals = torch.stack([self._acc[k].float() for k in keys]).cpu().numpy()
    out = {k: float(v) for k, v in zip(keys, vals)}
    for k in keys:
      self._acc[k].zero_()
    steps = max(out["steps"], 1.0)
    ep = max(out["ep_cnt"], 1.0)
    stats = {
      "env_steps": out["steps"],
      "mean_step_reward": out["rew_sum"] / steps,
      "episodes_finished": out["ep_cnt"],
      "mean_episode_return": out["ep_ret_sum"] / ep,
      "mean_episode_length": out["ep_len_sum"] / ep,
      "mean_episode_min_racket_ball_dist_m": out["ep_min_d_sum"] / ep,
      "termination_rate_per_env_step": (
        out["term_fall_h"] + out["term_tilt"] + out["term_nonfinite"]) / steps,
      "timeout_rate_per_env_step": out["term_timeout"] / steps,
      "terminations": {
        "fall_height": out["term_fall_h"],
        "fall_tilt": out["term_tilt"],
        "nonfinite_state": out["term_nonfinite"],
        "timeout_truncation": out["term_timeout"],
      },
      # The alignment ledger counts this lane's terminal predicates from
      # TERMINATION_TERMS.  Assert here rather than trusting the two to stay in
      # step: a fourth terminal reason added to `_terminate` and not to the
      # constant would make the ledger under-report the union it compares.
      "termination_terms_declared": list(TERMINATION_TERMS),
      "ball_reserves": out["reserves"],
    }
    reported = set(stats["terminations"]) - {"timeout_truncation"}
    if reported != set(TERMINATION_TERMS):
      raise RuntimeError(
        "TERMINATION_TERMS does not describe the terminal reasons this env "
        f"reports: declared={sorted(TERMINATION_TERMS)} "
        f"reported={sorted(reported)}")
    stats.update(reward_term_report(
      {k: out["r_" + k] / steps for k in self._rew_terms}, self.cfg))
    if self._cap_ok:
      stats["capacity"] = self.capacity_snapshot()
    # The binary contact block is always present, in every mode, even when it
    # says "not measured" -- a missing block is what sends a reader back to the
    # weighted reward term.  `ep` above is max(episodes, 1); the real count is
    # out["ep_cnt"], and passing the real one is what makes an empty window
    # report null instead of a fake 0.0.
    stats["contact"] = binary_contact_fields(
      probe_on=self._contact_ok,
      touched_episodes=out.get("ep_touched_racket", 0.0),
      episodes_finished=out["ep_cnt"],
      racket_substeps=out.get("contact_ball_racket_substeps", 0.0),
      table_substeps=out.get("contact_ball_table_substeps", 0.0))
    stats["robot_table"] = robot_table_contact_fields(
      probe_on=self._robot_table_ok,
      touched_episodes=out.get("ep_robot_touched_table", 0.0),
      episodes_finished=out["ep_cnt"],
      substeps=out.get("contact_robot_table_substeps", 0.0),
      n_robot_geoms=getattr(self, "_n_robot_geoms", 0))
    return stats

  # ---- capacity receipt ---------------------------------------------------

  def capacity_snapshot(self) -> dict:
    """Cumulative capacity numbers for the run so far.  One host sync.

    NOTE the change of meaning against receipts written before 2026-08-06: the
    peaks below are running maxima over the *whole run*, not over one PPO
    iteration -- hence the `_running` suffix on the key names.  They are also
    reporting only.  The pass/fail decision is `overflow_flags`, which is the
    engine's own bitmask and nothing of our own devising.
    """
    torch = self._torch
    if not self._cap_ok:
      return {"probe_enabled": False, "gate": "none"}
    bits = torch.bitwise_and(
      torch.bitwise_right_shift(self._ov_acc.unsqueeze(-1), self._ov_shifts), 1)
    packed = torch.cat([
      self._peak_nefc.max().reshape(1),
      self._peak_nacon.reshape(1),
      self._peak_ncollision.reshape(1),
      (self._ov_acc != 0).sum().to(torch.int32).reshape(1),
      bits.amax(dim=0).to(torch.int32),
    ]).cpu().tolist()
    nefc, nacon, ncol, flagged = (int(packed[0]), int(packed[1]),
                                  int(packed[2]), int(packed[3]))
    mask = 0
    for i, b in enumerate(packed[4:]):
      mask |= int(b) << i
    self._cap_flags_seen |= mask
    return capacity_fields(
      njmax=self.njmax_alloc, naconmax=self.naconmax_alloc,
      nefc_peak=nefc, nacon_peak=nacon, ncollision_peak=ncol,
      worlds_flagged=flagged, overflow_mask=mask,
      samples_step=self._cap_samples_step,
      samples_forward=self._cap_samples_forward,
      ncollision_known=self._ncollision_arr is not None)


# ==========================================================================
# Runner glue.
# ==========================================================================


def _rsl_rl_version() -> str:
  try:
    from importlib.metadata import version

    return version("rsl-rl-lib")
  except Exception:  # pragma: no cover
    return "?"


def build_agent_cfg(seed: int, iterations: int, num_steps_per_env: int,
                    experiment: str, entropy_coef: float = 0.002,
                    init_std: float = 1.0) -> dict:
  from mjlab.rl.config import (RslRlModelCfg, RslRlOnPolicyRunnerCfg,
                               RslRlPpoAlgorithmCfg)

  cfg = RslRlOnPolicyRunnerCfg(
    seed=seed,
    num_steps_per_env=num_steps_per_env,
    max_iterations=iterations,
    save_interval=max(iterations, 1),
    experiment_name=experiment,
    logger="tensorboard",
    obs_groups={"actor": ["policy"], "critic": ["policy"]},
    clip_actions=None,
    upload_model=False,
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution",
                        "init_std": init_std, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu",
                         obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      # rsl-rl's default 0.005 is a *per-dimension* entropy bonus.  With 31
      # action dims the bonus gradient is 31 * coef per unit log-std, which on
      # this reward scale outbid the whole stance-quality budget: the pilot's
      # policy std GREW 1.00 -> 1.16 over 60 iterations, i.e. PPO was paid to
      # stay noisy.  Measured, then halved-and-a-bit.
      entropy_coef=entropy_coef,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
  )
  return asdict(cfg)


def train(args) -> int:
  import torch
  from mjlab.rl import MjlabOnPolicyRunner
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  device = args.device
  sim_cfg = SimCfg(nworld=args.nworld, cone=args.cone,
                   add_pairs=not args.no_pairs,
                   njmax=args.njmax, nconmax=args.nconmax)
  task_cfg = TaskCfg(episode_length_s=args.episode_s,
                     action_scale_mode=args.action_scale_mode)

  # Everything from here down runs inside one try/finally so that the receipt
  # gets written on EVERY exit path, including the ones that matter most.  The
  # gate can fire during scene construction (the very first `sim.forward()`
  # already collides everything), which used to leave no `.json` at all --
  # the run with the most to prove was the one with no evidence.
  out = Path(args.out_prefix + ".json")
  out.parent.mkdir(parents=True, exist_ok=True)
  jsonl = Path(args.out_prefix + ".jsonl")
  jsonl.parent.mkdir(parents=True, exist_ok=True)
  records: list[dict] = []
  env = None
  runner = None
  jf = None
  log_dir = Path(args.log_root) / args.experiment / f"{args.tag}_seed{args.seed}"
  build_s = None
  smi_start: dict = {}
  device_id: dict | None = None
  status, exit_code, error = "started", 1, None

  try:
    t_build = time.perf_counter()
    # T11(b): the binary contact counter now runs during TRAINING too, not just
    # in `--eval`.  "did the racket touch the ball this episode" is the only
    # contact number on this curve that means anything physical; while it was
    # eval-only the training plots had nothing to show but the weighted `touch`
    # reward term, and that term got read as a contact probability.
    env = A3ReadyBallVecEnv(sim_cfg, task_cfg, device=device,
                            xml_path=Path(args.xml_path) if args.xml_path else None,
                            ready_pose_path=(Path(args.ready_pose)
                                             if args.ready_pose else None),
                            seed=args.seed,
                            count_contacts=not args.no_contact_probe,
                            capacity_probe=not args.no_capacity_probe)
    build_s = time.perf_counter() - t_build
    print(f"[a3_train_ppo] scene built in {build_s:.1f}s: nworld={env.num_envs} "
          f"nq={env.mj_model.nq} nu={env.num_actions} obs={env.num_obs} "
          f"decimation={env.decimation} step_dt={env.step_dt:.4f} "
          f"max_ep_len={env.max_episode_length}", flush=True)

    agent_cfg = build_agent_cfg(args.seed, args.iterations,
                                args.num_steps_per_env, args.experiment,
                                entropy_coef=args.entropy_coef,
                                init_std=args.init_std)
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = MjlabOnPolicyRunner(env, agent_cfg, str(log_dir), device)

    # --- receipt hook: one JSON line per PPO iteration -------------------
    jf = jsonl.open("w")
    orig_log = runner.logger.log
    collection_size = env.num_envs * args.num_steps_per_env
    t_start = time.perf_counter()

    def log_hook(**kw):
      stats = env.pop_stats()
      it = int(kw["it"])
      ct, lt = float(kw["collect_time"]), float(kw["learn_time"])
      rec = {
        "iter": it,
        "wall_s_total": time.perf_counter() - t_start,
        "collect_s": ct, "learn_s": lt, "iter_s": ct + lt,
        "env_steps_per_s": collection_size / max(ct + lt, 1e-9),
        "env_steps_per_s_collect_only": collection_size / max(ct, 1e-9),
        "physics_steps_per_s": collection_size * env.decimation / max(ct, 1e-9),
        "learning_rate": float(kw["learning_rate"]),
        "action_std": float(kw["action_std"].mean().item()),
        "losses": {k: float(v) for k, v in kw["loss_dict"].items()},
      }
      rec.update(stats)
      cap = stats.get("capacity")
      if cap is not None and cap.get("overflow_mask"):
        # Belt to the env-step gate's braces.  Reaching here means the per-step
        # gate somehow did not fire; still fail closed, still name the flags.
        raise CapacityOverflow(cap["overflow_mask"], f"iteration {it}")
      if runner.logger.rewbuffer:
        rec["rsl_rl_mean_reward"] = statistics.mean(runner.logger.rewbuffer)
        rec["rsl_rl_mean_ep_len"] = statistics.mean(runner.logger.lenbuffer)
      records.append(rec)
      jf.write(json.dumps(rec) + "\n")
      jf.flush()
      touch_rate = _dig(rec, BINARY_CONTACT_KEY)
      print(f"[it {it:4d}] R_ep={rec['mean_episode_return']:8.2f} "
            f"r_step={rec['mean_step_reward']:6.3f} "
            f"len={rec['mean_episode_length']:6.1f} "
            f"term/step={rec['termination_rate_per_env_step']:.4f} "
            f"minD={rec['mean_episode_min_racket_ball_dist_m']:.3f}m "
            # The headline metric goes on the console line, in percent, with
            # "n/a" (never 0) when this window had nothing to divide by.
            + ("touchEp=n/a " if touch_rate is None
               else f"touchEp={100.0 * touch_rate:5.1f}% ")
            + f"fps={rec['env_steps_per_s']:.0f} "
            + (f"nefc={cap['nefc_peak_per_world_running']}/"
               f"{cap['njmax_allocated_per_world']} "
               f"con={cap['naconmax_binding_peak_all_worlds']}/"
               f"{cap['naconmax_allocated_all_worlds']} "
               f"ovf={cap['overflow_mask']} " if cap else "")
            + f"({ct:.2f}s+{lt:.2f}s)", flush=True)
      if args.rsl_rl_console:
        return orig_log(**kw)
      return None

    runner.logger.log = log_hook  # type: ignore[method-assign]

    smi_start = plant._nvidia_smi()
    device_id = _device_identity(device, smi_start)
    runner.learn(num_learning_iterations=args.iterations,
                 init_at_random_ep_len=True)
    status, exit_code = "completed", 0
  except CapacityOverflow as exc:
    status, exit_code = "gate_fired", 1
    error = {"type": "CapacityOverflow", "message": str(exc),
             "overflow_mask": exc.mask, "overflow_flags": exc.flags,
             "where": exc.where, "traceback": traceback.format_exc()}
    _warn_block("CAPACITY GATE FIRED", str(exc))
  except BaseException as exc:  # noqa: BLE001 - receipt first, then re-report
    status, exit_code = "crashed", 1
    error = {"type": type(exc).__name__, "message": str(exc),
             "traceback": traceback.format_exc()}
    _warn_block("RUN CRASHED", f"{type(exc).__name__}: {exc}")
  finally:
    if jf is not None:
      _safe(jf.close)
    smi_end = _safe(plant._nvidia_smi, default={})
    if device_id is None:
      device_id = _safe(_device_identity, device, smi_end, default={})
    if status == "completed" and runner is not None:
      ckpt = log_dir / f"model_{max(args.iterations - 1, 0)}.pt"
      if not ckpt.is_file():
        _safe(runner.save, str(ckpt))
    warp_warn = (scan_warp_overflow_warnings(args.warn_scan_log)
                 if args.warn_scan_log else None)
    if env is None:
      capacity = {"verdict": "UNREADABLE", "probe_enabled": None,
                  "note": "the scene never finished building, so there is no "
                          "capacity state to read; never treat as PASS"}
    else:
      capacity = _safe(_capacity_summary, env, warp_warn,
                       default={"verdict": "UNREADABLE",
                                "note": "capacity state could not be read back "
                                        "after the failure (typically a CUDA "
                                        "fault); treat as OVERFLOW, never PASS"})
    capacity = _merge_gate_error(capacity, error)
    if warp_warn and warp_warn.get("lines"):
      _warn_block(
        "MUJOCO-WARP PRINTED OVERFLOW WARNINGS",
        f"{warp_warn['lines']} line(s) in {warp_warn['log_path']}: "
        f"{warp_warn.get('by_marker')}\n  "
        + "\n  ".join(warp_warn.get("examples", [])))
    if capacity.get("verdict") in ("OVERFLOW", "OVERFLOW_PRINTF_ONLY",
                                   "UNREADABLE") and exit_code == 0:
      # The gate can also be reached through the receipt, not just the raise.
      status, exit_code = "gate_fired", 1
    if capacity.get("verdict") == "NO_SAMPLES":
      _warn_block("CAPACITY NOT MEASURED",
                  "the gate was wired but no physics step was ever taken, so "
                  "verdict=NO_SAMPLES.  This run says nothing about njmax/"
                  "nconmax and must not be cited as evidence that they hold.")
      if args.iterations > 0 and exit_code == 0:
        status, exit_code = "gate_fired", 1
    if capacity.get("verdict") == "NOT_MEASURED":
      _warn_block("CAPACITY GATE OFF",
                  "--no-capacity-probe was passed: this run has no capacity "
                  "gate and cannot be cited as evidence that the caps held.")
    learning = _safe(_learning_summary, records, default={})
    if not _dig(learning, "binary_contact_rate.measured"):
      _warn_block(
        "CONTACT RATE NOT MEASURED",
        "this run has no binary per-episode racket-ball contact rate "
        f"({_dig(learning, 'binary_contact_rate.reason')}).  Its reward terms "
        "are weighted sums, not contact probabilities -- do not quote "
        "`touch_term_weighted` as one.  --report will refuse this receipt.")
    summary = {
      "status": status,
      "exit_code": exit_code,
      "error": error,
      "tag": args.tag,
      "seed": args.seed,
      "device": device,
      "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
      "device_identity": device_id,
      "device_identity_end": _safe(_device_identity, device, smi_end, default={}),
      "argv": sys.argv,
      "scene": _safe(_scene_summary, env, sim_cfg, build_s, default={}),
      "agent": _safe(_agent_summary, env, args, task_cfg, records, default={}),
      "task_cfg": asdict(task_cfg),
      "isaac_alignment": _safe(alignment_receipt_block, default={}),
      "capacity": capacity,
      "warp_stdout_overflow_scan": warp_warn,
      "throughput": _safe(_throughput_summary, records, env, args, default={}),
      "learning": learning,
      "nvidia_smi_start": smi_start,
      "nvidia_smi_end": smi_end,
      "torch_cuda_mem_reserved_MiB": _safe(
        lambda: torch.cuda.memory_reserved() / 2**20, default=None),
      "torch_cuda_max_mem_allocated_MiB": _safe(
        lambda: torch.cuda.max_memory_allocated() / 2**20, default=None),
      "log_dir": str(log_dir),
      "jsonl": str(jsonl),
      "records": records,
    }
    _safe(out.write_text, json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in
                      ("status", "exit_code", "capacity", "throughput",
                       "learning")}, indent=2, default=str), flush=True)
    # The last thing on stdout is the only metric that may be quoted, in the
    # only form it may be quoted in -- and it says out loud that one run is not
    # a result.  Everything above it is context.
    bc = _dig(learning, "binary_contact_rate") or {}
    if bc.get("measured"):
      f0 = bc.get("fraction_of_episodes_with_a_racket_touch_first")
      f1 = bc.get("fraction_of_episodes_with_a_racket_touch_last")
      print("[a3_train_ppo][HEADLINE] binary per-episode racket-ball contact "
            f"rate {100.0 * (f0 or 0.0):.2f}% -> {100.0 * (f1 or 0.0):.2f}% "
            "(this ONE run only; not a result until it is put next to the "
            "zero-policy baseline and a second run -- use --report)",
            flush=True)
    print(f"[a3_train_ppo] wrote {out} (status={status}, exit={exit_code})",
          flush=True)
  return exit_code


def _warn_block(title: str, body: str) -> None:
  """Anything a human must not miss goes through here, on stdout AND stderr.

  MEMORY rule "WARN must reach the summary": a counter nobody reads is the same
  as no counter at all.  Every line here is prefixed so a log scan can find it.
  """
  line = f"[a3_train_ppo][WARN][{title}] {body}"
  print(line, flush=True)
  print(line, file=sys.stderr, flush=True)


def _safe(fn, *a, default=None, **kw):
  """Run `fn`; on any failure return `default`.  Used only on receipt paths.

  The receipt has to survive the very failures it is documenting -- after a
  CUDA fault most GPU reads raise, and losing the whole receipt to that is the
  P9 defect ("the run that most needs evidence is the one with none").
  """
  try:
    return fn(*a, **kw)
  except BaseException as exc:  # noqa: BLE001
    if isinstance(default, dict):
      d = dict(default)
      d["read_back_error"] = repr(exc)
      return d
    return default


def _merge_gate_error(capacity: dict, error: dict | None) -> dict:
  """Keep the flags the exception carried even when the GPU can no longer talk.

  When the gate fires during scene construction -- the very first
  `sim.forward()` already collides everything -- there is no env left to read
  the bitmask off, but the raised exception knows it.  Losing the named axis at
  exactly that moment is how a fired gate used to leave no usable evidence.
  """
  if not error or error.get("type") != "CapacityOverflow":
    return capacity
  out = dict(capacity)
  out.setdefault("overflow_mask", error.get("overflow_mask"))
  out.setdefault("overflow_flags", error.get("overflow_flags"))
  out["overflow_reported_by"] = error.get("where")
  if out.get("verdict") in (None, "UNREADABLE"):
    out["verdict"] = "OVERFLOW"
  return out


def _agent_summary(env, args, task_cfg, records) -> dict:
  return {
    "num_steps_per_env": args.num_steps_per_env,
    "iterations_requested": args.iterations,
    "iterations_completed": len(records),
    "obs_dim": env.num_obs, "action_dim": env.num_actions,
    "episode_length_s": task_cfg.episode_length_s,
    "max_episode_length_steps": env.max_episode_length,
    "entropy_coef": args.entropy_coef,
    "init_std": args.init_std,
    "action_scale_mode": task_cfg.action_scale_mode,
    "raw_action_clip": task_cfg.action_clip,
    "action_scale_rad_min_max": [float(env.act_scale.min()),
                                 float(env.act_scale.max())],
    "hidden_dims": [512, 256, 128],
    "obs_normalization": True,
    "rsl_rl_version": _rsl_rl_version(),
  }


def _scene_summary(env, sim_cfg, build_s) -> dict:
  return {
    "nworld": env.num_envs, "cone": sim_cfg.cone, "pairs": sim_cfg.add_pairs,
    "njmax": int(getattr(env.sim.wp_data, "njmax", -1)),
    "naconmax": int(getattr(env.sim.wp_data, "naconmax", -1)),
    "cuda_graph": bool(env.sim.use_cuda_graph),
    "timestep": env.physics_dt, "decimation": env.decimation,
    "policy_dt": env.step_dt,
    "nq": int(env.mj_model.nq), "nv": int(env.mj_model.nv),
    "nu": int(env.mj_model.nu), "nbody": int(env.mj_model.nbody),
    "ngeom": int(env.mj_model.ngeom), "npair": int(env.mj_model.npair),
    "ready_row_map_agrees_with_json": env.row_map_agrees,
    "build_seconds": build_s,
  }


def evaluate(args) -> int:
  """Score a fixed policy on the same reward the training run optimizes.

  Two modes, and the pair is the point: ``--eval zero`` is the zero raw-action
  policy, whose target is the pinned Isaac runtime default offset (not the
  split-ready reset), and ``--eval ckpt`` is a trained checkpoint,
  deterministic.  If the trained number does not beat the zero policy on the
  *same* reward, "the curve went up" would only mean PPO learned to stop
  shouting at its own actuators.
  """
  import torch
  from mjlab.rl import MjlabOnPolicyRunner
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()
  torch.manual_seed(args.seed)
  sim_cfg = SimCfg(nworld=args.nworld, cone=args.cone,
                   add_pairs=not args.no_pairs,
                   njmax=args.njmax, nconmax=args.nconmax)
  task_cfg = TaskCfg(episode_length_s=args.episode_s,
                     action_scale_mode=args.action_scale_mode)

  # Same shape as train(): scene construction is inside the try, because the
  # gate can fire on the very first forward() and a run that fires the gate
  # must still leave a receipt behind.
  env = None
  t0 = time.perf_counter()
  device_id = None
  status, exit_code, error, stats = "started", 1, None, {}
  try:
    env = A3ReadyBallVecEnv(sim_cfg, task_cfg, device=args.device,
                            xml_path=Path(args.xml_path) if args.xml_path else None,
                            ready_pose_path=(Path(args.ready_pose)
                                             if args.ready_pose else None),
                            seed=args.seed,
                            count_contacts=not args.no_contact_probe,
                            capacity_probe=not args.no_capacity_probe)
    policy = None
    if args.eval == "ckpt":
      assert args.eval_ckpt, "--eval ckpt needs --eval-ckpt PATH"
      agent_cfg = build_agent_cfg(args.seed, 1, args.num_steps_per_env,
                                  args.experiment,
                                  entropy_coef=args.entropy_coef,
                                  init_std=args.init_std)
      runner = MjlabOnPolicyRunner(env, agent_cfg, None, args.device)
      runner.load(args.eval_ckpt, map_location=args.device)
      policy = runner.get_inference_policy(args.device)

    obs = env.get_observations()
    zeros = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    env.pop_stats()                    # discard the reset-only warmup
    device_id = _device_identity(args.device, plant._nvidia_smi())
    t0 = time.perf_counter()
    with torch.inference_mode():
      for _ in range(args.eval_steps):
        act = zeros if policy is None else policy(obs)
        obs, _, _, _ = env.step(act)
    torch.cuda.synchronize()
    status, exit_code = "completed", 0
  except CapacityOverflow as exc:
    status, exit_code = "gate_fired", 1
    error = {"type": "CapacityOverflow", "message": str(exc),
             "overflow_mask": exc.mask, "overflow_flags": exc.flags,
             "where": exc.where, "traceback": traceback.format_exc()}
    _warn_block("CAPACITY GATE FIRED (eval)", str(exc))
  except BaseException as exc:  # noqa: BLE001
    status, exit_code = "crashed", 1
    error = {"type": type(exc).__name__, "message": str(exc),
             "traceback": traceback.format_exc()}
    _warn_block("EVAL CRASHED", f"{type(exc).__name__}: {exc}")
  finally:
    wall = time.perf_counter() - t0
    smi_end = _safe(plant._nvidia_smi, default={})
    if device_id is None:
      device_id = _safe(_device_identity, args.device, smi_end, default={})
    stats = _safe(env.pop_stats, default={}) if env is not None else {}
    warp_warn = (scan_warp_overflow_warnings(args.warn_scan_log)
                 if args.warn_scan_log else None)
    if env is None:
      capacity = {"verdict": "UNREADABLE",
                  "note": "the scene never finished building; never PASS"}
    else:
      capacity = _safe(_capacity_summary, env, warp_warn,
                       default={"verdict": "UNREADABLE"})
    capacity = _merge_gate_error(capacity, error)
    # T5: the eval path is gated exactly like training now -- the gate lives in
    # env.step(), which eval calls.  When the probe is switched off the receipt
    # says NOT_GATED out loud rather than staying silent about it.
    gated = bool(getattr(env, "_cap_ok", False))
    if not gated:
      _warn_block("EVAL NOT GATED",
                  "--no-capacity-probe: this eval has no capacity gate and its "
                  "numbers cannot be used to claim the caps held.")
    if capacity.get("verdict") in ("OVERFLOW", "OVERFLOW_PRINTF_ONLY",
                                   "UNREADABLE") and exit_code == 0:
      status, exit_code = "gate_fired", 1
    if capacity.get("verdict") == "NO_SAMPLES":
      _warn_block("CAPACITY NOT MEASURED (eval)",
                  "no physics step was taken; verdict=NO_SAMPLES.")
      if args.eval_steps > 0 and exit_code == 0:
        status, exit_code = "gate_fired", 1
    contact = (stats or {}).get("contact") or {}
    if not contact.get("measured"):
      _warn_block(
        "CONTACT RATE NOT MEASURED (eval)",
        f"reason={contact.get('reason')}: this eval reports "
        "fraction_of_episodes_with_a_racket_touch = null, not 0.  A "
        "zero-policy baseline receipt in this state cannot anchor a report.")
    out = {
      "status": status,
      "exit_code": exit_code,
      "error": error,
      "mode": args.eval,
      "checkpoint": args.eval_ckpt,
      "nworld": getattr(env, "num_envs", None),
      "policy_steps": args.eval_steps,
      "wall_s": wall,
      "env_steps_per_s": (getattr(env, "num_envs", 0) * args.eval_steps
                          / max(wall, 1e-9)),
      "capacity_gate": "ENFORCED" if gated else "NOT_GATED",
      "isaac_alignment": _safe(alignment_receipt_block, default={}),
      "capacity": capacity,
      "warp_stdout_overflow_scan": warp_warn,
      "device_identity": device_id,
      "argv": sys.argv,
      "stats": stats,
      "nvidia_smi": smi_end,
    }
    _safe(Path(args.out_prefix + ".json").write_text,
          json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str), flush=True)
  return exit_code


def _capacity_summary(env, warp_warn: dict | None = None) -> dict:
  """Run-level constraint/contact capacity receipt.

  `verdict` is five-valued, and PASS is the only one that needs evidence:

  * ``NOT_MEASURED``      -- the probe was switched off on the command line.
  * ``NO_SAMPLES``        -- the probe was on but never saw a single physics
                             step (e.g. ``--iterations 0``).  A run that
                             measured nothing must never be able to say PASS;
                             this is exactly the hole that let a zero-length
                             run publish ``njmax_headroom_x: 572.0``.
  * ``OVERFLOW``          -- the engine set at least one bit in ``d.overflow``.
  * ``OVERFLOW_PRINTF_ONLY`` -- our GPU read saw nothing but the engine printed
                             overflow warnings on stdout.  The two channels
                             disagree, so the run fails either way.
  * ``PASS_NO_OVERFLOW``  -- >0 stepped samples and every bit stayed clear.
  """
  njmax = int(getattr(env.sim.wp_data, "njmax", -1))
  naconmax = int(getattr(env.sim.wp_data, "naconmax", -1))
  base = {
    "njmax_allocated_per_world": njmax,
    "naconmax_allocated_all_worlds": naconmax,
    "nconmax_allocated_per_world": naconmax // max(1, env.num_envs),
  }
  if not getattr(env, "_cap_ok", False):
    return {**base, "probe_enabled": False, "gate": "none",
            "capacity_samples_stepped": 0, "verdict": "NOT_MEASURED"}
  snap = env.capacity_snapshot()
  out = {**base, **snap, "scene_includes_table_and_ball": True}
  if snap.get("capacity_samples_stepped", 0) <= 0:
    out["verdict"] = "NO_SAMPLES"
    out["note"] = ("the capacity probe was wired but no physics step was ever "
                   "taken, so nothing at all is known about capacity under load")
  elif snap.get("overflow_mask"):
    out["verdict"] = "OVERFLOW"
  else:
    out["verdict"] = "PASS_NO_OVERFLOW"
  if warp_warn is not None:
    out["warp_overflow_printf_lines"] = int(warp_warn.get("lines", 0))
    if warp_warn.get("lines") and out["verdict"] == "PASS_NO_OVERFLOW":
      out["verdict"] = "OVERFLOW_PRINTF_ONLY"
      out["note"] = ("d.overflow read clean but mujoco-warp printed overflow "
                     "warnings on stdout; the two channels disagree, so this "
                     "run is not usable either way")
  return out


def _throughput_summary(records, env, args) -> dict:
  if not records:
    return {}
  # iteration 0 pays CUDA-graph warmup / JIT; report both with and without.
  fps = [r["env_steps_per_s"] for r in records]
  it_s = [r["iter_s"] for r in records]
  col = [r["collect_s"] for r in records]
  lrn = [r["learn_s"] for r in records]
  tail = slice(1, None) if len(records) > 1 else slice(0, None)
  return {
    "env_steps_per_s_mean_incl_it0": float(np.mean(fps)),
    "env_steps_per_s_mean_excl_it0": float(np.mean(fps[tail])),
    "env_steps_per_s_median": float(np.median(fps[tail])),
    "physics_steps_per_s_collect_median": float(np.median(
      [r["physics_steps_per_s"] for r in records][tail])),
    "wall_s_per_iteration_mean_excl_it0": float(np.mean(it_s[tail])),
    "collect_s_mean_excl_it0": float(np.mean(col[tail])),
    "learn_s_mean_excl_it0": float(np.mean(lrn[tail])),
    "collect_fraction": float(np.mean(col[tail]) /
                              max(np.mean(it_s[tail]), 1e-9)),
    "total_wall_s": float(records[-1]["wall_s_total"]),
    "total_env_steps": float(sum(r["env_steps"] for r in records)),
  }


def _tied_ranks(y) -> "np.ndarray":
  """Spearman ranks with ties averaged, which is what Spearman actually is.

  ``argsort(argsort(y))`` -- what this file used before -- breaks ties by
  position, so a perfectly FLAT curve came out ranked 0,1,2,... and scored
  rho = +1.0, "rising monotonically".  The binary contact rate is flat at 0.0
  for the whole early part of a run, so that bug would have printed a rising
  trend for a policy that had never once touched the ball.
  """
  y = np.asarray(y, dtype=float)
  order = np.argsort(y, kind="mergesort")
  ranks = np.empty(len(y), dtype=float)
  ranks[order] = np.arange(len(y), dtype=float)
  s = y[order]
  i = 0
  while i < len(s):
    j = i
    while j + 1 < len(s) and s[j + 1] == s[i]:
      j += 1
    if j > i:
      ranks[order[i:j + 1]] = float(np.mean(np.arange(i, j + 1, dtype=float)))
    i = j + 1
  return ranks


def _spearman(y) -> float:
  y = np.asarray(y, dtype=float)
  n = len(y)
  if n < 3:
    return float("nan")
  rx = _tied_ranks(np.arange(n, dtype=float))
  ry = _tied_ranks(y)
  rx = rx - rx.mean()
  ry = ry - ry.mean()
  den = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
  # A constant series has zero rank variance: the honest answer is "no trend
  # measurable", i.e. nan -- not +1.0.
  return float((rx * ry).sum() / den) if den > 0 else float("nan")


def _learning_summary(records) -> dict:
  if not records:
    return {}
  ret = [r["mean_episode_return"] for r in records]
  step_r = [r["mean_step_reward"] for r in records]
  ep_len = [r["mean_episode_length"] for r in records]
  term = [r["termination_rate_per_env_step"] for r in records]
  mind = [r["mean_episode_min_racket_ball_dist_m"] for r in records]
  k = max(1, len(records) // 10)
  return {
    "iterations": len(records),
    "mean_episode_return_first": float(np.mean(ret[:k])),
    "mean_episode_return_last": float(np.mean(ret[-k:])),
    "mean_episode_return_gain": float(np.mean(ret[-k:]) - np.mean(ret[:k])),
    "mean_step_reward_first": float(np.mean(step_r[:k])),
    "mean_step_reward_last": float(np.mean(step_r[-k:])),
    "mean_episode_length_first": float(np.mean(ep_len[:k])),
    "mean_episode_length_last": float(np.mean(ep_len[-k:])),
    "termination_rate_first": float(np.mean(term[:k])),
    "termination_rate_last": float(np.mean(term[-k:])),
    "min_racket_ball_dist_first_m": float(np.mean(mind[:k])),
    "min_racket_ball_dist_last_m": float(np.mean(mind[-k:])),
    "spearman_return_vs_iteration": _spearman(ret),
    "spearman_step_reward_vs_iteration": _spearman(step_r),
    "spearman_min_dist_vs_iteration": _spearman(mind),
    "monotone_rising": bool(np.mean(ret[-k:]) > np.mean(ret[:k])),
    "curve_mean_episode_return": [float(v) for v in ret],
    "curve_mean_step_reward": [float(v) for v in step_r],
    "curve_mean_episode_length": [float(v) for v in ep_len],
    "curve_termination_rate": [float(v) for v in term],
    "curve_min_racket_ball_dist_m": [float(v) for v in mind],
    "curve_action_std": [float(r["action_std"]) for r in records],
    "curve_reward_terms": {
      t: [float(r["reward_terms_mean"][t]) for r in records]
      for t in records[0]["reward_terms_mean"]
    },
    "reward_terms_first": {t: float(np.mean([r["reward_terms_mean"][t]
                                             for r in records[:k]]))
                           for t in records[0]["reward_terms_mean"]},
    "reward_terms_last": {t: float(np.mean([r["reward_terms_mean"][t]
                                            for r in records[-k:]]))
                          for t in records[0]["reward_terms_mean"]},
    "reward_terms_max_possible": records[-1].get("reward_terms_max_possible"),
    "reward_terms_are_weighted_not_probabilities": True,
    "reward_terms_note": _NOT_A_PROBABILITY,
    "binary_contact_rate": _binary_contact_summary(records, k),
    "robot_table_contact": _robot_table_summary(records),
    "reporting": {
      "headline_metric": BINARY_CONTACT_KEY,
      "headline_needs_a_zero_policy_baseline": True,
      "headline_needs_at_least_two_runs": True,
      "do_not_quote_as_a_contact_rate": sorted(KERNEL_REWARD_TERMS),
      "how": ("python a3_train_ppo.py --report RUN_A.json RUN_B.json "
              "--report-zero-policy EVAL_zero.json --out REPORT.json"),
      "note": ("report the binary per-episode contact rate against the "
               "do-nothing policy measured on the same scene, over at least "
               "two runs with the run-to-run band shown.  A single run is a "
               "sample: this config gave touch = 0.21/0.46/0.59/0.61 on four "
               "identical tries, and the 0.21 is the one that got published."),
    },
  }


def _robot_table_summary(records) -> dict:
  """Did the robot ever touch the table during this run?  Pure.

  Reported as ``max`` over the curve, not ``last``: the question is not "is it
  leaning on the table right now", it is "does this curve contain behaviour
  Isaac would have terminated".  One iteration is enough to contaminate it.
  """
  curve = [_dig(r, ROBOT_TABLE_CONTACT_KEY) for r in records]
  seen = [float(v) for v in curve if v is not None]
  if not seen:
    return {"measured": False, "peak_fraction_of_episodes": None,
            "curve": curve,
            "reason": "no table geom, probe off, or no episode ever ended",
            "isaac_twin": "robot_hit_table (terminal in the Isaac ActionBall run)"}
  return {
    "measured": True,
    "peak_fraction_of_episodes": max(seen),
    "last_fraction_of_episodes": seen[-1],
    "iterations_measured": len(seen),
    "curve": curve,
    "terminal_here": False,
    "isaac_twin": "robot_hit_table (terminal in the Isaac ActionBall run)",
  }


def _binary_contact_summary(records, k: int) -> dict:
  """The headline metric, summarised: curve, first/last decile, and honesty.

  ``measured`` is the flag ``--report`` gates on.  It is True only when at
  least one iteration actually produced a rate; a run with the probe off, or
  one whose windows never closed an episode, says False and carries no numbers
  at all rather than zeros.
  """
  curve = _binary_contact_curve(records)
  seen = [v for v in curve if v is not None]
  probe = _dig(records[-1], "contact.probe")
  if not seen:
    return {
      "measured": False,
      "probe": probe,
      "curve_fraction_of_episodes_with_a_racket_touch": curve,
      "iterations_measured": 0,
      "reason": ("CONTACT_PROBE_OFF" if probe == "OFF"
                 else "NO_EPISODES_FINISHED_IN_ANY_ITERATION"),
      "note": "this run cannot support any claim about racket-ball contact",
    }
  first = [v for v in curve[:k] if v is not None]
  last = [v for v in curve[-k:] if v is not None]
  return {
    "measured": True,
    "probe": probe,
    "curve_fraction_of_episodes_with_a_racket_touch": curve,
    "iterations_measured": len(seen),
    "iterations_total": len(curve),
    "fraction_of_episodes_with_a_racket_touch_first": (
      float(np.mean(first)) if first else None),
    "fraction_of_episodes_with_a_racket_touch_last": (
      float(np.mean(last)) if last else None),
    "fraction_of_episodes_with_a_racket_touch_max": float(max(seen)),
    "spearman_vs_iteration": _spearman(seen),
    "denominator": "episodes that ENDED inside that iteration",
    "note": ("binary: an episode counts once if the racket and the ball were "
             "ever in contact during it.  This is the contact metric; the "
             "weighted touch reward term is not one."),
  }


# ==========================================================================
# N-seed band analysis (the only reproducibility claim warp supports).
# ==========================================================================


def analyze(paths, out_path) -> int:
  # A "band" over one run is not a band -- it is a point with the error bars
  # drawn at zero width, which is worse than no error bars at all.  This used
  # to be allowed and would happily print `rel_spread_max_pct: 0.0`.
  if len(paths) < 2:
    _warn_block(
      "BAND REFUSED",
      f"--analyze needs at least 2 runs, got {len(paths)}.  mujoco-warp is "
      "non-deterministic and this config swings ~3x between identical runs, "
      "so a single-run 'band' reports zero spread and is a false claim of "
      "reproducibility.  Re-run with another seed (or another repeat).")
    return 2
  curves, names = [], []
  for p in paths:
    p = Path(p)
    if p.suffix == ".jsonl":
      recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    else:
      recs = json.loads(p.read_text())["records"]
    curves.append(recs)
    names.append(p.stem)
  n = min(len(c) for c in curves)
  band: dict[str, Any] = {"runs": names, "iterations_compared": n,
                          "n_seeds": len(curves)}
  # The binary contact rate is the headline metric, so it gets a band like
  # everything else -- and `np.nan` where an iteration closed no episode, so a
  # missing measurement never averages in as a zero.
  band["binary_contact_rate"] = _contact_band(curves, n)
  for key in ("mean_episode_return", "mean_step_reward", "mean_episode_length",
              "termination_rate_per_env_step",
              "mean_episode_min_racket_ball_dist_m", "env_steps_per_s"):
    arr = np.array([[c[i][key] for i in range(n)] for c in curves], dtype=float)
    mean = arr.mean(axis=0)
    lo, hi = arr.min(axis=0), arr.max(axis=0)
    spread = hi - lo
    denom = np.maximum(np.abs(mean), 1e-9)
    k = max(1, n // 10)
    band[key] = {
      "per_seed_first": [float(a[:k].mean()) for a in arr],
      "per_seed_last": [float(a[-k:].mean()) for a in arr],
      "band_mean": [float(v) for v in mean],
      "band_lo": [float(v) for v in lo],
      "band_hi": [float(v) for v in hi],
      "abs_spread_mean": float(spread.mean()),
      "abs_spread_max": float(spread.max()),
      "rel_spread_mean_pct": float(100.0 * (spread / denom).mean()),
      "rel_spread_max_pct": float(100.0 * (spread / denom).max()),
      "final_values": [float(a[-1]) for a in arr],
      "final_abs_spread": float(spread[-1]),
      "final_rel_spread_pct": float(100.0 * spread[-1] / denom[-1]),
      "last_decile_band_overlaps_first_decile": bool(
        max(float(a[-k:].mean()) for a in arr) > min(float(a[:k].mean()) for a in arr)
        and min(float(a[-k:].mean()) for a in arr)
        < max(float(a[:k].mean()) for a in arr)),
    }
  ret = band["mean_episode_return"]
  band["verdict"] = {
    "all_seeds_rose": all(l > f for f, l in zip(ret["per_seed_first"],
                                                ret["per_seed_last"])),
    "learning_gain_vs_seed_spread": float(
      (np.mean(ret["per_seed_last"]) - np.mean(ret["per_seed_first"]))
      / max(ret["abs_spread_mean"], 1e-9)),
  }
  Path(out_path).write_text(json.dumps(band, indent=2))
  print(json.dumps({k: v for k, v in band.items()
                    if k in ("runs", "iterations_compared", "verdict")}, indent=2))
  print(json.dumps({"mean_episode_return": {
    k: v for k, v in band["mean_episode_return"].items()
    if not k.startswith("band_")}}, indent=2))
  print(json.dumps({"binary_contact_rate": {
    k: v for k, v in band["binary_contact_rate"].items()
    if not k.startswith("band_")}}, indent=2, default=str))
  print(f"[a3_train_ppo] wrote {out_path}")
  return 0


def _contact_band(curves, n: int) -> dict:
  """Run-to-run band of the binary contact rate, NaN-aware.

  Iterations in which a run closed no episode carry ``None``; they become NaN
  here and are skipped by the nan-aware reductions, so "not measured" never
  averages in as a contact rate of zero.
  """
  arr = np.array([[_dig(c[i], BINARY_CONTACT_KEY) for i in range(n)]
                  for c in curves], dtype=object)
  arr = np.array([[np.nan if v is None else float(v) for v in row]
                  for row in arr], dtype=float)
  measured = int(np.isfinite(arr).sum())
  if measured == 0:
    return {"measured": False, "n_runs": len(curves),
            "reason": "no run on this list carries a binary contact rate "
                      "(pre-2026-08-06 receipts, or --no-contact-probe)",
            "note": "do not substitute the weighted touch reward term"}
  with np.errstate(all="ignore"):
    lo = np.nanmin(arr, axis=0)
    hi = np.nanmax(arr, axis=0)
    mean = np.nanmean(arr, axis=0)
  k = max(1, n // 10)
  per_run_first, per_run_last = [], []
  for row in arr:
    f = row[:k][np.isfinite(row[:k])]
    l = row[-k:][np.isfinite(row[-k:])]
    per_run_first.append(float(f.mean()) if f.size else None)
    per_run_last.append(float(l.mean()) if l.size else None)
  spread = hi - lo
  finite_last = [v for v in per_run_last if v is not None]
  return {
    "measured": True,
    "n_runs": len(curves),
    "iterations_compared": n,
    "iterations_with_a_measurement": measured,
    "band_mean": [None if not np.isfinite(v) else float(v) for v in mean],
    "band_lo": [None if not np.isfinite(v) else float(v) for v in lo],
    "band_hi": [None if not np.isfinite(v) else float(v) for v in hi],
    "per_run_first": per_run_first,
    "per_run_last": per_run_last,
    "final_decile_band": _band(per_run_last),
    "final_decile_spread_x": (
      (max(finite_last) / min(finite_last))
      if finite_last and min(finite_last) > 0 else None),
    "abs_spread_mean": float(np.nanmean(spread)) if measured else None,
    "note": ("binary per-episode racket-ball contact rate; the band is over "
             "runs, and this is the number to quote -- always next to the "
             "zero-policy baseline measured on the same scene"),
  }


def report(run_paths, zero_policy_path, out_path, eval_paths=None) -> int:
  """Print the ONE reporting format this lane's evidence supports.

  PLAIN LANGUAGE.  This exists because the previous headline
  ("touch 4e-5 -> 0.21") was a weighted reward term read as a probability, from
  a single run, with no do-nothing baseline next to it -- three separate ways
  to be wrong at once, and together they made a policy that learned quite a lot
  look like one that barely moved.  So the format is fixed here in code:
  binary per-episode contact rate, against the zero policy on the same scene,
  over at least two runs, with the run-to-run band shown.  Anything the
  evidence does not support is REFUSED by name and exits 2 -- it does not
  degrade into a weaker claim.
  """
  runs = []
  for p in run_paths:
    p = Path(p)
    runs.append((p.stem, json.loads(p.read_text())))
  baseline = None
  if zero_policy_path:
    bp = Path(zero_policy_path)
    baseline = (bp.stem, json.loads(bp.read_text()))
  evals = []
  for p in (eval_paths or []):
    p = Path(p)
    evals.append((p.stem, json.loads(p.read_text())))

  refusals = report_refusals(runs, baseline, evals)
  if refusals:
    for code, why in refusals:
      _warn_block(f"REPORT REFUSED: {code}", why)
    _safe(Path(out_path).write_text, json.dumps(
      {"status": "refused", "exit_code": 2,
       "refusals": [{"code": c, "why": w} for c, w in refusals],
       "runs": [n for n, _ in runs],
       "evals": [n for n, _ in evals],
       "zero_policy": baseline[0] if baseline else None}, indent=2))
    print(f"[a3_train_ppo] REFUSED, wrote {out_path}")
    return 2

  bname, b = baseline
  base_rate = float(_dig(b, "stats.contact.fraction_of_episodes_with_a_racket_touch"))
  last = [float(_dig(r, "learning.binary_contact_rate."
                        "fraction_of_episodes_with_a_racket_touch_last"))
          for _, r in runs]
  first = [_dig(r, "learning.binary_contact_rate."
                   "fraction_of_episodes_with_a_racket_touch_first")
           for _, r in runs]
  band_last = _band(last)
  weighted = {t: [_dig(r, "learning.reward_terms_last." + t) for _, r in runs]
              for t in sorted(KERNEL_REWARD_TERMS)}
  out = {
    "status": "reported",
    "exit_code": 0,
    "headline_metric": BINARY_CONTACT_KEY,
    "runs": [n for n, _ in runs],
    "seeds": [r.get("seed") for _, r in runs],
    "iterations": [_dig(r, "learning.iterations") for _, r in runs],
    "zero_policy_baseline": {
      "receipt": bname,
      "fraction_of_episodes_with_a_racket_touch": base_rate,
      "policy_steps": b.get("policy_steps"),
      "nworld": b.get("nworld"),
    },
    "trained_on_policy_training_curve": {
      "per_run_first_decile": first,
      "per_run_last_decile": last,
      "band": band_last,
      "note": ("measured DURING training, so the policy still carries its "
               "exploration noise; this is the conservative number"),
    },
    # Suffixed on purpose.  An unqualified `gain_vs_zero_policy_x` sitting next
    # to a sentence quoting the deterministic eval is precisely the kind of
    # "which number is this?" ambiguity this whole section exists to kill.
    "gain_vs_zero_policy_x_on_policy_training_curve": [
      (v / base_rate) if base_rate > 0 else None for v in last],
    "run_to_run_spread_x_on_policy_training_curve": (
      (max(last) / min(last)) if min(last) > 0 else None),
    "weighted_reward_terms_for_context_only": {
      "values": weighted,
      "max_possible": _dig(runs[0][1], "learning.reward_terms_max_possible"),
      "warning": _NOT_A_PROBABILITY,
    },
    # Mandatory scope line.  Everything above is a statement about THIS lane;
    # this block says, from each receipt's own live ledger, whether it is also
    # a statement about the Isaac A211/C211 run.  Receipts written before
    # 2026-08-06 have no ledger and report `null` rather than a comfortable
    # default -- "not recorded" and "aligned" must never render the same.
    "isaac_alignment_scope": _report_alignment_scope(runs, evals),
  }
  # The deterministic evaluation, when it was handed over: same metric, same
  # scene, no exploration noise.  This is where "0.12% -> 49.2%/97.8%" came
  # from, so the sentence leads with it when it exists and says which it is.
  ev_rates = [float(_dig(e, "stats.contact."
                            "fraction_of_episodes_with_a_racket_touch"))
              for _, e in evals]
  if ev_rates:
    out["trained_deterministic_eval"] = {
      "receipts": [n for n, _ in evals],
      "per_run": ev_rates,
      "band": _band(ev_rates),
      "gain_vs_zero_policy_x": [(v / base_rate) if base_rate > 0 else None
                                for v in ev_rates],
      "policy_steps": [e.get("policy_steps") for _, e in evals],
    }
  headline = ev_rates or last
  which = ("deterministic eval" if ev_rates
           else "on-policy training curve, final decile")
  hband = _band(headline)
  out["headline_measurement"] = which
  out["headline_per_run"] = headline
  out["headline_band"] = hband
  out["headline_gain_vs_zero_policy_x"] = [
    (v / base_rate) if base_rate > 0 else None for v in headline]
  out["headline_run_to_run_spread_x"] = (
    (max(headline) / min(headline)) if min(headline) > 0 else None)
  out["sentence"] = (
    f"binary per-episode racket-ball contact rate ({which}): zero policy "
    f"{100.0 * base_rate:.2f}%  ->  trained "
    + " / ".join(f"{100.0 * v:.1f}%" for v in headline)
    + f"  (band {100.0 * hband['lo']:.1f}--{100.0 * hband['hi']:.1f}%"
      f" over {len(headline)} runs)")
  Path(out_path).write_text(json.dumps(out, indent=2))
  print("[a3_train_ppo][REPORT] " + out["sentence"], flush=True)
  print("[a3_train_ppo][REPORT] 人话: 零策略基本碰不到球, 训练后每局摸到球的"
        "比例见上; 括号里是 run 之间的散布, 单次跑不作数.", flush=True)
  print("[a3_train_ppo][REPORT] the weighted `touch_term_weighted` reward term "
        f"(ceiling {_dig(runs[0][1], 'learning.reward_terms_max_possible.touch_term_weighted')}) "
        "is NOT a contact rate and is printed here only for context: "
        + ", ".join(f"{n}={v:.3f}" if isinstance(v, (int, float)) else f"{n}={v}"
                    for n, v in
                    zip(out["runs"], weighted["touch_term_weighted"])),
        flush=True)
  print(f"[a3_train_ppo] wrote {out_path}")
  return 0


def main(argv=None) -> int:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--nworld", type=int, default=4096)
  p.add_argument("--iterations", type=int, default=60)
  p.add_argument("--num-steps-per-env", type=int, default=24)
  p.add_argument("--episode-s", type=float, default=3.0)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--xml-path", default=None,
                 help="explicit frozen A3 MJCF input (no default-path claim)")
  p.add_argument("--ready-pose", default=None,
                 help="explicit frozen split-ready JSON input")
  p.add_argument("--cone", choices=("pyramidal", "elliptic"), default="elliptic")
  p.add_argument("--no-pairs", action="store_true")
  p.add_argument("--njmax", type=int, default=572)
  p.add_argument("--nconmax", type=int, default=128)
  p.add_argument("--action-scale-mode", choices=("flat", "vendor"), default="flat",
                 help="'vendor' = the Isaac/deploy per-joint 0.25*effort/kp decoder")
  p.add_argument("--entropy-coef", type=float, default=0.002)
  p.add_argument("--init-std", type=float, default=1.0)
  p.add_argument("--experiment", default="a3_court_ppo")
  p.add_argument("--tag", default="run")
  p.add_argument("--log-root", default="/workspace/mjlab_lane/logs")
  p.add_argument("--out-prefix", default=None)
  p.add_argument("--rsl-rl-console", action="store_true",
                 help="also print rsl-rl's own iteration block")
  p.add_argument("--smoke", action="store_true",
                 help="64 worlds / 3 iterations, for wiring checks")
  p.add_argument("--analyze", nargs="+", default=None,
                 help="two or more run .jsonl/.json files -> N-seed band.  "
                      "One file is refused: a band over one run reports zero "
                      "spread, which is a false reproducibility claim")
  p.add_argument("--report", nargs="+", default=None,
                 help="two or more training .json receipts -> the one "
                      "reportable headline: binary per-episode contact rate "
                      "vs the zero policy, with the run-to-run band.  Refuses "
                      "(exit 2) anything the evidence does not support")
  p.add_argument("--report-zero-policy", default=None,
                 help="the `--eval zero` receipt measured on the same scene; "
                      "required by --report, because 'the contact rate rose' "
                      "is meaningless without the do-nothing number")
  p.add_argument("--report-eval", nargs="+", default=None,
                 help="optional: one `--eval ckpt` receipt per run, in the "
                      "same order.  This is the deterministic-policy number "
                      "(the 49.2%%/97.8%% in the record); without it --report "
                      "falls back to the on-policy training curve and says so")
  p.add_argument("--out", default="BAND.json")
  p.add_argument("--eval", choices=("zero", "ckpt"), default=None,
                 help="score a fixed policy instead of training")
  p.add_argument("--eval-ckpt", default=None)
  p.add_argument("--eval-steps", type=int, default=750)
  p.add_argument("--no-contact-probe", action="store_true",
                 help="turn the per-substep ball<->racket contact counter OFF "
                      "(training AND eval).  The run then records "
                      "fraction_of_episodes_with_a_racket_touch = null, and "
                      "--report will refuse it -- only use it to time the "
                      "probe's own cost")
  p.add_argument("--no-capacity-probe", action="store_true",
                 help="turn the capacity gate OFF.  The run then records "
                      "verdict=NOT_MEASURED and cannot claim the njmax/nconmax "
                      "caps held -- only use it to time the gate's own cost")
  p.add_argument("--warn-scan-log", default=None,
                 help="path this run's stdout is being teed into.  At the end "
                      "we count mujoco-warp's own 'overflow' lines in it and "
                      "put them in the receipt; any such line fails the run "
                      "even if the GPU-side gate saw nothing")
  a = p.parse_args(argv)

  if a.analyze:
    return analyze(a.analyze, a.out)

  if a.report:
    return report(a.report, a.report_zero_policy, a.out, a.report_eval)

  if a.smoke:
    a.nworld, a.iterations = 64, 3
    a.tag = a.tag if a.tag != "run" else "smoke"
  if a.out_prefix is None:
    a.out_prefix = f"/workspace/mjlab_lane/TRAIN_{a.tag}_seed{a.seed}"
  if a.eval:
    return evaluate(a)
  return train(a)


if __name__ == "__main__":
  raise SystemExit(main())
