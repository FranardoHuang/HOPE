"""Regression tests for the mjlab-lane capacity gate (a3_train_ppo.py).

PLAIN LANGUAGE.  MuJoCo Warp does not stop when it runs out of pre-allocated
constraint rows or contact slots -- it drops the surplus and keeps going, so
the training curve stays pretty while the physics underneath it is wrong.  The
gate in ``a3_train_ppo.py`` exists to stop the run when that happens.  These
tests pin the parts of it that were measurably wrong on 2026-08-06:

  * the receipt divided the contact headroom by the wrong counter, which
    overstated it by roughly 2x (``naconmax`` also caps the broadphase
    candidate-pair array, which is always the bigger number);
  * a run that never took a single physics step could still publish
    ``verdict: PASS_NO_OVERFLOW``;
  * ``nefc == njmax`` was reported as an overflow although it fits exactly;
  * the engine's own stdout warnings were never counted anywhere.

The full acceptance for the gate is the mutation battery on the pod (feed it a
too-small ``--nconmax`` and check it actually fires).  What is testable without
a GPU is the arithmetic and the verdict logic, and that is what lives here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

def _lane_dir() -> Path:
  """Where the lane sources live: in the repo, or on the pod at /workspace."""
  here = (Path(__file__).resolve().parents[1]
          / "hope_training" / "whole_body_tracking" / "mjlab_lane")
  if (here / "a3_train_ppo.py").is_file():
    return here
  # The pod runs the same files out of /workspace/mjlab_lane (byte-identical
  # copies -- the sync step checks md5 both ways).
  return Path("/workspace/mjlab_lane")


@pytest.fixture(scope="module")
def mod():
  """Import a3_train_ppo, skipping where the mujoco stack is not installed."""
  pytest.importorskip("mujoco", reason="mjlab lane needs the mujoco stack")
  pytest.importorskip("mujoco_warp", reason="mjlab lane needs mujoco_warp")
  lane = _lane_dir()
  if not (lane / "a3_train_ppo.py").is_file():
    pytest.skip(f"mjlab lane sources not found at {lane}")
  sys.path.insert(0, str(lane))
  try:
    import a3_train_ppo  # noqa: PLC0415
  finally:
    sys.path.remove(str(lane))
  return a3_train_ppo


# --------------------------------------------------------------------------
# The bitmask we now gate on is the engine's, not one of our own invention.
# --------------------------------------------------------------------------


def test_flag_names_match_the_engines_own_enum(mod):
  from mujoco_warp._src.types import OverflowType

  expected = tuple(f.name for f in sorted(OverflowType, key=lambda f: f.value))
  assert mod.OVERFLOW_FLAGS == expected
  # Bit values must line up too, or a fired gate would name the wrong axis.
  for name, bit in mod.OVERFLOW_BIT.items():
    assert bit == int(getattr(OverflowType, name))


def test_broadphase_bit_decodes_by_name(mod):
  # BROADPHASE is the axis the old watchdog was blind to, and the one the
  # historical CUDA fault came from.  It must come back out by name.
  assert mod.decode_overflow_mask(mod.OVERFLOW_BIT["BROADPHASE"]) == ["BROADPHASE"]
  both = mod.OVERFLOW_BIT["BROADPHASE"] | mod.OVERFLOW_BIT["NARROWPHASE"]
  assert mod.decode_overflow_mask(both) == ["BROADPHASE", "NARROWPHASE"]
  assert mod.decode_overflow_mask(0) == []


def test_capacity_overflow_message_names_the_flags(mod):
  exc = mod.CapacityOverflow(mod.OVERFLOW_BIT["BROADPHASE"], "env step 16")
  text = str(exc)
  assert "CAPACITY_OVERFLOW" in text
  assert "BROADPHASE" in text
  assert "env step 16" in text


# --------------------------------------------------------------------------
# T2 / P10: the receipt arithmetic.
# --------------------------------------------------------------------------


def _fields(mod, **kw):
  base = dict(njmax=572, naconmax=32768, nefc_peak=100, nacon_peak=1000,
              ncollision_peak=2500, worlds_flagged=0, overflow_mask=0,
              samples_step=960, samples_forward=15)
  base.update(kw)
  return mod.capacity_fields(**base)


def test_contact_headroom_divides_by_the_broadphase_counter(mod):
  # naconmax caps candidate pairs too, and there are always more of those.
  f = _fields(mod, nacon_peak=1884, ncollision_peak=4694)
  assert f["naconmax_binding_peak_all_worlds"] == 4694
  assert f["naconmax_headroom_x"] == pytest.approx(32768 / 4694)
  # The old (wrong) answer divided by nacon and was 2.5x more generous.
  assert f["naconmax_headroom_x"] < 32768 / 1884


def test_nefc_exactly_at_njmax_is_not_an_overflow(mod):
  # forward.py drops a row on `nefc > njmax`; `==` fits exactly.  The old
  # `>=` comparison cried wolf on a run that was actually fine.
  f = _fields(mod, nefc_peak=572)
  assert f["nefc_over_njmax"] is False
  assert f["nefc_exactly_fills_njmax"] is True
  assert _fields(mod, nefc_peak=573)["nefc_over_njmax"] is True


def test_no_stepped_samples_yields_no_headroom_numbers(mod):
  f = _fields(mod, samples_step=0, samples_forward=1, nefc_peak=0,
              nacon_peak=0, ncollision_peak=0)
  assert f["njmax_headroom_x"] is None
  assert f["naconmax_headroom_x"] is None
  assert f["nefc_peak_per_world_running"] is None


# --------------------------------------------------------------------------
# T4 / T5: verdicts.  PASS is the only one that needs evidence.
# --------------------------------------------------------------------------


class _FakeEnv:
  def __init__(self, mod, *, cap_ok=True, **kw):
    self._mod = mod
    self._cap_ok = cap_ok
    self._kw = kw
    self.num_envs = 256
    self.sim = SimpleNamespace(wp_data=SimpleNamespace(njmax=572,
                                                       naconmax=32768))

  def capacity_snapshot(self):
    return _fields(self._mod, **self._kw)


def test_probe_off_is_not_measured_never_pass(mod):
  out = mod._capacity_summary(_FakeEnv(mod, cap_ok=False))
  assert out["verdict"] == "NOT_MEASURED"


def test_zero_stepped_samples_is_no_samples_never_pass(mod):
  out = mod._capacity_summary(_FakeEnv(mod, samples_step=0, samples_forward=1,
                                       nefc_peak=0, nacon_peak=0,
                                       ncollision_peak=0))
  assert out["verdict"] == "NO_SAMPLES"
  assert out["njmax_headroom_x"] is None


def test_clean_measured_run_passes(mod):
  assert mod._capacity_summary(_FakeEnv(mod))["verdict"] == "PASS_NO_OVERFLOW"


def test_any_engine_flag_is_an_overflow(mod):
  out = mod._capacity_summary(
    _FakeEnv(mod, overflow_mask=mod.OVERFLOW_BIT["BROADPHASE"],
             worlds_flagged=256))
  assert out["verdict"] == "OVERFLOW"
  assert out["overflow_flags"] == ["BROADPHASE"]


def test_engine_printf_alone_still_fails_the_run(mod):
  # Second channel: even if our GPU read came back clean, engine warnings on
  # stdout mean the two channels disagree and the run is not usable.
  out = mod._capacity_summary(_FakeEnv(mod), {"lines": 1134})
  assert out["verdict"] == "OVERFLOW_PRINTF_ONLY"
  assert out["warp_overflow_printf_lines"] == 1134


# --------------------------------------------------------------------------
# T12: the log scan must find the engine's lines and not our own receipt keys.
# --------------------------------------------------------------------------


def test_log_scan_counts_engine_lines_only(mod, tmp_path):
  log = tmp_path / "run.log"
  log.write_text(
    "[it 0] R_ep=1.0 fps=45000\n"
    "broadphase overflow - please increase nconmax to 11 or naconmax to 2561\n"
    "nefc overflow - please increase njmax to 71\n"
    "nvmax overflow: world 3 needs 9 active DOFs but nvmax = 8 (behavior undefined)\n"
    '  "overflow_mask": 0,\n'          # our own receipt key -- must NOT count
    '  "overflow_flags": [],\n'
  )
  out = mod.scan_warp_overflow_warnings(log)
  assert out["scanned"] is True
  assert out["lines"] == 3
  assert out["by_marker"] == {"broadphase overflow": 1, "nefc overflow": 1,
                              "nvmax overflow": 1}


def test_log_scan_catches_the_epa_line_that_never_says_overflow(mod, tmp_path):
  # EPA_HORIZON prints "Warning: EPA horizon = N isn't large enough." -- no
  # "overflow" in it, so every grep for that word has always missed it.
  log = tmp_path / "epa.log"
  log.write_text("Warning: EPA horizon = 12 isn't large enough.\n")
  out = mod.scan_warp_overflow_warnings(log)
  assert out["lines"] == 1
  assert out["by_marker"] == {"EPA horizon": 1}


def test_log_scan_on_a_clean_log_is_zero(mod, tmp_path):
  log = tmp_path / "clean.log"
  log.write_text('[it 0] ok\n  "overflow_mask": 0,\n  "overflow_flags": [],\n')
  assert mod.scan_warp_overflow_warnings(log)["lines"] == 0


def test_log_scan_missing_file_does_not_explode(mod, tmp_path):
  out = mod.scan_warp_overflow_warnings(tmp_path / "nope.log")
  assert out["scanned"] is False
  assert out["lines"] == 0


# --------------------------------------------------------------------------
# T7a: the receipt has to say which GPU it actually got, not which it asked for.
# --------------------------------------------------------------------------


def test_uuid_cross_check_against_nvidia_smi(mod):
  uuid = "GPU-473a79f3-8736-6c7f-c3db-290c6be385b8"
  smi = {"compute_procs": [f"2997556, {uuid}, 744 MiB",
                           "12345, GPU-deadbeef-0000-0000-0000-000000000000, 1 MiB"]}
  ok = mod._match_uuid_against_smi(uuid, 2997556, smi)
  assert ok["device_uuid_matches_nvidia_smi"] is True
  assert ok["nvidia_smi_lines_for_this_pid_on_another_uuid"] == []
  # A process that shows up on a *different* card must not be reported as a match.
  bad = mod._match_uuid_against_smi(uuid, 12345, smi)
  assert bad["device_uuid_matches_nvidia_smi"] is False
  assert bad["nvidia_smi_lines_for_this_pid_on_another_uuid"]


def test_no_smi_evidence_is_not_a_match(mod):
  out = mod._match_uuid_against_smi("GPU-abc", 999, {"compute_procs": []})
  assert out["device_uuid_matches_nvidia_smi"] is False
