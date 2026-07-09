"""Stage-1 question-bank wiring unit test — NO Isaac imports.

Loads ``tasks/tracking/mdp/stage1_question_bank.py`` directly by file path (the mdp package
``__init__`` pulls isaaclab, so the module under test is imported standalone — same pattern as
gen_stage1_questions.py loading virtual_ball.py). Covers:

* load_question_bank: per-clip shapes, clip keying (0=forehand, 1=backhand), Q padding to the
  shared max with per-clip counts, zero padding rows, difficulty passthrough, loud errors on a
  missing clip / mismatched arrays.
* select_questions: a fake resample-index selection (deterministic u draws) returns the FIXED
  contact point of each env's clip and the vel/normal/difficulty rows of the SAME question index,
  and padding rows are never selected (u -> 1 clamps to counts-1).
* grip calibration (0g): a synthetic wrist rotation + the registry grip -> the expected WORLD
  face normal AND grip-rotated blade position/velocity mount FK (pins the mount-offset-rotates
  decision — a CONSTANT ~1.6 cm blade shift, NOT the bake's ~13 cm wrist-chain re-solve), plus
  the generator's single-application guard (SystemExit) catching a double rally rotation.
* rally-yaw canonicalization: rotate by psi then -psi = identity; -psi-about-+Z convention;
  vertical vectors invariant.
* fail-closed bake detection (audit round 2): corrupted face_normal_reliable markers, *_cal
  stem/registry disagreement, _cal-sibling refusal, pinned-session mismatch, baked-under---grip
  off reporting.
* loader meta enforcement (audit round 2): banks without meta_json / non-JSON meta / flags not
  both true are refused with ValueError; allow_legacy=True is the explicit escape hatch.
* --anchor train-candidate: strike frame anchored at the FIRST entry of the registry's
  independent train_phase_candidates field; a clip without the field refuses (SystemExit);
  the default annotated anchor keeps the registry `phase` frame.

Run:  /opt/anaconda3/bin/python3 hope_training/whole_body_tracking/tests/test_stage1_wiring.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.join(
    HERE, "..", "source", "whole_body_tracking", "whole_body_tracking",
    "tasks", "tracking", "mdp", "stage1_question_bank.py",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("s1_qbank", os.path.abspath(MODULE_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _meta_bytes(grip_applied=True, rally_yaw_applied=True):
    """meta_json payload as the generator writes it (real JSON, loader-enforced flags)."""
    return np.frombuffer(json.dumps({"grip_applied": grip_applied,
                                     "rally_yaw_applied": rally_yaw_applied}).encode("utf-8"),
                         dtype=np.uint8)


def _write_bank(path, forehand_q=3, backhand_q=2):
    """Tiny synthetic bank with recognizable per-row values (row i of clip c encodes (c, i))."""
    def rows(c, q, base):
        return np.stack([[base + 10 * c + i, base + 0.1 * i, base - 0.1 * c] for i in range(q)]).astype(np.float64)

    flat = {
        "forehand/contact_pos_env": np.array([0.48, -0.38, 0.87]),
        "forehand/demanded_vel": rows(0, forehand_q, 100.0),
        "forehand/demanded_normal": rows(0, forehand_q, 1.0),
        "forehand/difficulty_deg": np.arange(forehand_q, dtype=np.float64) + 0.5,
        "backhand/contact_pos_env": np.array([0.52, -0.04, 1.05]),
        "backhand/demanded_vel": rows(1, backhand_q, 200.0),
        "backhand/demanded_normal": rows(1, backhand_q, 2.0),
        "backhand/difficulty_deg": np.arange(backhand_q, dtype=np.float64) + 20.5,
        # extra generator keys the loader must ignore
        "forehand/incoming_vel": rows(0, forehand_q, -3.0),
        "backhand/incoming_vel": rows(1, backhand_q, -3.0),
    }
    np.savez(path, meta_json=_meta_bytes(), **flat)
    return flat


# ------------------------------------------------------------------------------------------- #
# pytest fixtures — the same objects main() threads through the direct-run path, so the file
# works both ways: ``pytest test_stage1_wiring.py`` and ``python3 test_stage1_wiring.py``.
# ------------------------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def qb():
    return _load_module()


@pytest.fixture(scope="module")
def gen():
    return _load_script("gen_stage1_questions.py", "s1_gen")


@pytest.fixture(scope="module")
def asp(gen):
    return gen._load_analyzer()


@pytest.fixture(scope="module")
def _bank_on_disk(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("s1_bank") / "bank.npz")
    return path, _write_bank(path)


@pytest.fixture(scope="module")
def bank_path(_bank_on_disk):
    return _bank_on_disk[0]


@pytest.fixture(scope="module")
def flat(_bank_on_disk):
    return _bank_on_disk[1]


@pytest.fixture
def bank(qb, bank_path):
    return qb.load_question_bank(bank_path)


@pytest.fixture
def tmpdir(tmp_path):
    # plain-str tmp dir, matching what main() passes (tests os.path.join onto it)
    return str(tmp_path)


def test_load_shapes_keying_padding(qb, bank_path, flat):
    bank = qb.load_question_bank(bank_path)
    # shared Q_max = 3 (forehand), per-clip counts kept
    assert bank.contact_pos.shape == (2, 3), bank.contact_pos.shape
    assert bank.demanded_vel.shape == (2, 3, 3), bank.demanded_vel.shape
    assert bank.demanded_normal.shape == (2, 3, 3), bank.demanded_normal.shape
    assert bank.difficulty_deg.shape == (2, 3), bank.difficulty_deg.shape
    assert bank.counts.tolist() == [3, 2], bank.counts.tolist()
    # clip keying: row 0 = forehand, row 1 = backhand
    assert torch.allclose(bank.contact_pos[0], torch.tensor(flat["forehand/contact_pos_env"], dtype=torch.float32))
    assert torch.allclose(bank.contact_pos[1], torch.tensor(flat["backhand/contact_pos_env"], dtype=torch.float32))
    assert torch.allclose(
        bank.demanded_vel[1, :2], torch.tensor(flat["backhand/demanded_vel"], dtype=torch.float32)
    )
    # backhand row 2 is padding (beyond counts[1]=2) and must be zero
    assert torch.all(bank.demanded_vel[1, 2] == 0) and torch.all(bank.demanded_normal[1, 2] == 0)
    assert float(bank.difficulty_deg[1, 2]) == 0.0
    print("[ok] load: shapes / clip keying / padding / counts")


def test_select_fixed_point_and_matching_rows(qb, bank, flat):
    # fake resample: 6 envs, mixed clips, u chosen to hit known question indices
    clip = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    #                    q=0    q=1    q=2  |  q=0    q=1    q=1 (u->1 clamps below count=2)
    u = torch.tensor([0.0, 0.34, 0.99, 0.0, 0.51, 0.999999])
    pos, vel, nrm, diff = qb.select_questions(bank, clip, u)
    expected_q = [0, 1, 2, 0, 1, 1]
    names = ["forehand", "forehand", "forehand", "backhand", "backhand", "backhand"]
    for i, (name, q) in enumerate(zip(names, expected_q)):
        # position is ALWAYS the clip's fixed contact point, independent of q
        assert torch.allclose(pos[i], torch.tensor(flat[f"{name}/contact_pos_env"], dtype=torch.float32)), i
        # velocity and normal come from the SAME question row q
        assert torch.allclose(vel[i], torch.tensor(flat[f"{name}/demanded_vel"][q], dtype=torch.float32)), i
        assert torch.allclose(nrm[i], torch.tensor(flat[f"{name}/demanded_normal"][q], dtype=torch.float32)), i
        assert abs(float(diff[i]) - float(flat[f"{name}/difficulty_deg"][q])) < 1e-6, i
    # padding is unreachable: every backhand draw stays below count 2
    u_sweep = torch.rand(512)
    _, vel_b, _, _ = qb.select_questions(bank, torch.ones(512, dtype=torch.long), u_sweep)
    assert torch.all(vel_b[:, 0] >= 200.0), "a padding (zero) row leaked into backhand selection"
    print("[ok] select: fixed contact point per clip, matching vel/normal/difficulty rows, no padding leak")


def test_loud_errors(qb, tmpdir):
    # missing clip -> KeyError (valid meta so the meta gate is not what fires)
    p1 = os.path.join(tmpdir, "missing_clip.npz")
    np.savez(p1, meta_json=_meta_bytes(),
             **{"forehand/contact_pos_env": np.zeros(3),
                "forehand/demanded_vel": np.zeros((2, 3)),
                "forehand/demanded_normal": np.zeros((2, 3))})
    try:
        qb.load_question_bank(p1)
        raise AssertionError("missing backhand clip did not raise")
    except KeyError:
        pass
    # empty / mismatched question arrays -> ValueError
    p2 = os.path.join(tmpdir, "mismatched.npz")
    np.savez(p2, meta_json=_meta_bytes(),
             **{"forehand/contact_pos_env": np.zeros(3),
                "forehand/demanded_vel": np.zeros((2, 3)),
                "forehand/demanded_normal": np.zeros((3, 3)),
                "backhand/contact_pos_env": np.zeros(3),
                "backhand/demanded_vel": np.zeros((1, 3)),
                "backhand/demanded_normal": np.zeros((1, 3))})
    try:
        qb.load_question_bank(p2)
        raise AssertionError("mismatched vel/normal shapes did not raise")
    except ValueError:
        pass
    print("[ok] errors: missing clip raises KeyError, mismatched arrays raise ValueError")


def test_loader_meta_enforcement(qb, tmpdir):
    """Banks without meta_json / non-JSON meta / flags not both true -> ValueError;
    allow_legacy=True is the explicit escape hatch (audit round 2)."""
    arrays = {"forehand/contact_pos_env": np.zeros(3),
              "forehand/demanded_vel": np.ones((2, 3)),
              "forehand/demanded_normal": np.ones((2, 3)),
              "backhand/contact_pos_env": np.zeros(3),
              "backhand/demanded_vel": np.ones((1, 3)),
              "backhand/demanded_normal": np.ones((1, 3))}
    cases = {
        "no_meta.npz": {},                                    # missing meta_json
        "repr_meta.npz": {"meta_json": np.frombuffer(        # pre-audit python-repr meta
            repr({"grip_applied": True}).encode(), dtype=np.uint8)},
        "grip_off.npz": {"meta_json": _meta_bytes(grip_applied=False)},
        "no_rally.npz": {"meta_json": _meta_bytes(rally_yaw_applied=False)},
    }
    for fname, extra in cases.items():
        p = os.path.join(tmpdir, fname)
        np.savez(p, **arrays, **extra)
        try:
            qb.load_question_bank(p)
            raise AssertionError(f"{fname}: un-provable bank loaded without allow_legacy")
        except ValueError:
            pass
        bank = qb.load_question_bank(p, allow_legacy=True)   # explicit escape hatch loads
        assert bank.counts.tolist() == [2, 1]
    # flags both true -> loads without the escape hatch
    p_ok = os.path.join(tmpdir, "ok_meta.npz")
    np.savez(p_ok, meta_json=_meta_bytes(), **arrays)
    assert qb.load_question_bank(p_ok).counts.tolist() == [2, 1]
    print("[ok] loader meta: missing/repr/false-flag banks refused, allow_legacy escape hatch works")


def test_face_command_obs_vector(qb):
    """Contract-day 179-D lane layout: [normal (3), rho placeholder (1) = 0]."""
    normal = torch.randn(7, 3)
    v = qb.face_command_obs_vector(normal)
    assert v.shape == (7, 4), v.shape
    assert torch.equal(v[:, :3], normal), "normal columns must pass through unmodified"
    assert torch.all(v[:, 3] == 0.0), "rho placeholder column must be zero-filled"
    # zeros in (bank off) -> zeros out, still 4-D
    z = qb.face_command_obs_vector(torch.zeros(5, 3))
    assert z.shape == (5, 4) and torch.all(z == 0.0)
    print("[ok] obs: face_command_obs_vector = [normal(3), rho=0(1)] (N,4)")


def _load_script(fname, modname):
    """Load a scripts/*.py module by path (top-level imports are numpy-light; planner/torch
    imports live inside main())."""
    path = os.path.join(HERE, "..", "scripts", fname)
    spec = importlib.util.spec_from_file_location(modname, os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generator_split_determinism():
    """question_split: pure function of v_in (order/seed independent), ~80/20, disjoint sides."""
    gen = _load_script("gen_stage1_questions.py", "s1_gen")

    rng = np.random.default_rng(123)
    v = gen.sample_incoming(rng, 4000, 2.0, 5.0, 0.6, -2.0, 0.3)
    lab = np.array([gen.question_split(row) for row in v])
    # pure function of the row: shuffled evaluation order gives identical per-row labels
    perm = rng.permutation(len(v))
    lab_shuffled = np.empty_like(lab)
    lab_shuffled[perm] = np.array([gen.question_split(row) for row in v[perm]])
    assert (lab == lab_shuffled).all(), "split membership depended on evaluation order"
    frac = float((lab == "exam").mean())
    assert 0.15 < frac < 0.25, f"exam fraction {frac} outside the ~20% band"
    # disjointness: no row is ever on both sides (trivial by construction, asserted anyway)
    assert not (set(map(tuple, v[lab == "train"])) & set(map(tuple, v[lab == "exam"])))
    print(f"[ok] split: deterministic per-question membership, exam fraction {frac:.1%}, disjoint")


def _write_motion_npz(path, wrist_quat, wrist_pos=(0.3, -0.2, 1.0),
                      wrist_v=(0.1, 0.0, 0.0), wrist_w=(0.0, 0.0, 1.0), T=4):
    """Minimal synthetic motion npz for analyze(): identity bodies except the wrist (body 31)."""
    nb = 32
    pos = np.zeros((T, nb, 3)); pos[:, 31] = wrist_pos
    quat = np.zeros((T, nb, 4)); quat[..., 0] = 1.0; quat[:, 31] = wrist_quat
    lin = np.zeros((T, nb, 3)); lin[:, 31] = wrist_v
    ang = np.zeros((T, nb, 3)); ang[:, 31] = wrist_w
    np.savez(path, fps=np.array([50]), body_pos_w=pos, body_quat_w=quat,
             body_lin_vel_w=lin, body_ang_vel_w=ang)


def test_grip_calibration_synthetic(gen, asp, tmpdir):
    """Synthetic wrist rotation + known grip -> expected WORLD normal + rotated mount FK,
    and the single-application guard fires on a doubled rally rotation."""
    # Wrist = 90 deg about world +Z; expected R_wrist hardcoded (independent of quat_to_rot).
    s2 = np.sqrt(0.5)
    R_wrist = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    npz = os.path.join(tmpdir, "syn_motion.npz")
    wrist_pos, wrist_v, wrist_w = (0.3, -0.2, 1.0), (0.1, 0.0, 0.0), (0.0, 0.0, 1.0)
    _write_motion_npz(npz, (s2, 0.0, 0.0, s2), wrist_pos, wrist_v, wrist_w)

    # Independent Rg = Rz(+5)Rx(+40) built in-test; matches the registry's wrist-local blade dir.
    a, b = np.deg2rad(5.0), np.deg2rad(40.0)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    Rx = np.array([[1.0, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    Rg_expected = Rz @ Rx
    assert np.allclose(Rg_expected @ [0, 1, 0], [-0.0668, 0.7631, 0.6428], atol=1e-3), \
        "Rg @ y_hat does not match the registry's wrist-local blade dir ~[-0.07, 0.76, 0.64]"
    Rg = asp.grip_rotation(5.0, 40.0)
    assert np.allclose(Rg, Rg_expected, atol=1e-12), "grip_rotation != Rz(alpha_z) @ Rx(beta_x)"

    info = asp.analyze("synclip", npz, use_blade=True, grip_rot=Rg)
    exp_normal = R_wrist @ Rg_expected @ np.array([0.0, 1.0, 0.0])
    assert np.allclose(info["normal_w"][0], exp_normal, atol=1e-9), \
        f"grip world normal {info['normal_w'][0]} != R_wrist @ Rg @ y_hat {exp_normal}"
    # MOUNT-OFFSET DECISION pinned: blade position AND velocity use the grip-rotated mount
    # (matches the csv_to_npz_mujoco --grip-rot bake; ASSUMPTION flag for franco).
    exp_off = R_wrist @ Rg_expected @ asp.MOUNT_OFFSET
    assert np.allclose(info["racket_w"][0], np.asarray(wrist_pos) + exp_off, atol=1e-9), \
        "blade position must FK through the grip-ROTATED mount offset"
    exp_vel = np.asarray(wrist_v) + np.cross(wrist_w, exp_off)
    assert np.allclose(info["racket_v_w"][0], exp_vel, atol=1e-9), \
        "blade velocity must use omega x (grip-rotated offset)"
    # --grip off (grip_rot None) reproduces the raw wrist+Y proxy face
    raw = asp.analyze("synclip", npz, use_blade=True)
    assert np.allclose(raw["normal_w"][0], R_wrist @ [0.0, 1.0, 0.0], atol=1e-9)

    # SINGLE-APPLICATION GUARD: analyzer-path face + one rally rotation passes; doubled fails
    # with SystemExit (explicit raise — survives python -O, unlike a bare assert).
    n_pipe = gen.rally_canonicalize(info["normal_w"][0], 40.0)
    gen._face_self_check(asp, npz, 0, Rg, 40.0, n_pipe)
    doubled = gen.rally_canonicalize(n_pipe, 40.0)
    try:
        gen._face_self_check(asp, npz, 0, Rg, 40.0, doubled)
        raise AssertionError("a double-rotated face slipped through the guard")
    except SystemExit as exc:
        assert "single-application guard FAILED" in str(exc)
    print("[ok] grip: R_wrist@Rg@y_hat world normal, rotated mount pos/vel FK, guard catches x2 rally")


def test_anchor_train_candidate(gen, asp, tmpdir):
    """--anchor train-candidate: strike frame = round(candidates[0] * (T-1)) (FIRST registry
    entry, same phase->frame convention as _annotation_frame); absent field -> SystemExit;
    default annotated anchor keeps the registry `phase` frame."""
    npz = os.path.join(tmpdir, "anch_motion.npz")
    _write_motion_npz(npz, (1.0, 0.0, 0.0, 0.0), T=11)
    ann = {"phase": 0.9, "face_normal_reliable": False,
           "train_phase_candidates": [0.3, 0.6]}
    annotations = {"anch_motion": ann}
    # default (annotated) anchor: frame = round(0.9 * 10) = 9, phase recorded from the registry
    st = gen.strike_state_from_clip(asp, "bh", npz, annotations)
    assert st["strike_frame"] == 9 and st["anchor"] == "annotated", (st["strike_frame"], st["anchor"])
    assert abs(st["anchor_phase"] - 0.9) < 1e-12, st["anchor_phase"]
    # train-candidate anchor: FIRST candidate 0.3 (not 0.6) -> frame round(0.3 * 10) = 3
    st = gen.strike_state_from_clip(asp, "bh", npz, annotations, anchor="train-candidate")
    assert st["strike_frame"] == 3 and st["anchor"] == "train-candidate", \
        (st["strike_frame"], st["anchor"])
    assert abs(st["anchor_phase"] - 0.3) < 1e-12, st["anchor_phase"]
    # registry entry without the field refuses loudly (never falls back to `phase`)
    bare = {"anch_motion": {"phase": 0.9, "face_normal_reliable": False}}
    try:
        gen.strike_state_from_clip(asp, "bh", npz, bare, anchor="train-candidate")
        raise AssertionError("train-candidate anchor without the registry field did not refuse")
    except SystemExit as exc:
        assert "train_phase_candidates" in str(exc), exc
    print("[ok] anchor: train-candidate uses FIRST registry candidate, absent field refuses")


def test_bake_detection_fail_closed(gen, asp):
    """_grip_is_baked + resolve_grip_and_yaw refuse ambiguous registry states (audit round 2)."""
    def expect_exit(fn, *frags):
        try:
            fn()
            raise AssertionError(f"expected SystemExit mentioning {frags}")
        except SystemExit as exc:
            for f in frags:
                assert f in str(exc), f"{f!r} not in {exc}"

    # marker semantics: exactly False -> raw; 'true-calibrated...' -> baked; no entry -> raw
    assert gen._grip_is_baked("c", {"face_normal_reliable": False}) is False
    assert gen._grip_is_baked("c", {"face_normal_reliable": "true-calibrated"}) is True
    assert gen._grip_is_baked("c", None) is False
    # corrupted markers refuse loudly (fail-closed): bool True, strings, missing field
    for bad in (True, "yes", "false", None):
        expect_exit(lambda b=bad: gen._grip_is_baked("clipx", {"face_normal_reliable": b}),
                    "clipx", "bake marker")

    grip_block = {"sess": {"alpha_z_deg": 5, "beta_x_deg": 40}}
    raw = {"face_normal_reliable": False, "rally_yaw_deg": 10}
    baked = {"face_normal_reliable": "true-calibrated", "rally_yaw_deg": 10}
    # *_cal stem whose registry entry lacks the bake marker -> refuse
    expect_exit(lambda: gen.resolve_grip_and_yaw(
        asp, "fh", "/x/foo_cal.npz", {"foo_cal": dict(raw)}, grip_block, "registry"),
        "foo_cal", "bake marker")
    # raw clip with a *_cal sibling registered -> refuse (anchors must come from the _cal npz)
    expect_exit(lambda: gen.resolve_grip_and_yaw(
        asp, "fh", "/x/bar.npz", {"bar": dict(raw), "bar_cal": dict(baked)},
        grip_block, "registry"), "bar_cal", "_cal npz")
    # pinned session != the single available session -> refuse
    expect_exit(lambda: gen.resolve_grip_and_yaw(
        asp, "fh", "/x/baz.npz", {"baz": dict(raw, grip_session="other")},
        grip_block, "registry"), "grip_session", "'sess'")
    # baked clip reports mode 'baked' even under --grip off (meta must say grip_applied=True)
    r, desc, yaw, mode = gen.resolve_grip_and_yaw(
        asp, "fh", "/x/qux_cal.npz", {"qux_cal": dict(baked)}, grip_block, "off")
    assert mode == "baked" and r is None and yaw == 10.0, (mode, r, yaw)
    # raw clip without a sibling still resolves (registry mode, Rg present)
    r, desc, yaw, mode = gen.resolve_grip_and_yaw(
        asp, "fh", "/x/solo.npz", {"solo": dict(raw)}, grip_block, "registry")
    assert mode == "registry" and r is not None and yaw == 10.0, (mode, r, yaw)
    print("[ok] bake detection: fail-closed markers, _cal stem/sibling refusals, pinned-session"
          " mismatch, baked-under-off")


def test_rally_canonicalization(gen):
    """rally_canonicalize: rotate by psi then -psi = identity; -psi about +Z convention;
    vertical vectors and norms invariant; batch (N,3) supported."""
    rng = np.random.default_rng(7)
    v = rng.standard_normal((64, 3))
    for psi in (40.0, -60.0, 123.4):
        back = gen.rally_canonicalize(gen.rally_canonicalize(v, psi), -psi)
        assert np.allclose(back, v, atol=1e-12), f"psi={psi}: canonicalize(-psi) did not invert"
        assert np.allclose(np.linalg.norm(gen.rally_canonicalize(v, psi), axis=1),
                           np.linalg.norm(v, axis=1), atol=1e-12)
    # convention: canonicalize = rotate by -psi about +Z, so psi=+90 sends +X to -Y
    assert np.allclose(gen.rally_canonicalize(np.array([1.0, 0.0, 0.0]), 90.0),
                       [0.0, -1.0, 0.0], atol=1e-12)
    # vertical axis: z-vectors invariant; psi=0 = identity
    assert np.allclose(gen.rally_canonicalize(np.array([0.0, 0.0, 1.0]), 77.0),
                       [0.0, 0.0, 1.0], atol=1e-12)
    assert np.allclose(gen.rally_canonicalize(v, 0.0), v, atol=1e-12)
    print("[ok] rally: psi/-psi identity, -psi-about-+Z convention, vertical invariant")


def main():
    qb = _load_module()
    gen = _load_script("gen_stage1_questions.py", "s1_gen")
    asp = gen._load_analyzer()
    with tempfile.TemporaryDirectory() as tmpdir:
        bank_path = os.path.join(tmpdir, "bank.npz")
        flat = _write_bank(bank_path)
        test_load_shapes_keying_padding(qb, bank_path, flat)
        bank = qb.load_question_bank(bank_path)
        test_select_fixed_point_and_matching_rows(qb, bank, flat)
        test_loud_errors(qb, tmpdir)
        test_loader_meta_enforcement(qb, tmpdir)
        test_grip_calibration_synthetic(gen, asp, tmpdir)
        test_anchor_train_candidate(gen, asp, tmpdir)
    test_face_command_obs_vector(qb)
    test_generator_split_determinism()
    test_rally_canonicalization(gen)
    test_bake_detection_fail_closed(gen, asp)
    print("ALL STAGE-1 WIRING TESTS PASSED")


if __name__ == "__main__":
    main()
