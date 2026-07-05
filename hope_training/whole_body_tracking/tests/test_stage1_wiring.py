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

Run:  /opt/anaconda3/bin/python3 hope_training/whole_body_tracking/tests/test_stage1_wiring.py
"""

from __future__ import annotations

import importlib.util
import os
import tempfile

import numpy as np
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
    np.savez(path, **flat)
    return flat


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
    return bank


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
    # missing clip -> KeyError
    p1 = os.path.join(tmpdir, "missing_clip.npz")
    np.savez(p1, **{"forehand/contact_pos_env": np.zeros(3),
                    "forehand/demanded_vel": np.zeros((2, 3)),
                    "forehand/demanded_normal": np.zeros((2, 3))})
    try:
        qb.load_question_bank(p1)
        raise AssertionError("missing backhand clip did not raise")
    except KeyError:
        pass
    # empty / mismatched question arrays -> ValueError
    p2 = os.path.join(tmpdir, "mismatched.npz")
    np.savez(p2, **{"forehand/contact_pos_env": np.zeros(3),
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


def main():
    qb = _load_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        bank_path = os.path.join(tmpdir, "bank.npz")
        flat = _write_bank(bank_path)
        bank = test_load_shapes_keying_padding(qb, bank_path, flat)
        test_select_fixed_point_and_matching_rows(qb, bank, flat)
        test_loud_errors(qb, tmpdir)
    print("ALL STAGE-1 WIRING TESTS PASSED")


if __name__ == "__main__":
    main()
