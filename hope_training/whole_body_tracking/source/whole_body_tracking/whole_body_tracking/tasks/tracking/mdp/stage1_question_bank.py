"""Stage-1 question-bank loading + selection (torch/numpy ONLY — no Isaac imports).

The bank is produced offline by ``scripts/gen_stage1_questions.py``: per clip, a FIXED contact
point plus per-question inverse-solved racket velocity + face normal (the ANSWER to a sampled
incoming ball, StrikeSpecPlanner LM solve). Keys are ``"<clip>/contact_pos_env"`` (3,),
``"<clip>/demanded_vel"`` / ``"<clip>/demanded_normal"`` (Q, 3) and optionally
``"<clip>/difficulty_deg"`` (Q,), with clip names ordered by clip_id (forehand=0, backhand=1).
Positions are tracking-env-frame (env origin at the robot spawn, table surface z=0.76); world
targets are ``env_origin + pos``. This module is deliberately standalone (like virtual_ball.py)
so the loading/selection logic is unit-testable without the training env.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import torch


class QuestionBank(NamedTuple):
    """Per-clip question tensors. Clips share Q_max rows; rows >= counts[c] are zero padding
    (never selected — :func:`select_questions` draws indices below the per-clip count)."""

    contact_pos: torch.Tensor  # (C, 3) fixed strike point per clip, tracking-env frame
    demanded_vel: torch.Tensor  # (C, Q_max, 3) solved racket velocity per question
    demanded_normal: torch.Tensor  # (C, Q_max, 3) solved face normal per question
    counts: torch.Tensor  # (C,) long — valid question rows per clip
    difficulty_deg: torch.Tensor  # (C, Q_max) face-vs-clip-normal angle (0 where absent)


def load_question_bank(
    path: str, device: str | torch.device = "cpu", clip_names: Sequence[str] = ("forehand", "backhand")
) -> QuestionBank:
    """Load a stage-1 bank npz ONCE into per-clip float32 tensors on ``device``.

    Every name in ``clip_names`` must be present in the bank with >= 1 question — a missing or
    empty clip raises (do not train a clip on silent zeros). ``clip_names`` order defines the
    clip_id indexing (must match RacketTargetCommand._clip_names: 0=forehand, 1=backhand).
    """
    data = np.load(path)
    per_clip = []
    for name in clip_names:
        key = f"{name}/contact_pos_env"
        if key not in data:
            raise KeyError(
                f"question bank {path!r}: no entry for clip {name!r} (missing {key!r}); "
                f"regenerate with gen_stage1_questions.py --clip {name}:<motion.npz>"
            )
        contact = np.asarray(data[key], dtype=np.float32).reshape(3)
        vel = np.asarray(data[f"{name}/demanded_vel"], dtype=np.float32).reshape(-1, 3)
        nrm = np.asarray(data[f"{name}/demanded_normal"], dtype=np.float32).reshape(-1, 3)
        if len(vel) == 0 or vel.shape != nrm.shape:
            raise ValueError(
                f"question bank {path!r} clip {name!r}: demanded_vel {vel.shape} / "
                f"demanded_normal {nrm.shape} must be matching non-empty (Q, 3) arrays"
            )
        dkey = f"{name}/difficulty_deg"
        diff = (
            np.asarray(data[dkey], dtype=np.float32).reshape(-1)
            if dkey in data
            else np.zeros(len(vel), dtype=np.float32)
        )
        if len(diff) != len(vel):
            raise ValueError(
                f"question bank {path!r} clip {name!r}: difficulty_deg has {len(diff)} rows, "
                f"expected {len(vel)}"
            )
        per_clip.append((contact, vel, nrm, diff))

    q_max = max(len(v) for _, v, _, _ in per_clip)
    n_clips = len(per_clip)
    contact_pos = torch.zeros(n_clips, 3)
    demanded_vel = torch.zeros(n_clips, q_max, 3)
    demanded_normal = torch.zeros(n_clips, q_max, 3)
    counts = torch.zeros(n_clips, dtype=torch.long)
    difficulty = torch.zeros(n_clips, q_max)
    for c, (contact, vel, nrm, diff) in enumerate(per_clip):
        q = len(vel)
        contact_pos[c] = torch.from_numpy(contact)
        demanded_vel[c, :q] = torch.from_numpy(vel)
        demanded_normal[c, :q] = torch.from_numpy(nrm)
        difficulty[c, :q] = torch.from_numpy(diff)
        counts[c] = q
    return QuestionBank(
        contact_pos=contact_pos.to(device),
        demanded_vel=demanded_vel.to(device),
        demanded_normal=demanded_normal.to(device),
        counts=counts.to(device),
        difficulty_deg=difficulty.to(device),
    )


def face_command_obs_vector(normal: torch.Tensor) -> torch.Tensor:
    """Contract-day 179-D face-command lane: ``[demanded normal (3), rho placeholder (1)]``.

    rho (the S3 spin-lane scalar) is reserved NOW, zero-filled, so the actor layout matches the
    frozen 175 -> 179 contract-day decision and S3 needs no further contract change / ladder
    retrain. Pure tensor helper (unit-tested without the env); the obs term wraps it.
    """
    return torch.cat([normal, normal.new_zeros(normal.shape[:-1] + (1,))], dim=-1)


def select_questions(
    bank: QuestionBank, clip_ids: torch.Tensor, u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map uniform draws ``u`` in [0, 1) to one bank row per env for its clip.

    Returns ``(contact_pos, demanded_vel, demanded_normal, difficulty_deg)``, each row-selected by
    (clip_ids, floor(u * counts[clip])). The index is clamped below the per-clip count so zero
    padding rows are never selected. Pure tensor function — unit-tested without the env.
    """
    counts = bank.counts[clip_ids]
    q = (u * counts.to(u.dtype)).long().clamp_(min=0)
    q = torch.minimum(q, counts - 1)
    return (
        bank.contact_pos[clip_ids],
        bank.demanded_vel[clip_ids, q],
        bank.demanded_normal[clip_ids, q],
        bank.difficulty_deg[clip_ids, q],
    )
