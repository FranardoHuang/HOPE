"""Action-conditioned continuous-question producer contracts (CPU torch, Isaac Lab stubbed).

This file is intentionally about the producer boundary, not the adapter in isolation:

* every redraw remains in a flat proposal ledger after a later draw is admitted;
* fixed-direction acceptance is recomputed by the scorer rollout at its exact ``h``/horizon and
  explicit ``net_top_z`` (which already includes the ball radius);
* the emitted normal is signed to the runtime raw +Y reference face, including a backhand face;
* arbitrary N=5/N=93 action tables preserve shape and deterministic generator replay.
"""

from __future__ import annotations

import inspect
import os
import pathlib
import struct
import sys
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
REPO = pathlib.Path(HERE).resolve().parents[2]

from test_reward_flags_mdp import _PKG, _load  # noqa: E402 (installs the Isaac Lab stub)

MDP_DIR = str(
    REPO / "hope_training" / "whole_body_tracking" / "source" / "whole_body_tracking"
    / "whole_body_tracking" / "tasks" / "tracking" / "mdp"
)
sys.modules[_PKG].__path__ = [MDP_DIR]


def _mdp(name):
    return _load(f"{_PKG}.{name}", f"{name}.py")


@pytest.fixture(scope="module")
def cq():
    _mdp("virtual_ball")
    _mdp("strike_spec_torch")
    _mdp("stroke_prototypes_torch")
    _mdp("stroke_adapt_torch")
    return _mdp("continuous_questions")


@pytest.fixture(scope="module")
def prm():
    vb = _mdp("virtual_ball")
    return vb.load_venue_params(str(REPO / "configs" / "ball_physics_venue.yaml"))


def _protos(n, face_sign=1.0):
    signs = torch.as_tensor(face_sign, dtype=torch.float32)
    if signs.ndim == 0:
        signs = signs.expand(n).clone()
    elif tuple(signs.shape) != (n,):
        raise ValueError(f"face_sign test fixture must be scalar or ({n},)")
    return types.SimpleNamespace(
        v_hat_b=torch.tensor([[1.0, 0.0, 0.0]]).expand(n, 3).clone(),
        speed_min=torch.full((n,), 1.0),
        speed_max=torch.full((n,), 4.0),
        face_sign=signs,
    )


def _base_quat(n):
    q = torch.zeros(n, 4)
    q[:, 0] = 1.0
    return q


def _fake_solver(*args, **kwargs):
    p_strike, _v_ball, _w_ball, target_xy, d_hat_w = args[:5]
    n = p_strike.shape[0]
    speed = torch.full((n,), 2.0, device=p_strike.device, dtype=p_strike.dtype)
    return {
        "p_contact": p_strike,
        "v_r": speed[:, None] * d_hat_w,
        # Deliberately +x for every action.  The producer must hemisphere-align backhands.
        "n": torch.tensor([1.0, 0.0, 0.0], device=p_strike.device,
                          dtype=p_strike.dtype).expand(n, 3).clone(),
        "speed": speed,
        "landing_xy": target_xy.clone(),
        "resid_m": torch.zeros(n, device=p_strike.device, dtype=p_strike.dtype),
        "net_z": torch.full((n,), 99.0, device=p_strike.device, dtype=p_strike.dtype),
        "clears_net": torch.ones(n, device=p_strike.device, dtype=torch.bool),
        "ok": torch.ones(n, device=p_strike.device, dtype=torch.bool),
        "reason": torch.full((n,), -1, device=p_strike.device, dtype=torch.long),
        "q": torch.zeros(n, 3, device=p_strike.device, dtype=p_strike.dtype),
    }


def _identity_contact(v_minus, _v_r, _n, omega_minus, _prm):
    # Make the incoming x speed visible to the fake scorer below.
    return v_minus, omega_minus


def _fixed_cfg(cq, **kwargs):
    base = dict(
        fixed_direction=True,
        n_iters=1,
        max_redraw_rounds=1,
        aim_x_range=(2.5, 2.5),
        aim_y_range=(0.0, 0.0),
    )
    base.update(kwargs)
    return cq.ContinuousQuestionCfg(**base)


def _varied_protos(n):
    y = torch.linspace(-0.35, 0.35, n)
    d = torch.stack([torch.ones(n), y, torch.linspace(0.0, 0.2, n)], dim=-1)
    d = d / torch.linalg.norm(d, dim=-1, keepdim=True)
    return types.SimpleNamespace(
        v_hat_b=d,
        speed_min=torch.full((n,), 1.0),
        speed_max=torch.full((n,), 4.0),
        # The raw clip face alternates below; multiplying by this sign keeps
        # the physical contact face +X for the shared incoming-ball fixture.
        face_sign=torch.where(
            torch.arange(n) % 2 == 0,
            torch.ones(n),
            -torch.ones(n),
        ),
    )


def _external_rows(n):
    clip_ids = torch.arange(n, dtype=torch.long)
    p_contact = torch.stack([
        torch.linspace(0.52, 0.60, n),
        torch.linspace(-0.20, 0.20, n),
        torch.linspace(0.90, 1.10, n),
    ], dim=-1)
    v_ball_in = torch.stack([
        torch.linspace(-3.0, -4.0, n),
        torch.linspace(-0.2, 0.2, n),
        torch.linspace(-0.1, 0.1, n),
    ], dim=-1)
    w_ball_in = torch.stack([
        torch.linspace(-2.0, 2.0, n),
        torch.linspace(1.0, -1.0, n),
        torch.linspace(0.0, 3.0, n),
    ], dim=-1)
    aim_xy = torch.tensor([[2.5, 0.0]]).expand(n, 2).clone()
    ref_normal = torch.tensor([[1.0, 0.0, 0.0]]).expand(n, 3).clone()
    ref_normal[1::2, 0] = -1.0
    return {
        "clip_ids": clip_ids,
        "p_contact": p_contact,
        "v_ball_in": v_ball_in,
        "w_ball_in": w_ball_in,
        "aim_xy": aim_xy,
        "ref_normal": ref_normal,
        "base_quat": _base_quat(n),
    }


def _legal_scorer(p0, _v0, _w0, _prm, *, surface_z, net_x, h, n_steps):
    del surface_z, net_x, h, n_steps
    n = p0.shape[0]
    return {
        "land_xy": torch.tensor([2.5, 0.0], device=p0.device,
                                dtype=p0.dtype).expand(n, 2).clone(),
        "land_valid": torch.ones(n, device=p0.device, dtype=torch.bool),
        "net_z": torch.full((n,), 0.95, device=p0.device, dtype=p0.dtype),
        "net_valid": torch.ones(n, device=p0.device, dtype=torch.bool),
    }


def test_rejected_hard_draw_and_admitted_redraw_are_both_accounted(
    cq, prm, monkeypatch,
):
    """Round one's hard speed bucket cannot disappear when round two is admitted."""
    monkeypatch.setattr(cq, "solve_strike_specs_fixed_dir", _fake_solver)
    monkeypatch.setattr(cq, "predict_paddle_contact", _identity_contact)

    queued = iter([
        torch.tensor([[0.55, 0.00, 1.00]]),   # round 1 contact
        torch.tensor([[-5.1, 0.00, 0.00]]),   # round 1 hard speed bucket
        torch.tensor([[0.56, 0.01, 1.01]]),   # round 2 contact
        torch.tensor([[-3.0, 0.00, 0.00]]),   # round 2 easy speed bucket
    ])

    def fixed_boxes(_box_rows, _gen, device, dtype):
        return next(queued).to(device=device, dtype=dtype)

    monkeypatch.setattr(cq, "_uniform_box", fixed_boxes)

    def scorer(p0, v0, _w0, _prm, *, surface_z, net_x, h, n_steps):
        del p0, surface_z, net_x, h, n_steps
        hard = v0[:, 0] < -5.0
        n = v0.shape[0]
        return {
            "land_xy": torch.tensor([2.5, 0.0], device=v0.device,
                                    dtype=v0.dtype).expand(n, 2).clone(),
            "land_valid": torch.ones(n, device=v0.device, dtype=torch.bool),
            "net_z": torch.where(hard, torch.full_like(v0[:, 0], 0.90),
                                 torch.full_like(v0[:, 0], 0.94)),
            "net_valid": torch.ones(n, device=v0.device, dtype=torch.bool),
        }

    monkeypatch.setattr(cq, "coarse_landing", scorer)
    out = cq.generate(
        torch.tensor([0]), prm, surface_z=0.78, net_x=1.87,
        net_top_z=0.9325, cfg=_fixed_cfg(cq, max_redraw_rounds=2),
        ref_normal=torch.tensor([[1.0, 0.0, 0.0]]),
        protos=_protos(1, face_sign=1.0), base_quat=_base_quat(1),
        generator=torch.Generator().manual_seed(7),
    )

    assert out.ok.tolist() == [True]
    assert out.proposal_count == 2 == len(out.proposals)
    assert out.proposals.request_index.tolist() == [0, 0]
    assert out.proposals.clip_id.tolist() == [0, 0]
    assert out.proposals.round_index.tolist() == [1, 2]
    assert out.proposals.v_ball_in[:, 0].tolist() == pytest.approx([-5.1, -3.0])
    assert out.proposals.admitted.tolist() == [False, True]
    assert out.proposals.reason_code.tolist() == [5, -1]  # net_not_cleared, admitted
    # A curriculum can bucket by action and any raw ball axis without relying on the final redraw.
    hard_bucket = (out.proposals.clip_id == 0) & (out.proposals.v_ball_in[:, 0] < -5.0)
    assert int(hard_bucket.sum()) == 1
    assert not bool(out.proposals.admitted[hard_bucket].any())
    assert out.reason_counts == {"net_not_cleared": 1}


def test_fixed_direction_uses_exact_scorer_rollout_and_backhand_face(
    cq, prm, monkeypatch,
):
    """A clearance 0.5 mm above explicit net top passes; no second ball radius is added."""
    seen = {"solver": [], "scorer": []}

    def solver(*args, **kwargs):
        seen["solver"].append(dict(kwargs))
        return _fake_solver(*args, **kwargs)

    monkeypatch.setattr(cq, "solve_strike_specs_fixed_dir", solver)
    monkeypatch.setattr(cq, "predict_paddle_contact", _identity_contact)

    def scorer(p0, _v0, _w0, _prm, *, surface_z, net_x, h, n_steps):
        seen["scorer"].append((surface_z, net_x, h, n_steps))
        n = p0.shape[0]
        return {
            "land_xy": torch.tensor([2.5, 0.0], device=p0.device,
                                    dtype=p0.dtype).expand(n, 2).clone(),
            "land_valid": torch.ones(n, device=p0.device, dtype=torch.bool),
            # Correct threshold is 0.9325.  Adding ball radius again would reject 0.9330.
            "net_z": torch.full((n,), 0.9330, device=p0.device, dtype=p0.dtype),
            "net_valid": torch.ones(n, device=p0.device, dtype=torch.bool),
        }

    monkeypatch.setattr(cq, "coarse_landing", scorer)
    out = cq.generate(
        torch.tensor([0]), prm, surface_z=0.78, net_x=1.87, net_top_z=0.9325,
        cfg=_fixed_cfg(cq), ref_normal=torch.tensor([[-1.0, 0.0, 0.0]]),
        protos=_protos(1, face_sign=-1.0), base_quat=_base_quat(1),
        generator=torch.Generator().manual_seed(11), h=0.007, n_steps=137,
    )

    assert out.ok.tolist() == [True]
    assert float(torch.sum(out.n_racket[0] * torch.tensor([-1.0, 0.0, 0.0]))) > 0.0
    assert float(out.n_racket[0, 0]) < 0.0, "backhand raw +Y face was flipped to +x"
    assert seen["scorer"] == [(0.78, 1.87, 0.007, 137)]
    assert seen["solver"][0]["h"] == 0.007
    assert seen["solver"][0]["n_steps"] == 137
    assert seen["solver"][0]["net_height"] == pytest.approx(0.9325 - 0.78)
    assert seen["solver"][0]["ball_radius"] == 0.0
    assert seen["solver"][0]["net_margin_m"] == 0.0


def test_fixed_direction_speed_budget_caps_action_prototype(
    cq, prm, monkeypatch,
):
    seen = {}

    def solver(*args, **kwargs):
        seen["speed_max"] = args[6].clone()
        return _fake_solver(*args, **kwargs)

    monkeypatch.setattr(cq, "solve_strike_specs_fixed_dir", solver)
    monkeypatch.setattr(cq, "predict_paddle_contact", _identity_contact)
    monkeypatch.setattr(cq, "coarse_landing", _legal_scorer)
    rows = _external_rows(1)
    out = cq.solve_proposals(
        rows["clip_ids"],
        rows["p_contact"],
        rows["v_ball_in"],
        rows["w_ball_in"],
        rows["aim_xy"],
        rows["ref_normal"],
        protos=_protos(1),
        base_quat=rows["base_quat"],
        prm=prm,
        surface_z=0.78,
        net_x=1.87,
        net_top_z=0.9325,
        cfg=_fixed_cfg(cq, speed_budget=2.5),
    )
    assert out.ok.tolist() == [True]
    assert seen["speed_max"].tolist() == pytest.approx([2.5])


def test_selected_physical_face_must_approach_inside_contact_fit(
    cq, prm, monkeypatch,
):
    monkeypatch.setattr(cq, "solve_strike_specs_fixed_dir", _fake_solver)
    monkeypatch.setattr(cq, "predict_paddle_contact", _identity_contact)
    monkeypatch.setattr(cq, "coarse_landing", _legal_scorer)
    rows = _external_rows(1)
    rows["ref_normal"][0] = torch.tensor([-1.0, 0.0, 0.0])

    wrong_face = cq.solve_proposals(
        rows["clip_ids"],
        rows["p_contact"],
        rows["v_ball_in"],
        rows["w_ball_in"],
        rows["aim_xy"],
        rows["ref_normal"],
        protos=_protos(1, face_sign=1.0),
        base_quat=rows["base_quat"],
        prm=prm,
        surface_z=0.78,
        net_x=1.87,
        net_top_z=0.9325,
        cfg=_fixed_cfg(cq),
    )
    assert wrong_face.ok.tolist() == [False]
    assert wrong_face.proposals.reason_code.tolist() == [7]
    assert wrong_face.reason_counts == {
        "contact_normal_speed_out_of_fit": 1
    }

    correct_face = cq.solve_proposals(
        rows["clip_ids"],
        rows["p_contact"],
        rows["v_ball_in"],
        rows["w_ball_in"],
        rows["aim_xy"],
        rows["ref_normal"],
        protos=_protos(1, face_sign=-1.0),
        base_quat=rows["base_quat"],
        prm=prm,
        surface_z=0.78,
        net_x=1.87,
        net_top_z=0.9325,
        cfg=_fixed_cfg(cq),
    )
    assert correct_face.ok.tolist() == [True]


def test_contact_normal_speed_outside_venue_fit_is_rejected(
    cq, prm, monkeypatch,
):
    monkeypatch.setattr(cq, "solve_strike_specs_fixed_dir", _fake_solver)
    monkeypatch.setattr(cq, "predict_paddle_contact", _identity_contact)
    monkeypatch.setattr(cq, "coarse_landing", _legal_scorer)
    rows = _external_rows(1)
    rows["v_ball_in"][0] = torch.tensor([-10.0, 0.0, 0.0])
    out = cq.solve_proposals(
        rows["clip_ids"],
        rows["p_contact"],
        rows["v_ball_in"],
        rows["w_ball_in"],
        rows["aim_xy"],
        rows["ref_normal"],
        protos=_protos(1),
        base_quat=rows["base_quat"],
        prm=prm,
        surface_z=0.78,
        net_x=1.87,
        net_top_z=0.9325,
        cfg=_fixed_cfg(cq),
    )
    assert out.ok.tolist() == [False]
    assert out.proposals.reason_code.tolist() == [7]


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"net_top_z": None}, "finite scorer net_top_z"),
        ({"net_top_z": float("nan")}, "finite scorer net_top_z"),
        ({"net_top_z": 0.77}, "above the scorer landing plane"),
        ({"h": 0.0}, "finite positive scorer step"),
        ({"n_steps": 0}, "positive integer scorer horizon"),
        ({"n_steps": 2.5}, "positive integer scorer horizon"),
    ],
)
def test_fixed_direction_rejects_invalid_scorer_geometry(
    cq, prm, overrides, match,
):
    kwargs = dict(net_top_z=0.9325, h=0.01, n_steps=100)
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        cq.generate(
            torch.tensor([0]), prm, surface_z=0.78, net_x=1.87,
            cfg=_fixed_cfg(cq), ref_normal=torch.tensor([[1.0, 0.0, 0.0]]),
            protos=_protos(1), base_quat=_base_quat(1), **kwargs,
        )


@pytest.mark.parametrize("tol_m", [0.0, -0.01, float("nan"), float("inf")])
def test_fixed_direction_rejects_invalid_tolerance(cq, prm, tol_m):
    with pytest.raises(ValueError, match="tol_m must be finite and positive"):
        cq.generate(
            torch.tensor([0]), prm, surface_z=0.78, net_x=1.87, net_top_z=0.9325,
            cfg=_fixed_cfg(cq, tol_m=tol_m),
            ref_normal=torch.tensor([[1.0, 0.0, 0.0]]),
            protos=_protos(1), base_quat=_base_quat(1),
        )


@pytest.mark.parametrize(
    "ref_normal,base_quat,match",
    [
        (torch.zeros(1, 3), _base_quat(1), "finite non-zero raw"),
        (torch.tensor([[float("nan"), 0.0, 0.0]]), _base_quat(1), "finite non-zero raw"),
        (torch.tensor([[1.0, 0.0, 0.0]]), torch.zeros(1, 4), "unit wxyz"),
        (torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[2.0, 0.0, 0.0, 0.0]]),
         "unit wxyz"),
        (torch.tensor([[1.0, 0.0, 0.0]]),
         torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]), "finite"),
    ],
)
def test_fixed_direction_rejects_invalid_face_or_base(
    cq, prm, ref_normal, base_quat, match,
):
    with pytest.raises(ValueError, match=match):
        cq.generate(
            torch.tensor([0]), prm, surface_z=0.78, net_x=1.87, net_top_z=0.9325,
            cfg=_fixed_cfg(cq), ref_normal=ref_normal,
            protos=_protos(1), base_quat=base_quat,
        )


def test_fixed_direction_rejects_empty_request(cq, prm):
    with pytest.raises(ValueError, match="at least one requested row"):
        cq.generate(
            torch.empty(0, dtype=torch.long), prm,
            surface_z=0.78, net_x=1.87, net_top_z=0.9325,
            cfg=_fixed_cfg(cq), ref_normal=torch.empty(0, 3),
            protos=_protos(1), base_quat=torch.empty(0, 4),
        )


@pytest.mark.parametrize("n_actions", [5, 93])
def test_fixed_direction_arbitrary_n_shapes_and_determinism(
    cq, prm, monkeypatch, n_actions,
):
    monkeypatch.setattr(cq, "solve_strike_specs_fixed_dir", _fake_solver)
    monkeypatch.setattr(cq, "predict_paddle_contact", _identity_contact)

    def legal_scorer(p0, _v0, _w0, _prm, *, surface_z, net_x, h, n_steps):
        del surface_z, net_x, h, n_steps
        n = p0.shape[0]
        return {
            "land_xy": torch.tensor([2.5, 0.0], device=p0.device,
                                    dtype=p0.dtype).expand(n, 2).clone(),
            "land_valid": torch.ones(n, device=p0.device, dtype=torch.bool),
            "net_z": torch.full((n,), 0.95, device=p0.device, dtype=p0.dtype),
            "net_valid": torch.ones(n, device=p0.device, dtype=torch.bool),
        }

    monkeypatch.setattr(cq, "coarse_landing", legal_scorer)
    clip_ids = torch.arange(n_actions, dtype=torch.long)
    ref = torch.tensor([[1.0, 0.0, 0.0]]).expand(n_actions, 3).clone()
    ref[1::2, 0] = -1.0
    common = dict(
        clip_ids=clip_ids,
        prm=prm,
        surface_z=0.78,
        net_x=1.87,
        net_top_z=0.9325,
        cfg=_fixed_cfg(cq),
        ref_normal=ref,
        protos=_protos(n_actions, face_sign=ref[:, 0]),
        base_quat=_base_quat(n_actions),
    )
    a = cq.generate(**common, generator=torch.Generator().manual_seed(20260727))
    b = cq.generate(**common, generator=torch.Generator().manual_seed(20260727))

    assert a.p_contact.shape == (n_actions, 3)
    assert a.v_racket.shape == (n_actions, 3)
    assert a.n_racket.shape == (n_actions, 3)
    assert a.proposal_count == n_actions
    assert a.proposals.clip_id.tolist() == list(range(n_actions))
    assert bool(a.ok.all())
    assert bool((torch.sum(a.n_racket * ref, dim=-1) > 0.0).all())
    for field in (
        "p_contact", "v_racket", "n_racket", "v_ball_in", "w_ball_in",
        "aim_xy", "ok", "resid_m",
    ):
        assert torch.equal(getattr(a, field), getattr(b, field)), field
    for field in (
        "request_index", "clip_id", "round_index", "p_contact", "v_ball_in",
        "w_ball_in", "aim_xy", "reason_code", "admitted", "resid_m",
    ):
        assert torch.equal(getattr(a.proposals, field), getattr(b.proposals, field)), field


def test_selected_direction_is_full_table_equivalent_without_n_squared_allocation(cq):
    """The arbitrary-N fast path returns the same selected rows as the old full (M,K,3) table."""
    adapt = _mdp("stroke_adapt_torch")
    n = 93
    gen = torch.Generator().manual_seed(93)
    v_hat_b = torch.randn(n, 3, generator=gen)
    v_hat_b = v_hat_b / torch.linalg.norm(v_hat_b, dim=-1, keepdim=True)
    clip_ids = torch.randperm(n, generator=gen)
    yaw = torch.rand(n, generator=gen) * 2.0 - 1.0
    selected = cq._selected_direction_world(
        v_hat_b, clip_ids, yaw, torch.device("cpu"), torch.float32)
    full = adapt.direction_world(v_hat_b, yaw)
    expected = full[torch.arange(n), clip_ids]
    assert selected.shape == (n, 3)
    assert torch.allclose(selected, expected, rtol=0.0, atol=1e-7)


@pytest.mark.parametrize("n_actions", [1, 5, 93])
def test_solve_proposals_is_exact_once_fixed_action_and_input_immutable(
    cq, prm, monkeypatch, n_actions,
):
    """External proposals are certified once; no internal draw may replace any input row."""
    monkeypatch.setattr(cq, "solve_strike_specs_fixed_dir", _fake_solver)
    monkeypatch.setattr(cq, "predict_paddle_contact", _identity_contact)
    monkeypatch.setattr(cq, "coarse_landing", _legal_scorer)

    def forbidden_sampler(*_args, **_kwargs):
        raise AssertionError("solve_proposals must not sample")

    monkeypatch.setattr(cq, "_uniform_box", forbidden_sampler)
    rows = _external_rows(n_actions)
    protos = _varied_protos(n_actions)
    snapshots = {name: value.clone() for name, value in rows.items()}
    proto_snapshots = {
        "v_hat_b": protos.v_hat_b.clone(),
        "speed_min": protos.speed_min.clone(),
        "speed_max": protos.speed_max.clone(),
        "face_sign": protos.face_sign.clone(),
    }
    out = cq.solve_proposals(
        rows["clip_ids"], rows["p_contact"], rows["v_ball_in"], rows["w_ball_in"],
        rows["aim_xy"], rows["ref_normal"], protos=protos,
        base_quat=rows["base_quat"], prm=prm, surface_z=0.78, net_x=1.87,
        net_top_z=0.9325, cfg=_fixed_cfg(cq), h=0.007, n_steps=137,
    )

    assert out.proposal_count == n_actions == len(out.proposals)
    assert out.rounds_used == 1 and out.exhausted == 0
    assert out.proposals.request_index.tolist() == list(range(n_actions))
    assert out.proposals.clip_id.tolist() == list(range(n_actions))
    assert out.proposals.round_index.tolist() == [1] * n_actions
    assert out.proposals.admitted.tolist() == [True] * n_actions
    assert out.proposals.reason_code.tolist() == [-1] * n_actions
    assert out.proposal_host_packet.reason_codes == (-1,) * n_actions
    assert out.proposal_host_packet.admitted == (True,) * n_actions
    assert out.proposal_host_packet.racket_velocity_rows == tuple(
        tuple(float(value) for value in row)
        for row in out.v_racket.tolist()
    )
    assert out.proposal_host_packet.racket_normal_rows == tuple(
        tuple(float(value) for value in row)
        for row in out.n_racket.tolist()
    )
    assert out.proposal_host_packet.residual_rows == tuple(
        float(value) for value in out.resid_m.tolist()
    )
    assert torch.equal(out.proposals.p_contact, rows["p_contact"])
    assert torch.equal(out.proposals.v_ball_in, rows["v_ball_in"])
    assert torch.equal(out.proposals.w_ball_in, rows["w_ball_in"])
    assert torch.equal(out.proposals.aim_xy, rows["aim_xy"])
    assert torch.equal(out.proposals.ref_normal, rows["ref_normal"])
    assert torch.equal(out.proposals.base_quat, rows["base_quat"])
    assert torch.equal(out.p_contact, rows["p_contact"])
    assert torch.equal(out.v_ball_in, rows["v_ball_in"])
    assert torch.equal(out.w_ball_in, rows["w_ball_in"])
    assert torch.equal(out.aim_xy, rows["aim_xy"])
    # Identity base yaw + fake solver v_r=2*d proves each row used its exact selected action.
    actual_dir = out.v_racket / torch.linalg.norm(out.v_racket, dim=-1, keepdim=True)
    expected_dir = protos.v_hat_b[rows["clip_ids"]]
    # Two float32 unit normalisations (prototype -> world direction -> measured v direction).
    assert torch.allclose(actual_dir, expected_dir, rtol=0.0, atol=5e-7)
    assert bool((torch.sum(out.n_racket * rows["ref_normal"], dim=-1) > 0.0).all())

    for name, before in snapshots.items():
        assert torch.equal(rows[name], before), f"input mutated: {name}"
    for name, before in proto_snapshots.items():
        assert torch.equal(getattr(protos, name), before), f"prototype mutated: {name}"


def test_diagnostic_prevalidated_solver_matches_ordinary_valid_batch(
    cq, prm, monkeypatch,
):
    lm_authorities = []

    def recording_solver(*args, **kwargs):
        lm_authorities.append(
            kwargs.get("_diagnostic_fixed_try_lm_authority")
        )
        return _fake_solver(*args, **kwargs)

    monkeypatch.setattr(
        cq, "solve_strike_specs_fixed_dir", recording_solver
    )
    monkeypatch.setattr(cq, "predict_paddle_contact", _identity_contact)
    monkeypatch.setattr(cq, "coarse_landing", _legal_scorer)
    rows = _external_rows(5)
    protos = _varied_protos(5)
    kwargs = {
        "protos": protos,
        "base_quat": rows["base_quat"],
        "prm": prm,
        "surface_z": 0.78,
        "net_x": 1.87,
        "net_top_z": 0.9325,
        "cfg": _fixed_cfg(cq),
        "h": 0.007,
        "n_steps": 137,
    }

    ordinary = cq.solve_proposals(
        rows["clip_ids"],
        rows["p_contact"],
        rows["v_ball_in"],
        rows["w_ball_in"],
        rows["aim_xy"],
        rows["ref_normal"],
        **kwargs,
    )
    diagnostic = cq.solve_proposals(
        rows["clip_ids"],
        rows["p_contact"],
        rows["v_ball_in"],
        rows["w_ball_in"],
        rows["aim_xy"],
        rows["ref_normal"],
        _diagnostic_prevalidated_authority=(
            cq._DIAGNOSTIC_PREVALIDATED_SOLVE_AUTHORITY
        ),
        **kwargs,
    )
    assert lm_authorities == [
        None,
        cq._DIAGNOSTIC_FIXED_TRY_LM_AUTHORITY,
    ]

    for field in (
        "p_contact",
        "v_racket",
        "n_racket",
        "v_ball_in",
        "w_ball_in",
        "aim_xy",
        "ok",
        "resid_m",
        "attempted_v_ball_in",
    ):
        assert torch.equal(
            getattr(diagnostic, field), getattr(ordinary, field)
        ), field
    for field in (
        "request_index",
        "clip_id",
        "round_index",
        "p_contact",
        "v_ball_in",
        "w_ball_in",
        "aim_xy",
        "reason_code",
        "admitted",
        "resid_m",
        "ref_normal",
        "base_quat",
    ):
        assert torch.equal(
            getattr(diagnostic.proposals, field),
            getattr(ordinary.proposals, field),
        ), field
    assert diagnostic.reason_counts == ordinary.reason_counts
    assert diagnostic.exhausted == ordinary.exhausted
    assert diagnostic.proposal_count == ordinary.proposal_count
    assert (
        diagnostic.proposal_host_packet
        == ordinary.proposal_host_packet
    )


def _host_packet_float32_bits(packet):
    return (
        packet.reason_codes,
        packet.admitted,
        tuple(
            tuple(struct.pack("<f", value) for value in row)
            for row in packet.racket_velocity_rows
        ),
        tuple(
            tuple(struct.pack("<f", value) for value in row)
            for row in packet.racket_normal_rows
        ),
        tuple(
            struct.pack("<f", value)
            for value in packet.residual_rows
        ),
    )


def test_diagnostic_host_only_solver_omits_public_result_scaffolding(cq):
    source = inspect.getsource(
        cq._solve_proposals_diagnostic_host_only
    )
    assert source.count("_solve_fixed_direction_batch(") == 1
    assert source.count("_build_proposal_host_packet(") == 1
    assert "ProposalLedger(" not in source
    assert "QuestionDrawResult(" not in source
    for unused_output in (
        "p_out",
        "v_in_out",
        "w_in_out",
        "aim_out",
        "attempted_v_ball_in",
    ):
        assert unused_output not in source


@pytest.mark.parametrize(
    "selection",
    (
        (0, 1, 2, 3, 4, 5),
        (2,),
        (5, 1, 3, 0, 4, 2),
    ),
    ids=("full", "single", "permuted"),
)
def test_diagnostic_host_only_packet_matches_public_solver_for_any_row_order(
    cq, prm, monkeypatch, selection,
):
    monkeypatch.setattr(cq, "solve_strike_specs_fixed_dir", _fake_solver)
    monkeypatch.setattr(cq, "predict_paddle_contact", _identity_contact)

    def scorer(p0, v0, _w0, _prm, *, surface_z, net_x, h, n_steps):
        del surface_z, net_x, h, n_steps
        rejected = v0[:, 0] < -3.5
        n = p0.shape[0]
        return {
            "land_xy": torch.tensor(
                [2.5, 0.0], device=p0.device, dtype=p0.dtype
            ).expand(n, 2).clone(),
            "land_valid": torch.ones(
                n, device=p0.device, dtype=torch.bool
            ),
            "net_z": torch.where(
                rejected,
                torch.full_like(v0[:, 0], 0.90),
                torch.full_like(v0[:, 0], 0.95),
            ),
            "net_valid": torch.ones(
                n, device=p0.device, dtype=torch.bool
            ),
        }

    monkeypatch.setattr(cq, "coarse_landing", scorer)
    rows = _external_rows(6)
    protos = _varied_protos(6)
    index = torch.tensor(selection, dtype=torch.long)
    selected = {
        name: value.index_select(0, index)
        for name, value in rows.items()
    }
    kwargs = {
        "protos": protos,
        "base_quat": selected["base_quat"],
        "prm": prm,
        "surface_z": 0.78,
        "net_x": 1.87,
        "net_top_z": 0.9325,
        "cfg": _fixed_cfg(cq),
        "h": 0.007,
        "n_steps": 137,
    }
    public = cq.solve_proposals(
        selected["clip_ids"],
        selected["p_contact"],
        selected["v_ball_in"],
        selected["w_ball_in"],
        selected["aim_xy"],
        selected["ref_normal"],
        **kwargs,
    )
    host_packet, reason_counts = (
        cq._solve_proposals_diagnostic_host_only(
            selected["clip_ids"],
            selected["p_contact"],
            selected["v_ball_in"],
            selected["w_ball_in"],
            selected["aim_xy"],
            selected["ref_normal"],
            _diagnostic_prevalidated_authority=(
                cq._DIAGNOSTIC_PREVALIDATED_SOLVE_AUTHORITY
            ),
            **kwargs,
        )
    )

    assert _host_packet_float32_bits(
        host_packet
    ) == _host_packet_float32_bits(public.proposal_host_packet)
    assert reason_counts == public.reason_counts


def test_diagnostic_host_only_chunked_packet_matches_one_permuted_batch(
    cq, prm, monkeypatch,
):
    monkeypatch.setattr(cq, "solve_strike_specs_fixed_dir", _fake_solver)
    monkeypatch.setattr(cq, "predict_paddle_contact", _identity_contact)

    def scorer(p0, v0, _w0, _prm, *, surface_z, net_x, h, n_steps):
        del surface_z, net_x, h, n_steps
        rejected = v0[:, 0] < -3.5
        n = p0.shape[0]
        return {
            "land_xy": torch.tensor(
                [2.5, 0.0], device=p0.device, dtype=p0.dtype
            ).expand(n, 2).clone(),
            "land_valid": torch.ones(
                n, device=p0.device, dtype=torch.bool
            ),
            "net_z": torch.where(
                rejected,
                torch.full_like(v0[:, 0], 0.90),
                torch.full_like(v0[:, 0], 0.95),
            ),
            "net_valid": torch.ones(
                n, device=p0.device, dtype=torch.bool
            ),
        }

    monkeypatch.setattr(cq, "coarse_landing", scorer)
    rows = _external_rows(6)
    protos = _varied_protos(6)
    order = torch.tensor([5, 1, 3, 0, 4, 2], dtype=torch.long)

    def solve(index):
        selected = {
            name: value.index_select(0, index)
            for name, value in rows.items()
        }
        return cq._solve_proposals_diagnostic_host_only(
            selected["clip_ids"],
            selected["p_contact"],
            selected["v_ball_in"],
            selected["w_ball_in"],
            selected["aim_xy"],
            selected["ref_normal"],
            protos=protos,
            base_quat=selected["base_quat"],
            prm=prm,
            surface_z=0.78,
            net_x=1.87,
            net_top_z=0.9325,
            cfg=_fixed_cfg(cq),
            h=0.007,
            n_steps=137,
            _diagnostic_prevalidated_authority=(
                cq._DIAGNOSTIC_PREVALIDATED_SOLVE_AUTHORITY
            ),
        )

    full_packet, full_counts = solve(order)
    chunks = (
        order[:2],
        order[2:3],
        order[3:],
    )
    chunk_results = [solve(index) for index in chunks]
    chunk_packets = [result[0] for result in chunk_results]
    chunk_counts = {}
    for _packet, counts in chunk_results:
        for name, count in counts.items():
            chunk_counts[name] = chunk_counts.get(name, 0) + count

    joined = cq.ProposalHostPacket(
        reason_codes=sum(
            (packet.reason_codes for packet in chunk_packets), ()
        ),
        admitted=sum(
            (packet.admitted for packet in chunk_packets), ()
        ),
        racket_velocity_rows=sum(
            (packet.racket_velocity_rows for packet in chunk_packets), ()
        ),
        racket_normal_rows=sum(
            (packet.racket_normal_rows for packet in chunk_packets), ()
        ),
        residual_rows=sum(
            (packet.residual_rows for packet in chunk_packets), ()
        ),
    )
    assert _host_packet_float32_bits(
        joined
    ) == _host_packet_float32_bits(full_packet)
    assert chunk_counts == full_counts


def test_exact_proposal_host_packet_has_one_transfer_and_no_scalar_sync(cq):
    source = inspect.getsource(cq._build_proposal_host_packet)
    assert source.count(".cpu().tolist()") == 1
    assert source.count(".tolist()") == 1
    assert ".item()" not in source
    assert ".numpy()" not in source


@pytest.mark.parametrize(
    "reason,admitted,match",
    (
        (0, True, "admitted row must carry reason code -1"),
        (-1, False, "rejected row has an invalid reason code"),
        (100, False, "invalid reason code"),
    ),
)
def test_exact_proposal_host_packet_rejects_discrete_contract_drift(
    cq, reason, admitted, match,
):
    with pytest.raises(RuntimeError, match=match):
        cq._build_proposal_host_packet(
            reason_codes=torch.tensor([reason], dtype=torch.long),
            admitted=torch.tensor([admitted], dtype=torch.bool),
            racket_velocity=torch.zeros((1, 3), dtype=torch.float32),
            racket_normal=torch.zeros((1, 3), dtype=torch.float32),
            residual=torch.zeros((1,), dtype=torch.float32),
        )


def test_diagnostic_prevalidated_solver_rejects_forged_authority(
    cq, prm,
):
    rows = _external_rows(1)
    with pytest.raises(PermissionError, match="exact private authority"):
        cq.solve_proposals(
            rows["clip_ids"],
            rows["p_contact"],
            rows["v_ball_in"],
            rows["w_ball_in"],
            rows["aim_xy"],
            rows["ref_normal"],
            protos=_protos(1),
            base_quat=rows["base_quat"],
            prm=prm,
            surface_z=0.78,
            net_x=1.87,
            net_top_z=0.9325,
            cfg=_fixed_cfg(cq),
            _diagnostic_prevalidated_authority=object(),
        )


@pytest.mark.parametrize("invalid_kind", ("nan_input", "bad_proto"))
def test_diagnostic_prevalidated_solver_async_rejects_invalid_dynamic_state(
    cq, prm, invalid_kind,
):
    rows = _external_rows(1)
    protos = _protos(1)
    if invalid_kind == "nan_input":
        rows["p_contact"][0, 0] = float("nan")
    else:
        protos.speed_min[0] = -1.0

    with pytest.raises(
        RuntimeError,
        match="diagnostic producer emitted invalid prevalidated solver inputs",
    ):
        cq.solve_proposals(
            rows["clip_ids"],
            rows["p_contact"],
            rows["v_ball_in"],
            rows["w_ball_in"],
            rows["aim_xy"],
            rows["ref_normal"],
            protos=protos,
            base_quat=rows["base_quat"],
            prm=prm,
            surface_z=0.78,
            net_x=1.87,
            net_top_z=0.9325,
            cfg=_fixed_cfg(cq),
            _diagnostic_prevalidated_authority=(
                cq._DIAGNOSTIC_PREVALIDATED_SOLVE_AUTHORITY
            ),
        )


def test_solve_proposals_failure_is_nan_but_proposal_and_reason_survive(
    cq, prm, monkeypatch,
):
    monkeypatch.setattr(cq, "solve_strike_specs_fixed_dir", _fake_solver)
    monkeypatch.setattr(cq, "predict_paddle_contact", _identity_contact)

    def scorer(p0, v0, _w0, _prm, *, surface_z, net_x, h, n_steps):
        del surface_z, net_x, h, n_steps
        rejected = v0[:, 0] < -5.0
        n = p0.shape[0]
        return {
            "land_xy": torch.tensor([2.5, 0.0], device=p0.device,
                                    dtype=p0.dtype).expand(n, 2).clone(),
            "land_valid": torch.ones(n, device=p0.device, dtype=torch.bool),
            "net_z": torch.where(rejected, torch.full_like(v0[:, 0], 0.90),
                                 torch.full_like(v0[:, 0], 0.95)),
            "net_valid": torch.ones(n, device=p0.device, dtype=torch.bool),
        }

    monkeypatch.setattr(cq, "coarse_landing", scorer)
    rows = _external_rows(2)
    rows["v_ball_in"][:, 0] = torch.tensor([-3.0, -5.1])
    out = cq.solve_proposals(
        rows["clip_ids"], rows["p_contact"], rows["v_ball_in"], rows["w_ball_in"],
        rows["aim_xy"], rows["ref_normal"], protos=_varied_protos(2),
        base_quat=rows["base_quat"], prm=prm, surface_z=0.78, net_x=1.87,
        net_top_z=0.9325, cfg=_fixed_cfg(cq),
    )

    assert out.ok.tolist() == [True, False]
    assert out.proposal_count == 2 and out.rounds_used == 1 and out.exhausted == 1
    assert out.proposals.request_index.tolist() == [0, 1]
    assert out.proposals.admitted.tolist() == [True, False]
    assert out.proposals.reason_code.tolist() == [-1, 5]
    assert out.proposal_host_packet.reason_codes == (-1, 5)
    assert out.proposal_host_packet.admitted == (True, False)
    assert out.proposal_host_packet.racket_velocity_rows[0] == tuple(
        float(value) for value in out.v_racket[0].tolist()
    )
    assert all(
        value != value
        for value in out.proposal_host_packet.racket_velocity_rows[1]
    )
    assert out.proposal_host_packet.residual_rows == tuple(
        float(value) for value in out.resid_m.tolist()
    )
    assert torch.equal(out.proposals.v_ball_in, rows["v_ball_in"])
    assert out.reason_counts == {"net_not_cleared": 1}
    for field in (
        "p_contact", "v_racket", "n_racket", "v_ball_in", "w_ball_in", "aim_xy",
    ):
        assert bool(torch.isnan(getattr(out, field)[1]).all()), field
    assert torch.isfinite(out.resid_m[1])
    assert torch.equal(out.attempted_v_ball_in, rows["v_ball_in"])


def test_solve_proposals_rejects_empty_and_free_direction(cq, prm):
    rows = _external_rows(1)
    with pytest.raises(ValueError, match="fixed-direction only"):
        cq.solve_proposals(
            rows["clip_ids"], rows["p_contact"], rows["v_ball_in"], rows["w_ball_in"],
            rows["aim_xy"], rows["ref_normal"], protos=_protos(1),
            base_quat=rows["base_quat"], prm=prm, surface_z=0.78, net_x=1.87,
            net_top_z=0.9325, cfg=cq.ContinuousQuestionCfg(fixed_direction=False),
        )

    empty = _external_rows(0)
    with pytest.raises(ValueError, match="at least one proposal row"):
        cq.solve_proposals(
            empty["clip_ids"], empty["p_contact"], empty["v_ball_in"],
            empty["w_ball_in"], empty["aim_xy"], empty["ref_normal"],
            protos=_protos(1), base_quat=empty["base_quat"], prm=prm,
            surface_z=0.78, net_x=1.87, net_top_z=0.9325, cfg=_fixed_cfg(cq),
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("bad_shape", "p_contact must have shape"),
        ("nan_ball", "v_ball_in must contain only finite"),
        ("negative_action", "clip_ids out of range"),
        ("large_action", "clip_ids out of range"),
        ("clip_dtype", "dtype torch.long"),
        ("mixed_dtype", "share p_contact device/dtype"),
    ],
)
def test_solve_proposals_strict_shape_finite_and_action_range(
    cq, prm, mutation, match,
):
    rows = _external_rows(2)
    protos = _protos(2)
    if mutation == "bad_shape":
        rows["p_contact"] = rows["p_contact"][:, :2]
    elif mutation == "nan_ball":
        rows["v_ball_in"][0, 0] = float("nan")
    elif mutation == "negative_action":
        rows["clip_ids"][0] = -1
    elif mutation == "large_action":
        rows["clip_ids"][1] = 2
    elif mutation == "clip_dtype":
        rows["clip_ids"] = rows["clip_ids"].to(torch.int32)
    elif mutation == "mixed_dtype":
        rows["ref_normal"] = rows["ref_normal"].double()
    with pytest.raises(ValueError, match=match):
        cq.solve_proposals(
            rows["clip_ids"], rows["p_contact"], rows["v_ball_in"], rows["w_ball_in"],
            rows["aim_xy"], rows["ref_normal"], protos=protos,
            base_quat=rows["base_quat"], prm=prm, surface_z=0.78, net_x=1.87,
            net_top_z=0.9325, cfg=_fixed_cfg(cq),
        )
