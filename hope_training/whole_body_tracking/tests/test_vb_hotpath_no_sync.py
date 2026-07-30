"""Source and tensor-equivalence guards for the virtual-ball masked-mean hot path.

Run on the Pod training environment:

    python -m pytest \
      hope_training/whole_body_tracking/tests/test_vb_hotpath_no_sync.py -q

These tests intentionally avoid importing Isaac Lab.  They lock the small, behavior-preserving
CUDA synchronization removal in ``RacketTargetCommand._vb_evaluate``: valid-row means update the
two held metrics, while an empty mask leaves both tensors unchanged.
"""

from __future__ import annotations

import ast
from pathlib import Path

import torch


COMMANDS = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "hope_commands.py"
)


def _masked_mean_or_previous(
    values: torch.Tensor,
    selected: torch.Tensor,
    previous: torch.Tensor,
) -> torch.Tensor:
    """Mirror the branch-free reduction used by the production method."""

    count = selected.sum()
    denom = count.clamp_min(1).to(dtype=values.dtype)
    mean = torch.where(
        selected,
        values,
        torch.zeros_like(values),
    ).sum() / denom
    return torch.where(count > 0, mean, previous)


def _vb_evaluate_source() -> str:
    source = COMMANDS.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_vb_evaluate":
            return ast.get_source_segment(source, node) or ""
    raise AssertionError("_vb_evaluate was not found")


def test_masked_means_match_the_old_selected_row_updates():
    selected = torch.tensor([False, True, False, True, False])
    previous_land = torch.full((5,), 7.25)
    previous_spin = torch.full((5,), -3.5)
    # Non-selected non-finite values prove that masking occurs before reduction: 0 * NaN would
    # poison the result, whereas the historic boolean-indexed mean ignored those rows.
    land_error = torch.tensor([float("nan"), 0.25, float("inf"), 0.75, -float("inf")])
    spin_revs = torch.tensor([float("nan"), -2.0, float("inf"), 6.0, -float("inf")])

    expected_land = previous_land.clone()
    expected_spin = previous_spin.clone()
    expected_land[:] = land_error[selected].mean()
    expected_spin[:] = spin_revs[selected].mean()

    actual_land = _masked_mean_or_previous(land_error, selected, previous_land)
    actual_spin = _masked_mean_or_previous(spin_revs, selected, previous_spin)

    assert torch.equal(actual_land, expected_land)
    assert torch.equal(actual_spin, expected_spin)


def test_empty_mask_preserves_every_held_metric_element():
    selected = torch.zeros(4, dtype=torch.bool)
    previous_land = torch.tensor([0.1, 0.2, 0.3, 0.4])
    previous_spin = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    values = torch.full((4,), float("nan"))

    actual_land = _masked_mean_or_previous(values, selected, previous_land)
    actual_spin = _masked_mean_or_previous(values, selected, previous_spin)

    assert torch.equal(actual_land, previous_land)
    assert torch.equal(actual_spin, previous_spin)
    assert torch.isfinite(actual_land).all()
    assert torch.isfinite(actual_spin).all()


def test_vb_evaluate_has_no_host_branch_for_fired_valid_means():
    source = _vb_evaluate_source()

    # The strike-free return has broader cache/EMA semantics and is intentionally outside this
    # behavior-preserving change. Diagnostic may batch the predicate with its identity checks,
    # while formal/default retains the historical sequential host checks; both feed the same lazy
    # return before venue/contact/rollout.
    assert "exact_any_host" in source
    assert "exact_any = bool(exact_strike.any())" in source
    assert "if not exact_any:" in source
    assert source.index("if not exact_any:") < source.index(
        "if self._vb_params is None:"
    )
    assert "if bool(fired_valid.any()):" not in source
    assert "fired_valid_count = fired_valid.sum()" in source
    assert source.count("has_fired_valid,") == 2
    assert source.count("torch.where(") >= 4
