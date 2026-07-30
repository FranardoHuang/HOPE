"""Franco 2026-07-28 bundle: diagnostic bypass brand + phase-gated reference terminations.

Host-only seams: the Isaac modules are not imported.  The pure gating helpers
are executed from the shipped AST; the bypass/brand wiring is pinned as source
text the same way the runtime wiring tests pin transaction order.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
MDP = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)
HOPE_REWARDS = (MDP / "hope_rewards.py").read_text(encoding="utf-8")
HOPE_COMMANDS = (MDP / "hope_commands.py").read_text(encoding="utf-8")
COMMANDS = (MDP / "commands.py").read_text(encoding="utf-8")
TRAIN = (
    ROOT / "scripts" / "train.py"
).read_text(encoding="utf-8")


def _functions(source: str, path: Path, names):
    tree = ast.parse(source, filename=str(path))
    wanted = set(names)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in nodes} == wanted
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias("annotations")],
                level=0,
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, str(path), "exec"), namespace)
    return tuple(namespace[name] for name in names)


MASK_FN, GATE_FN = _functions(
    HOPE_REWARDS,
    MDP / "hope_rewards.py",
    (
        "_action_ball_reference_terminations_mask",
        "_gate_reference_termination",
    ),
)


def _env_with(term):
    def get_term(name):
        if name == "racket_target" and term is not None:
            return term
        raise KeyError(name)

    return SimpleNamespace(
        command_manager=SimpleNamespace(get_term=get_term)
    )


def test_gate_is_identity_outside_action_ball():
    verdict = torch.tensor([True, True, False])
    plain = SimpleNamespace()  # no command_manager at all
    assert torch.equal(GATE_FN(plain, verdict), verdict)
    no_term = _env_with(None)
    assert torch.equal(GATE_FN(no_term, verdict), verdict)
    # A racket term without the gate attribute (non-action_ball modes).
    bare = _env_with(SimpleNamespace())
    assert torch.equal(GATE_FN(bare, verdict), verdict)
    # Action-ball center phase: mask all-True is a no-op.
    center = _env_with(
        SimpleNamespace(
            action_ball_reference_terminations_enabled=lambda: torch.tensor(
                [True, True, True]
            )
        )
    )
    assert torch.equal(GATE_FN(center, verdict), verdict)


def test_gate_disables_reference_terminations_past_center():
    verdict = torch.tensor([True, True, False, True])
    mask = torch.tensor([True, False, True, False])
    env = _env_with(
        SimpleNamespace(
            action_ball_reference_terminations_enabled=lambda: mask
        )
    )
    assert torch.equal(
        GATE_FN(env, verdict), torch.tensor([True, False, False, False])
    )
    assert MASK_FN(env) is mask


def test_all_three_reference_terminations_are_gated():
    for name in (
        "bad_anchor_pos_z_only_hold_aware",
        "bad_anchor_ori_hold_aware",
        "bad_motion_body_pos_z_only_hold_aware",
    ):
        body = HOPE_REWARDS.split(f"def {name}(")[1].split("\ndef ")[0]
        assert "_gate_reference_termination(" in body, name


def test_diagnostic_bypass_default_off_and_branded():
    # Default false is declared on the cfg dataclass itself.
    assert (
        "action_ball_diagnostic_unauthorized: bool = False" in HOPE_COMMANDS
    )
    # The runtime physical ready-entry contract stays enforced in bypass mode:
    # only the registry admission is skipped.
    assert (
        "if not self._canonical_diagnostic_unauthorized:\n"
        "                self._validate_canonical_registry_motion_bytes()\n"
        "            self._validate_canonical_ready_clips()" in COMMANDS
    )
    # The curriculum never binds a diagnostic authority: formal observe/update
    # stay fail-loud, which is the promotion-path brand enforcement.
    assert (
        "None if diagnostic_unauthorized else evaluator_authority"
        in HOPE_COMMANDS
    )
    # Brands: hard contract, stdout receipt, run name, applied receipt.
    assert 'payload["diagnostic_unauthorized"] = True' in HOPE_COMMANDS
    assert 'receipt["diagnostic_unauthorized"] = True' in HOPE_COMMANDS
    assert "DIAGNOSTIC_UNAUTHORIZED" in TRAIN
    assert "; diagnostic_unauthorized=true" in TRAIN
    # A diagnostic run must brand its runtime hard contract or fail loud.
    assert (
        "diagnostic action-ball run produced an unbranded" in TRAIN
    )


def test_exact_resume_embeds_branded_hard_contract():
    # The exact-resume payload embeds the hard contract verbatim and restore
    # requires equality, so a branded checkpoint can never be resumed by a
    # formal run (and vice versa) without a fail-loud mismatch.
    assert '"hard_contract": hard_contract,' in HOPE_COMMANDS
    assert (
        'state["hard_contract"] != hard_contract' in HOPE_COMMANDS
    )
