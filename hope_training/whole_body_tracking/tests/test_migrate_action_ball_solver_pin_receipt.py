"""The v2->v3 migration receipt must not say anything it did not check.

人话:那份收据以前写着"每一个命名题目或计算答案的符号在两个 revision 上都相同"。
这句话有两处越界:

1. 它只比了**覆盖面里**的符号,排除清单一个没比 —— 而排除清单正是本轮收窄的产物,
   拿它当"什么都没变"的证据是循环论证。
2. 它对**造产物的那套工具**结构性失明。实测:2026-08-06 和 2026-08-07 两条 N1 tape
   是在两个不同版本的 ``training_contract.py``(``8c9eec4c`` vs ``c6608440``)上建的,
   两份 build report 里白纸黑字,而这支脚本一个字都看不见。emit 出来的题目数字确实
   一个没变(那是另外逐 leaf 对拍出来的),所以结论不受影响 —— 但话要说准。

这份测试把"说准"钉死:收据必须自陈它的作用域、必须列出它没比的东西、
必须把生产血统的移动测量出来而不是假设它没动。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


WHOLE_BODY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WHOLE_BODY_ROOT.parents[1]
SCRIPT = (
    WHOLE_BODY_ROOT
    / "scripts"
    / "migrate_action_ball_solver_pin_to_semantic_surface.py"
)
SURFACE_SOURCE = (
    WHOLE_BODY_ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
    / "action_ball_solver_semantic_surface.py"
)

#: The two tape build reports whose producing ``training_contract.py`` differ.
#: This is the measured instance of the blind spot, not a hypothetical.
TAPE_BUILD_REPORTS = (
    (
        "configs/action_ball_n1_measured_20260806/fresh_tape_seed0_20260806_r1/"
        "offline_n1_tape_build_report.v1.091b3d189dde.json",
        "8c9eec4c33c54a94a5d8c36c18f8df1b0672ded1a33f49e90a069d29a27579b2",
    ),
    (
        "configs/action_ball_n1_measured_20260807/v3pin_tape_seed0_20260807_r1/"
        "offline_n1_tape_build_report.v1.58eb977b1566.json",
        "c6608440c30e1b867c9856253dd59cef6c5b1130a144745281b1ac87506d318c",
    ),
)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MIGRATE = _load(SCRIPT, "_test_migrate_action_ball_solver_pin")
SURFACE = _load(SURFACE_SOURCE, "_test_migrate_surface")


def test_the_producing_toolchain_is_measured_not_assumed():
    """Every producer source the receipt reports on must actually exist."""

    assert MIGRATE.PRODUCER_LINEAGE_SOURCES
    for relative in MIGRATE.PRODUCER_LINEAGE_SOURCES:
        assert (REPO_ROOT / relative).is_file(), relative


def test_training_contract_is_the_named_instance_of_the_blind_spot():
    """The two shipped tapes really were built on two different contracts."""

    training_contract = next(
        relative
        for relative in MIGRATE.PRODUCER_LINEAGE_SOURCES
        if relative.endswith("utils/training_contract.py")
    )
    observed = []
    for relative, expected in TAPE_BUILD_REPORTS:
        path = REPO_ROOT / relative
        if not path.is_file():
            pytest.skip(f"{relative} is not in this checkout")
        report = json.loads(path.read_text(encoding="utf-8"))
        shas = {
            contract["payload"]["implementation_source_sha256"][
                "training_contract"
            ]
            for contract in report["producer_contracts"][
                "desired_contact"
            ].values()
        }
        assert shas == {expected}, (relative, shas)
        observed.append(expected)
    assert observed[0] != observed[1], (
        "this test only means something while the two lineages really do carry "
        "different producing contracts"
    )
    # The judgement, encoded: training_contract.py is already pinned where it
    # belongs (the fixed-tape producer contract, which moved with it) and must
    # NOT be added to the solver semantic surface -- it names no question and
    # computes no answer, and the solver never imports it.
    assert training_contract not in SURFACE.PINNED_SOURCES
    assert not any(
        name.endswith("training_contract.py") for name in SURFACE.PINNED_SOURCES
    )
    solver_sources = (
        WHOLE_BODY_ROOT
        / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
    )
    # The five pure solver sources never mention it at all.
    for name in SURFACE.FULLY_ENUMERATED_SOURCES:
        assert "training_contract" not in (
            solver_sources / name
        ).read_text(encoding="utf-8"), name
    # ``hope_commands.py`` does mention it -- it owns dozens of unrelated
    # command terms, including the task-first training contract builder -- so
    # the claim has to be the narrow one: no COVERED symbol reaches it.  Saying
    # "the solver never imports it" about a 26k-line module would be the same
    # kind of over-broad sentence this test exists to delete.
    covered_nodes = SURFACE._symbol_nodes(
        (solver_sources / "hope_commands.py").read_text(encoding="utf-8"),
        filename="hope_commands.py",
    )
    for symbol in SURFACE.COVERED["hope_commands.py"]:
        node = covered_nodes.get(symbol)
        if node is None:
            continue
        referenced = SURFACE._referenced_names(node)
        assert not any(
            "training_contract" in name for name in referenced
        ), symbol


def test_the_receipt_scopes_its_claim_and_names_what_it_did_not_check():
    """Read the emitted strings, not the intent behind them."""

    source = SCRIPT.read_text(encoding="utf-8")
    assert "Scoped to the symbols this surface COVERS" in source
    assert '"not_claimed"' in source
    assert '"coverage_this_receipt_does_not_check"' in source
    assert '"producer_lineage_outside_this_pin"' in source
    # The old, unscoped sentence must be gone, not merely surrounded.
    assert "Only the pin's own declaration half moved. Every symbol" not in (
        source
    )


def test_the_bridge_symbols_are_the_only_allowed_introduction():
    """A brand-new covered symbol is refused unless it is named here.

    The migration proves "questions did not move" by comparing covered symbol
    digests.  A symbol that exists on only one side cannot be compared at all,
    so the script has to either refuse or declare the exception out loud.
    """

    allowed = MIGRATE.ALLOWED_INTRODUCED_SYMBOLS["hope_commands.py"]
    covered = set(SURFACE.COVERED["hope_commands.py"])
    assert allowed <= covered, allowed - covered
    assert allowed == {
        "action_ball_declared_solver_knobs",
        "action_ball_solver_cfg_from_declaration",
        "action_ball_assert_solver_runtime_matches_declaration",
        "_ACTION_BALL_SOLVER_FIXED_DIRECTION",
        "_ACTION_BALL_VIRTUAL_BALL_PARAM_NAMES",
    }
    # Introducing a symbol is not the same as being allowed to change one.
    moved_allowance = MIGRATE.ALLOWED_MOVED_SYMBOLS["hope_commands.py"]
    assert allowed.isdisjoint(moved_allowance)


def test_the_declaration_bridge_is_not_exempt_from_the_migration_gate():
    """The three edits that escaped the pin now make the migration refuse.

    Each of them lands on a covered symbol that is on neither allowance list, so
    ``ALLOWED_MOVED_SYMBOLS`` rejects it and the script prints it rather than
    signing for it.
    """

    covered = set(SURFACE.COVERED["hope_commands.py"])
    allowance = set(MIGRATE.ALLOWED_MOVED_SYMBOLS["hope_commands.py"]) | set(
        MIGRATE.ALLOWED_INTRODUCED_SYMBOLS["hope_commands.py"]
    )
    for symbol in (
        "action_ball_solver_cfg_from_declaration",
        "action_ball_declared_solver_knobs",
        "action_ball_assert_solver_runtime_matches_declaration",
    ):
        assert symbol in covered
    # The mapping and the cross-check may be *introduced*, never silently
    # changed: only the two declaration-half symbols may move.
    assert set(MIGRATE.ALLOWED_MOVED_SYMBOLS["hope_commands.py"]) == {
        "action_ball_solver_profile_contract",
        "_ACTION_BALL_SOLVER_PROFILE_SCHEMA_VERSION",
    }
    assert "action_ball_solver_cfg_from_declaration" not in (
        MIGRATE.ALLOWED_MOVED_SYMBOLS["hope_commands.py"]
    )
    assert allowance < covered


def test_script_still_refuses_a_non_v2_pins_document(tmp_path):
    """The one behavioural end-to-end check that needs no repo materialisation."""

    pins = tmp_path / "pins.json"
    pins.write_text(
        json.dumps(
            {
                "solver_profile_sha256": "0" * 64,
                "solver_payload": {"schema_version": 3},
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(REPO_ROOT),
            "--pins",
            str(pins),
            "--from-rev",
            "HEAD",
            "--out",
            str(tmp_path / "receipt.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "only converts solver profile schema v2" in (
        result.stderr + result.stdout
    )
