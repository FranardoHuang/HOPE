import hashlib
import json
from pathlib import Path
import subprocess
import sys


import test_action_ball_banded_question_bank as fixture


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_action_ball_banded_question_bank.py"
B = fixture.B


def _run(*arguments):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, arguments)],
        cwd=ROOT.parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _write_input(path, receipts, *, rejected=0):
    payload = {
        "schema_version": 1,
        "kind": "action_ball_offline_solved_receipts",
        "source_id": path.stem,
        "solver_mode": "current_lm_only",
        "offline_producer_source_sha256": "a" * 64,
        "offline_input_root_sha256": "b" * 64,
        "solve_ledger": {
            "proposed_count": len(receipts) + rejected,
            "admitted_count": len(receipts),
            "rejections": (
                [] if rejected == 0 else [{"reason": "solver_rejected", "count": rejected}]
            ),
        },
        "receipts": [receipt.to_dict() for receipt in receipts],
    }
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_coverage(path, receipts):
    unique = {}
    for receipt in receipts:
        unique[(receipt.action_uid, receipt.levels_sha256)] = {
            "action_uid": receipt.action_uid,
            "levels_sha256": receipt.levels_sha256,
            "domain_levels": receipt.domain_levels.to_dict(),
        }
    payload = {
        "schema_version": 1,
        "kind": "action_ball_reachable_domain_level_blocks",
        "arm_catalog_sha256": fixture.R.ARM_CATALOG_SHA256,
        "arm_keys": list(fixture.R.ARM_KEYS),
        "expected_action_uids": sorted({receipt.action_uid for receipt in receipts}),
        "reachable_arm_keys_by_action": [
            {
                "action_uid": uid,
                "reachable_arm_keys": list(fixture.R.ARM_KEYS),
            }
            for uid in sorted({receipt.action_uid for receipt in receipts})
        ],
        "reachable_blocks": [unique[key] for key in sorted(unique)],
    }
    document = {**payload, "canonical_sha256": B._sha256_json(payload)}
    path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_offline_builder_is_deterministic_exact_lineage_and_no_clobber(tmp_path):
    receipt = fixture._receipt()
    source = tmp_path / "receipts.json"
    source_sha = _write_input(source, (receipt,), rejected=2)
    coverage = tmp_path / "coverage.json"
    coverage_sha = _write_coverage(coverage, (receipt,))
    first = tmp_path / "bank-a.json"
    second = tmp_path / "bank-b.json"
    common = (
        "--input", f"{source}={source_sha}",
        "--coverage", f"{coverage}={coverage_sha}",
        "--split-seed", "41",
    )

    result_a = _run(*common, "--output", first)
    result_b = _run(*common, "--output", second)
    assert result_a.returncode == 0, result_a.stderr
    assert result_b.returncode == 0, result_b.stderr
    assert first.read_bytes() == second.read_bytes()

    raw = first.read_bytes()
    bank = B.load_banded_question_bank(
        first, expected_file_sha256=hashlib.sha256(raw).hexdigest()
    )
    lineage = bank.canonical_payload["producer_lineage"]
    assert lineage["row_order"] == "canonical_receipt_sha256"
    assert lineage["inputs"] == [
        {
            "source_id": source.stem,
            "solver_mode": "current_lm_only",
            "block_key_sha256": bank.blocks[0].key_sha256,
            "file_sha256": source_sha,
            "offline_producer_source_sha256": "a" * 64,
            "offline_input_root_sha256": "b" * 64,
            "proposed_count": 3,
            "admitted_count": 1,
            "rejections": [{"reason": "solver_rejected", "count": 2}],
            "receipt_canonical_sha256": [receipt.canonical_sha256],
        }
    ]
    assert bank.canonical_payload["online_solver_calls_per_reset"] == 0
    assert bank.canonical_payload["offline_solve_ledger"] == {
        "proposed_count": 3,
        "admitted_count": 1,
        "rejected_count": 2,
        "by_reason": [{"reason": "solver_rejected", "count": 2}],
        "by_block": [
            {
                "block_key_sha256": bank.blocks[0].key_sha256,
                "proposed_count": 3,
                "admitted_count": 1,
                "rejected_count": 2,
                "by_reason": [{"reason": "solver_rejected", "count": 2}],
            }
        ],
    }

    clobber = _run(*common, "--output", first)
    assert clobber.returncode != 0
    assert "no-clobber" in clobber.stderr
    assert first.read_bytes() == raw


def test_offline_builder_rejects_input_sha_before_publication(tmp_path):
    receipt = fixture._receipt()
    source = tmp_path / "receipt.json"
    _write_input(source, (receipt,))
    coverage = tmp_path / "coverage.json"
    coverage_sha = _write_coverage(coverage, (receipt,))
    output = tmp_path / "bank.json"
    result = _run(
        "--input",
        f"{source}={'0' * 64}",
        "--coverage",
        f"{coverage}={coverage_sha}",
        "--split-seed",
        "0",
        "--output",
        output,
    )
    assert result.returncode != 0
    assert "input file SHA mismatch" in result.stderr
    assert not output.exists()


def test_offline_builder_groups_center_and_exact_per_level_blocks(tmp_path):
    center = fixture._receipt()
    levels = fixture.R.ActionDomainLevels(contact_y_upper=0.25)
    expanded_birth = fixture._birth_at_levels(levels)
    expanded = fixture._fixture_call(fixture.FIXTURE._source_task, expanded_birth)
    source = tmp_path / "center.json"
    source_sha = _write_input(source, (center,))
    expanded_source = tmp_path / "expanded.json"
    expanded_source_sha = _write_input(expanded_source, (expanded,))
    coverage = tmp_path / "coverage.json"
    coverage_sha = _write_coverage(coverage, (expanded, center))
    output = tmp_path / "bank.json"
    result = _run(
        "--input",
        f"{source}={source_sha}",
        "--input",
        f"{expanded_source}={expanded_source_sha}",
        "--coverage",
        f"{coverage}={coverage_sha}",
        "--split-seed",
        "7",
        "--output",
        output,
    )
    assert result.returncode == 0, result.stderr
    raw = output.read_bytes()
    bank = B.load_banded_question_bank(
        output, expected_file_sha256=hashlib.sha256(raw).hexdigest()
    )
    assert len(bank.blocks) == 2
    assert {block.key["levels_sha256"] for block in bank.blocks} == {
        fixture.R.ActionDomainLevels().canonical_sha256,
        levels.canonical_sha256,
    }


def test_offline_builder_rejects_incomplete_reachable_coverage_before_publication(
    tmp_path,
):
    center = fixture._receipt()
    levels = fixture.R.ActionDomainLevels(contact_y_upper=0.25)
    expanded = fixture._fixture_call(
        fixture.FIXTURE._source_task, fixture._birth_at_levels(levels)
    )
    source = tmp_path / "center.json"
    source_sha = _write_input(source, (center,))
    expanded_source = tmp_path / "expanded.json"
    expanded_source_sha = _write_input(expanded_source, (expanded,))
    coverage = tmp_path / "center-only-coverage.json"
    coverage_sha = _write_coverage(coverage, (center,))
    output = tmp_path / "bank.json"
    result = _run(
        "--input",
        f"{source}={source_sha}",
        "--input",
        f"{expanded_source}={expanded_source_sha}",
        "--coverage",
        f"{coverage}={coverage_sha}",
        "--split-seed",
        "9",
        "--output",
        output,
    )
    assert result.returncode != 0
    assert "exactly cover" in result.stderr
    assert not output.exists()


def test_offline_builder_rejects_unconserved_rejection_denominator(tmp_path):
    receipt = fixture._receipt()
    source = tmp_path / "bad-ledger.json"
    source_sha = _write_input(source, (receipt,), rejected=2)
    document = json.loads(source.read_text(encoding="utf-8"))
    document["solve_ledger"]["rejections"][0]["count"] = 1
    source.write_text(json.dumps(document), encoding="utf-8")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    coverage = tmp_path / "coverage.json"
    coverage_sha = _write_coverage(coverage, (receipt,))
    output = tmp_path / "bank.json"
    result = _run(
        "--input",
        f"{source}={source_sha}",
        "--coverage",
        f"{coverage}={coverage_sha}",
        "--split-seed",
        "3",
        "--output",
        output,
    )
    assert result.returncode != 0
    assert "does not conserve P=A+R" in result.stderr
    assert not output.exists()
