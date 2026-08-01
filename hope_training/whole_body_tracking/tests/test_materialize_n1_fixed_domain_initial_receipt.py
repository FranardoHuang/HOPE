from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import re

import pytest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "materialize_n1_fixed_domain_initial_receipt.py"
)
SPEC = importlib.util.spec_from_file_location(
    "materialize_n1_fixed_domain_initial_receipt", SOURCE
)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)

REPO_ROOT = SOURCE.parents[3]
R5_BUNDLES = {
    "bh_loop_c": (
        "configs/n1_contact_vendor_a3_20260801_r5/bh_loop_c/"
        "bh_loop_c.bundle.v2.bf0ae909e108.json",
        "bf0ae909e108ff7d96a9173fe30d69716a23379682e766e98acf615b3a8ac4d5",
    ),
    "bh_block": (
        "configs/n1_contact_vendor_a3_20260801_r5/bh_block/"
        "bh_block.bundle.v2.497c4bbd5658.json",
        "497c4bbd5658f32e2e38a7f529207ce303121d36669cb1c3fb654b743437ae8a",
    ),
}


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _copy(root: Path, relative: str) -> None:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / relative, destination)


def _pin_fixture_bundle(root: Path, action_id: str) -> None:
    path, digest = R5_BUNDLES[action_id]
    registry = root / M.REGISTRY_REPO_PATH
    text = registry.read_text()
    pattern = re.compile(
        r"contact_bundle=ArtifactPin\(\n"
        rf"\s*\"configs/n1_contact_vendor_a3_20260802_r9/{action_id}/\"\n"
        r"\s*\"[^\n]+\",\n"
        r"\s*(?:None|\"[0-9a-f]{64}\"),\n\s*\),\n"
        r"\s*fixed_domain_initial_receipt=ArtifactPin\(\n"
        r"\s*\"configs/n1_fixed_domain_initial_20260802_r9/\"\n"
        rf"\s*\"{action_id}\.fixed_domain_initial\.v1\.json\",\n"
        r"\s*(?:None|\"[0-9a-f]{64}\"),",
        re.MULTILINE,
    )
    replacement = (
        "contact_bundle=ArtifactPin(\n"
        f"        {path!r},\n"
        f"        {digest!r},\n"
        "    ),\n"
        "    fixed_domain_initial_receipt=ArtifactPin(\n"
        "        \"configs/n1_fixed_domain_initial_20260802_r9/\"\n"
        f"        \"{action_id}.fixed_domain_initial.v1.json\",\n"
        "        None,"
    )
    text, count = pattern.subn(replacement, text)
    assert count == 1
    registry.write_text(text)


def _fixture_repo(tmp_path: Path, action_id: str = "bh_loop_c") -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    for path in (
        M.PRODUCER_REPO_PATH,
        M.REGISTRY_REPO_PATH,
        M.TRAIN_REPO_PATH,
        *(f"{M.MDP_DIR}/{name}.py" for name in M.MDP_MODULES),
    ):
        _copy(root, path)
    _pin_fixture_bundle(root, action_id)
    bundle_path, _ = R5_BUNDLES[action_id]
    _copy(root, bundle_path)
    bundle = json.loads((root / bundle_path).read_text())
    _copy(root, bundle["manifest"]["path"])
    _copy(root, bundle["source_manifest"]["path"])
    _git(root, "init")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Receipt Test",
        "-c",
        "user.email=receipt-test@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    return root


def _commit(root: Path, message: str) -> None:
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Receipt Test",
        "-c",
        "user.email=receipt-test@example.invalid",
        "commit",
        "-m",
        message,
    )


def _bundle(root: Path, action_id: str = "bh_loop_c") -> tuple[Path, dict]:
    registry = M._load_file_module(
        root / M.REGISTRY_REPO_PATH,
        f"_fixed_domain_fixture_registry_{action_id}_{id(root)}",
    )
    config = registry.get_action_config(action_id)
    path = root / config.contact_bundle.path
    return path, json.loads(path.read_text())


def _replace_registry_bundle_sha(
    root: Path, *, old_sha: str, new_sha: str
) -> None:
    path = root / M.REGISTRY_REPO_PATH
    text = path.read_text()
    assert text.count(old_sha) == 1
    path.write_text(text.replace(old_sha, new_sha))
    # The digest replacement preserves file size and may happen inside one
    # filesystem timestamp tick.  Remove importlib's timestamp-based bytecode
    # cache so the next isolated registry import consumes the edited source.
    shutil.rmtree(path.parent / "__pycache__", ignore_errors=True)


def _write_json(path: Path, value: object) -> str:
    raw = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _repin_selected_manifest(
    root: Path, mutator, *, action_id: str = "bh_loop_c"
) -> None:
    bundle_path, bundle = _bundle(root, action_id)
    old_bundle_sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    manifest_path = root / bundle["manifest"]["path"]
    manifest = json.loads(manifest_path.read_text())
    mutator(manifest)
    bundle["manifest"]["sha256"] = _write_json(manifest_path, manifest)
    new_bundle_sha = _write_json(bundle_path, bundle)
    _replace_registry_bundle_sha(
        root, old_sha=old_bundle_sha, new_sha=new_bundle_sha
    )
    _commit(root, "repin selected manifest")


def _repin_source_manifest(
    root: Path, mutator, *, action_id: str = "bh_loop_c"
) -> None:
    bundle_path, bundle = _bundle(root, action_id)
    old_bundle_sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    manifest_path = root / bundle["source_manifest"]["path"]
    manifest = json.loads(manifest_path.read_text())
    mutator(manifest)
    bundle["source_manifest"]["sha256"] = _write_json(manifest_path, manifest)
    new_bundle_sha = _write_json(bundle_path, bundle)
    _replace_registry_bundle_sha(
        root, old_sha=old_bundle_sha, new_sha=new_bundle_sha
    )
    _commit(root, "repin historical source manifest")


def _reseal(document: dict) -> dict:
    document["content_sha256"] = M.canonical_sha256(document["content"])
    return document


@pytest.mark.parametrize("action_id", ("bh_loop_c", "bh_block"))
def test_builds_complete_fixed_domain_from_code_owned_inputs(
    tmp_path: Path, action_id: str
) -> None:
    root = _fixture_repo(tmp_path, action_id)
    document = M.build_receipt(root, action_id)
    assert document["schema_version"] == 1
    assert document["kind"] == M.RECEIPT_KIND
    assert document["content_sha256"] == M.canonical_sha256(
        document["content"]
    )
    payload = document["content"]
    assert payload["action_order"] == [action_id]
    assert payload["domain_epoch"] == 0
    assert len(payload["domain_levels"]) == 32
    assert set(payload["domain_levels"].values()) == {0.0}
    assert {axis["arm"] for axis in payload["domain_axes"]} == set(
        payload["domain_levels"]
    )
    assert len(payload["domain_axes"]) == 32
    assert all(axis["width"] == axis["initial"] for axis in payload["domain_axes"])
    assert all(
        0.0 <= axis["initial"] <= axis["maximum"] <= axis["cap"] + 1.0e-12
        for axis in payload["domain_axes"]
    )
    assert set(payload["inputs"]) == {
        "producer_source",
        "runtime_sampling_wiring",
        *(f"mdp_{name}" for name in M.MDP_MODULES),
        "contact_bundle",
        "action_manifest",
        "source_manifest",
    }
    assert payload["registry_action_source_identity_sha256"] == M.canonical_sha256(
        payload["registry_action_source_identity"]
    )
    assert M.validate_receipt_document(document) == document


def test_loop_static_and_adaptive_share_one_lane_independent_receipt(
    tmp_path: Path,
) -> None:
    root = _fixture_repo(tmp_path)
    first = M.build_receipt(root, "bh_loop_c")
    second = M.build_receipt(root, "bh_loop_c")
    assert first == second
    assert first["content"]["authorized_lane_ids"] == [
        "bh_loop_c_static_v1",
        "bh_loop_c_monotonic_fresh_canary_v1",
    ]


def test_historical_source_manifest_is_identity_not_current_formal_policy(
    tmp_path: Path,
) -> None:
    root = _fixture_repo(tmp_path)
    _, bundle = _bundle(root)
    source = json.loads((root / bundle["source_manifest"]["path"]).read_text())
    assert source["holdout"]["samples_per_action"] == 512
    assert M.build_receipt(root, "bh_loop_c")["content"]["action_order"] == [
        "bh_loop_c"
    ]


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (lambda value: value.__setitem__("mobility_mode", "move"), "mobility_mode"),
        (lambda value: value.__setitem__("action_order", ["bh_block"]), "action_order"),
    ),
)
def test_historical_source_manifest_identity_fields_remain_fail_closed(
    tmp_path: Path, mutator, message: str
) -> None:
    root = _fixture_repo(tmp_path)
    _repin_source_manifest(root, mutator)
    with pytest.raises(M.FixedDomainReceiptRefused, match=message):
        M.build_receipt(root, "bh_loop_c")


def test_no_move_masks_and_zeros_every_base_travel_arm(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    payload = M.build_receipt(root, "bh_loop_c")["content"]
    axes = {row["arm"]: row for row in payload["domain_axes"]}
    for arm in (
        "base_travel_x_lower",
        "base_travel_x_upper",
        "base_travel_y_lower",
        "base_travel_y_upper",
    ):
        assert axes[arm]["mask"] is False
        assert {
            axes[arm][key] for key in ("initial", "maximum", "width", "cap")
        } == {0.0}
    assert axes["landing_aim_y_lower"]["mask"] is False
    assert axes["landing_aim_y_upper"]["mask"] is False


def test_cell_mixture_is_exact_runtime_one_three_one_contract(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    mixture = M.build_receipt(root, "bh_loop_c")["content"]["cell_mixture"]
    assert mixture["value"] == {
        "center_slots": 1,
        "interior_slots": 3,
        "frontier_slots": 1,
        "interior_level_scale": 0.8,
        "frontier_band_fraction": 0.2,
        "schedule": ["interior", "center", "interior", "frontier", "interior"],
    }
    assert mixture["canonical_sha256"] == M.canonical_sha256(mixture["value"])


def test_materialize_uses_registry_path_is_canonical_and_no_clobber(
    tmp_path: Path,
) -> None:
    root = _fixture_repo(tmp_path)
    result = M.materialize(root, "bh_loop_c")
    raw = result.path.read_bytes()
    document = json.loads(raw)
    assert raw == M._canonical_bytes(document) + b"\n"
    assert result.file_sha256 == hashlib.sha256(raw).hexdigest()
    assert result.content_sha256 == document["content_sha256"]
    assert result.repo_path == (
        "configs/n1_fixed_domain_initial_20260802_r9/"
        "bh_loop_c.fixed_domain_initial.v1.json"
    )
    with pytest.raises(M.FixedDomainReceiptRefused, match="overwrite"):
        M.materialize(root, "bh_loop_c")


def test_cli_emits_payload_and_file_sha_summary(tmp_path: Path, capsys) -> None:
    root = _fixture_repo(tmp_path)
    assert (
        M.main(
            (
                "--repo-root",
                str(root),
                "--action-id",
                "bh_loop_c",
                "--materialize",
            )
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["kind"] == M.MATERIALIZATION_KIND
    assert len(summary["content_sha256"]) == 64
    assert len(summary["file_sha256"]) == 64
    assert (root / summary["path"]).is_file()


def _repin_fixed_domain_receipt(
    root: Path, *, action_id: str, file_sha256: str
) -> None:
    registry = root / M.REGISTRY_REPO_PATH
    text = registry.read_text()
    pattern = re.compile(
        r"(fixed_domain_initial_receipt=ArtifactPin\(\n"
        r"\s*\"configs/n1_fixed_domain_initial_20260802_r9/\"\n"
        rf"\s*\"{action_id}\.fixed_domain_initial\.v1\.json\",\n"
        r"\s*)None(,\n\s*\),)",
        re.MULTILINE,
    )
    text, count = pattern.subn(rf"\g<1>{file_sha256!r}\g<2>", text)
    assert count == 1
    registry.write_text(text)
    _commit(root, "pin fixed-domain receipt")


def test_registry_none_to_file_sha_repin_keeps_receipt_bytes_verifiable(
    tmp_path: Path, capsys
) -> None:
    root = _fixture_repo(tmp_path)
    result = M.materialize(root, "bh_loop_c")
    before = result.path.read_bytes()
    before_document = json.loads(before)
    _repin_fixed_domain_receipt(
        root,
        action_id="bh_loop_c",
        file_sha256=result.file_sha256,
    )

    rebuilt = M.build_receipt(
        root,
        "bh_loop_c",
        require_materialized_output=True,
    )
    verified = M.verify_materialized(root, "bh_loop_c")
    assert rebuilt == before_document
    assert M._canonical_bytes(rebuilt) + b"\n" == before
    assert verified.file_sha256 == result.file_sha256
    assert verified.content_sha256 == result.content_sha256
    assert (
        M.main(
            (
                "--repo-root",
                str(root),
                "--action-id",
                "bh_loop_c",
                "--verify",
            )
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["kind"] == M.VERIFICATION_KIND
    assert summary["file_sha256"] == result.file_sha256
    with pytest.raises(M.FixedDomainReceiptRefused, match="already registry-materialized"):
        M.materialize(root, "bh_loop_c")


def test_refuses_bundle_sha_mismatch_even_when_changed_bytes_are_committed(
    tmp_path: Path,
) -> None:
    root = _fixture_repo(tmp_path)
    bundle_path, bundle = _bundle(root)
    bundle["scope"] = "full"
    _write_json(bundle_path, bundle)
    _commit(root, "change bundle without registry repin")
    with pytest.raises(M.FixedDomainReceiptRefused, match="SHA-256 mismatch"):
        M.build_receipt(root, "bh_loop_c")


def test_refuses_non_normalized_manifest_path(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    bundle_path, bundle = _bundle(root)
    old_bundle_sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    bundle["manifest"]["path"] = "../escape.json"
    new_bundle_sha = _write_json(bundle_path, bundle)
    _replace_registry_bundle_sha(
        root, old_sha=old_bundle_sha, new_sha=new_bundle_sha
    )
    _commit(root, "point bundle outside root")
    with pytest.raises(M.FixedDomainReceiptRefused, match="normalized repo path"):
        M.build_receipt(root, "bh_loop_c")


def test_refuses_selected_manifest_action_order_row_mismatch(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    _repin_selected_manifest(
        root,
        lambda manifest: manifest["actions"][0].__setitem__(
            "action_id", "bh_block"
        ),
    )
    with pytest.raises(M.FixedDomainReceiptRefused, match="action"):
        M.build_receipt(root, "bh_loop_c")


def test_refuses_no_move_manifest_with_nonzero_base_travel(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)

    def mutate(manifest: dict) -> None:
        profile = manifest["actions"][0]["ball_profile"]
        profile["base_travel_std_upper_initial_m"][0] = 0.01
        profile["base_travel_std_upper_max_m"][0] = 0.01
        profile["base_travel_max_b_yaw_xy_m"][0] = 0.01

    _repin_selected_manifest(root, mutate)
    with pytest.raises(M.FixedDomainReceiptRefused, match="base_travel"):
        M.build_receipt(root, "bh_loop_c")


def test_refuses_nonfinite_manifest_json(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    bundle_path, bundle = _bundle(root)
    old_bundle_sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    manifest_path = root / bundle["manifest"]["path"]
    manifest = json.loads(manifest_path.read_text())
    manifest["actions"][0]["ball_profile"]["incoming_speed_center_mps"] = float(
        "nan"
    )
    raw = (json.dumps(manifest, sort_keys=True) + "\n").encode("utf-8")
    assert b"NaN" in raw
    manifest_path.write_bytes(raw)
    bundle["manifest"]["sha256"] = hashlib.sha256(raw).hexdigest()
    new_bundle_sha = _write_json(bundle_path, bundle)
    _replace_registry_bundle_sha(
        root, old_sha=old_bundle_sha, new_sha=new_bundle_sha
    )
    _commit(root, "commit nonfinite manifest")
    with pytest.raises(M.FixedDomainReceiptRefused, match="non-finite"):
        M.build_receipt(root, "bh_loop_c")


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda content: content["authority"].__setitem__(
                "curriculum_promotion", True
            ),
            "fixed/non-promotable",
        ),
        (
            lambda content: content["domain_axes"][0].__setitem__(
                "width", -0.1
            ),
            "finite and >= 0.0",
        ),
        (
            lambda content: content["domain_axes"][16].__setitem__(
                "cap", 0.1
            ),
            "base_travel",
        ),
    ),
)
def test_receipt_validator_refuses_nonfixed_negative_or_nonzero_travel(
    tmp_path: Path, mutation, message: str
) -> None:
    root = _fixture_repo(tmp_path)
    forged = deepcopy(M.build_receipt(root, "bh_loop_c"))
    mutation(forged["content"])
    _reseal(forged)
    with pytest.raises(M.FixedDomainReceiptRefused, match=message):
        M.validate_receipt_document(forged)


def test_refuses_untracked_or_dirty_scientific_input(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    path = root / M.MDP_DIR / "action_ball_sampling.py"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(M.FixedDomainReceiptRefused, match="differ from HEAD"):
        M.build_receipt(root, "bh_loop_c")


def test_refuses_committed_runtime_wiring_without_default_sampling_mixture(
    tmp_path: Path,
) -> None:
    root = _fixture_repo(tmp_path)
    train = root / M.TRAIN_REPO_PATH
    text = train.read_text()
    assert text.count("sampling_mixture=SamplingMixture(),") == 1
    train.write_text(
        text.replace(
            "sampling_mixture=SamplingMixture(),",
            "sampling_mixture=None,",
        )
    )
    _commit(root, "remove production sampling mixture")
    with pytest.raises(M.FixedDomainReceiptRefused, match="default SamplingMixture"):
        M.build_receipt(root, "bh_loop_c")
