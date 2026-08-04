from __future__ import annotations

import ast
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "action_ball_211_transition_preflight.py"
)
SPEC = importlib.util.spec_from_file_location(
    "_test_action_ball_211_transition_preflight", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
P = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = P
SPEC.loader.exec_module(P)
REAL_SCAN_LIVE_RESERVATIONS = P._scan_live_reservations


BOOT_A = "11111111-2222-3333-4444-555555555555"
BOOT_B = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
COMMIT = "1" * 40
GPU_UUIDS = {
    0: "GPU-00000000-0000-0000-0000-000000000000",
    1: "GPU-11111111-1111-1111-1111-111111111111",
    2: "GPU-22222222-2222-2222-2222-222222222222",
}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    a_root = (
        checkout
        / P.WBT_RELATIVE
        / "logs"
        / "rsl_rl"
        / P.A_EXPERIMENT_NAME
    )
    c_root = (
        checkout
        / P.WBT_RELATIVE
        / "logs"
        / "rsl_rl"
        / P.C_EXPERIMENT_NAME
    )
    a_root.mkdir(parents=True)
    c_root.mkdir(parents=True)
    lock_root = tmp_path / "locks"
    lock_root.mkdir()
    lock_paths = {index: lock_root / ("gpu%d.lock" % index) for index in range(3)}
    for path in lock_paths.values():
        path.write_bytes(b"lock\n")
    monkeypatch.setattr(P, "GPU_LOCK_PATHS", lock_paths)

    source = {
        "checkout": str(checkout),
        "commit_sha": COMMIT,
        "clean": True,
        "runtime_sources": {
            label: {"path": relative, "sha256": ("%x" % (index + 1)) * 64}
            for index, (label, relative) in enumerate(P.RUNTIME_SOURCE_SPECS)
        },
    }
    state = {
        "source": source,
        "boot_id": BOOT_A,
        "writers": [],
        "reservations": {0: [], 1: [], 2: []},
        "queries": {
            index: {
                "index": index,
                "uuid": GPU_UUIDS[index],
                "total_memory_mib": 24576,
                "free_memory_mib": 24000 - index,
                "processes": [],
            }
            for index in range(3)
        },
        "query_order": [],
        "all_locks_held": [],
    }

    def source_document(observed_checkout, observed_commit):
        assert observed_checkout == checkout
        assert observed_commit == COMMIT
        return json.loads(json.dumps(state["source"]))

    def query_gpu(_admission, index, uuid):
        assert uuid == GPU_UUIDS[index]
        state["query_order"].append(index)
        held = []
        for lock_index in range(3):
            descriptor = os.open(lock_paths[lock_index], os.O_RDWR)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held.append(lock_index)
            finally:
                os.close(descriptor)
        state["all_locks_held"].append(held)
        return json.loads(json.dumps(state["queries"][index]))

    def reservations(
        _admission,
        *,
        checkout,
        commit,
        gpu_index,
        gpu_uuid,
    ):
        assert checkout == harness_checkout
        assert commit == COMMIT
        assert gpu_uuid == GPU_UUIDS[gpu_index]
        return list(state["reservations"][gpu_index])

    harness_checkout = checkout
    monkeypatch.setattr(P, "_source_document", source_document)
    monkeypatch.setattr(P, "_verify_four_grid_authority", lambda _checkout: None)
    monkeypatch.setattr(P, "_load_vendor_admission", lambda _checkout: object())
    monkeypatch.setattr(P, "_query_gpu", query_gpu)
    monkeypatch.setattr(P, "_scan_live_reservations", reservations)
    monkeypatch.setattr(P, "_scan_live_writers", lambda: list(state["writers"]))
    monkeypatch.setattr(P, "_boot_id", lambda: state["boot_id"])
    monkeypatch.setattr(P, "_observed_at", lambda: "2026-08-04T12:34:56.123456Z")

    return {
        "checkout": checkout,
        "a_root": a_root,
        "c_root": c_root,
        "locks": lock_paths,
        "namespaces": {
            "a0": str(a_root / "a0-fresh"),
            "a1": str(a_root / "a1-fresh"),
            "c0": str(c_root / "c0-fresh"),
            "c1": str(c_root / "c1-fresh"),
        },
        "output": tmp_path / "transition.json",
        "state": state,
    }


def _produce(harness, **changes):
    arguments = {
        "checkout": str(harness["checkout"]),
        "commit_sha": COMMIT,
        "gpu_uuids": dict(GPU_UUIDS),
        "namespaces": dict(harness["namespaces"]),
        "output": str(harness["output"]),
    }
    arguments.update(changes)
    return P.produce_receipt(**arguments)


def _assert_lock_is_free(path):
    descriptor = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _rewrite_canonical(path, document):
    raw = P.canonical_bytes(document) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _rewrite_with_fresh_content_sha(path, document):
    unsigned = dict(document)
    unsigned.pop("content_sha256")
    document["content_sha256"] = P.canonical_sha256(unsigned)
    return _rewrite_canonical(path, document)


def test_producer_holds_ordered_three_lock_common_cut_and_validator_accepts(harness):
    observed_lock_order = []
    original = P._acquire_exclusive_gpu_lock

    def wrapped(index, path):
        observed_lock_order.append(index)
        return original(index, path)

    P._acquire_exclusive_gpu_lock = wrapped
    try:
        result = _produce(harness)
    finally:
        P._acquire_exclusive_gpu_lock = original

    assert observed_lock_order == [0, 1, 2]
    assert harness["state"]["query_order"] == [0, 1, 2]
    assert harness["state"]["all_locks_held"] == [[0, 1, 2]] * 3
    assert result["schema_version"] == 1
    assert result["kind"] == "action_ball_211_transition_preflight_v1"
    assert result["status"] == "PASS"
    assert result["diagnostic_unauthorized"] is True
    assert result["machine_verified"] == {
        "common_cut_before_first_scale4096": True,
        "legacy_live_or_pending_count": 0,
        "cross_time_atomicity_claimed": False,
        "cross_checkout_legacy_pending_completeness_claimed": False,
    }
    assert [row["index"] for row in result["gpus"]] == [0, 1, 2]
    assert [row["uuid"] for row in result["gpus"]] == [
        GPU_UUIDS[0],
        GPU_UUIDS[1],
        GPU_UUIDS[2],
    ]
    assert all(row["compute_processes"] == [] for row in result["gpus"])
    assert all(row["live_reservations"] == [] for row in result["gpus"])
    assert [row["cell_id"] for row in result["targets"]] == [
        spec[1] for spec in P.TARGET_SPECS
    ]
    assert all(row["namespace_absent"] is True for row in result["targets"])
    assert stat.S_IMODE(harness["output"].stat().st_mode) == 0o600
    assert harness["output"].read_bytes() == (
        P.canonical_bytes(
            {key: value for key, value in result.items() if key != "artifact"}
        )
        + b"\n"
    )
    validated = P.validate_receipt(
        harness["output"],
        result["artifact"]["sha256"],
        harness["checkout"],
        COMMIT,
    )
    assert validated == result
    for path in harness["locks"].values():
        _assert_lock_is_free(path)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda state: state["queries"][1].__setitem__(
                "processes",
                [{"pid": 7, "process_name": "python", "used_gpu_memory_mib": 1}],
            ),
            "live compute processes",
        ),
        (
            lambda state: state["queries"][2].__setitem__(
                "free_memory_mib", P.MINIMUM_FREE_MEMORY_MIB - 1
            ),
            "UUID or memory",
        ),
        (
            lambda state: state["queries"][0].__setitem__("uuid", GPU_UUIDS[1]),
            "UUID or memory",
        ),
    ],
)
def test_gpu_identity_memory_and_compute_are_fail_closed(harness, mutation, message):
    mutation(harness["state"])
    with pytest.raises(P.TransitionPreflightRefused, match=message):
        _produce(harness)
    assert not harness["output"].exists()
    for path in harness["locks"].values():
        _assert_lock_is_free(path)


def test_duplicate_physical_gpu_uuid_is_rejected_before_locking(harness):
    uuids = dict(GPU_UUIDS)
    uuids[2] = uuids[1]
    with pytest.raises(P.TransitionPreflightRefused, match="must be unique"):
        _produce(harness, gpu_uuids=uuids)
    assert harness["state"]["query_order"] == []


def test_live_physical_or_legacy_reservation_is_rejected(harness):
    harness["state"]["reservations"][0] = [{"namespace": "/legacy/live"}]
    with pytest.raises(P.TransitionPreflightRefused, match="live reservations"):
        _produce(harness)
    assert not harness["output"].exists()


def test_live_legacy_or_grid_writer_is_rejected(harness):
    harness["state"]["writers"] = [
        {
            "pid": 9,
            "proc_starttime_ticks": 10,
            "writer_sources": ["launch_action_ball_a225_four_arm_diagnostic.py"],
            "cmdline_sha256": "2" * 64,
        }
    ]
    with pytest.raises(P.TransitionPreflightRefused, match="writer is live"):
        _produce(harness)
    assert harness["state"]["query_order"] == []


@pytest.mark.parametrize("kind", ["directory", "broken_symlink"])
def test_existing_or_symlink_target_namespace_is_rejected(harness, kind):
    target = Path(harness["namespaces"]["a1"])
    if kind == "directory":
        target.mkdir()
    else:
        target.symlink_to(target.parent / "missing-target")
    with pytest.raises(
        P.TransitionPreflightRefused, match="already exists|symlinked namespace"
    ):
        _produce(harness)
    assert not harness["output"].exists()


def test_target_namespaces_must_be_unique_and_have_exact_family_parent(harness):
    duplicate = dict(harness["namespaces"])
    duplicate["a1"] = duplicate["a0"]
    with pytest.raises(P.TransitionPreflightRefused, match="must be unique"):
        _produce(harness, namespaces=duplicate)

    wrong_parent = dict(harness["namespaces"])
    wrong_parent["c0"] = str(harness["a_root"] / "c0-wrong-family")
    with pytest.raises(P.TransitionPreflightRefused, match="wrong direct parent"):
        _produce(harness, namespaces=wrong_parent)


def test_existing_scale_claim_in_another_namespace_blocks_first_scale_receipt(harness):
    namespace = harness["a_root"] / "already-spent-scale"
    namespace.mkdir()
    payload = {
        "spec": {
            "stage": "scale4096",
            "arm_id": P.TARGET_SPECS[0][1],
        }
    }
    claim_sha = P.canonical_sha256(payload)
    _rewrite_canonical(
        namespace / "launch_claim.json",
        {
            "schema_version": 2,
            "kind": P.ALLOWED_CLAIM_KINDS[0],
            "launch_claim_sha256": claim_sha,
            "canonical_payload": payload,
        },
    )
    with pytest.raises(P.TransitionPreflightRefused, match="already claimed"):
        _produce(harness)
    assert not harness["output"].exists()
    assert harness["state"]["query_order"] == []


def test_lock_conflict_releases_earlier_locks_and_writes_nothing(harness):
    blocker = os.open(harness["locks"][1], os.O_RDWR)
    fcntl.flock(blocker, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(P.TransitionPreflightRefused, match="already held"):
            _produce(harness)
        _assert_lock_is_free(harness["locks"][0])
        assert not harness["output"].exists()
    finally:
        os.close(blocker)


def test_lock_path_replacement_while_held_refuses_before_receipt(
    harness, monkeypatch
):
    original_lock = harness["locks"][1]
    displaced_lock = original_lock.with_name("gpu1.displaced")

    def replace_lock_during_cut():
        original_lock.rename(displaced_lock)
        original_lock.write_bytes(b"replacement\n")
        return "2026-08-04T12:34:56.123456Z"

    monkeypatch.setattr(P, "_observed_at", replace_lock_during_cut)
    with pytest.raises(P.TransitionPreflightRefused, match="pathname identity changed"):
        _produce(harness)
    assert not harness["output"].exists()
    _assert_lock_is_free(harness["locks"][0])
    _assert_lock_is_free(harness["locks"][2])


def test_physical_registry_rejects_entry_labeled_for_another_gpu(harness):
    lock = harness["locks"][0]
    registry = lock.parent / (lock.name + P.GPU_RESERVATION_REGISTRY_SUFFIX)
    registry.mkdir()
    claim_sha = "a" * 64
    document = {
        "schema_version": 1,
        "kind": "measured_vendor_v2_gpu_slot_reservation_v1",
        "owner_pid": 123,
        "owner_proc_starttime_ticks": 456,
        "gpu_index": 1,
        "gpu_uuid": GPU_UUIDS[1],
        "namespace": str(harness["a_root"] / "old-reservation"),
        "checkout": str(harness["checkout"]),
        "commit_sha": COMMIT,
        "launch_claim_sha256": claim_sha,
        "max_compute_pids": 2,
        "minimum_free_memory_mib": 8192,
        "allow_vendor_v2_colocation": True,
    }
    _rewrite_canonical(registry / (claim_sha + ".json"), document)

    class EmptyAdmission:
        def _live_reservations(self, *args, **kwargs):
            return []

    with pytest.raises(
        P.TransitionPreflightRefused,
        match="identity differs from its GPU lock registry",
    ):
        REAL_SCAN_LIVE_RESERVATIONS(
            EmptyAdmission(),
            checkout=harness["checkout"],
            commit=COMMIT,
            gpu_index=0,
            gpu_uuid=GPU_UUIDS[0],
        )


def test_receipt_output_is_o_excl_and_preserves_existing_bytes(harness):
    harness["output"].write_bytes(b"do-not-clobber\n")
    with pytest.raises(P.TransitionPreflightRefused, match="output.*already exists"):
        _produce(harness)
    assert harness["output"].read_bytes() == b"do-not-clobber\n"
    assert harness["state"]["query_order"] == []


def test_exclusive_writer_fsyncs_file_and_parent(harness, monkeypatch):
    observed_fsyncs = []
    original = os.fsync

    def recording_fsync(descriptor):
        observed_fsyncs.append(os.fstat(descriptor).st_mode)
        return original(descriptor)

    monkeypatch.setattr(P.os, "fsync", recording_fsync)
    _produce(harness)
    assert len(observed_fsyncs) == 2
    assert stat.S_ISREG(observed_fsyncs[0])
    assert stat.S_ISDIR(observed_fsyncs[1])


def test_validator_rejects_noncanonical_bytes_even_with_matching_file_sha(harness):
    result = _produce(harness)
    document = json.loads(harness["output"].read_text(encoding="utf-8"))
    raw = json.dumps(document, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    harness["output"].write_bytes(raw)
    with pytest.raises(P.TransitionPreflightRefused, match="canonical JSON"):
        P.validate_receipt(
            str(harness["output"]),
            hashlib.sha256(raw).hexdigest(),
            str(harness["checkout"]),
            COMMIT,
        )
    assert result["content_sha256"] == document["content_sha256"]


def test_validator_rejects_symlink_receipt(harness):
    result = _produce(harness)
    link = harness["output"].with_name("transition-link.json")
    link.symlink_to(harness["output"])
    with pytest.raises(P.TransitionPreflightRefused, match="cannot be opened"):
        P.validate_receipt(
            str(link),
            result["artifact"]["sha256"],
            str(harness["checkout"]),
            COMMIT,
        )


def test_validator_rejects_receipt_reached_through_symlinked_parent(harness):
    result = _produce(harness)
    linked_parent = harness["output"].parent / "linked-parent"
    linked_parent.symlink_to(harness["output"].parent, target_is_directory=True)
    linked_receipt = linked_parent / harness["output"].name
    with pytest.raises(P.TransitionPreflightRefused, match="bounded regular file"):
        P.validate_receipt(
            str(linked_receipt),
            result["artifact"]["sha256"],
            str(harness["checkout"]),
            COMMIT,
        )


def test_validator_rejects_receipt_from_an_old_boot(harness):
    result = _produce(harness)
    harness["state"]["boot_id"] = BOOT_B
    with pytest.raises(P.TransitionPreflightRefused, match="another or invalid host boot"):
        P.validate_receipt(
            str(harness["output"]),
            result["artifact"]["sha256"],
            str(harness["checkout"]),
            COMMIT,
        )


def test_validator_rejects_source_or_runtime_pin_drift(harness):
    result = _produce(harness)
    harness["state"]["source"]["runtime_sources"]["a211_launcher"][
        "sha256"
    ] = "f" * 64
    with pytest.raises(P.TransitionPreflightRefused, match="runtime-source bytes differ"):
        P.validate_receipt(
            str(harness["output"]),
            result["artifact"]["sha256"],
            str(harness["checkout"]),
            COMMIT,
        )


def test_validator_rejects_replaced_lock_inode(harness):
    result = _produce(harness)
    lock = harness["locks"][2]
    old = lock.with_name("gpu2.old")
    lock.rename(old)
    lock.write_bytes(b"replacement\n")
    with pytest.raises(P.TransitionPreflightRefused, match="lock inode changed"):
        P.validate_receipt(
            str(harness["output"]),
            result["artifact"]["sha256"],
            str(harness["checkout"]),
            COMMIT,
        )


def test_validator_rejects_canonical_extra_field_with_recomputed_hashes(harness):
    _produce(harness)
    document = json.loads(harness["output"].read_text(encoding="utf-8"))
    document["unexpected_authority"] = True
    file_sha = _rewrite_with_fresh_content_sha(harness["output"], document)
    with pytest.raises(P.TransitionPreflightRefused, match="receipt keys differ"):
        P.validate_receipt(
            str(harness["output"]),
            file_sha,
            str(harness["checkout"]),
            COMMIT,
        )


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda document: document.__setitem__("schema_version", True), "identity"),
        (
            lambda document: document["machine_verified"].__setitem__(
                "legacy_live_or_pending_count", False
            ),
            "machine verification semantics",
        ),
        (lambda document: document["source"].__setitem__("clean", 1), "source"),
        (lambda document: document["gpus"][0].__setitem__("index", False), "GPU 0"),
        (
            lambda document: document["targets"][0].__setitem__(
                "gpu_index", False
            ),
            "target a0",
        ),
    ],
)
def test_validator_rejects_boolean_integer_aliases(harness, mutation, message):
    _produce(harness)
    document = json.loads(harness["output"].read_text(encoding="utf-8"))
    mutation(document)
    file_sha = _rewrite_with_fresh_content_sha(harness["output"], document)
    with pytest.raises(P.TransitionPreflightRefused, match=message):
        P.validate_receipt(
            str(harness["output"]),
            file_sha,
            str(harness["checkout"]),
            COMMIT,
        )


def test_source_document_requires_clean_exact_commit_and_pins_runtime_bytes(
    tmp_path, monkeypatch
):
    checkout = tmp_path / "source"
    checkout.mkdir()
    repository_root = SCRIPT.parents[1]
    for _label, relative in P.RUNTIME_SOURCE_SPECS:
        destination = checkout / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository_root / relative, destination)
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=transition-test",
            "-c",
            "user.email=transition-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "exact source",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(P, "_script_checkout", lambda: checkout)

    source = P._source_document(checkout, commit)
    assert source["checkout"] == str(checkout)
    assert source["commit_sha"] == commit
    assert source["clean"] is True
    assert set(source["runtime_sources"]) == {
        label for label, _relative in P.RUNTIME_SOURCE_SPECS
    }
    for label, relative in P.RUNTIME_SOURCE_SPECS:
        assert source["runtime_sources"][label] == {
            "path": relative,
            "sha256": hashlib.sha256((checkout / relative).read_bytes()).hexdigest(),
        }

    launcher = checkout / dict(P.RUNTIME_SOURCE_SPECS)["a211_launcher"]
    launcher.write_bytes(launcher.read_bytes() + b"\n# dirty transition test\n")
    with pytest.raises(P.TransitionPreflightRefused, match="not clean"):
        P._source_document(checkout, commit)


def test_vendor_v2_admission_mechanics_are_loaded_from_the_checkout(
    tmp_path, monkeypatch
):
    repository_root = SCRIPT.parents[1]
    admission = P._load_vendor_admission(repository_root)
    assert type(admission).__name__ == "VendorV2GPUAdmission"
    assert admission.physical_reservation_registry is True
    P._verify_four_grid_authority(repository_root)
    lock_paths = {
        index: tmp_path / ("gpu%d.lock" % index) for index in range(3)
    }
    for path in lock_paths.values():
        path.write_bytes(b"lock\n")
    monkeypatch.setattr(P, "GPU_LOCK_PATHS", lock_paths)
    empty_checkout = tmp_path / "empty-checkout"
    empty_checkout.mkdir()
    assert P._scan_live_reservations(
        admission,
        checkout=empty_checkout,
        commit=COMMIT,
        gpu_index=0,
        gpu_uuid=GPU_UUIDS[0],
    ) == []


def test_boot_id_reader_accepts_procfs_style_zero_stat_size(tmp_path, monkeypatch):
    virtual_boot_id = tmp_path / "boot_id"
    virtual_boot_id.write_bytes(b"")
    assert virtual_boot_id.stat().st_size == 0
    monkeypatch.setattr(P, "BOOT_ID_PATH", virtual_boot_id)
    monkeypatch.setattr(P.os, "read", lambda _descriptor, _count: (BOOT_A + "\n").encode())
    assert P._boot_id() == BOOT_A


def test_script_parses_with_python_38_grammar():
    ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT), feature_version=8)
