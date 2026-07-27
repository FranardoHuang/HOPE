from __future__ import annotations

import hashlib
import json

import pytest

from hope_planner.action_catalog import (
    ACTION_CATALOG_SCHEMA_V1,
    MAX_ACTION_UID,
    ActionCatalog,
    ActionRecord,
    derive_action_uid,
)


def _content_hash(action_id):
    return hashlib.sha256(("clip:" + action_id).encode("utf-8")).hexdigest()


def _identity_rows(count):
    return [
        {
            "action_id": "action_{:03d}".format(index),
            "family": "forehand" if index % 2 == 0 else "backhand",
            "content_sha256": _content_hash("action_{:03d}".format(index)),
        }
        for index in range(count)
    ]


@pytest.mark.parametrize("count", [1, 2, 5, 6, 93])
def test_build_round_trip_and_lookup_for_required_catalog_sizes(count):
    catalog = ActionCatalog.build(_identity_rows(count))

    assert catalog.schema_version == ACTION_CATALOG_SCHEMA_V1
    assert len(catalog) == count
    assert [record.slot for record in catalog] == list(range(count))
    assert len(set(catalog.action_ids)) == count
    assert len(set(catalog.action_uids)) == count
    assert all(1 <= uid <= MAX_ACTION_UID for uid in catalog.action_uids)

    for record in catalog:
        assert catalog.by_id(record.action_id) is record
        assert catalog.by_uid(record.action_uid) is record
        assert catalog.by_slot(record.slot) is record
        assert catalog[record.slot] is record

    mapping = catalog.to_mapping()
    assert ActionCatalog.from_mapping(mapping) == catalog
    assert ActionCatalog.from_mapping(
        json.loads(json.dumps(mapping, sort_keys=True))
    ) == catalog


def test_two_forehands_have_distinct_content_bound_uids_and_local_slots():
    rows = [
        {
            "action_id": "fh_loop_low",
            "family": "forehand",
            "content_sha256": _content_hash("fh_loop_low"),
        },
        {
            "action_id": "fh_loop_high",
            "family": "forehand",
            "content_sha256": _content_hash("fh_loop_high"),
        },
    ]
    catalog = ActionCatalog.build(rows)

    assert catalog[0].family == catalog[1].family == "forehand"
    assert catalog[0].slot == 0
    assert catalog[1].slot == 1
    assert catalog[0].action_uid != catalog[1].action_uid
    assert catalog[0].action_uid == derive_action_uid(
        "fh_loop_low", "forehand", _content_hash("fh_loop_low")
    )


def test_reorder_preserves_uids_rebuilds_dense_slots_and_changes_catalog_hash():
    catalog = ActionCatalog.build(_identity_rows(6))
    original_uids = {
        record.action_id: record.action_uid for record in catalog
    }

    reordered = catalog.reorder(tuple(reversed(catalog.action_ids)))

    assert reordered.action_ids == tuple(reversed(catalog.action_ids))
    assert [record.slot for record in reordered] == list(range(6))
    assert {
        record.action_id: record.action_uid for record in reordered
    } == original_uids
    assert reordered.catalog_sha256 != catalog.catalog_sha256
    assert ActionCatalog.from_mapping(reordered.to_mapping()) == reordered


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"unknown": 1}),
        lambda value: value.pop("schema_version"),
        lambda value: value.update({"schema_version": True}),
        lambda value: value.update({"schema_version": 1.0}),
        lambda value: value.update({"schema_version": 2}),
        lambda value: value.update({"catalog_sha256": "A" * 64}),
        lambda value: value.update({"catalog_sha256": "0" * 64}),
    ],
)
def test_catalog_mapping_rejects_unknown_missing_bad_integer_and_bad_self_hash(
    mutate,
):
    mapping = ActionCatalog.build(_identity_rows(2)).to_mapping()
    mutate(mapping)
    with pytest.raises(ValueError):
        ActionCatalog.from_mapping(mapping)


def test_mapping_rows_must_be_in_dense_slot_order_and_nonempty():
    mapping = ActionCatalog.build(_identity_rows(2)).to_mapping()
    mapping["actions"] = list(reversed(mapping["actions"]))
    with pytest.raises(ValueError, match="dense slots"):
        ActionCatalog.from_mapping(mapping)

    mapping = ActionCatalog.build(_identity_rows(2)).to_mapping()
    mapping["actions"][1]["slot"] = 0
    with pytest.raises(ValueError, match="duplicate slot"):
        ActionCatalog.from_mapping(mapping)

    with pytest.raises(ValueError, match="at least one"):
        ActionCatalog.build([])


@pytest.mark.parametrize(
    "field,value",
    [
        ("action_id", " action_000"),
        ("action_id", "action_000 "),
        ("family", " forehand"),
        ("family", "forehand "),
    ],
)
def test_identity_strings_must_be_trimmed_at_catalog_producer(field, value):
    rows = _identity_rows(1)
    rows[0][field] = value
    with pytest.raises(ValueError, match="trimmed"):
        ActionCatalog.build(rows)


@pytest.mark.parametrize(
    "action_id",
    (
        "fore\u0000hand",
        "fore\u200bhand",
        "fore\u0301hand",
    ),
)
def test_action_identity_rejects_control_format_and_non_nfc_text(action_id):
    rows = _identity_rows(1)
    rows[0]["action_id"] = action_id
    with pytest.raises(ValueError):
        ActionCatalog.build(rows)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.update({"unknown": 1}),
        lambda row: row.pop("family"),
        lambda row: row.update({"slot": False}),
        lambda row: row.update({"slot": 0.0}),
        lambda row: row.update({"action_uid": False}),
        lambda row: row.update({"action_uid": 1.0}),
        lambda row: row.update({"content_sha256": "A" * 64}),
        lambda row: row.update({"content_sha256": "0" * 63}),
    ],
)
def test_full_rows_reject_unknown_missing_bool_float_and_bad_hash(mutate):
    mapping = ActionCatalog.build(_identity_rows(2)).to_mapping()
    mutate(mapping["actions"][0])
    with pytest.raises(ValueError):
        ActionCatalog.from_mapping(mapping)


def test_duplicate_id_uid_and_uid_content_mismatch_fail_closed():
    rows = _identity_rows(2)
    rows[1] = dict(rows[0])
    with pytest.raises(ValueError, match="duplicate action_id"):
        ActionCatalog.build(rows)

    mapping = ActionCatalog.build(_identity_rows(2)).to_mapping()
    mapping["actions"][1]["action_uid"] = mapping["actions"][0]["action_uid"]
    with pytest.raises(ValueError):
        ActionCatalog.from_mapping(mapping)

    valid = ActionRecord.build(
        action_id="fh_loop",
        slot=0,
        family="forehand",
        content_sha256=_content_hash("fh_loop"),
    )
    with pytest.raises(ValueError, match="does not match"):
        ActionRecord(
            action_id=valid.action_id,
            action_uid=valid.action_uid,
            slot=valid.slot,
            family=valid.family,
            content_sha256=_content_hash("different_clip"),
        )


@pytest.mark.parametrize(
    "lookup,args",
    [
        ("by_id", ("does_not_exist",)),
        ("by_uid", (1,)),
        ("by_slot", (99,)),
    ],
)
def test_unknown_id_uid_and_slot_fail_strictly(lookup, args):
    catalog = ActionCatalog.build(_identity_rows(2))
    # Among three positive integers at least one is absent from a two-row
    # catalog.  This remains deterministic even under a contrived hash result.
    if lookup == "by_uid":
        args = (
            next(uid for uid in range(1, 4) if uid not in catalog.action_uids),
        )
    with pytest.raises(KeyError):
        getattr(catalog, lookup)(*args)


def test_create_does_not_sort_or_repair_records():
    first = ActionRecord.build(
        action_id="a",
        slot=0,
        family="forehand",
        content_sha256=_content_hash("a"),
    )
    second = ActionRecord.build(
        action_id="b",
        slot=1,
        family="backhand",
        content_sha256=_content_hash("b"),
    )
    with pytest.raises(ValueError, match="dense slots"):
        ActionCatalog.create((second, first))


def test_reorder_rejects_duplicate_missing_and_unknown_ids():
    catalog = ActionCatalog.build(_identity_rows(2))
    with pytest.raises(ValueError, match="duplicate"):
        catalog.reorder((catalog.action_ids[0], catalog.action_ids[0]))
    with pytest.raises(ValueError, match="every action"):
        catalog.reorder((catalog.action_ids[0],))
    with pytest.raises(KeyError, match="unknown"):
        catalog.reorder((catalog.action_ids[0], "unknown"))
