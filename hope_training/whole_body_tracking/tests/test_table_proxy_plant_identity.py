"""Mutation suite for the A3 table-collision proxy's plant identity.

人话:这个文件回答一个问题 —— 换机器人之后,桌面守卫读的那份碰撞代理,
到底能不能证明"它是当前这台机器人身上量出来的"。
下面每一条都是"故意做坏一件事,门必须开火";同时也有"正确的那份必须放行",
免得只剩一个不分青红皂白拒收所有东西的门。

The four scenarios the 2026-08-08 review asked for, in order:

  (i)   the RETIRED 0409 proxy with every digest re-stamped to the 0807 pins
        -- exactly the deception that got through on 2026-08-07 -- must still
        be refused;
  (ii)  the real 0807 proxy must pass, on both engines;
  (iii) tampering with the proxy bytes must still be refused, so the older,
        weaker byte-integrity claim was not traded away for the new one;
  (iv)  deleting ONLY the 20 OmniPicker3 left-gripper components and leaving
        everything else consistent must be caught.

HOST NOTE: needs torch for the Isaac half, so it does NOT run on the py3.8
host.  Run it on a pod checkout::

    python -m pytest \
      hope_training/whole_body_tracking/tests/test_table_proxy_plant_identity.py -q
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
REPO = pathlib.Path(HERE).resolve().parents[2]
WBT_ROOT = REPO / "hope_training" / "whole_body_tracking"
if str(WBT_ROOT) not in sys.path:
    sys.path.insert(0, str(WBT_ROOT))

# Named to sort AFTER ``test_table_obstacle_termination`` on purpose: this
# module reuses that one's isaaclab stub and table-tennis package install, and
# importing it earlier in a full-suite run reorders those global installs for
# every other module that shares them.
from test_table_obstacle_termination import (  # noqa: E402
    BODIES,
    COLLISION_PROXY_PATH,
    COLLISION_PROXY_SHA256,
    term_mod,  # noqa: F401  (pytest fixture)
)

RETIRED_0409_PROXY = (
    REPO
    / "configs/a3_table_collision_proxy_20260731"
    / "a3_table_collision_components.v1.json"
)
PLANT_URDF = REPO / "agi/URDF/A3P-P1-32dof-0807-OP3-pingpang/urdf/model.urdf"


def _seal(document: dict) -> bytes:
    """Re-stamp ``content_sha256`` the way the producer does, then serialize."""

    unsigned = dict(document)
    unsigned.pop("content_sha256", None)
    document = dict(document)
    document["content_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return (
        json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        + b"\n"
    )


def _live_document() -> dict:
    return json.loads((REPO / COLLISION_PROXY_PATH).read_text(encoding="ascii"))


def _write(tmp_path, document: dict, name: str = "proxy.json"):
    payload = _seal(document)
    path = tmp_path / name
    path.write_bytes(payload)
    return str(path), hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# (ii) the honest article passes -- on both engines
# ---------------------------------------------------------------------------


def test_the_real_0807_proxy_is_accepted_by_the_isaac_gate(term_mod):
    owners, centers, axes = term_mod._load_table_collision_proxy_artifact(
        str(REPO / COLLISION_PROXY_PATH), COLLISION_PROXY_SHA256, tuple(BODIES)
    )
    assert len(owners) == len(centers) == len(axes) == 63
    assert set(owners) == set(range(32))


def test_the_real_0807_proxy_is_accepted_by_the_mujoco_gate():
    from mujoco_native import table_termination as term

    term._load_collision_components_cached.cache_clear()
    components = term.load_collision_components()
    assert components.owner_indices.shape == (63,)
    assert components.artifact_sha256 == (
        term.EXPECTED_COLLISION_PROXY_ARTIFACT_SHA256
    )
    term._load_collision_components_cached.cache_clear()


def test_both_engines_read_the_same_artifact_and_the_same_asset_hash(term_mod):
    from mujoco_native import table_termination as term

    assert (
        REPO / COLLISION_PROXY_PATH
    ).resolve() == term.COLLISION_PROXY_ARTIFACT.resolve()
    assert COLLISION_PROXY_SHA256 == term.EXPECTED_COLLISION_PROXY_ARTIFACT_SHA256
    assert (
        term_mod._A3_COLLISION_PROXY_ISAACLAB_ASSET_HASH
        == term.EXPECTED_ISAACLAB_ASSET_HASH
    )
    assert (
        term_mod._A3_COLLISION_PROXY_SOURCE_URDF_SHA256
        == term.EXPECTED_PLANT_SOURCE_URDF_SHA256
    )
    assert (
        term_mod._A3_COLLISION_PROXY_LEFT_GRIPPER_SOURCE_LINKS
        == term.EXPECTED_LEFT_GRIPPER_SOURCE_LINKS
    )


# ---------------------------------------------------------------------------
# (i) the retired 0409 proxy, every digest re-stamped, must still be refused
# ---------------------------------------------------------------------------


def _restamped_0409_document() -> dict:
    """The 0409 proxy wearing all of the 0807 plant's paperwork.

    Every field a name-comparing gate looks at is rewritten to the correct
    0807 value.  The geometry underneath is still the retired robot's.
    """

    retired = json.loads(RETIRED_0409_PROXY.read_text(encoding="ascii"))
    live = _live_document()
    forged = dict(retired)
    for key in (
        "schema_version",
        "artifact_type",
        "source_urdf",
        "runtime_usd_bundle",
        "plant_identity",
        "mujoco_actual_collision_binding",
        "left_gripper_source_links",
        "decomposition",
        "source_component_count",
    ):
        forged[key] = live[key]
    return forged


def test_retired_0409_proxy_with_every_digest_restamped_is_still_refused(
    term_mod, tmp_path
):
    assert RETIRED_0409_PROXY.is_file(), (
        "the retired 0409 proxy is kept on purpose: it is the negative fixture"
    )
    forged = _restamped_0409_document()
    # Everything a name/digest comparison can see now agrees with the 0807 pin.
    assert forged["source_urdf"]["sha256"] == (
        term_mod._A3_COLLISION_PROXY_SOURCE_URDF_SHA256
    )
    assert forged["runtime_usd_bundle"]["bundle_tree_sha256"] == (
        term_mod._A3_COLLISION_PROXY_RUNTIME_USD_TREE_SHA256
    )
    assert forged["plant_identity"]["isaaclab_asset_hash"] == (
        term_mod._A3_COLLISION_PROXY_ISAACLAB_ASSET_HASH
    )
    path, sha = _write(tmp_path, forged)
    with pytest.raises(RuntimeError, match="component count is malformed"):
        term_mod._load_table_collision_proxy_artifact(path, sha, tuple(BODIES))


def test_restamped_0409_proxy_is_refused_by_the_mujoco_gate_too(tmp_path):
    from mujoco_native import table_termination as term

    path, sha = _write(tmp_path, _restamped_0409_document(), "mj_forged.json")
    term._load_collision_components_cached.cache_clear()
    with pytest.raises(term.TableTerminationContractError, match="63 components"):
        term._load_collision_components_cached(path, sha)
    term._load_collision_components_cached.cache_clear()


def test_proxy_without_actual_mujoco_collision_binding_is_refused_by_both_lanes(
    term_mod, tmp_path
):
    document = _live_document()
    document.pop("mujoco_actual_collision_binding")
    path, sha = _write(tmp_path, document, "missing_mujoco_binding.json")
    with pytest.raises(RuntimeError, match="MuJoCo wrist collision inventory"):
        term_mod._load_table_collision_proxy_artifact(path, sha, tuple(BODIES))

    from mujoco_native import table_termination as term

    term._load_collision_components_cached.cache_clear()
    with pytest.raises(
        term.TableTerminationContractError,
        match="MuJoCo wrist collision inventory",
    ):
        term._load_collision_components_cached(path, sha)
    term._load_collision_components_cached.cache_clear()


def test_restamped_0409_proxy_survives_the_count_check_and_dies_on_the_gripper(
    term_mod, tmp_path
):
    """Strip the count check's help; the semantic requirement must stand alone.

    A reviewer will reasonably ask whether 63-vs-43 is the only thing catching
    the forged 0409 document.  It is not: padding the retired geometry back up
    to 63 rows still leaves the twenty gripper links absent.
    """

    forged = _restamped_0409_document()
    filler = dict(forged["components"][0])
    padded = list(forged["components"])
    while len(padded) < 63:
        clone = dict(filler)
        clone["component_id"] = f"zzz_pad_{len(padded):03d}"
        padded.append(clone)
    forged["components"] = sorted(padded, key=lambda row: row["component_id"])
    forged["component_count"] = len(forged["components"])
    path, sha = _write(tmp_path, forged)
    with pytest.raises(RuntimeError, match="metadata is malformed"):
        term_mod._load_table_collision_proxy_artifact(path, sha, tuple(BODIES))


# ---------------------------------------------------------------------------
# (iii) byte integrity was not traded away for the identity proof
# ---------------------------------------------------------------------------


def test_a_single_appended_byte_is_still_refused(term_mod, tmp_path):
    payload = (REPO / COLLISION_PROXY_PATH).read_bytes() + b"\n"
    path = tmp_path / "tampered.json"
    path.write_bytes(payload)
    with pytest.raises(RuntimeError, match="artifact SHA mismatch"):
        term_mod._load_table_collision_proxy_artifact(
            str(path), COLLISION_PROXY_SHA256, tuple(BODIES)
        )


def test_a_geometry_edit_with_a_refreshed_file_sha_is_still_refused(
    term_mod, tmp_path
):
    """The self-seal is the second line: a fresh file SHA does not buy passage."""

    document = _live_document()
    document["components"][0]["local_center_owner_m"][0] += 0.05
    payload = (
        json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        + b"\n"
    )
    path = tmp_path / "regeometried.json"
    path.write_bytes(payload)
    with pytest.raises(RuntimeError, match="content SHA mismatch"):
        term_mod._load_table_collision_proxy_artifact(
            str(path), hashlib.sha256(payload).hexdigest(), tuple(BODIES)
        )


# ---------------------------------------------------------------------------
# (iv) dropping only the twenty gripper components must be caught
# ---------------------------------------------------------------------------


def _without_gripper(document: dict, *, also_drop_declaration: bool) -> dict:
    from mujoco_native import table_termination as term

    gripper = set(term.EXPECTED_LEFT_GRIPPER_SOURCE_LINKS)
    trimmed = dict(document)
    trimmed["components"] = [
        row
        for row in document["components"]
        if row["source_link_name"] not in gripper
    ]
    trimmed["component_count"] = len(trimmed["components"])
    if also_drop_declaration:
        trimmed.pop("left_gripper_source_links", None)
    return trimmed


def test_dropping_only_the_twenty_gripper_components_is_refused(
    term_mod, tmp_path
):
    trimmed = _without_gripper(_live_document(), also_drop_declaration=False)
    assert trimmed["component_count"] == 43
    path, sha = _write(tmp_path, trimmed)
    with pytest.raises(RuntimeError, match="component count is malformed"):
        term_mod._load_table_collision_proxy_artifact(path, sha, tuple(BODIES))


def test_dropping_the_gripper_and_padding_the_count_back_is_still_refused(
    term_mod, tmp_path
):
    """The count is a coarse check; the named links are the real requirement."""

    trimmed = _without_gripper(_live_document(), also_drop_declaration=True)
    filler = dict(trimmed["components"][0])
    padded = list(trimmed["components"])
    while len(padded) < 63:
        clone = dict(filler)
        clone["component_id"] = f"zzz_pad_{len(padded):03d}"
        padded.append(clone)
    trimmed["components"] = sorted(padded, key=lambda row: row["component_id"])
    trimmed["component_count"] = len(trimmed["components"])
    path, sha = _write(tmp_path, trimmed)
    with pytest.raises(RuntimeError, match="split parent mapping"):
        term_mod._load_table_collision_proxy_artifact(path, sha, tuple(BODIES))


def test_dropping_the_gripper_is_refused_by_the_mujoco_gate_too(tmp_path):
    from mujoco_native import table_termination as term

    trimmed = _without_gripper(_live_document(), also_drop_declaration=False)
    path, sha = _write(tmp_path, trimmed, "mj_trimmed.json")
    term._load_collision_components_cached.cache_clear()
    with pytest.raises(term.TableTerminationContractError, match="63 components"):
        term._load_collision_components_cached(path, sha)
    term._load_collision_components_cached.cache_clear()


# ---------------------------------------------------------------------------
# The derivation proof itself, one leg weakened at a time
# ---------------------------------------------------------------------------


def test_proxy_without_a_derivation_proof_is_refused(term_mod, tmp_path):
    document = _live_document()
    document.pop("plant_identity")
    path, sha = _write(tmp_path, document)
    with pytest.raises(RuntimeError, match="no derivation proof"):
        term_mod._load_table_collision_proxy_artifact(path, sha, tuple(BODIES))


def test_proxy_carrying_someone_elses_asset_hash_is_refused(term_mod, tmp_path):
    document = _live_document()
    document["plant_identity"]["isaaclab_asset_hash"] = "0" * 32
    path, sha = _write(tmp_path, document)
    with pytest.raises(RuntimeError, match="no derivation proof"):
        term_mod._load_table_collision_proxy_artifact(path, sha, tuple(BODIES))


def test_proxy_whose_carried_converter_config_was_edited_is_refused(
    term_mod, tmp_path
):
    """The carried config.yaml is what makes the proof redoable offline.

    Edit it and the artifact no longer agrees with the six-file bundle pin it
    also carries, so the offline re-derivation would be run against a
    configuration nobody reviewed.
    """

    document = _live_document()
    document["plant_identity"]["converter_config_yaml"] += "\n# nudged\n"
    document["plant_identity"]["converter_config_sha256"] = hashlib.sha256(
        document["plant_identity"]["converter_config_yaml"].encode("ascii")
    ).hexdigest()
    path, sha = _write(tmp_path, document)
    with pytest.raises(RuntimeError, match="converter configuration"):
        term_mod._load_table_collision_proxy_artifact(path, sha, tuple(BODIES))


def test_mujoco_gate_redoes_the_derivation_and_a_forged_config_fails(tmp_path):
    """MuJoCo has no Pod bundle, and still refuses a doctored derivation.

    The carried converter configuration is cross-checked against the six-file
    pin AND fed to the re-derivation, so a forged config is refused twice.
    Here the file-map check is the one that speaks first.
    """

    from mujoco_native import table_termination as term

    document = _live_document()
    document["plant_identity"]["converter_config_yaml"] = (
        document["plant_identity"]["converter_config_yaml"].replace(
            "merge_fixed_joints: true", "merge_fixed_joints: false"
        )
    )
    forged_sha = hashlib.sha256(
        document["plant_identity"]["converter_config_yaml"].encode("ascii")
    ).hexdigest()
    document["plant_identity"]["converter_config_sha256"] = forged_sha
    for row in document["runtime_usd_bundle"]["files"]:
        if row["path"] == "config.yaml":
            row["sha256"] = forged_sha
    path, sha = _write(tmp_path, document, "mj_forged_config.json")
    term._load_collision_components_cached.cache_clear()
    with pytest.raises(
        term.TableTerminationContractError, match="six-file pin"
    ):
        term._load_collision_components_cached(path, sha)
    term._load_collision_components_cached.cache_clear()


def test_mujoco_derivation_is_load_bearing_on_its_own(tmp_path, monkeypatch):
    """Isolate the derivation: point every NAME at the retired robot too.

    With the plant URDF on disk swapped for the 0409 one and its digest pin
    moved to match, every string comparison in this gate agrees.  Only the
    re-derived IsaacLab asset hash still disagrees, and it is enough.
    """

    from mujoco_native import table_termination as term

    retired_urdf = (
        REPO / "agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf"
    )
    retired_sha = hashlib.sha256(retired_urdf.read_bytes()).hexdigest()
    monkeypatch.setattr(term, "PLANT_SOURCE_URDF", retired_urdf)
    monkeypatch.setattr(term, "EXPECTED_PLANT_SOURCE_URDF_SHA256", retired_sha)
    document = _live_document()
    document["source_urdf"]["sha256"] = retired_sha
    path, sha = _write(tmp_path, document, "mj_isolated.json")
    term._load_collision_components_cached.cache_clear()
    with pytest.raises(
        term.TableTerminationContractError, match="not derived from the reviewed plant"
    ):
        term._load_collision_components_cached(path, sha)
    term._load_collision_components_cached.cache_clear()


def test_mujoco_gate_refuses_a_proxy_naming_another_plant_urdf(tmp_path):
    from mujoco_native import table_termination as term

    document = _live_document()
    document["source_urdf"]["sha256"] = (
        "0d83529cf808e2e68036f8168bd8b7a1c9a97d9c536eb9a14981ea4105d6b9ae"
    )
    path, sha = _write(tmp_path, document, "mj_other_urdf.json")
    term._load_collision_components_cached.cache_clear()
    with pytest.raises(
        term.TableTerminationContractError, match="reviewed A3 plant URDF"
    ):
        term._load_collision_components_cached(path, sha)
    term._load_collision_components_cached.cache_clear()


def test_the_offline_rederivation_actually_reproduces_the_stored_asset_hash():
    """The claim is checkable from the repository alone; check it here."""

    import yaml

    from mujoco_native import table_termination as term

    identity = _live_document()["plant_identity"]
    config = yaml.safe_load(identity["converter_config_yaml"])
    assert term._rederive_isaaclab_asset_hash(config, PLANT_URDF) == (
        identity["isaaclab_asset_hash"]
    )
    assert identity["isaaclab_asset_hash"] == term.EXPECTED_ISAACLAB_ASSET_HASH
    # And it is genuinely a function of the URDF bytes, not of the names in it.
    assert term._rederive_isaaclab_asset_hash(
        config,
        REPO / "agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf",
    ) != identity["isaaclab_asset_hash"]


def test_the_rederivation_drops_exactly_the_three_upstream_path_keys():
    import yaml

    from mujoco_native import table_termination as term

    identity = _live_document()["plant_identity"]
    config = yaml.safe_load(identity["converter_config_yaml"])
    expected = term._rederive_isaaclab_asset_hash(config, PLANT_URDF)
    moved = dict(config)
    moved["asset_path"] = "/somewhere/else/model.urdf"
    moved["usd_dir"] = "/elsewhere"
    assert term._rederive_isaaclab_asset_hash(moved, PLANT_URDF) == expected
    coarser = dict(config)
    coarser["merge_fixed_joints"] = not coarser["merge_fixed_joints"]
    assert term._rederive_isaaclab_asset_hash(coarser, PLANT_URDF) != expected


# ---------------------------------------------------------------------------
# The live-bundle half of the proof, exercised without a Pod bundle
# ---------------------------------------------------------------------------


def _synthetic_bundle(tmp_path, *, asset_path: str, urdf_for_hash):
    import yaml

    from mujoco_native import table_termination as term

    root = tmp_path / "bundle"
    root.mkdir()
    config = yaml.safe_load(_live_document()["plant_identity"]["converter_config_yaml"])
    config["asset_path"] = asset_path
    text = yaml.dump(config, sort_keys=False)
    (root / "config.yaml").write_text(text, encoding="ascii")
    loaded = yaml.safe_load(text)
    (root / ".asset_hash").write_text(
        term._rederive_isaaclab_asset_hash(loaded, urdf_for_hash), encoding="ascii"
    )
    return root


def test_live_bundle_derivation_accepts_a_cache_of_this_plant(
    term_mod, tmp_path, monkeypatch
):
    root = _synthetic_bundle(
        tmp_path,
        asset_path=(
            "/pod/checkout/hope_training/whole_body_tracking/source/"
            "whole_body_tracking/whole_body_tracking/assets/"
            "agibot_a3p_p1_0807_v1/urdf/model.urdf"
        ),
        urdf_for_hash=PLANT_URDF,
    )
    stored = (root / ".asset_hash").read_text(encoding="ascii").strip()
    monkeypatch.setattr(
        term_mod, "_A3_COLLISION_PROXY_ISAACLAB_ASSET_HASH", stored
    )
    assert term_mod._verify_live_bundle_is_a_cache_of_this_plant(root) == stored


def test_live_bundle_from_the_retired_robot_is_refused(term_mod, tmp_path):
    """2026-08-07's deception, reproduced: a 0409 cache wearing fresh digests."""

    root = _synthetic_bundle(
        tmp_path,
        asset_path=(
            "/pod/checkout/hope_training/whole_body_tracking/source/"
            "whole_body_tracking/whole_body_tracking/assets/"
            "agibot_a3p_p1_0807_v1/urdf/model.urdf"
        ),
        urdf_for_hash=(
            REPO / "agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf"
        ),
    )
    with pytest.raises(RuntimeError, match="not a cache of the proxied plant"):
        term_mod._verify_live_bundle_is_a_cache_of_this_plant(root)


def test_live_bundle_naming_another_asset_package_is_refused(term_mod, tmp_path):
    root = _synthetic_bundle(
        tmp_path,
        asset_path="/pod/checkout/.../assets/agibot_a3/urdf/model.urdf",
        urdf_for_hash=PLANT_URDF,
    )
    with pytest.raises(RuntimeError, match="different asset package"):
        term_mod._verify_live_bundle_is_a_cache_of_this_plant(root)


def test_live_bundle_derivation_is_not_satisfied_by_the_pin_alone(
    term_mod, tmp_path
):
    """A bundle that derives correctly, but to a value nobody reviewed.

    This is the "someone re-ran the converter with a different setting" case:
    the cache really is a cache of this URDF, so the derivation holds, and the
    guard still refuses because the reviewed pin moved underneath it.
    """

    import yaml

    from mujoco_native import table_termination as term

    root = tmp_path / "bundle"
    root.mkdir()
    config = yaml.safe_load(
        _live_document()["plant_identity"]["converter_config_yaml"]
    )
    config["asset_path"] = (
        "/pod/checkout/hope_training/whole_body_tracking/source/"
        "whole_body_tracking/whole_body_tracking/assets/"
        "agibot_a3p_p1_0807_v1/urdf/model.urdf"
    )
    config["merge_fixed_joints"] = not config["merge_fixed_joints"]
    text = yaml.dump(config, sort_keys=False)
    (root / "config.yaml").write_text(text, encoding="ascii")
    derived = term._rederive_isaaclab_asset_hash(yaml.safe_load(text), PLANT_URDF)
    (root / ".asset_hash").write_text(derived, encoding="ascii")
    assert derived != term_mod._A3_COLLISION_PROXY_ISAACLAB_ASSET_HASH
    with pytest.raises(RuntimeError, match="differs from the reviewed pin"):
        term_mod._verify_live_bundle_is_a_cache_of_this_plant(root)


def test_split_proxy_parent_indices_must_be_contiguous(term_mod, tmp_path):
    document = _live_document()
    split_rows = [
        row
        for row in document["components"]
        if row["proxy_box_count"] == 2
    ]
    assert [row["proxy_box_index"] for row in split_rows] == [0, 1]
    split_rows[1]["proxy_box_index"] = 0
    path, sha = _write(tmp_path, document, "split_gap.json")
    with pytest.raises(RuntimeError, match="split parent mapping"):
        term_mod._load_table_collision_proxy_artifact(path, sha, tuple(BODIES))
