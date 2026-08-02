from __future__ import annotations

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/scripts/benchmark_mujoco_physical_ball_tax.py"
)
SPEC = importlib.util.spec_from_file_location("benchmark_mujoco_physical_ball_tax", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
B = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = B
SPEC.loader.exec_module(B)


def test_contract_uses_exact_runtime_ball_and_dt():
    row = B.load_ball_and_step_contract(B.DEFAULT_CONTRACT)
    assert row["radius_m"] == pytest.approx(0.02)
    assert row["mass_kg"] == pytest.approx(0.0034)
    assert row["inertia_coeff"] == pytest.approx(2.0 / 3.0)
    assert row["physics_step_dt_s"] == pytest.approx(0.005)
    assert len(row["sha256"]) == 64


def test_scene_diff_is_exactly_one_native_ball_and_pairs():
    table_scene = B._load_table_scene_module()
    contract = B.load_ball_and_step_contract(B.DEFAULT_CONTRACT)
    canonical = B.DEFAULT_MJCF.read_bytes()
    off_xml, off_receipt = B.assemble_scene_xml(
        canonical,
        table_scene=table_scene,
        ball_contract=contract,
        with_ball=False,
    )
    on_xml, on_receipt = B.assemble_scene_xml(
        canonical,
        table_scene=table_scene,
        ball_contract=contract,
        with_ball=True,
    )
    off = ET.fromstring(off_xml)
    on = ET.fromstring(on_xml)
    assert off.find(f".//body[@name='{B.BALL_BODY_NAME}']") is None
    body = on.find(f".//body[@name='{B.BALL_BODY_NAME}']")
    assert body is not None
    assert body.find(f"./freejoint[@name='{B.BALL_JOINT_NAME}']") is not None
    geom = body.find(f"./geom[@name='{B.BALL_GEOM_NAME}']")
    assert geom is not None
    assert geom.attrib["type"] == "sphere"
    assert float(geom.attrib["size"]) == pytest.approx(0.02)
    assert geom.attrib["contype"] == "1"
    assert geom.attrib["conaffinity"] == "7"
    pairs = on.findall(f"./contact/pair[@geom1='{B.BALL_GEOM_NAME}']")
    assert {pair.attrib["geom2"] for pair in pairs} == {
        B.RACKET_GEOM_NAME,
        B.TABLE_GEOM_NAME,
        *B.NET_GEOM_NAMES,
    }
    assert off_receipt["canonical_mjcf_sha256"] == on_receipt["canonical_mjcf_sha256"]
    assert off_receipt["with_ball"] is False
    assert on_receipt["with_ball"] is True


def test_protocol_refuses_4096_extrapolation_by_construction():
    parser = B._parser()
    args = parser.parse_args(["--out", "/tmp/unused.json"])
    assert args.num_envs == (1, 8, 32, 64)
    assert 4096 not in args.num_envs
    assert "MUST NOT be linearly extrapolated" in B.__doc__


def test_num_env_parser_is_fail_closed():
    assert B._parse_num_envs("1,8,32,64") == (1, 8, 32, 64)
    with pytest.raises(Exception):
        B._parse_num_envs("1,1")
    with pytest.raises(Exception):
        B._parse_num_envs("0,8")
