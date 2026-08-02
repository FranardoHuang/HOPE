from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/resign_chingmu_catalog_from_measured_hit.py"
SPEC = importlib.util.spec_from_file_location("resign_chingmu_catalog", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    catalog = {
        "n_clips": 73,
        "excluded": ["Take_085_unit00_FH"],
        "clips": [
            {
                "clip_id": index,
                "uid": f"Take_fixture_{index:02d}_BH",
                "mount_normal_sign": -1,
                "mount_normal_sign_source": "family-default",
            }
            for index in range(73)
        ],
    }
    source = tmp_path / "CLIP_ORDER.json"
    source.write_text(json.dumps(catalog, sort_keys=True))
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    reports = tmp_path / "reports"
    reports.mkdir()
    gates = {
        "hit_position_le_0p05_m": True,
        "hit_face_le_5_deg": True,
        "hit_long_axis_le_5_deg": True,
        "hit_so3_le_5_deg": True,
        "hit_velocity_direction_observable": True,
        "hit_velocity_direction_le_15_deg": True,
        "hit_velocity_relative_le_0p20": True,
    }
    for index, row in enumerate(catalog["clips"]):
        report = {
            "schema_version": 3,
            "kind": "chingmu_canonical_racket_full_phase_retarget_v3",
            "action_id": row["uid"],
            "sources": {"catalog": {"sha256": source_sha}},
            "teacher": {
                "robot_mount_normal_sign": 1.0 if index == 0 else -1.0,
            },
            "gates": gates,
        }
        (reports / f"{row['uid']}.json").write_text(
            json.dumps(report, sort_keys=True)
        )
    return source, reports


def test_resign_catalog_publishes_complete_measured_sign_map(tmp_path):
    source, reports = _write_fixture(tmp_path)
    output = tmp_path / "CLIP_ORDER_RESIGNED.json"
    result = MODULE.resign(
        source_catalog=source,
        reports_dir=reports,
        output=output,
    )
    document = json.loads(output.read_text())
    assert result["actions"] == 73
    assert result["flipped_uids"] == ["Take_fixture_00_BH"]
    assert result["old_counts"] == {"+1": 0, "-1": 73}
    assert result["new_counts"] == {"+1": 1, "-1": 72}
    assert document["clips"][0]["mount_normal_sign"] == 1
    assert document["clips"][0]["mount_normal_sign_source"].startswith(
        MODULE.SIGN_SOURCE_PREFIX
    )
    assert document["sign_authority"]["flipped_uids"] == [
        "Take_fixture_00_BH"
    ]
