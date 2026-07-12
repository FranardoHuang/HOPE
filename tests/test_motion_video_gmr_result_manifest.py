from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
GVHMR_RESULT_PATH = ROOT / "configs" / "motion_video_gvhmr_results_20260711.json"
GMR_RESULT_PATH = ROOT / "configs" / "motion_video_gmr_results_20260711.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
QUEUE_STATE_SHA256 = "ae147e9504396daf6990c6f2002f7fb9b27c1836fa195c7386840d45052992ee"
SOURCE_BUNDLE_SHA256 = "5b94af15f4a367dff8d7dc6c1cf14d26be6a649a25df6e1c1046b0e6ab72e2de"
DIAGNOSTIC_RESULT_AUDITOR_SHA256 = (
    "a9b75cfe746bdb9d31426a2707733de364b1188368d10c263ca8947981881d92"
)


EXPECTED_REMOTE_EVIDENCE = {
    "franco_forehand_block": {
        "binding_sha256": "f2176247b9ef68628e2040e311c46dbffb92ab453f177bef40f4eafc6501edd6",
        "output_sha256": "0e7e674ecba2459b4db6d2c49fb8498a35db2fe8291782eadb7214933be39be5",
        "output_bytes": 20062,
        "run_log_sha256": "4698666f4e3b892645638a9e6d8ea025e38a63b64939347cf9ae09cd9112d47b",
        "run_log_bytes": 3071,
        "structural_audit_sha256": "6fc5b40195a30364ab53233a2ff2989cceb69159463abd6f2ac14d405b75afd7",
        "finite_elements": 2471,
        "root_rotation_max_norm_error": 9.992007221626409e-16,
        "warmup_rounds": 22,
        "warmup_max_dq": 7.81e-05,
    },
    "franco_backhand_block": {
        "binding_sha256": "064d1ae2b6b7f4b9efc747287b469774c775334e053b56b3a36a93ef72448ec2",
        "output_sha256": "1303ed8c34b1edd7884a68ce78f3408be751423fdc153bd9e5f8d3cc21a2d710",
        "output_bytes": 19454,
        "run_log_sha256": "31638c522be43e30827e66215be64559847c1e08707da32e3fb215ffc86abe3b",
        "run_log_bytes": 3071,
        "structural_audit_sha256": "570a83821a552a5b2079f2103e4b9dbffa8f29ef6fa069eb7a043a8a64b258ca",
        "finite_elements": 2395,
        "root_rotation_max_norm_error": 8.881784197001252e-16,
        "warmup_rounds": 21,
        "warmup_max_dq": 8.21e-05,
    },
    "franco_forehand_loop": {
        "binding_sha256": "82b81ad569c0220b5c459431c9dfd029f29ebe1b2cea99b4364d6756d4815316",
        "output_sha256": "b8767dbacf8dadba371d0933755b7d812081f65c924a3c106567297b910ce889",
        "output_bytes": 21582,
        "run_log_sha256": "87f017b5d684d99211992334770cfc5ccce1c87622f5694d02cba7f0b9d21d4f",
        "run_log_bytes": 3070,
        "structural_audit_sha256": "781321ec6c0cacc449cd008f8e8d4292b9c7ed4edeca0926f744b92633a3f58f",
        "finite_elements": 2661,
        "root_rotation_max_norm_error": 9.992007221626409e-16,
        "warmup_rounds": 20,
        "warmup_max_dq": 7.51e-05,
    },
    "franco_backhand_loop_a": {
        "binding_sha256": "c39c2984336875f4fda03ff9472befe1b81d5b9ca8b5d270008aa8d762a5fc8f",
        "output_sha256": "b9d1ae3a31b2cdda0458a1c949a85d2cf56b55176f9b0c7b49d2f2e24c582755",
        "output_bytes": 21278,
        "run_log_sha256": "b5694f389b056c711e1aefb669ae1b3b6bb1c08cdb745997d10ca9a12adc39a2",
        "run_log_bytes": 3072,
        "structural_audit_sha256": "e31e008e4d4aae8ea6d84759ccbd59f73a6dfea9c6c949a7f8ec455e09e0c922",
        "finite_elements": 2623,
        "root_rotation_max_norm_error": 7.771561172376096e-16,
        "warmup_rounds": 18,
        "warmup_max_dq": 6.41e-05,
    },
    "franco_backhand_loop_b": {
        "binding_sha256": "1f68229ce59ec9eb317ce587f7afea91dcf52582cd6e43177417032250ea2a34",
        "output_sha256": "0c61b752ce13cf3094275ff3d1d731827ba0d3e44650f6dbc5cab0dd11412729",
        "output_bytes": 27966,
        "run_log_sha256": "0c138d772bd08a961ebfdeb79767bd47f6c6cf1fd6b3796f5afb209d879b7b12",
        "run_log_bytes": 3072,
        "structural_audit_sha256": "08b5d05d3caa65c7d80bf3206ef60c4579c45c0f33038a46bad8f27e4da1baa7",
        "finite_elements": 3459,
        "root_rotation_max_norm_error": 9.992007221626409e-16,
        "warmup_rounds": 17,
        "warmup_max_dq": 6.89e-05,
    },
    "franco_backhand_loop_c": {
        "binding_sha256": "7d28d451c4d159b2dc8b7c312fbe8a770c2e79691e3415eeeb01fa9f8624227c",
        "output_sha256": "cf4a9834b51c940bb41cd39d370f6c42c44be6d16166c2d9322a838550dab397",
        "output_bytes": 30094,
        "run_log_sha256": "d2f3a98f84558eab497470a6c4959d52b1375a02fcb9abba90c70de9a8617429",
        "run_log_bytes": 3072,
        "structural_audit_sha256": "b7bc911dcceb3a8fcec40025b6ef47dcd7b9c13790bbbd42c51f2a1ad654db4b",
        "finite_elements": 3725,
        "root_rotation_max_norm_error": 1.1102230246251565e-15,
        "warmup_rounds": 19,
        "warmup_max_dq": 6.42e-05,
    },
    "v6_forehand_block": {
        "binding_sha256": "c52b71f8891dfd5e9b8068de7ba9dc0ff14375a1f453edbad24254d2f5e0f2d1",
        "output_sha256": "ca97c556f7c554a7ee14da4a0db849b4bbc3add20f8d59e244ce248e260b3f54",
        "output_bytes": 10942,
        "run_log_sha256": "e9d9cc1ea014c7c537597a074fe35fe49ccab321c201de2d422a233b1160a35e",
        "run_log_bytes": 3067,
        "structural_audit_sha256": "6a359e085db9a3a9e5ea66086afa3183dd97cd4d84ccab0b44fde80f96086616",
        "finite_elements": 1331,
        "root_rotation_max_norm_error": 1.1102230246251565e-15,
        "warmup_rounds": 25,
        "warmup_max_dq": 8.82e-05,
    },
    "v6_backhand_block": {
        "binding_sha256": "8e0df13978e9bfaa5576387c470fa644cef587ede4f8d1ce815646885dde8fe6",
        "output_sha256": "2b9d3ee7fff3e8d9be028d2bc8430c65babbeddb9fe3b9d0c139321219447cc7",
        "output_bytes": 10942,
        "run_log_sha256": "f1070adb4ef738fa4c6b09269a38b92432c361bf826994697378a81d331ca5b2",
        "run_log_bytes": 3067,
        "structural_audit_sha256": "76749730994778d145e654fd5b3cc0b2b402c4ca327a79c00f8d2b15760e9db9",
        "finite_elements": 1331,
        "root_rotation_max_norm_error": 8.881784197001252e-16,
        "warmup_rounds": 26,
        "warmup_max_dq": 9.04e-05,
    },
    "v7_forehand_block": {
        "binding_sha256": "5d45794ddf8d4922eef5515eef830a035a4352034af78da365dea2f1752fccab",
        "output_sha256": "7a21c658de090dafcc1b2223636742dd0423e8406dc06a8a97d5e3afdc7cd3fa",
        "output_bytes": 19758,
        "run_log_sha256": "1580de2225478cd90e725030207e5beed2c90b3768a076472764d3c10aa86d98",
        "run_log_bytes": 3067,
        "structural_audit_sha256": "145809028bd507e6debcfe4020fae5efd7fbdfcc492f37bebfe5e73199e77163",
        "finite_elements": 2433,
        "root_rotation_max_norm_error": 9.992007221626409e-16,
        "warmup_rounds": 24,
        "warmup_max_dq": 6.84e-05,
    },
    "v7_backhand_block": {
        "binding_sha256": "14aa4b7418faf3a6a5897e519f5373bc37b706260147eb9fb81f19a90d83a799",
        "output_sha256": "6c3370d5089d03db3e6d9953d37d2b7dc5969b69ce2827c96abc9e972a517179",
        "output_bytes": 19758,
        "run_log_sha256": "3f4b2ca224baa7832597206e4e70bf9fe07f941afe4c34a548835a2517b6f9c2",
        "run_log_bytes": 3067,
        "structural_audit_sha256": "1cbdae5405a5455d7f7d2780582d894320ba882c307f40578cb7cc70eb7dd949",
        "finite_elements": 2433,
        "root_rotation_max_norm_error": 1.1102230246251565e-15,
        "warmup_rounds": 28,
        "warmup_max_dq": 7.85e-05,
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tracked_gmr_results_bind_queue_tools_source_bundle_and_gvhmr_inputs():
    gvhmr = json.loads(GVHMR_RESULT_PATH.read_text(encoding="utf-8"))
    gmr = json.loads(GMR_RESULT_PATH.read_text(encoding="utf-8"))
    contract = gmr["processing_contract"]

    assert gmr["status"] == "complete"
    assert gmr["queue_state_sha256"] == QUEUE_STATE_SHA256
    assert gmr["source_manifest_sha256"] == _sha256(GVHMR_RESULT_PATH)
    assert contract["manifest_sha256"] == _sha256(GVHMR_RESULT_PATH)
    assert contract["queue_tool_sha256"] == _sha256(
        ROOT / "scripts" / "run_motion_video_gmr_queue.py"
    )
    # This result is immutable evidence from the original per-video-beta lane.
    # The later canonical-beta lane extended the auditor CLI, so requiring the
    # historical result to equal today's tool bytes would silently rewrite its
    # provenance.  Keep the recorded auditor SHA and let the newer result bind
    # the newer tool independently.
    assert contract["result_auditor_sha256"] == DIAGNOSTIC_RESULT_AUDITOR_SHA256
    assert contract["result_auditor_sha256"] != _sha256(
        ROOT / "scripts" / "audit_gmr_result.py"
    )
    assert contract["gmr_source_bundle"] == {
        "bytes": 282953810,
        "path": "/workspace/codexschema/motion_video_intake_20260711/gmr_provenance/GMR_aabea2e.bundle",
        "sha256": SOURCE_BUNDLE_SHA256,
        "verified_commit": contract["gmr_commit"],
    }
    assert contract["gmr_commit"] == "aabea2eee4be4bc16d4be17dac5ffa85e5a31539"
    assert contract["gmr_worktree_clean"] is True
    assert contract["body_shape_contract"] == "diagnostic_video_betas"
    assert contract["formal_eligible"] is False

    source_rows = {row["asset_id"]: row for row in gvhmr["results"]}
    rows = gmr["results"]
    assert [row["asset_id"] for row in rows] == [
        row["asset_id"] for row in gvhmr["results"]
    ]
    assert set(source_rows) == set(EXPECTED_REMOTE_EVIDENCE)
    assert len(rows) == 10
    for row in rows:
        source = source_rows[row["asset_id"]]
        expected = EXPECTED_REMOTE_EVIDENCE[row["asset_id"]]
        assert row["source_path"] == source["result_path"]
        assert row["source_sha256"] == source["result_sha256"]
        assert row["source_frames"] == source["frames"]
        assert row["frames"] == source["frames"]
        assert row["status"] == "complete"
        assert row["structural_status"] == "pass"
        for field in (
            "binding_sha256",
            "output_sha256",
            "run_log_sha256",
            "structural_audit_sha256",
        ):
            assert row[field] == expected[field]
            assert SHA256.fullmatch(row[field])
        for field in ("output_bytes", "run_log_bytes", "finite_elements"):
            assert row[field] == expected[field]
        assert row["fps"] == 30.0
        assert row["root_rotation_convention"] == "xyzw"
        assert (
            row["root_rotation_max_norm_error"]
            == expected["root_rotation_max_norm_error"]
        )
        assert row["root_rotation_max_norm_error"] < 1e-3
        assert row["finite_elements"] == row["frames"] * (3 + 4 + 31) + 1
        assert row["shapes"] == {
            "dof_pos": [row["frames"], 31],
            "root_pos": [row["frames"], 3],
            "root_rot": [row["frames"], 4],
        }
        assert row["warmup"] == {
            "parser": "built-in",
            "rounds": expected["warmup_rounds"],
            "max_dq": expected["warmup_max_dq"],
            "threshold_strict_lt": contract["warmup_threshold_strict_lt"],
            "max_rounds": contract["warmup_max_rounds"],
        }
        assert row["warmup"]["max_dq"] < row["warmup"]["threshold_strict_lt"]


def test_gmr_results_remain_diagnostic_and_explicitly_uncalibrated():
    gmr = json.loads(GMR_RESULT_PATH.read_text(encoding="utf-8"))

    assert gmr["formal_eligible"] is False
    assert gmr["formal_blockers"]
    assert gmr["body_shape_contract"] == "diagnostic_video_betas"
    assert gmr["ground_root_calibration"] == {
        "status": "not_performed",
        "ground_calibrated": False,
        "root_calibrated": False,
    }
    for row in gmr["results"]:
        assert row["formal_eligible"] is False
        assert row["body_shape_contract"] == "diagnostic_video_betas"
        assert row["ground_calibrated"] is False
        assert row["root_calibrated"] is False
