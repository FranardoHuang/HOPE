#!/usr/bin/env bash
set -euo pipefail

# Reproduce the one-action diagnostic materialization on the exact pinned Pod checkout.
# OUTPUT_ROOT must be a fresh path. This script never writes into the versioned candidate.
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to a fresh no-clobber directory}"

REPO_ROOT="${REPO_ROOT:-/workspace/franco/a3vendor_final_pin}"
V4D_ROOT="${V4D_ROOT:-/workspace/codexschema/chingmu_racket_v4d_exact_20260803.kRiC8j}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/hope_mjeval_venv/bin/python}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-$REPO_ROOT/assets/motions/chingmu_n1_take061u04_mechanical_candidate_v5_20260803}"
ACTION_ID=Take_061_unit04_BH
MODEL_XML="$REPO_ROOT/agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"
JOINT_CONTRACT="$REPO_ROOT/configs/a3_joint_order_bijection_v1.json"
SOURCE_MOTION="$REPO_ROOT/assets/motions/chingmu73_20260728/hope_$ACTION_ID.npz"
RETARGET="$CANDIDATE_ROOT/$ACTION_ID.v70a150.pkl"
RETARGET_REPORT="$CANDIDATE_ROOT/$ACTION_ID.v70a150.report.json"
RAW_OUTPUT="$OUTPUT_ROOT/hope_$ACTION_ID.v5.raw.npz"
FINAL_OUTPUT="$OUTPUT_ROOT/hope_$ACTION_ID.measured_v5.npz"
AUDIT_OUTPUT="$OUTPUT_ROOT/$ACTION_ID.measured_v5.audit.json"

mkdir "$OUTPUT_ROOT"

test "$(shasum -a 256 "$MODEL_XML" | awk '{print $1}')" = \
  2ab1cd31bffaaef979b4d9f35699bf1e6bec3a127be96c9266af131eee3feb97
test "$(shasum -a 256 "$RETARGET" | awk '{print $1}')" = \
  6e58f04bcf2c66abc8696fcb27bcc354edd06d44ac473611cd682b4c01e0edce
test "$(shasum -a 256 "$RETARGET_REPORT" | awk '{print $1}')" = \
  1d131a543a64f80e077f46cacb75076012c03a6c3208945bfde4030e2e67f807

"$PYTHON_BIN" "$V4D_ROOT/materialize_measured_racket_motion_npz.py" \
  --motion "$SOURCE_MOTION" \
  --retarget "$RETARGET" \
  --retarget-report "$RETARGET_REPORT" \
  --manifest "$V4D_ROOT/chingmu_manifest_v1.json" \
  --catalog "$V4D_ROOT/CLIP_ORDER.json" \
  --uid "$ACTION_ID" \
  --xml "$MODEL_XML" \
  --joint-order-contract "$JOINT_CONTRACT" \
  --output "$RAW_OUTPUT"

export RAW_OUTPUT FINAL_OUTPUT RETARGET RETARGET_REPORT V4D_ROOT
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import hashlib
import json
import os

import numpy as np

raw = Path(os.environ["RAW_OUTPUT"])
out = Path(os.environ["FINAL_OUTPUT"])
retarget = Path(os.environ["RETARGET"])
report_path = Path(os.environ["RETARGET_REPORT"])
v4d_root = Path(os.environ["V4D_ROOT"])

sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
report = json.loads(report_path.read_text())
assert report["admitted"] is True
assert report["mechanical_admission"] is False
assert report["authorization"]["diagnostic_unauthorized"] is True
assert report["authorization"]["training"] is False
constraints = report["optimization"]["motion_constraints"]
assert constraints["urdf_soft_limit_margin_fraction"] == 0.01
assert constraints["urdf_velocity_limit_fraction"] == 0.70
assert constraints["acceleration_proxy_rad_s2"] == 150.0

with np.load(raw, allow_pickle=False) as archive:
    arrays = {key: np.asarray(archive[key]) for key in archive.files}
arrays.update(
    {
        "measured_racket_retarget_admission_semantics": np.asarray(
            report["retarget_admission_semantics"]
        ),
        "measured_racket_mechanical_admission": np.asarray([0], dtype=np.int64),
        "diagnostic_unauthorized": np.asarray([1], dtype=np.int64),
        "training_authorized": np.asarray([0], dtype=np.int64),
        "promotion_authorized": np.asarray([0], dtype=np.int64),
        "deployment_authorized": np.asarray([0], dtype=np.int64),
        "mechanical_candidate_recipe_id": np.asarray(
            "urdf-soft01-vel70-accproxy150-v1"
        ),
        "urdf_soft_limit_margin_fraction": np.asarray([0.01], dtype=np.float64),
        "urdf_velocity_limit_fraction": np.asarray([0.70], dtype=np.float64),
        "acceleration_proxy_rad_s2": np.asarray([150.0], dtype=np.float64),
        "acceleration_proxy_semantics": np.asarray(
            "diagnostic_second_difference_smoothness_cap_not_hardware_authority"
        ),
        "source_retarget_pkl_sha256": np.asarray(sha256(retarget)),
        "retarget_report_sha256": np.asarray(sha256(report_path)),
        "raw_materialization_sha256": np.asarray(sha256(raw)),
        "materializer_sha256": np.asarray(
            sha256(v4d_root / "materialize_measured_racket_motion_npz.py")
        ),
    }
)
with out.open("xb") as stream:
    np.savez_compressed(stream, **arrays)
print(json.dumps({"output": str(out), "sha256": sha256(out)}, sort_keys=True))
PY

"$PYTHON_BIN" "$V4D_ROOT/audit_materialized_measured_racket_npz.py" \
  --motion "$FINAL_OUTPUT" \
  --xml "$MODEL_XML" \
  --joint-order-contract "$JOINT_CONTRACT" \
  --manifest "$V4D_ROOT/chingmu_manifest_v1.json" \
  --catalog "$V4D_ROOT/CLIP_ORDER.json" \
  --uid "$ACTION_ID" \
  --report "$AUDIT_OUTPUT"
