"""Dependency-light contract tests for the reconstructed fit-lineage oracle."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
ORACLE = ROOT / "hope_training/ball_physics_fit/reference_oracle.py"
PARITY_TEST = Path(__file__).with_name("test_ball_physics_vs_record.py")


def _load_oracle():
    spec = importlib.util.spec_from_file_location("test_fit_lineage_reference", ORACLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _params(ref):
    return ref.SpinEquationParams(e_eff=0.8, a_t=0.3, b_t=0.0, mu_safety=0.5)


def test_contact_normal_is_scale_invariant_and_unit_at_the_fit_kernel():
    ref = _load_oracle()
    kwargs = {
        "v_minus": np.array([1.0, -0.4, -2.0]),
        "v_r": np.array([0.2, 0.0, 0.0]),
        "omega_minus": np.array([0.0, 20.0, -5.0]),
        "params": _params(ref),
    }
    unit = ref.predict_contact(n=np.array([0.0, 0.0, 1.0]), **kwargs)
    scaled = ref.predict_contact(n=np.array([0.0, 0.0, 1.0e-300]), **kwargs)
    np.testing.assert_allclose(unit["v_plus"], scaled["v_plus"], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(unit["omega_plus"], scaled["omega_plus"], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(scaled["n"], axis=1), [1.0], atol=1e-15)


@pytest.mark.parametrize(
    "normal",
    (
        np.array([0.0, 0.0, 0.0]),
        np.array([np.nan, 0.0, 1.0]),
        np.array([np.inf, 0.0, 1.0]),
    ),
)
def test_invalid_contact_normal_fails_loudly(normal):
    ref = _load_oracle()
    with pytest.raises(ValueError):
        ref.predict_contact(
            v_minus=np.array([0.0, 0.0, -1.0]),
            v_r=np.zeros(3),
            n=normal,
            omega_minus=np.zeros(3),
            params=_params(ref),
        )


def test_zero_table_normal_fails_before_simulation():
    ref = _load_oracle()
    table = ref.Table(center_m=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 0.0], surface_z_m=0.0)
    with pytest.raises(ValueError, match="zero normal"):
        ref.simulate(
            [0.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
            0.1,
            table,
            0.01,
        )


def test_reference_lineage_binds_all_three_byte_sources():
    ref = _load_oracle()
    lineage = ref.reference_lineage()
    assert lineage["schema_version"] == 1
    assert set(lineage["files_sha256"]) == {
        "reference_oracle.py",
        "contact_model.py",
        "ball_physics_venue.yaml",
    }
    assert all(len(value) == 64 for value in lineage["files_sha256"].values())
    canonical = "".join(
        f"{name}:{lineage['files_sha256'][name]}\n"
        for name in sorted(lineage["files_sha256"])
    )
    assert lineage["combined_sha256"] == hashlib.sha256(canonical.encode("ascii")).hexdigest()


def test_explicit_missing_record_dir_cannot_turn_into_a_dependency_skip(tmp_path):
    missing = tmp_path / "missing-record"
    env = dict(os.environ)
    env["RECORD_DIR"] = str(missing)
    proc = subprocess.run(
        [sys.executable, str(PARITY_TEST)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "refusing to skip or fall back" in proc.stderr
