"""Host-only contracts for the independent arbitrary-N bank producer."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent / "scripts"
for directory in (SCRIPTS, TESTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import canonical_motion_arbitrary_bank as arbitrary  # noqa: E402
import canonical_motion_compiler as compiler  # noqa: E402
import materialize_arbitrary_motion_bank_recipe as recipe_builder  # noqa: E402
import test_canonical_motion_compiler as compiler_fixtures  # noqa: E402
from mujoco_motion_player import (  # noqa: E402
    RUNTIME_BODY_NAMES,
    RUNTIME_JOINT_NAMES,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return _sha(path)


def _write_motion(
    path: Path,
    *,
    frames: int,
    motion_index: int = 0,
    hold: bool = False,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    joint_pos = np.zeros((frames, 31), dtype=np.float32)
    if not hold:
        progress = np.linspace(0.0, 0.18 + motion_index * 1.0e-4, frames)
        joint_pos[:, 18] = progress
        joint_pos[:, 20] = -0.4 * progress
    body_pos = np.zeros((frames, 32, 3), dtype=np.float32)
    pelvis = RUNTIME_BODY_NAMES.index("pelvis_link")
    body_pos[:, pelvis, 2] = 1.0
    body_quat = np.zeros((frames, 32, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    np.savez(
        path,
        fps=np.asarray([50.0], dtype=np.float32),
        joint_pos=joint_pos,
        joint_vel=np.zeros((frames, 31), dtype=np.float32),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=np.zeros((frames, 32, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((frames, 32, 3), dtype=np.float32),
        kinematics_schema_version=np.asarray([2], dtype=np.int64),
        body_pos_point=np.asarray("link_origin"),
        body_lin_vel_point=np.asarray("center_of_mass"),
        body_names=np.asarray(RUNTIME_BODY_NAMES),
    )
    return _sha(path)


class BankFixture:
    """One self-contained content-bound arbitrary-N fixture."""

    def __init__(self, root: Path, count: int, monkeypatch):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.count = count
        self.capsule_directory = root / "capsule"
        self.capsule_path = (
            self.capsule_directory / "SOURCE_CAPSULE_RECEIPT.json"
        )
        self.recipe_path = root / "arbitrary_recipe.json"
        self.template_path = root / "template.json"
        self.ready_path = root / "ready_source_hold.npz"
        self.acceleration_path = root / "acceleration_receipt.json"
        self.producer_path = root / "canonical_motion_arbitrary_bank.py"
        self.motion_ids = tuple(
            f"take_{index:03d}_unit00_bh" for index in range(count)
        )

        original_producer = Path(arbitrary.__file__).resolve()
        self.producer_path.write_bytes(original_producer.read_bytes())
        monkeypatch.setattr(arbitrary, "__file__", str(self.producer_path))

        _write_motion(self.ready_path, frames=3, hold=True)
        _write_json(self.template_path, {"fixture": "compiler template"})
        _write_json(
            self.acceleration_path,
            {
                "acceleration_rad_s2": [80.0] * 31,
                "all_positive": True,
                "joint_names": list(RUNTIME_JOINT_NAMES),
                "limiting_source_frame": ["fixture:f0"] * 31,
                "method": "host fixture positive diagonal envelope",
                "minimum_effort_margin_nm": [10.0] * 31,
            },
        )

        actions = []
        for index, motion_id in enumerate(self.motion_ids):
            motion_path = self.capsule_directory / "motions" / f"{motion_id}.npz"
            motion_sha = _write_motion(
                motion_path,
                frames=9,
                motion_index=index,
            )
            station = [index * 0.01 - 0.6, -0.8 + index * 0.001]
            base_spawn = [station[0] + 0.5, station[1] + 0.7625]
            metadata_path = (
                self.capsule_directory
                / "metadata"
                / f"{motion_id}.meta.json"
            )
            _write_json(
                metadata_path,
                {"station_xy_hope_m": station},
            )
            actions.append(
                {
                    "action_id": motion_id,
                    "family": "backhand",
                    "motion_path": f"motions/{motion_id}.npz",
                    "motion_sha256": motion_sha,
                    "metadata_path": f"metadata/{motion_id}.meta.json",
                    "metadata_sha256": _sha(metadata_path),
                    "base_spawn_center_w_xy_m": base_spawn,
                    "T": 9,
                    "fps": 50.0,
                    "hit_frame_50": 4,
                    "reference_t_hit_s": 0.08,
                    "reference_t_cycle_s": 0.16,
                }
            )
        self.capsule = {
            "schema_version": 1,
            "consumer_interface": arbitrary.SOURCE_CAPSULE_INTERFACE,
            "verdict": "PASS_SOURCE_INVENTORY_ONLY",
            "authorization": {
                "compiler_candidate_authorized": False,
                "motion_admission_present": False,
                "training_authorized": False,
                "deployment_authorized": False,
                "hardware_authorized": False,
            },
            "actions": actions,
        }
        _write_json(self.capsule_path, self.capsule)
        self.recipe = {
            "schema_version": 1,
            "recipe_type": arbitrary.RECIPE_TYPE,
            "bank_id": f"fixture_arbitrary_{count}",
            "publication_class": arbitrary.PUBLICATION_CLASS,
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
            "producer": self._binding(self.producer_path),
            "source_capsule": self._binding(self.capsule_path),
            "ordered_motion_ids": list(self.motion_ids),
            "shared_ready": {
                "source_motion_path": self._relative(self.ready_path),
                "source_motion_sha256": _sha(self.ready_path),
                "source_frame": 1,
                "hold_tolerances": {
                    "joint_position_rad": 1.0e-6,
                    "root_position_m": 1.0e-6,
                    "root_orientation_rad": 1.0e-6,
                    "joint_velocity_rad_s": 1.0e-6,
                    "body_linear_velocity_m_s": 1.0e-6,
                    "body_angular_velocity_rad_s": 1.0e-6,
                },
                "evidence_status": (
                    "SOURCE_HOLD_ONLY_NOT_GROUNDED_CERTIFICATE"
                ),
            },
            "marker_policy": {
                "mode": "source_hit_centered_marker_only_v1",
                "half_width_frames": 1,
                "minimum_source_preparation_frames": 4,
                "minimum_source_recovery_frames": 4,
                "minimum_compiled_recovery_s": 0.04,
            },
            "compiler_template": self._binding(self.template_path),
            "compiler_options": {
                "joint_acceleration_receipt": self._binding(
                    self.acceleration_path
                ),
                "full_root_position_lower": [0.0, -0.4, 0.85, -1, -1, -1],
                "full_root_position_upper": [0.4, 0.1, 1.05, 1, 1, 1],
                "full_root_velocity": [1, 1, 0.5, 2, 2, 2],
                "full_root_acceleration": [10, 10, 5, 20, 20, 20],
                "samples_per_scaled_unit": 6,
                "min_connector_intervals": 5,
                "min_core_intervals": 5,
                "grid_subdivisions": 4,
                "search_workers": 1,
                "search_parallel_backend": "thread",
            },
            "required_output_matrix": {
                "motion_ids": list(self.motion_ids),
                "scopes": ["upper", "full"],
                "candidate_count": count * 2,
            },
            "placement_contract": dict(arbitrary._PLACEMENT_CONTRACT),
            "non_claims": [
                "grounded_ready_certificate",
                "dynamics_or_balance",
                "table_or_collision_safety",
                "physical_ball_return",
                "training_authorization",
                "hardware_authorization",
            ],
        }
        self.write_recipe()

        template_fixture = root / "template_fixture"
        template_fixture.mkdir(parents=True, exist_ok=True)
        template = compiler_fixtures._make_recipe(template_fixture)
        self.recipe["shared_ready"]["canonical_ready"] = self._binding(
            template.ready.path
        )
        self.write_recipe()
        monkeypatch.setattr(
            arbitrary,
            "load_canonical_motion_recipe",
            lambda *args, **kwargs: template,
        )

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _binding(self, path: Path) -> dict[str, str]:
        return {"path": self._relative(path), "sha256": _sha(path)}

    def write_capsule(self) -> None:
        _write_json(self.capsule_path, self.capsule)
        self.recipe["source_capsule"] = self._binding(self.capsule_path)
        self.write_recipe()

    def write_recipe(self) -> None:
        _write_json(self.recipe_path, self.recipe)

    def load(self) -> arbitrary.LoadedArbitraryRecipe:
        return arbitrary.load_arbitrary_bank_recipe(
            self.recipe_path,
            repo_root=self.root,
        )

    def recipe_builder_kwargs(self, output: Path) -> dict:
        return {
            "repo_root": self.root,
            "bank_id": f"built_arbitrary_{self.count}",
            "source_capsule_path": self.capsule_path,
            "expected_source_capsule_sha256": _sha(self.capsule_path),
            "compiler_template_path": self.template_path,
            "expected_compiler_template_sha256": _sha(self.template_path),
            "source_hold_motion_path": self.ready_path,
            "expected_source_hold_motion_sha256": _sha(self.ready_path),
            "source_hold_frame": 1,
            "hold_tolerances": {
                "joint_position_rad": 1.0e-6,
                "root_position_m": 1.0e-6,
                "root_orientation_rad": 1.0e-6,
                "joint_velocity_rad_s": 1.0e-6,
                "body_linear_velocity_m_s": 1.0e-6,
                "body_angular_velocity_rad_s": 1.0e-6,
            },
            "acceleration_receipt_path": self.acceleration_path,
            "expected_acceleration_receipt_sha256": _sha(
                self.acceleration_path
            ),
            "marker_half_width_frames": 1,
            "minimum_source_preparation_frames": 4,
            "minimum_source_recovery_frames": 4,
            "minimum_compiled_recovery_s": 0.04,
            "full_root_position_lower": [0, -0.4, 0.85, -1, -1, -1],
            "full_root_position_upper": [0.4, 0.1, 1.05, 1, 1, 1],
            "full_root_velocity": [1, 1, 0.5, 2, 2, 2],
            "full_root_acceleration": [10, 10, 5, 20, 20, 20],
            "samples_per_scaled_unit": 6,
            "min_connector_intervals": 5,
            "min_core_intervals": 5,
            "grid_subdivisions": 4,
            "search_workers": 1,
            "search_parallel_backend": "thread",
            "output_path": output,
        }


@pytest.mark.parametrize("count", [1, 5, 73])
def test_n1_n5_n73_dry_fixture_binds_exact_order_and_pair_count(
    tmp_path: Path,
    monkeypatch,
    count: int,
):
    fixture = BankFixture(tmp_path, count, monkeypatch)

    loaded = fixture.load()
    receipt = arbitrary.dry_run_receipt(loaded)

    assert loaded.motion_ids == fixture.motion_ids
    assert receipt["ordered_motion_ids"] == list(fixture.motion_ids)
    assert receipt["required_output_matrix"]["candidate_count"] == 2 * count
    assert receipt["authorization"] == {
        "compiler_outputs_present": False,
        "bank_gate_pass": False,
        "training_authorized": False,
        "deployment_authorized": False,
        "hardware_authorized": False,
    }
    assert not any(tmp_path.glob("*canonical_v2.npz"))


def test_swept_clearance_view_preserves_first_middle_last_and_exact_template(
    tmp_path: Path,
    monkeypatch,
):
    fixture = BankFixture(tmp_path, 5, monkeypatch)
    loaded = fixture.load()

    view = arbitrary.swept_clearance_recipe_contract(loaded)

    assert view["required_output_matrix"] == {
        "motion_ids": list(fixture.motion_ids),
        "scopes": ["upper", "full"],
        "candidate_count": 10,
    }
    assert view["required_output_matrix"]["motion_ids"][0] == fixture.motion_ids[0]
    assert view["required_output_matrix"]["motion_ids"][2] == fixture.motion_ids[2]
    assert view["required_output_matrix"]["motion_ids"][-1] == fixture.motion_ids[-1]
    assert view["canonical_ready"] == {
        "path": str(loaded.canonical_recipe.ready.path.resolve()),
        "sha256": loaded.canonical_recipe.ready.sha256,
    }
    for name in ("mjcf", "urdf", "body_order"):
        assert view["model_contract"][f"{name}_path"] == str(
            loaded.canonical_recipe.model_paths[name].resolve()
        )
        assert view["model_contract"][f"{name}_sha256"] == (
            loaded.canonical_recipe.model_hashes[name]
        )


def test_swept_clearance_view_rejects_embedded_order_and_model_byte_drift(
    tmp_path: Path,
    monkeypatch,
):
    fixture = BankFixture(tmp_path / "order", 3, monkeypatch)
    loaded = fixture.load()
    loaded.canonical_recipe.raw["required_output_matrix"][
        "motion_ids"
    ].reverse()
    with pytest.raises(
        arbitrary.ArbitraryBankError,
        match="changed arbitrary-N order",
    ):
        arbitrary.swept_clearance_recipe_contract(loaded)

    fixture = BankFixture(tmp_path / "model", 3, monkeypatch)
    loaded = fixture.load()
    mjcf = loaded.canonical_recipe.model_paths["mjcf"]
    mjcf.write_bytes(mjcf.read_bytes() + b"\n<!-- drift -->\n")
    with pytest.raises(
        arbitrary.ArbitraryBankError,
        match="mjcf bytes drifted",
    ):
        arbitrary.swept_clearance_recipe_contract(loaded)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda fixture: fixture.capsule["actions"].pop(),
            "action count differs",
        ),
        (
            lambda fixture: fixture.capsule["actions"].reverse(),
            "changed ordered identity",
        ),
        (
            lambda fixture: fixture.capsule["actions"][0].__setitem__(
                "reference_t_hit_s", 0.09
            ),
            "t_hit/t_cycle do not bind",
        ),
    ],
)
def test_capsule_completeness_order_and_timing_fail_closed(
    tmp_path: Path,
    monkeypatch,
    mutation,
    message: str,
):
    fixture = BankFixture(tmp_path, 5, monkeypatch)
    mutation(fixture)
    fixture.write_capsule()

    with pytest.raises(arbitrary.ArbitraryBankError, match=message):
        fixture.load()


def test_source_sha_drift_and_ready_claim_fail_closed(
    tmp_path: Path,
    monkeypatch,
):
    fixture = BankFixture(tmp_path, 1, monkeypatch)
    first_motion = fixture.capsule_directory / fixture.capsule["actions"][0][
        "motion_path"
    ]
    first_motion.write_bytes(first_motion.read_bytes() + b"drift")

    with pytest.raises(arbitrary.ArbitraryBankError, match="bytes drifted"):
        fixture.load()

    fixture = BankFixture(tmp_path / "ready", 1, monkeypatch)
    fixture.recipe["shared_ready"]["evidence_status"] = (
        "PASS_GROUNDED_READY"
    )
    fixture.write_recipe()
    with pytest.raises(
        arbitrary.ArbitraryBankError,
        match="may not claim grounded admission",
    ):
        fixture.load()


def test_station_swap_is_rejected_and_base_local_frame_is_se2_equivariant(
    tmp_path: Path,
    monkeypatch,
):
    fixture = BankFixture(tmp_path, 2, monkeypatch)
    first, second = fixture.capsule["actions"]
    first["base_spawn_center_w_xy_m"] = list(
        second["base_spawn_center_w_xy_m"]
    )
    fixture.write_capsule()
    with pytest.raises(
        arbitrary.ArbitraryBankError,
        match="station/base_spawn mapping changed",
    ):
        fixture.load()

    base = np.asarray([0.2, -0.3])
    local = np.asarray([0.45, 0.12])
    yaw = 0.47
    world = arbitrary.base_yaw_local_xy_to_world(local, base, yaw)
    shift = np.asarray([-0.8, 0.6])
    extra_yaw = -0.31
    cosine, sine = np.cos(extra_yaw), np.sin(extra_yaw)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    transformed_base = shift + rotation @ base
    transformed_world = shift + rotation @ world
    recovered = arbitrary.world_xy_to_base_yaw_local(
        transformed_world,
        transformed_base,
        yaw + extra_yaw,
    )
    assert np.allclose(recovered, local, rtol=0.0, atol=1.0e-12)


def test_compile_emits_complete_pair_and_refuses_clobber(
    tmp_path: Path,
    monkeypatch,
):
    fixture = BankFixture(tmp_path, 1, monkeypatch)
    loaded = fixture.load()
    monkeypatch.setattr(
        compiler,
        "build_schema2_candidate",
        compiler_fixtures._fake_schema2_builder,
    )
    monkeypatch.setattr(
        compiler,
        "build_canonical_geometry",
        compiler_fixtures._compiler_plumbing_geometry,
    )
    monkeypatch.setattr(
        compiler,
        "retime_path",
        compiler_fixtures._fast_marker_only_retime,
    )
    output = tmp_path / "compiled"

    arbitrary.compile_arbitrary_bank(
        loaded,
        output_directory=output,
        backend=compiler_fixtures.FakePlantBackend(),
    )

    manifest = json.loads(
        (output / compiler.BUILD_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    arbitrary.validate_arbitrary_build_manifest(manifest, loaded)
    assert [
        (row["motion_id"], row["scope"]) for row in manifest["outputs"]
    ] == [
        (fixture.motion_ids[0], "upper"),
        (fixture.motion_ids[0], "full"),
    ]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        arbitrary.compile_arbitrary_bank(
            loaded,
            output_directory=output,
            backend=compiler_fixtures.FakePlantBackend(),
        )


def test_recipe_builder_derives_exact_order_and_refuses_clobber(
    tmp_path: Path,
    monkeypatch,
):
    fixture = BankFixture(tmp_path, 5, monkeypatch)
    output = tmp_path / "built_recipe.json"

    published, document, digest = recipe_builder.materialize_recipe(
        **fixture.recipe_builder_kwargs(output)
    )

    assert published == output
    assert _sha(output) == digest
    assert document["ordered_motion_ids"] == list(fixture.motion_ids)
    assert document["required_output_matrix"] == {
        "motion_ids": list(fixture.motion_ids),
        "scopes": ["upper", "full"],
        "candidate_count": 10,
    }
    assert document["placement_contract"] == dict(
        arbitrary._PLACEMENT_CONTRACT
    )
    loaded = arbitrary.load_arbitrary_bank_recipe(
        output, repo_root=tmp_path
    )
    assert loaded.motion_ids == fixture.motion_ids

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        recipe_builder.materialize_recipe(
            **fixture.recipe_builder_kwargs(output)
        )
    assert _sha(output) == digest


def test_recipe_builder_lost_publish_race_never_deletes_winner(
    tmp_path: Path,
    monkeypatch,
):
    fixture = BankFixture(tmp_path, 1, monkeypatch)
    output = tmp_path / "race_recipe.json"
    winner = b"concurrent-winner\n"

    def lose_publish_race(_temporary, destination):
        Path(destination).write_bytes(winner)
        raise FileExistsError("fixture concurrent publisher won")

    monkeypatch.setattr(recipe_builder.os, "link", lose_publish_race)

    with pytest.raises(FileExistsError, match="concurrent publisher"):
        recipe_builder.materialize_recipe(
            **fixture.recipe_builder_kwargs(output)
        )
    assert output.read_bytes() == winner
