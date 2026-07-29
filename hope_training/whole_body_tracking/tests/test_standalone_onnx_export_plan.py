"""Focused zero-write tests for standalone ONNX export plan mode."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from contextlib import contextmanager

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/standalone_onnx_export.py"


class _FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data)

    def is_floating_point(self):
        return np.issubdtype(self.data.dtype, np.floating)

    def is_complex(self):
        return np.issubdtype(self.data.dtype, np.complexfloating)

    @property
    def shape(self):
        return self.data.shape


class _FakeFiniteResult:
    def __init__(self, value):
        self.value = bool(value)

    def all(self):
        return self

    def item(self):
        return self.value


class _FakeModule:
    def eval(self):
        return self

    def state_dict(self):
        return {}


class _FakeIdentity(_FakeModule):
    def __call__(self, value):
        return value


class _FakeActor(_FakeModule):
    def __init__(self, state):
        self._state = state
        self._layers = [SimpleNamespace(in_features=179), SimpleNamespace(out_features=31)]

    def __getitem__(self, index):
        return self._layers[index]

    def state_dict(self):
        return self._state


def _load_exporter(monkeypatch):
    fake_nn = ModuleType("torch.nn")
    fake_nn.Module = _FakeModule
    fake_nn.Identity = _FakeIdentity
    fake_torch = ModuleType("torch")
    fake_torch.nn = fake_nn
    fake_torch.float32 = "float32"
    fake_torch.load = lambda *args, **kwargs: None
    fake_torch.is_tensor = lambda value: isinstance(value, _FakeTensor)
    fake_torch.isfinite = lambda value: _FakeFiniteResult(np.isfinite(value.data).all())
    fake_torch.as_tensor = lambda value, dtype=None: _FakeTensor(value)
    fake_torch.zeros = lambda *shape: _FakeTensor(np.zeros(shape, dtype=np.float32))
    fake_torch.onnx = SimpleNamespace(export=lambda *args, **kwargs: None)
    fake_onnx = ModuleType("onnx")
    fake_onnx.load = lambda _path: None
    fake_onnx.save = lambda _model, _path: None
    fake_onnx.checker = SimpleNamespace(check_model=lambda _model: None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", fake_nn)
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)
    old_argv = sys.argv
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.argv = [str(SCRIPT), "--plan"]
    spec = importlib.util.spec_from_file_location("standalone_onnx_export_plan_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = old_argv
        sys.dont_write_bytecode = old_dont_write_bytecode
    return module


def _value_info(name: str, width: int):
    return SimpleNamespace(
        name=name,
        type=SimpleNamespace(
            tensor_type=SimpleNamespace(
                shape=SimpleNamespace(
                    dim=[SimpleNamespace(dim_value=1), SimpleNamespace(dim_value=width)]
                )
            )
        ),
    )


def _donor(*, clip_hashes: tuple[str, str]):
    metadata = {
        "hope_metadata_schema_version": "2",
        "joint_names": "joint",
        "joint_stiffness": "1",
        "clip_seg_lengths": "2,3",
        "clip_strike_phases": "0.5,0.5",
        "motion_clip_sha256": ",".join(clip_hashes),
        "actor_obs_contract": "legacy",
        "actor_obs_total_dim": "179",
        "actor_obs_term_dims": "179",
        "observation_names": "obs",
    }
    return SimpleNamespace(
        metadata_props=[SimpleNamespace(key=key, value=value) for key, value in metadata.items()],
        graph=SimpleNamespace(
            node=[],
            input=[_value_info("obs", 179)],
            output=[_value_info("actions", 31)],
        ),
    )


def _tree_snapshot(root: Path):
    if not root.exists():
        return None
    return {
        str(path.relative_to(root)): (
            "dir" if path.is_dir() else path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in sorted([root, *root.rglob("*")])
    }


def _install_valid_inputs(
    monkeypatch,
    exporter,
    *,
    bad_harvest: bool = False,
    checkpoint_iteration=6700,
    nonfinite_actor: bool = False,
    nonfinite_obs_norm: bool = False,
):
    hashes = {
        "checkpoint.pt": "c" * 64,
        "donor.onnx": "d" * 64,
        "fh.npz": "f" * 64,
        "bh.npz": "b" * 64,
    }
    checkpoint = {
        "iter": checkpoint_iteration,
        "model_state_dict": {
            "actor.0.weight": _FakeTensor(np.zeros((31, 179), dtype=np.float32)),
            "actor.0.bias": _FakeTensor(np.zeros(31, dtype=np.float32)),
        },
        "obs_norm_state_dict": {
            "mean": _FakeTensor(np.zeros(179, dtype=np.float32)),
            "std": _FakeTensor(np.ones(179, dtype=np.float32)),
            "count": _FakeTensor(np.asarray(1, dtype=np.int64)),
        },
    }
    if nonfinite_actor:
        checkpoint["model_state_dict"]["actor.0.weight"].data[0, 0] = np.inf
    if nonfinite_obs_norm:
        checkpoint["obs_norm_state_dict"]["mean"].data[0] = np.nan
    load_calls = []

    def fake_torch_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        return checkpoint

    monkeypatch.setattr(exporter.torch, "load", fake_torch_load)
    monkeypatch.setattr(
        exporter.onnx,
        "load",
        lambda _path: _donor(clip_hashes=(hashes["fh.npz"], hashes["bh.npz"])),
    )
    monkeypatch.setattr(exporter.onnx.checker, "check_model", lambda _model: None)
    monkeypatch.setattr(exporter, "_donor_activation", lambda _donor: (_FakeIdentity(), 0))
    monkeypatch.setattr(
        exporter,
        "build_actor",
        lambda msd, _activation, expected_activations: _FakeActor(msd),
    )
    monkeypatch.setattr(exporter, "_sha256_file", lambda path: hashes[Path(path).name])

    frames = 4 if bad_harvest else 5
    harvest = {
        "total": frames,
        "donor_sha256": np.asarray(hashes["donor.onnx"]),
        "donor_obs_dim": 179,
        **{
            key: np.zeros((frames, 1), dtype=np.float32)
            for key in exporter.MOTION_KEYS
        },
    }

    def fake_np_load(path):
        name = Path(path).name
        if name == "fh.npz":
            return {"joint_pos": np.zeros((2, 1), dtype=np.float32)}
        if name == "bh.npz":
            return {"joint_pos": np.zeros((3, 1), dtype=np.float32)}
        if name == "harvest.npz":
            return harvest
        raise AssertionError(f"unexpected np.load path: {path}")

    monkeypatch.setattr(exporter.np, "load", fake_np_load)
    return load_calls


def _argv(out: Path):
    return [
        str(SCRIPT),
        "--ckpt", "checkpoint.pt",
        "--fh", "fh.npz",
        "--bh", "bh.npz",
        "--donor", "donor.onnx",
        "--harvest", "harvest.npz",
        "--out", str(out),
        "--plan",
    ]


def _normal_argv(out: Path):
    return _argv(out)[:-1]


@pytest.mark.parametrize("preexisting", [False, True])
def test_plan_success_is_zero_write_for_missing_or_existing_output(
    tmp_path, monkeypatch, capsys, preexisting
):
    exporter = _load_exporter(monkeypatch)
    out = tmp_path / "exported"
    if preexisting:
        out.mkdir()
        (out / "keep.bin").write_bytes(b"unchanged")
        (out / "policy.onnx").write_bytes(b"existing-policy")
    before = _tree_snapshot(tmp_path)
    load_calls = _install_valid_inputs(monkeypatch, exporter)
    monkeypatch.setattr(sys, "argv", _argv(out))

    assert exporter.main() == 0

    assert _tree_snapshot(tmp_path) == before
    assert load_calls[0][1]["weights_only"] is True
    plan = json.loads(capsys.readouterr().out)
    assert plan == {
        "action_ball_diagnostic_unauthorized": False,
        "artifact_written": False,
        "checkpoint_iteration": 6700,
        "graph_export_not_executed": True,
        "input_dim": 179,
        "materials_validated": True,
        "obs_norm_baked": False,
        "output_dim": 31,
        "plan": True,
        "formal_face179_materials_validated": False,
        "train_bank_validated": False,
        "training_contract_present": False,
        "training_contract_schema": None,
        "would_write": [str((out / "policy.onnx").resolve())],
    }


def test_plan_validation_failure_is_zero_write(tmp_path, monkeypatch):
    exporter = _load_exporter(monkeypatch)
    out = tmp_path / "exported"
    out.mkdir()
    (out / "keep.bin").write_bytes(b"unchanged")
    before = _tree_snapshot(tmp_path)
    _install_valid_inputs(monkeypatch, exporter, bad_harvest=True)
    monkeypatch.setattr(sys, "argv", _argv(out))

    with pytest.raises(SystemExit, match="harvest length"):
        exporter.main()

    assert _tree_snapshot(tmp_path) == before


@pytest.mark.parametrize("checkpoint_iteration", [6700.5, "6700", True])
def test_plan_rejects_non_integer_checkpoint_iteration_without_writing(
    tmp_path, monkeypatch, checkpoint_iteration
):
    exporter = _load_exporter(monkeypatch)
    out = tmp_path / "exported"
    before = _tree_snapshot(tmp_path)
    _install_valid_inputs(
        monkeypatch,
        exporter,
        checkpoint_iteration=checkpoint_iteration,
    )
    monkeypatch.setattr(sys, "argv", _argv(out))

    with pytest.raises(SystemExit, match="integer checkpoint iteration"):
        exporter.main()

    assert _tree_snapshot(tmp_path) == before


@pytest.mark.parametrize("kind", ["actor", "obs_norm"])
def test_plan_rejects_nonfinite_checkpoint_material_without_writing(
    tmp_path, monkeypatch, kind
):
    exporter = _load_exporter(monkeypatch)
    out = tmp_path / "exported"
    before = _tree_snapshot(tmp_path)
    _install_valid_inputs(
        monkeypatch,
        exporter,
        nonfinite_actor=kind == "actor",
        nonfinite_obs_norm=kind == "obs_norm",
    )
    monkeypatch.setattr(sys, "argv", _argv(out))

    with pytest.raises(SystemExit, match="NaN/Inf"):
        exporter.main()

    assert _tree_snapshot(tmp_path) == before


def test_normal_export_still_uses_non_weights_only_atomic_graph_path(
    tmp_path, monkeypatch, capsys
):
    exporter = _load_exporter(monkeypatch)
    out = tmp_path / "exported"
    load_calls = _install_valid_inputs(monkeypatch, exporter)
    events = []
    saved_models = {}
    donor = _donor(clip_hashes=("f" * 64, "b" * 64))
    donor.metadata_props.extend(
        [
            SimpleNamespace(
                key="action_ball_diagnostic_unauthorized",
                value="stale-donor-brand",
            ),
            SimpleNamespace(
                key="formal_evidence_bookable",
                value="1",
            ),
        ]
    )

    class MetadataProps(list):
        def add(self):
            value = SimpleNamespace(key="", value="")
            self.append(value)
            return value

    exported_model = SimpleNamespace(metadata_props=MetadataProps())

    def fake_onnx_load(path):
        if Path(path).name == "donor.onnx":
            return donor
        return saved_models.get(str(path), exported_model)

    def fake_onnx_save(model, path):
        events.append("save")
        saved_models[str(path)] = model
        Path(path).write_bytes(b"fake-onnx")

    def fake_graph_export(_module, _inputs, path, **_kwargs):
        events.append("export")
        Path(path).write_bytes(b"fake-graph")

    @contextmanager
    def fake_atomic(final_path):
        events.append("atomic")
        temporary = Path(f"{final_path}.tmp")
        try:
            yield temporary
            temporary.replace(final_path)
        finally:
            temporary.unlink(missing_ok=True)

    monkeypatch.setattr(exporter.onnx, "load", fake_onnx_load)
    monkeypatch.setattr(exporter.onnx, "save", fake_onnx_save)
    monkeypatch.setattr(exporter.torch.onnx, "export", fake_graph_export)
    monkeypatch.setattr(exporter, "atomic_output_path", fake_atomic)
    monkeypatch.setattr(sys, "argv", _normal_argv(out))

    assert exporter.main() == 0

    assert load_calls[0][1]["weights_only"] is False
    assert events[:2] == ["atomic", "export"]
    assert "save" in events
    assert (out / "policy.onnx").read_bytes() == b"fake-onnx"
    final_metadata = {
        entry.key: entry.value for entry in exported_model.metadata_props
    }
    assert "action_ball_diagnostic_unauthorized" not in final_metadata
    assert "formal_evidence_bookable" not in final_metadata
    assert "[standalone-export] SUCCESS" in capsys.readouterr().out


def test_plan_branch_follows_formal_material_checks_and_precedes_all_writes():
    main_source = SCRIPT.read_text(encoding="utf-8").split("def main() -> int:", 1)[1]
    plan_payload = main_source.index('"artifact_written": False')
    plan = main_source.rindex("if args.plan:", 0, plan_payload)
    for validation in (
        "require_checkpoint_contract_binding(",
        "_bind_schema3_donor_metadata(",
        "validate_runtime_motion_contract(",
        "derive_stage1_normal_envelope(",
        'donor_meta["motion_harvest_donor_sha256"]',
    ):
        assert main_source.index(validation) < plan
    for write in (
        "os.makedirs(args.out",
        "with atomic_output_path(out_path)",
        "torch.onnx.export(",
        "onnx.save(",
    ):
        assert plan < main_source.index(write)
    full_source = SCRIPT.read_text(encoding="utf-8")
    assert full_source.index("sys.dont_write_bytecode = True") < full_source.index(
        '_TC = _load_light_module('
    )
    assert 'weights_only=args.plan' in main_source
    assert "onnx.checker.check_model(donor)" in main_source
    assert main_source.index("_require_module_finite(actor") < plan
    assert main_source.index("_require_module_finite(normalizer") < plan
