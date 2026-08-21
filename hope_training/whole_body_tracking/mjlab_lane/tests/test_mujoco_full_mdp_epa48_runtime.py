from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
import types

import pytest


LANE = Path(__file__).resolve().parents[1]


def _load():
    path = LANE / "mujoco_full_mdp_epa48_runtime.py"
    spec = importlib.util.spec_from_file_location(
        "mujoco_full_mdp_epa48_runtime_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_assets(module):
    paths = (module.BUILD_RECEIPT_PATH, module.EPA48_WHEEL_PATH,
             module.RSL3_WHEEL_PATH)
    missing = [path for path in paths if not path.exists() and not path.is_symlink()]
    if missing:
        pytest.skip(
            "ignored EPA48/RSL3 assets are not restored: "
            + ", ".join(str(path) for path in missing)
        )


def _wheel_payloads(module):
    _require_assets(module)
    return module.RSL3_WHEEL_PATH.read_bytes(), module.EPA48_WHEEL_PATH.read_bytes()


def _fake_modules(monkeypatch, module, site, *, version=None, horizon=48):
    package = types.ModuleType("mujoco_warp")
    package.__file__ = str(site / "mujoco_warp" / "__init__.py")
    package.__version__ = module.EPA48_VERSION if version is None else version
    types_module = types.ModuleType("mujoco_warp._src.types")
    types_module.__file__ = str(site / "mujoco_warp" / "_src" / "types.py")
    types_module.MJ_MAX_EPAHORIZON = horizon
    monkeypatch.setitem(sys.modules, "mujoco_warp", package)
    monkeypatch.setitem(sys.modules, "mujoco_warp._src.types", types_module)
    return package, types_module


def _cleanup(site):
    while str(site) in sys.path:
        sys.path.remove(str(site))
    for name in tuple(sys.modules):
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in ("mujoco_warp", "rsl_rl")
        ):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def test_runtime_identity_is_exact_epa_evidence_not_authority():
    module = _load()
    identity = module.expected_mujoco_warp_runtime_identity()
    assert set(identity) == {
        "schema_version", "distribution", "fork_id", "version", "epa_horizon",
        "types_py_sha256", "wheel_sha256", "build_receipt_sha256", "import_scope",
    }
    assert identity["version"] == "3.10.0.3+hope.epa48.1"
    assert identity["epa_horizon"] == 48
    assert identity["types_py_sha256"] == (
        "391e421eeede84389d6c7daeae39b19ce43132d29c11f7f3c328a50011c7a696"
    )
    assert identity["wheel_sha256"] == (
        "58f47b1c3b4249d82666f25d3a302ff5a215043a3d7a3b9445a5ca7ef15b561a"
    )
    assert identity["build_receipt_sha256"] == (
        "336f6454296d3c062e26fb0c330d6dbca4b2fd0ad4e50f386f8a647db013e041"
    )
def test_real_offline_dual_wheels_bind_one_fresh_site(monkeypatch, tmp_path):
    module = _load()
    _require_assets(module)
    site = tmp_path / "runtime_site"
    monkeypatch.setattr(
        module, "_import_runtime_modules",
        lambda actual: _fake_modules(monkeypatch, module, actual),
    )
    try:
        assert module.bind_fresh_epa48_runtime_site(site) == (
            module.expected_mujoco_warp_runtime_identity()
        )
        assert hashlib.sha256(
            (site / "mujoco_warp" / "_src" / "types.py").read_bytes()
        ).hexdigest() == module.EPA48_TYPES_SHA256
        assert (site / "rsl_rl" / "runners" / "on_policy_runner.py").is_file()
    finally:
        _cleanup(site)


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None
        for name in ("mujoco", "warp", "torch", "tensordict")),
    reason="actual EPA48/RSL3 import requires the isolated MJLab runtime",
)
def test_real_site_imports_epa48_and_rsl3_from_exact_origins(tmp_path):
    module = _load()
    _require_assets(module)
    site = tmp_path / "real_import_site"
    try:
        module.bind_fresh_epa48_runtime_site(site)
        imported = [
            importlib.import_module("mujoco_warp"),
            importlib.import_module("mujoco_warp._src.types"),
            importlib.import_module("rsl_rl"),
            importlib.import_module("rsl_rl.runners.on_policy_runner"),
        ]
        for value in imported:
            Path(value.__file__).resolve(strict=True).relative_to(site)
        assert imported[0].__version__ == module.EPA48_VERSION
        assert imported[1].MJ_MAX_EPAHORIZON == 48
        distribution = module.importlib.metadata.distribution("rsl-rl-lib")
        assert distribution.version == module.RSL3_VERSION
        assert Path(distribution.locate_file("")).resolve(strict=True) == site
    finally:
        _cleanup(site)


@pytest.mark.parametrize(
    "module_name", ("mujoco_warp._src.types", "rsl_rl.runners", "mjlab")
)
def test_preloaded_runtime_fails_before_asset_reads(monkeypatch, tmp_path, module_name):
    module = _load()
    monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))
    monkeypatch.setattr(
        module, "_verified_wheel_payloads",
        lambda: pytest.fail("preloaded runtime must fail before asset reads"),
    )
    with pytest.raises(module.RuntimeBindingError, match="preloaded"):
        module.bind_fresh_epa48_runtime_site(tmp_path / "site")


def test_fresh_site_rejects_invalid_or_aliased_targets(monkeypatch, tmp_path):
    module = _load()
    monkeypatch.setattr(
        module, "_verified_wheel_payloads",
        lambda: pytest.fail("invalid site must fail before asset reads"),
    )
    with pytest.raises(module.RuntimeBindingError, match="absolute fresh"):
        module.bind_fresh_epa48_runtime_site(Path("relative-site"))
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(module.RuntimeBindingError, match="already exists"):
        module.bind_fresh_epa48_runtime_site(existing)
    with pytest.raises(module.RuntimeBindingError, match="parent is missing"):
        module.bind_fresh_epa48_runtime_site(tmp_path / "missing" / "site")
    broken = tmp_path / "broken"
    broken.symlink_to(tmp_path / "absent", target_is_directory=True)
    with pytest.raises(module.RuntimeBindingError, match="symlink"):
        module.bind_fresh_epa48_runtime_site(broken)
    listed = tmp_path / "listed"
    monkeypatch.syspath_prepend(str(listed))
    with pytest.raises(module.RuntimeBindingError, match="already present"):
        module.bind_fresh_epa48_runtime_site(listed)


def test_stable_asset_read_rejects_missing_and_symlink(tmp_path):
    module = _load()
    with pytest.raises(module.RuntimeBindingError, match="missing"):
        module._stable_regular_bytes(tmp_path / "missing.whl", "wheel")
    target = tmp_path / "target.whl"
    target.write_bytes(b"wheel")
    alias = tmp_path / "alias.whl"
    alias.symlink_to(target)
    with pytest.raises(module.RuntimeBindingError, match="missing|regular"):
        module._stable_regular_bytes(alias, "wheel")


@pytest.mark.parametrize(
    "changed,error",
    (("receipt", "receipt SHA"), ("epa", "EPA48 wheel SHA"),
     ("rsl", "RSL-RL 3 wheel SHA")),
)
def test_each_fixed_asset_hash_is_enforced(monkeypatch, changed, error):
    module = _load()
    payloads = {
        "EPA48 build receipt": b"receipt",
        "EPA48 wheel": b"epa",
        "RSL-RL 3 wheel": b"rsl",
    }
    monkeypatch.setattr(
        module, "_stable_regular_bytes", lambda _path, label: payloads[label]
    )
    for name, label in (
        ("BUILD_RECEIPT_SHA256", "EPA48 build receipt"),
        ("EPA48_WHEEL_SHA256", "EPA48 wheel"),
        ("RSL3_WHEEL_SHA256", "RSL-RL 3 wheel"),
    ):
        monkeypatch.setattr(module, name, hashlib.sha256(payloads[label]).hexdigest())
    monkeypatch.setattr(
        module,
        {"receipt": "BUILD_RECEIPT_SHA256", "epa": "EPA48_WHEEL_SHA256",
         "rsl": "RSL3_WHEEL_SHA256"}[changed],
        "0" * 64,
    )
    with pytest.raises(module.RuntimeBindingError, match=error):
        module._verified_wheel_payloads()


@pytest.mark.parametrize(
    "kind,target",
    (("distribution", "mujoco-warp"), ("distribution", "rsl-rl-lib"),
     ("spec", "mujoco_warp"), ("spec", "rsl_rl")),
)
def test_foreign_distribution_or_import_winner_is_rejected(
    monkeypatch, tmp_path, kind, target
):
    module = _load()
    site = tmp_path / "site"
    monkeypatch.setattr(module, "_verified_wheel_payloads", lambda: _wheel_payloads(module))
    if kind == "distribution":
        original = module.importlib.metadata.distribution
        foreign = types.SimpleNamespace(
            version=(module.EPA48_VERSION if target == "mujoco-warp" else module.RSL3_VERSION),
            locate_file=lambda _name: Path("/foreign/site"),
        )
        monkeypatch.setattr(
            module.importlib.metadata, "distribution",
            lambda name: foreign if name == target else original(name),
        )
        error = "root differs|origin/version differs"
    else:
        original = module.importlib.util.find_spec
        monkeypatch.setattr(
            module.importlib.util, "find_spec",
            lambda name: (types.SimpleNamespace(origin=f"/foreign/{name}.py")
                          if name == target else original(name)),
        )
        error = "candidate is foreign"
    with pytest.raises(module.RuntimeBindingError, match=error):
        module.bind_fresh_epa48_runtime_site(site)
    assert str(site) not in sys.path


@pytest.mark.parametrize(
    "version,horizon,change_file",
    (("3.10.0.3", 48, False), ("3.10.0.3+hope.epa48.1", 24, False),
     ("3.10.0.3+hope.epa48.1", 48, True)),
)
def test_loaded_version_horizon_and_types_sha_are_enforced(
    monkeypatch, tmp_path, version, horizon, change_file
):
    module = _load()
    site = tmp_path / "site"
    monkeypatch.setattr(module, "_verified_wheel_payloads", lambda: _wheel_payloads(module))

    def fake_import(actual):
        if change_file:
            with (actual / "mujoco_warp" / "_src" / "types.py").open("ab") as stream:
                stream.write(b"\n# changed during import\n")
        return _fake_modules(
            monkeypatch, module, actual, version=version, horizon=horizon
        )

    monkeypatch.setattr(module, "_import_runtime_modules", fake_import)
    with pytest.raises(module.RuntimeBindingError, match="version/horizon/origin"):
        module.bind_fresh_epa48_runtime_site(site)
    assert str(site) not in sys.path


def test_failed_import_restores_path_and_cleans_partial_prefixes(monkeypatch, tmp_path):
    module = _load()
    site = tmp_path / "site"
    monkeypatch.setattr(module, "_verified_wheel_payloads", lambda: (b"rsl", b"epa"))
    monkeypatch.setattr(module, "_extract_exact_wheel", lambda *_: None)
    monkeypatch.setattr(module, "_require_site_candidates", lambda *_: None)
    path_before = list(sys.path)

    def fail(_site):
        for name in ("mujoco_warp.partial", "rsl_rl.partial", "mjlab.partial"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        sys.path.append("/fault-injected/runtime-path")
        raise ModuleNotFoundError("fault-injected dependency")

    monkeypatch.setattr(module, "_import_runtime_modules", fail)
    with pytest.raises(ModuleNotFoundError, match="fault-injected"):
        module.bind_fresh_epa48_runtime_site(site)
    assert sys.path == path_before
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in sys.modules for prefix in module._PRELOAD_PREFIXES
    )
