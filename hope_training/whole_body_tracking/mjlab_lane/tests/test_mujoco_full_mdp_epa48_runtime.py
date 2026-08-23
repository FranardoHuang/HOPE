from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
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


def _preverified(module, rsl3_wheel: bytes, epa48_wheel: bytes):
    return module._VerifiedRuntimeStackPreimport(
        module.expected_mjlab_runtime_identity(), rsl3_wheel, epa48_wheel
    )


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


def _make_mjlab_tree(tmp_path):
    site = tmp_path / "site-packages"
    files = {
        "mjlab/__init__.py": b'__version__ = "1.5.3"\n',
        "mjlab/core.py": b"VALUE = 1\n",
        "mjlab/nested/tool.py": b"def tool():\n    return 1\n",
        "mjlab/scene/__init__.py": b"",
        "mjlab/scene/scene.xml": b"<mujoco model='scene'/>\n",
    }
    for relative, payload in files.items():
        path = site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return site


def _selected_tree_measure(site):
    package = site / "mjlab"
    paths = sorted(
        (
            path for path in package.rglob("*")
            if path.is_file()
            and (path.suffix == ".py" or path == package / "scene" / "scene.xml")
        ),
        key=lambda path: path.relative_to(site).as_posix(),
    )
    items = []
    for path in paths:
        payload = path.read_bytes()
        items.append([
            path.relative_to(site).as_posix(), len(payload),
            hashlib.sha256(payload).hexdigest(),
        ])
    encoded = json.dumps(
        items, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return len(items), sum(item[1] for item in items), hashlib.sha256(encoded).hexdigest()


def _patch_expected_mjlab_tree(monkeypatch, module, site):
    count, byte_count, digest = _selected_tree_measure(site)
    monkeypatch.setattr(module, "MJLAB_SELECTED_FILE_COUNT", count)
    monkeypatch.setattr(module, "MJLAB_SELECTED_BYTE_COUNT", byte_count)
    monkeypatch.setattr(module, "MJLAB_SELECTED_TREE_SHA256", digest)


class _FakeEntryPoints(tuple):
    def select(self, **criteria):
        return type(self)(
            item for item in self
            if all(getattr(item, key, None) == value
                   for key, value in criteria.items())
        )


def _install_fake_mjlab_candidate(
    monkeypatch, module, site, *, version=None, spec_site=None, entry_points=()
):
    distribution = types.SimpleNamespace(
        version=module.MJLAB_VERSION if version is None else version,
        locate_file=lambda _name: site,
    )
    monkeypatch.setattr(
        module.importlib.metadata, "distribution", lambda name: distribution
    )
    winner = site if spec_site is None else spec_site
    spec = types.SimpleNamespace(
        origin=str(winner / "mjlab" / "__init__.py"),
        submodule_search_locations=[str(winner / "mjlab")],
    )
    monkeypatch.setattr(
        module.importlib.util, "find_spec", lambda name: spec if name == "mjlab" else None
    )
    monkeypatch.setattr(
        module.importlib.metadata, "entry_points",
        lambda: _FakeEntryPoints(entry_points),
    )
    return spec


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


def test_expected_runtime_stack_identity_is_exact_path_free_and_single_source():
    module = _load()
    identity = module.expected_runtime_stack_identity()
    assert set(identity) == {"schema_version", "mujoco_warp", "rsl_rl", "mjlab"}
    assert identity["schema_version"] == 1
    assert identity["mujoco_warp"] == module.expected_mujoco_warp_runtime_identity()
    assert identity["rsl_rl"] == {
        "distribution": "rsl-rl-lib",
        "version": "3.1.2",
        "wheel_sha256": (
            "406867356b70920e99ed8fd12c5b3463a64895407cc3ed96c917fddb9bfae06d"
        ),
        "import_scope": "fresh_run_local_site",
    }
    assert identity["mjlab"] == module.expected_mjlab_runtime_identity()
    assert "mujoco_warp_runtime" not in identity
    assert not any(value.startswith("/") for component in identity.values()
                   if isinstance(component, dict)
                   for value in component.values() if isinstance(value, str))


def test_expected_mjlab_identity_is_exact_and_path_free():
    module = _load()
    identity = module.expected_mjlab_runtime_identity()
    assert identity == {
        "schema_version": 1,
        "distribution": "mjlab",
        "version": "1.5.3",
        "import_scope": "verified_venv_distribution",
        "selected_tree_scope": "mjlab/**/*.py+mjlab/scene/scene.xml",
        "selected_file_count": 193,
        "selected_byte_count": 1_399_177,
        "selected_tree_sha256": (
            "88c9725d0416b4ac3e21f6752ad423c13ea3b8cfb9e23ca664f8aba146cec33d"
        ),
        "mjlab_tasks_entry_point_count": 0,
    }
    assert not any(value.startswith("/") for value in identity.values()
                   if isinstance(value, str))


def test_mjlab_preimport_tree_verifier_returns_path_free_identity(
    monkeypatch, tmp_path
):
    module = _load()
    site = _make_mjlab_tree(tmp_path)
    _patch_expected_mjlab_tree(monkeypatch, module, site)
    _install_fake_mjlab_candidate(monkeypatch, module, site)
    identity = module.verify_mjlab_runtime_preimport()
    assert identity == module.expected_mjlab_runtime_identity()
    assert str(site) not in json.dumps(identity, sort_keys=True)


def test_mjlab_preimport_rejects_an_already_loaded_package(monkeypatch, tmp_path):
    module = _load()
    monkeypatch.setitem(sys.modules, "mjlab", types.ModuleType("mjlab"))
    monkeypatch.setattr(
        module, "_canonical_mjlab_roots",
        lambda: pytest.fail("preloaded package must fail before candidate reads"),
    )
    with pytest.raises(module.RuntimeBindingError, match="already imported"):
        module.verify_mjlab_runtime_preimport()


@pytest.mark.parametrize(
    "mutation,error",
    (
        ("python", "SHA differs"),
        ("scene", "SHA differs"),
        ("add", "file count differs"),
        ("missing", "file count differs"),
        ("symlink", "contains a symlink"),
    ),
)
def test_mjlab_same_version_tree_mutations_are_rejected(
    monkeypatch, tmp_path, mutation, error
):
    module = _load()
    site = _make_mjlab_tree(tmp_path)
    _patch_expected_mjlab_tree(monkeypatch, module, site)
    _install_fake_mjlab_candidate(monkeypatch, module, site)
    if mutation == "python":
        (site / "mjlab" / "core.py").write_bytes(b"VALUE = 2\n")
    elif mutation == "scene":
        (site / "mjlab" / "scene" / "scene.xml").write_bytes(
            b"<mujoco model='other'/>\n"
        )
    elif mutation == "add":
        (site / "mjlab" / "extra.py").write_bytes(b"EXTRA = True\n")
    elif mutation == "missing":
        (site / "mjlab" / "nested" / "tool.py").unlink()
    else:
        target = tmp_path / "foreign.py"
        target.write_bytes(b"VALUE = 1\n")
        selected = site / "mjlab" / "core.py"
        selected.unlink()
        selected.symlink_to(target)
    with pytest.raises(module.RuntimeBindingError, match=error):
        module.verify_mjlab_runtime_preimport()


def test_mjlab_distribution_version_and_foreign_import_root_are_rejected(
    monkeypatch, tmp_path
):
    module = _load()
    site = _make_mjlab_tree(tmp_path)
    foreign = _make_mjlab_tree(tmp_path / "foreign")
    _patch_expected_mjlab_tree(monkeypatch, module, site)
    _install_fake_mjlab_candidate(
        monkeypatch, module, site, version="1.5.3+same-name"
    )
    with pytest.raises(module.RuntimeBindingError, match="version differs"):
        module.verify_mjlab_runtime_preimport()
    _install_fake_mjlab_candidate(monkeypatch, module, site, spec_site=foreign)
    with pytest.raises(module.RuntimeBindingError, match="origin differs"):
        module.verify_mjlab_runtime_preimport()


def test_mjlab_distribution_root_must_be_canonical(monkeypatch, tmp_path):
    module = _load()
    site = _make_mjlab_tree(tmp_path)
    alias = tmp_path / "site-alias"
    alias.symlink_to(site, target_is_directory=True)
    _patch_expected_mjlab_tree(monkeypatch, module, site)
    _install_fake_mjlab_candidate(monkeypatch, module, alias, spec_site=site)
    with pytest.raises(module.RuntimeBindingError, match="canonical directory"):
        module.verify_mjlab_runtime_preimport()


def test_mjlab_tasks_entry_point_is_forbidden(monkeypatch, tmp_path):
    module = _load()
    site = _make_mjlab_tree(tmp_path)
    _patch_expected_mjlab_tree(monkeypatch, module, site)
    entry_point = types.SimpleNamespace(
        group="mjlab.tasks", name="ambient-task", value="foreign:register"
    )
    _install_fake_mjlab_candidate(
        monkeypatch, module, site, entry_points=(entry_point,)
    )
    with pytest.raises(module.RuntimeBindingError, match="forbidden: ambient-task"):
        module.verify_mjlab_runtime_preimport()


def test_mjlab_tree_must_match_across_both_enumeration_passes(
    monkeypatch, tmp_path
):
    module = _load()
    site = _make_mjlab_tree(tmp_path)
    _patch_expected_mjlab_tree(monkeypatch, module, site)
    _install_fake_mjlab_candidate(monkeypatch, module, site)
    original = module._enumerate_mjlab_selected_tree
    calls = 0

    def changed_between_passes(distribution_root, package_root):
        nonlocal calls
        calls += 1
        result = original(distribution_root, package_root)
        if calls == 2:
            relative, path, fingerprint = result[0]
            changed = fingerprint[:-1] + (fingerprint[-1] + 1,)
            return ((relative, path, changed),) + result[1:]
        return result

    monkeypatch.setattr(
        module, "_enumerate_mjlab_selected_tree", changed_between_passes
    )
    with pytest.raises(module.RuntimeBindingError, match="across enumeration passes"):
        module.verify_mjlab_runtime_preimport()


def test_loaded_mjlab_modules_must_all_stay_under_verified_package_root(
    monkeypatch, tmp_path
):
    module = _load()
    site = _make_mjlab_tree(tmp_path)
    _patch_expected_mjlab_tree(monkeypatch, module, site)
    package_spec = _install_fake_mjlab_candidate(monkeypatch, module, site)
    package = types.ModuleType("mjlab")
    package.__file__ = package_spec.origin
    package.__spec__ = package_spec
    core = types.ModuleType("mjlab.core")
    core.__file__ = str(site / "mjlab" / "core.py")
    core.__spec__ = types.SimpleNamespace(origin=core.__file__)
    monkeypatch.setitem(sys.modules, "mjlab", package)
    monkeypatch.setitem(sys.modules, "mjlab.core", core)
    assert module.verify_loaded_mjlab_runtime_modules() == (
        module.expected_mjlab_runtime_identity()
    )

    foreign_file = tmp_path / "foreign_loaded.py"
    foreign_file.write_bytes(b"FOREIGN = True\n")
    foreign = types.ModuleType("mjlab.foreign")
    foreign.__file__ = str(foreign_file)
    foreign.__spec__ = types.SimpleNamespace(origin=foreign.__file__)
    monkeypatch.setitem(sys.modules, "mjlab.foreign", foreign)
    with pytest.raises(module.RuntimeBindingError, match="module is foreign"):
        module.verify_loaded_mjlab_runtime_modules()


def test_loaded_mjlab_module_must_belong_to_selected_tree(monkeypatch, tmp_path):
    module = _load()
    site = _make_mjlab_tree(tmp_path)
    nonselected_path = site / "mjlab" / "native.so"
    nonselected_path.write_bytes(b"not-an-imported-extension")
    _patch_expected_mjlab_tree(monkeypatch, module, site)
    package_spec = _install_fake_mjlab_candidate(monkeypatch, module, site)
    package = types.ModuleType("mjlab")
    package.__file__ = package_spec.origin
    package.__spec__ = package_spec
    nonselected = types.ModuleType("mjlab.native")
    nonselected.__file__ = str(nonselected_path)
    nonselected.__spec__ = types.SimpleNamespace(origin=nonselected.__file__)
    monkeypatch.setitem(sys.modules, "mjlab", package)
    monkeypatch.setitem(sys.modules, "mjlab.native", nonselected)
    with pytest.raises(module.RuntimeBindingError, match="outside the selected code tree"):
        module.verify_loaded_mjlab_runtime_modules()


def test_postimport_mjlab_verifier_requires_loaded_root_module(monkeypatch, tmp_path):
    module = _load()
    site = _make_mjlab_tree(tmp_path)
    _patch_expected_mjlab_tree(monkeypatch, module, site)
    _install_fake_mjlab_candidate(monkeypatch, module, site)
    with pytest.raises(module.RuntimeBindingError, match="is not loaded"):
        module.verify_loaded_mjlab_runtime_modules()


def test_real_offline_dual_wheels_bind_one_fresh_site(monkeypatch, tmp_path):
    module = _load()
    _require_assets(module)
    site = tmp_path / "runtime_site"
    verified = _preverified(module, *module._verified_wheel_payloads())
    monkeypatch.setattr(
        module, "_import_runtime_modules",
        lambda actual: _fake_modules(monkeypatch, module, actual),
    )
    try:
        assert module.bind_fresh_epa48_runtime_site(
            site, preimport_verification=verified
        ) == (
            module.expected_runtime_stack_identity()
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


def test_cold_verification_payloads_are_reused_by_binding(monkeypatch, tmp_path):
    module = _load()
    calls = []
    monkeypatch.setattr(
        module, "verify_mjlab_runtime_preimport",
        lambda: module.expected_mjlab_runtime_identity(),
    )

    def wheel_payloads():
        calls.append("hash_wheels")
        return b"verified-rsl", b"verified-epa"

    monkeypatch.setattr(module, "_verified_wheel_payloads", wheel_payloads)
    verified = module.verify_runtime_stack_preimport()
    extracted = []
    monkeypatch.setattr(
        module, "_extract_exact_wheel",
        lambda payload, _site, label: extracted.append((label, payload)),
    )
    monkeypatch.setattr(module, "_require_site_candidates", lambda _site: None)
    monkeypatch.setattr(module, "_import_runtime_modules", lambda _site: (object(), object()))
    monkeypatch.setattr(module, "_require_loaded_runtime", lambda *_args: None)
    identity = module.bind_fresh_epa48_runtime_site(
        tmp_path / "runtime_site", preimport_verification=verified
    )
    assert calls == ["hash_wheels"]
    assert extracted == [
        ("RSL-RL 3 wheel", b"verified-rsl"),
        ("EPA48 wheel", b"verified-epa"),
    ]
    assert identity == module.expected_runtime_stack_identity()
    identity["mjlab"]["version"] = "mutated"
    assert module.verified_runtime_stack_identity(verified) == (
        module.expected_runtime_stack_identity()
    )


def test_runtime_stack_identity_rejects_a_forged_verification():
    module = _load()
    with pytest.raises(module.RuntimeBindingError, match="verification is missing"):
        module.verified_runtime_stack_identity(object())


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
    verified = _preverified(module, *_wheel_payloads(module))
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
        module.bind_fresh_epa48_runtime_site(
            site, preimport_verification=verified
        )
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
    verified = _preverified(module, *_wheel_payloads(module))

    def fake_import(actual):
        if change_file:
            with (actual / "mujoco_warp" / "_src" / "types.py").open("ab") as stream:
                stream.write(b"\n# changed during import\n")
        return _fake_modules(
            monkeypatch, module, actual, version=version, horizon=horizon
        )

    monkeypatch.setattr(module, "_import_runtime_modules", fake_import)
    with pytest.raises(module.RuntimeBindingError, match="version/horizon/origin"):
        module.bind_fresh_epa48_runtime_site(
            site, preimport_verification=verified
        )
    assert str(site) not in sys.path


def test_failed_import_restores_path_and_cleans_partial_prefixes(monkeypatch, tmp_path):
    module = _load()
    site = tmp_path / "site"
    verified = _preverified(module, b"rsl", b"epa")
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
        module.bind_fresh_epa48_runtime_site(
            site, preimport_verification=verified
        )
    assert sys.path == path_before
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in sys.modules for prefix in module._PRELOAD_PREFIXES
    )
