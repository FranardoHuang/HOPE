"""Dependency-light tests for the optional Isaac Kit runtime thread caps."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "scripts/train.py"


def _module(name: str, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_train_module(monkeypatch):
    hydra = _module("hydra", main=lambda **kwargs: (lambda function: function))

    class FakeOmegaConf:
        @staticmethod
        def resolve(cfg):
            return None

        @staticmethod
        def set_struct(cfg, value):
            return None

    monkeypatch.setitem(sys.modules, "hydra", hydra)
    monkeypatch.setitem(
        sys.modules,
        "omegaconf",
        _module(
            "omegaconf",
            ListConfig=type("ListConfig", (list,), {}),
            OmegaConf=FakeOmegaConf,
        ),
    )
    module_name = "train_thread_caps_under_test"
    spec = importlib.util.spec_from_file_location(module_name, TRAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


class _Cfg(dict):
    __getattr__ = dict.__getitem__


class _Settings:
    def __init__(self, values):
        self.values = values

    def get(self, key):
        return self.values.get(key)


def test_resolve_kit_thread_caps_builds_exact_args_and_absence_is_compatible(monkeypatch):
    train = _load_train_module(monkeypatch)
    assert train._resolve_kit_thread_caps({}) == (None, None, None)
    assert train._resolve_kit_thread_caps({
        "kit_carb_tasking_thread_count": 16,
        "kit_tbb_thread_count": 16,
    }) == (
        "--/plugins/carb.tasking.plugin/threadCount=16 "
        "--/plugins/omni.tbb.globalcontrol/maxThreadCount=16",
        16,
        16,
    )


@pytest.mark.parametrize(
    "cfg",
    [
        {"kit_carb_tasking_thread_count": 16},
        {"kit_tbb_thread_count": 16},
    ],
)
def test_resolve_kit_thread_caps_rejects_one_sided_configuration(monkeypatch, cfg):
    train = _load_train_module(monkeypatch)
    with pytest.raises(ValueError, match="must be supplied together"):
        train._resolve_kit_thread_caps(cfg)


@pytest.mark.parametrize("value", [True, False, 2.0, "2", 0, -1])
@pytest.mark.parametrize(
    "key", ["kit_carb_tasking_thread_count", "kit_tbb_thread_count"]
)
def test_resolve_kit_thread_caps_rejects_invalid_values(monkeypatch, key, value):
    train = _load_train_module(monkeypatch)
    cfg = {"kit_carb_tasking_thread_count": 16, "kit_tbb_thread_count": 16}
    cfg[key] = value
    error = TypeError if type(value) is not int else ValueError
    with pytest.raises(error):
        train._resolve_kit_thread_caps(cfg)


def test_verify_kit_thread_caps_reads_exact_settings_and_prints_marker(monkeypatch, capsys):
    train = _load_train_module(monkeypatch)
    settings = _Settings({
        train._KIT_CARB_TASKING_THREAD_SETTING: 16,
        train._KIT_TBB_THREAD_SETTING: 16,
        train._KIT_USE_OMNI_JOB_SETTING: False,
    })
    train._verify_kit_thread_caps(settings, 16, 16)
    assert capsys.readouterr().out == (
        "[train.py] KIT_THREAD_CAP_OK: "
        "carb.tasking=16 omni.tbb=16 useOmniJob=false\n"
    )


@pytest.mark.parametrize(
    "values,message",
    [
        ({"carb": 15, "tbb": 16, "use": False}, "carb.tasking thread cap mismatch"),
        ({"carb": 16, "tbb": 15, "use": False}, "omni.tbb thread cap mismatch"),
        ({"carb": 16, "tbb": 16, "use": True}, "useOmniJob must be exactly false"),
    ],
)
def test_verify_kit_thread_caps_rejects_runtime_mismatch(monkeypatch, values, message):
    train = _load_train_module(monkeypatch)
    settings = _Settings({
        train._KIT_CARB_TASKING_THREAD_SETTING: values["carb"],
        train._KIT_TBB_THREAD_SETTING: values["tbb"],
        train._KIT_USE_OMNI_JOB_SETTING: values["use"],
    })
    with pytest.raises(RuntimeError, match=message):
        train._verify_kit_thread_caps(settings, 16, 16)


def test_main_passes_and_verifies_kit_args(monkeypatch, capsys):
    train = _load_train_module(monkeypatch)
    captured = {}
    order = []

    class FakeSimulationApp:
        def close(self):
            captured["closed"] = True
            order.append("closed")

    class FakeAppLauncher:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            self.app = FakeSimulationApp()
            order.append("app_started")

    isaaclab = _module("isaaclab")
    isaaclab.__path__ = []
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(
        sys.modules, "isaaclab.app", _module("isaaclab.app", AppLauncher=FakeAppLauncher)
    )
    monkeypatch.setitem(sys.modules, "wandb", _module("wandb", run=None))
    settings = _Settings({
        train._KIT_CARB_TASKING_THREAD_SETTING: 16,
        train._KIT_TBB_THREAD_SETTING: 16,
        train._KIT_USE_OMNI_JOB_SETTING: False,
    })
    monkeypatch.setitem(
        sys.modules,
        "carb",
        _module("carb", settings=_module("carb.settings", get_settings=lambda: settings)),
    )
    monkeypatch.setattr(
        train,
        "_require_action_ball_runtime_unloaded_before_app_start",
        lambda: order.append("pre_app_checked"),
    )
    monkeypatch.setattr(
        train,
        "_attest_action_ball_runtime_after_app_start",
        lambda: order.append("runtime_attested"),
    )

    def _run(cfg):
        captured["ran"] = True
        order.append("run")

    monkeypatch.setattr(train, "_run", _run)
    cfg = _Cfg(
        headless=True,
        device="cuda:0",
        video=False,
        kit_carb_tasking_thread_count=16,
        kit_tbb_thread_count=16,
    )
    train.main(cfg)
    assert captured["ran"] is True
    assert captured["closed"] is True
    assert captured["kwargs"]["kit_args"] == (
        "--/plugins/carb.tasking.plugin/threadCount=16 "
        "--/plugins/omni.tbb.globalcontrol/maxThreadCount=16"
    )
    assert order == [
        "pre_app_checked",
        "app_started",
        "runtime_attested",
        "run",
        "closed",
    ]
    assert "KIT_THREAD_CAP_OK" in capsys.readouterr().out


def test_runtime_attestation_is_absent_or_complete(monkeypatch):
    train = _load_train_module(monkeypatch)
    names = (
        "HOPE_ACTION_BALL_RUNTIME_ATTESTATION",
        "HOPE_ACTION_BALL_RUNTIME_RECEIPT_PATH",
        "HOPE_ACTION_BALL_RUNTIME_KIT_PYTHON_SHA256",
        "HOPE_ACTION_BALL_RUNTIME_RSL_ZIP_SHA256",
        "HOPE_ACTION_BALL_RUNTIME_VENV_SITE",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    assert train._attest_action_ball_runtime_after_app_start() is None
    monkeypatch.setenv("HOPE_ACTION_BALL_RUNTIME_ATTESTATION", "sealed_rsl_v1")
    with pytest.raises(RuntimeError, match="pre-AppLauncher runtime unload proof"):
        train._attest_action_ball_runtime_after_app_start()


def test_runtime_attestation_rejects_preloaded_runtime(monkeypatch):
    train = _load_train_module(monkeypatch)
    values = {
        "HOPE_ACTION_BALL_RUNTIME_ATTESTATION": "sealed_rsl_v1",
        "HOPE_ACTION_BALL_RUNTIME_RECEIPT_PATH": "/tmp/receipt",
        "HOPE_ACTION_BALL_RUNTIME_KIT_PYTHON_SHA256": "a" * 64,
        "HOPE_ACTION_BALL_RUNTIME_RSL_ZIP_SHA256": "b" * 64,
        "HOPE_ACTION_BALL_RUNTIME_VENV_SITE": "/tmp/site-packages",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setitem(sys.modules, "rsl_rl.foreign_plugin", _module("foreign"))
    with pytest.raises(RuntimeError, match="Hydra preloaded"):
        train._require_action_ball_runtime_unloaded_before_app_start()
    assert train._ACTION_BALL_RUNTIME_PRE_APP_STATE == "consumed"

    monkeypatch.delitem(sys.modules, "rsl_rl.foreign_plugin")
    with pytest.raises(RuntimeError, match="not reusable"):
        train._require_action_ball_runtime_unloaded_before_app_start()


def test_runtime_attestation_consumes_pre_app_proof_before_post_app_imports(
    monkeypatch,
):
    train = _load_train_module(monkeypatch)
    values = {
        "HOPE_ACTION_BALL_RUNTIME_ATTESTATION": "sealed_rsl_v1",
        "HOPE_ACTION_BALL_RUNTIME_RECEIPT_PATH": "/tmp/receipt",
        "HOPE_ACTION_BALL_RUNTIME_KIT_PYTHON_SHA256": "a" * 64,
        "HOPE_ACTION_BALL_RUNTIME_RSL_ZIP_SHA256": "b" * 64,
        "HOPE_ACTION_BALL_RUNTIME_VENV_SITE": "/tmp/site-packages",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    train._require_action_ball_runtime_unloaded_before_app_start()
    assert train._ACTION_BALL_RUNTIME_PRE_APP_STATE == "checked"

    # Model a module imported by AppLauncher itself, after the pre-App proof.
    monkeypatch.setitem(sys.modules, "torch.post_app", _module("torch.post_app"))
    with pytest.raises(RuntimeError, match="AppLauncher preloaded"):
        train._attest_action_ball_runtime_after_app_start()
    assert train._ACTION_BALL_RUNTIME_PRE_APP_STATE == "consumed"

    monkeypatch.delitem(sys.modules, "torch.post_app")
    with pytest.raises(RuntimeError, match="missing or consumed"):
        train._attest_action_ball_runtime_after_app_start()


def test_runtime_attestation_consumes_proof_before_post_app_env_validation(
    monkeypatch,
):
    train = _load_train_module(monkeypatch)
    values = {
        "HOPE_ACTION_BALL_RUNTIME_ATTESTATION": "sealed_rsl_v1",
        "HOPE_ACTION_BALL_RUNTIME_RECEIPT_PATH": "/tmp/receipt",
        "HOPE_ACTION_BALL_RUNTIME_KIT_PYTHON_SHA256": "a" * 64,
        "HOPE_ACTION_BALL_RUNTIME_RSL_ZIP_SHA256": "b" * 64,
        "HOPE_ACTION_BALL_RUNTIME_VENV_SITE": "/tmp/site-packages",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    train._require_action_ball_runtime_unloaded_before_app_start()

    monkeypatch.delenv("HOPE_ACTION_BALL_RUNTIME_VENV_SITE")
    with pytest.raises(RuntimeError, match="changed after AppLauncher"):
        train._attest_action_ball_runtime_after_app_start()
    assert train._ACTION_BALL_RUNTIME_PRE_APP_STATE == "consumed"

    monkeypatch.setenv(
        "HOPE_ACTION_BALL_RUNTIME_VENV_SITE", values["HOPE_ACTION_BALL_RUNTIME_VENV_SITE"]
    )
    with pytest.raises(RuntimeError, match="missing or consumed"):
        train._attest_action_ball_runtime_after_app_start()


def test_runtime_attestation_failure_closes_started_app(monkeypatch):
    train = _load_train_module(monkeypatch)
    captured = []

    class FakeCoreApp:
        def post_quit(self, code):
            captured.append(("post_quit", code))

    class FakeSimulationApp:
        app = FakeCoreApp()

        def close(self):
            captured.append(("close", None))

    class FakeAppLauncher:
        def __init__(self, **kwargs):
            self.app = FakeSimulationApp()

    isaaclab = _module("isaaclab")
    isaaclab.__path__ = []
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(
        sys.modules, "isaaclab.app", _module("isaaclab.app", AppLauncher=FakeAppLauncher)
    )
    monkeypatch.setitem(sys.modules, "wandb", _module("wandb", run=None))
    monkeypatch.setattr(
        train,
        "_require_action_ball_runtime_unloaded_before_app_start",
        lambda: None,
    )
    monkeypatch.setattr(
        train,
        "_attest_action_ball_runtime_after_app_start",
        lambda: (_ for _ in ()).throw(RuntimeError("attestation failed")),
    )
    monkeypatch.setattr(
        train,
        "_run",
        lambda cfg: (_ for _ in ()).throw(AssertionError("run must not start")),
    )
    cfg = _Cfg(
        headless=True,
        device="cuda:0",
        video=False,
        kit_carb_tasking_thread_count=None,
        kit_tbb_thread_count=None,
    )
    train.main(cfg)
    assert captured == [("post_quit", 1), ("close", None)]


def test_runtime_attestation_call_is_after_app_start_before_training():
    main_source = TRAIN_PATH.read_text(encoding="utf-8").split(
        "def main(cfg):", 1
    )[1]
    pre_app = main_source.index(
        "_require_action_ball_runtime_unloaded_before_app_start()"
    )
    app_started = main_source.index("simulation_app = app_launcher.app")
    attest = main_source.index("_attest_action_ball_runtime_after_app_start()")
    training = main_source.index("_run(cfg)")
    assert pre_app < app_started < attest < training
