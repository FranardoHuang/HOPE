"""Canonical, Kit-free loader for focused ``full_mdp_env`` tests."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile


CANONICAL_MODULE = "whole_body_tracking.tasks.tracking.full_mdp_env"
_LIVE_STAGES: list[tempfile.TemporaryDirectory[str]] = []


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _stage_import_root(module_path: Path) -> tempfile.TemporaryDirectory[str]:
    source_package = module_path.resolve().parents[2]
    stage = tempfile.TemporaryDirectory(prefix="full-mdp-canonical-")
    root = Path(stage.name)

    for package_dir in (source_package, *source_package.rglob("*")):
        if not package_dir.is_dir() or not (package_dir / "__init__.py").is_file():
            continue
        relative = package_dir.relative_to(source_package.parent)
        _write(
            root / relative / "__init__.py",
            f"__path__.append({str(package_dir)!r})\n",
        )

    _write(root / "isaaclab/__init__.py", "")
    _write(root / "isaaclab/envs/__init__.py", "")
    _write(root / "isaaclab/envs/common.py", "VecEnvStepReturn = tuple\n")
    _write(
        root / "isaaclab/envs/manager_based_rl_env.py",
        "class ManagerBasedRLEnv:\n"
        "    base_constructions = 0\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        type(self).base_constructions += 1\n"
        "        raise AssertionError('focused tests must not construct a Kit env')\n"
        "    def close(self):\n"
        "        return None\n",
    )
    _write(
        root / "isaaclab/envs/manager_based_rl_env_cfg.py",
        "class ManagerBasedRLEnvCfg:\n"
        "    pass\n",
    )
    return stage


def _namespace_snapshot() -> dict[str, object]:
    return {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == "whole_body_tracking"
        or name.startswith("whole_body_tracking.")
        or name == "isaaclab"
        or name.startswith("isaaclab.")
    }


def _clear_namespace() -> None:
    for name in tuple(sys.modules):
        if (
            name == "whole_body_tracking"
            or name.startswith("whole_body_tracking.")
            or name == "isaaclab"
            or name.startswith("isaaclab.")
        ):
            sys.modules.pop(name, None)


def load_canonical_full_mdp_env(
    module_path: Path, *, retain_namespace: bool
):
    """Execute the real source under its launcher-visible canonical name."""

    prior = _namespace_snapshot()
    stage = _stage_import_root(module_path)
    _clear_namespace()
    sys.path.insert(0, stage.name)
    try:
        module = importlib.import_module(CANONICAL_MODULE)
        isaaclab_path = Path(sys.modules["isaaclab"].__file__).resolve()
        assert Path(stage.name).resolve() in isaaclab_path.parents
        assert module.__name__ == CANONICAL_MODULE
        assert Path(module.__file__).resolve() == module_path.resolve()
        assert module.reward_contract.__name__ == (
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_full_mdp_reward_contract"
        )
    except BaseException:
        _clear_namespace()
        sys.modules.update(prior)
        stage.cleanup()
        raise
    finally:
        sys.path.remove(stage.name)

    if retain_namespace:
        _LIVE_STAGES.append(stage)
    else:
        _clear_namespace()
        sys.modules.update(prior)
        stage.cleanup()
    return module


def probe_canonical_full_mdp_env_subprocess(module_path: Path) -> None:
    """Prove canonical cold import without an ambient Kit/Omni package."""

    stage = _stage_import_root(module_path)
    env = dict(os.environ)
    prior_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = stage.name + (
        "" if not prior_pythonpath else os.pathsep + prior_pythonpath
    )
    env["PYTHONSAFEPATH"] = "1"
    code = (
        "import importlib, pathlib, sys; "
        f"m=importlib.import_module({CANONICAL_MODULE!r}); "
        f"assert pathlib.Path(m.__file__).resolve()==pathlib.Path({str(module_path.resolve())!r}); "
        f"assert m.__name__=={CANONICAL_MODULE!r}; "
        "p=pathlib.Path(sys.modules['isaaclab'].__file__).resolve(); "
        f"r=pathlib.Path({stage.name!r}).resolve(); assert r==p or r in p.parents; "
        "assert m.reward_contract.__name__=='whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_reward_contract'; "
        "assert 'omni' not in sys.modules and 'omni.kit' not in sys.modules"
    )
    try:
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            cwd=module_path.parents[5],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    finally:
        stage.cleanup()
