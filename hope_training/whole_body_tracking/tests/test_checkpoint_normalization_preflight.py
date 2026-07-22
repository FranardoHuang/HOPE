"""Host-only tests for the pre-Kit checkpoint normalization preflight.

人话:不 import 任何 Isaac 包、不需要真 torch(用 stub 顶替 ``import torch``),在普通主机上
验证 2x2 真值表、目录选最大编号 model_N.pt、以及 train.py 确实在启动 Kit 之前调用预检并把
checkpoint_path 重绑到精确文件。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/utils"
    / "checkpoint_normalization_preflight.py"
)
TRAIN_PATH = ROOT / "scripts/train.py"

SPEC = importlib.util.spec_from_file_location("preflight_under_test", MODULE_PATH)
PREFLIGHT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PREFLIGHT)


# --------------------------------------------------------------------------- #
# 2x2 真值表
# --------------------------------------------------------------------------- #

def test_truth_table_diagonal_cells_pass():
    # 都关:raw checkpoint + empirical off -> 放行,报告"无归一化状态"。
    assert (
        PREFLIGHT.validate_checkpoint_normalization(
            {"model_state_dict": {}}, empirical_normalization=False
        )
        is False
    )
    # 都开:normalized checkpoint + empirical on -> 放行,报告"有归一化状态"。
    assert (
        PREFLIGHT.validate_checkpoint_normalization(
            {"model_state_dict": {}, "obs_norm_state_dict": {}},
            empirical_normalization=True,
        )
        is True
    )


def test_raw_checkpoint_with_normalization_enabled_fails_with_exact_override():
    with pytest.raises(RuntimeError, match=r"algo\.runner\.empirical_normalization=false") as excinfo:
        PREFLIGHT.validate_checkpoint_normalization(
            {"model_state_dict": {}}, empirical_normalization=True
        )
    # 修复命令必须是 run-local CLI override,不许怂恿改全局 ppo.yaml。
    assert "Do not change cfg/algo/ppo.yaml" in str(excinfo.value)


def test_normalized_checkpoint_with_normalization_disabled_fails():
    with pytest.raises(RuntimeError, match=r"algo\.runner\.empirical_normalization=true"):
        PREFLIGHT.validate_checkpoint_normalization(
            {"model_state_dict": {}, "obs_norm_state_dict": {}},
            empirical_normalization=False,
        )


def test_non_mapping_checkpoint_counts_as_raw():
    # 老式"裸 state_dict 序列化成别的对象"也算 raw:empirical on 时必须炸。
    with pytest.raises(RuntimeError, match="trained_with_obs_norm=False"):
        PREFLIGHT.validate_checkpoint_normalization(
            ["not", "a", "mapping"], empirical_normalization=True
        )


# --------------------------------------------------------------------------- #
# 路径解析:目录选最大编号 + fail-loud
# --------------------------------------------------------------------------- #

def test_resolve_none_means_train_from_scratch():
    assert PREFLIGHT.resolve_checkpoint_path(None) is None


@pytest.mark.parametrize("bogus", ["", "  ", "none", "None", "null"])
def test_resolve_stringly_null_fails_loud(bogus):
    # "none"/"null"/空串到了 _run 会被当路径然后必然 FileNotFoundError;预检直接拒收。
    with pytest.raises(ValueError, match="checkpoint_path=null"):
        PREFLIGHT.resolve_checkpoint_path(bogus)


def test_resolve_missing_path_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        PREFLIGHT.resolve_checkpoint_path(tmp_path / "no_such_model.pt")


def test_resolve_directory_without_numbered_checkpoints_fails(tmp_path):
    (tmp_path / "model_latest.pt").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="no model_<iteration>"):
        PREFLIGHT.resolve_checkpoint_path(tmp_path)


def test_resolve_directory_selects_latest_numeric_checkpoint(tmp_path):
    for name in ("model_2.pt", "model_10.pt", "model_latest.pt", "model_3_backup.pt"):
        (tmp_path / name).write_bytes(b"x")
    resolved = PREFLIGHT.resolve_checkpoint_path(tmp_path)
    # 数值比较(10 > 2),不是字典序("model_2" > "model_10");非纯数字编号的文件不参与。
    assert resolved == (tmp_path / "model_10.pt").resolve()


def test_resolve_plain_file_passes_through(tmp_path):
    target = tmp_path / "model_7.pt"
    target.write_bytes(b"x")
    assert PREFLIGHT.resolve_checkpoint_path(str(target)) == target.resolve()


# --------------------------------------------------------------------------- #
# 端到端预检:stub torch,验证"加载的就是重绑的那个精确文件"
# --------------------------------------------------------------------------- #

def _stub_torch(load_results: dict, seen_paths: list):
    """人话:顶替 ``import torch`` 的假模块,按文件名回放 checkpoint dict 并记录加载了谁。"""

    def _load(path, map_location=None, weights_only=None):
        # 预检必须和 _run 的正式加载同参数,别的调用方式一律当 bug。
        assert map_location == "cpu"
        assert weights_only is False
        seen_paths.append(Path(path))
        return load_results[Path(path).name]

    stub = types.ModuleType("torch")
    stub.load = _load
    return stub


def test_preflight_none_checkpoint_skips_without_torch():
    # 不热启动就不该碰 torch:即使 torch 不存在也要能安静返回 None。
    with mock.patch.dict(sys.modules, {"torch": None}):
        assert (
            PREFLIGHT.preflight_checkpoint_normalization(
                None, empirical_normalization=True
            )
            is None
        )


def test_preflight_directory_loads_exact_latest_file_and_passes(tmp_path, capsys):
    for name in ("model_2.pt", "model_10.pt"):
        (tmp_path / name).write_bytes(b"x")
    seen = []
    stub = _stub_torch({"model_10.pt": {"model_state_dict": {}}}, seen)
    with mock.patch.dict(sys.modules, {"torch": stub}):
        resolved = PREFLIGHT.preflight_checkpoint_normalization(
            tmp_path, empirical_normalization=False
        )
    # 返回值 = 实际被加载的精确文件 = 目录里编号最大的:调用方拿它重绑防竞态。
    assert resolved == (tmp_path / "model_10.pt").resolve()
    assert seen == [resolved]
    assert "CHECKPOINT NORMALIZATION PREFLIGHT PASS (before Kit)" in capsys.readouterr().out


def test_preflight_mismatch_fails_before_any_kit_work(tmp_path):
    (tmp_path / "model_5.pt").write_bytes(b"x")
    stub = _stub_torch({"model_5.pt": {"model_state_dict": {}}}, [])
    with mock.patch.dict(sys.modules, {"torch": stub}):
        with pytest.raises(RuntimeError, match=r"empirical_normalization=false"):
            PREFLIGHT.preflight_checkpoint_normalization(
                tmp_path, empirical_normalization=True
            )


# --------------------------------------------------------------------------- #
# train.py 接线:预检 + 重绑必须发生在 Kit 启动之前
# --------------------------------------------------------------------------- #

def test_train_main_preflights_and_rebinds_before_kit_launch():
    main_source = TRAIN_PATH.read_text().split("def main(cfg):", 1)[1]
    kit_import = main_source.index("from isaaclab.app import AppLauncher")
    pre_kit = main_source[:kit_import]
    call = pre_kit.index("preflight_checkpoint_normalization(")
    rebind = pre_kit.index("cfg.checkpoint_path = str(resolved_checkpoint)")
    assert call < rebind
    # 模块必须按文件路径加载(包 __init__ 会连带 import Isaac 任务,Kit 没起时不能走包导入)。
    assert "checkpoint_normalization_preflight.py" in pre_kit
    assert "spec_from_file_location" in pre_kit
    # 缺 empirical_normalization key 不许带默认值猜(与 runner_kwargs 的 fail-loud 语义同源)。
    assert "algo.runner.empirical_normalization is missing" in pre_kit
