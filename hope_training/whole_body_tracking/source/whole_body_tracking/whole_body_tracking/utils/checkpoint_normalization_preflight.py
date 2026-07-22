"""Fail-loud checkpoint/runner observation-normalization preflight (runs BEFORE Kit).

人话:热启动/续训时,checkpoint 里"有没有 obs 归一化状态(obs_norm_state_dict)"必须和本次
启动解析出的 ``algo.runner.empirical_normalization`` 一致。不一致的话,老路径要等 Isaac Kit
起完、``runner.load``/``ckpt_compat`` 才炸出费解的 ``KeyError: obs_norm_state_dict``,白烧几
分钟 GPU 启动费。``scripts/train.py`` 的 ``main()`` 在启动 Kit 之前用本模块在 CPU 上先加载
checkpoint、查 2x2 真值表,错位格立刻 RuntimeError 并给出单条 CLI 修复命令。

本模块刻意不 import 任何 Isaac 包(torch 也只在真正要读 checkpoint 时才 import):它必须能在
Kit 启动前、以及 host-only 单测里直接按文件路径加载。注意不能走 ``import
whole_body_tracking.utils.…`` 的常规包导入 —— 包 ``__init__.py`` 会连带注册 Isaac 任务。
"""

from __future__ import annotations

import pathlib
import re
from collections.abc import Mapping


def resolve_checkpoint_path(value) -> pathlib.Path | None:
    """Resolve a checkpoint file, or the latest numbered ``model_N.pt`` in a run directory.

    人话:``None``(Hydra 的 ``checkpoint_path=null``)= 不热启动,返回 ``None``;字符串一律
    当路径。传目录就挑编号最大的 ``model_<iteration>.pt``(``model_latest.pt`` 之类不带纯数字
    编号的文件不参与)。路径不存在现在就报错,不留给 Kit 起完后的 ``_run`` 再炸。
    """

    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in ("", "none", "null"):
        # 人话:这些字符串不是"不加载"的合法写法 —— 到了 _run 里会被当成路径然后必然
        # FileNotFoundError。想从零训练请传 checkpoint_path=null(Hydra 才会给出真正的 None)。
        raise ValueError(
            f"[train.py] checkpoint_path={value!r} is a string, not a real null; "
            "pass checkpoint_path=null to train from scratch."
        )
    path = pathlib.Path(text).expanduser().resolve()
    if path.is_dir():
        numbered = []
        for candidate in path.glob("model_*.pt"):
            match = re.fullmatch(r"model_(\d+)\.pt", candidate.name)
            if match is not None:
                numbered.append((int(match.group(1)), candidate))
        if not numbered:
            raise FileNotFoundError(
                f"[train.py] checkpoint directory has no model_<iteration>.pt: {path}"
            )
        path = max(numbered, key=lambda item: item[0])[1]
    if not path.is_file():
        raise FileNotFoundError(f"[train.py] checkpoint_path does not exist: {path}")
    return path


def validate_checkpoint_normalization(
    checkpoint,
    *,
    empirical_normalization: bool,
) -> bool:
    """Validate the two-by-two checkpoint/config normalization truth table.

    人话:对角线(都开 / 都关)放行,返回 checkpoint 是否带归一化状态;两个错位格都是"换了
    一个语义不同的 actor",不是兼容性小问题,立刻 RuntimeError 并给出单条 CLI 修复命令。这与
    Kit 起完后 ``ckpt_compat._validate_and_restore_actor_normalizer`` 的判定同一张真值表,只是
    把炸点从"几分钟后"提前到"现在"。
    """

    trained_with_obs_norm = (
        isinstance(checkpoint, Mapping) and "obs_norm_state_dict" in checkpoint
    )
    configured = bool(empirical_normalization)
    if trained_with_obs_norm == configured:
        return trained_with_obs_norm

    if trained_with_obs_norm:
        guidance = (
            "This checkpoint contains obs_norm_state_dict; relaunch with the CLI override "
            "algo.runner.empirical_normalization=true."
        )
    else:
        guidance = (
            "This is a raw-observation checkpoint; relaunch with the run-local CLI override "
            "algo.runner.empirical_normalization=false. Do not change cfg/algo/ppo.yaml "
            "globally — every other lineage trains through it."
        )
    raise RuntimeError(
        "checkpoint/config observation-normalization mismatch: "
        f"checkpoint trained_with_obs_norm={trained_with_obs_norm}, "
        f"resolved algo.runner.empirical_normalization={configured}. {guidance}"
    )


def preflight_checkpoint_normalization(
    checkpoint_path,
    *,
    empirical_normalization: bool,
) -> pathlib.Path | None:
    """Load checkpoint metadata on CPU and validate normalization before Kit starts.

    人话:返回通过预检的精确文件路径(目录已解析成具体 ``model_N.pt``),调用方要把
    ``cfg.checkpoint_path`` 重绑到这个文件 —— 防止并发训练继续往同一目录写更新的
    ``model_N.pt``,导致 Kit 起完后真正加载的字节和预检看过的不一致(竞态)。
    """

    path = resolve_checkpoint_path(checkpoint_path)
    if path is None:
        return None

    import torch

    # 与 _run 里的正式加载同参数(map_location="cpu", weights_only=False):预检看的必须和
    # 之后真加载的是同一种解码方式。
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    trained_with_obs_norm = validate_checkpoint_normalization(
        checkpoint,
        empirical_normalization=empirical_normalization,
    )
    print(
        "[train.py] CHECKPOINT NORMALIZATION PREFLIGHT PASS (before Kit): "
        f"path={path}, checkpoint_obs_norm={trained_with_obs_norm}, "
        f"runner_empirical_normalization={bool(empirical_normalization)}",
        flush=True,
    )
    return path
