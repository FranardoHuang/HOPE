"""给训练存档的观测输入补零列(如 175->177 或 175->179 或 179->181):新输入列权重=0,第 0 步行为与原模型逐位相同。

用法: python pad_obs_cols.py <源存档.pt> <输出.pt> <旧维数> <新维数>
前提: 新观测维追加在向量末尾(站位/拍面指令通道都是这么设计的)。

新增 4 列 actor 首层权重 = 0,归一化器 mean=0/var=1 => 第 0 步行为与 P2 逐位相同,
之后靠梯度学会使用新通道。优化器动量同步扩列(新列动量 0 = 冷启动)。

[收编说明 2026-07-09,station-anchor-obs-0709 分支] 本文件与 pod 上的
/workspace/shared/pad_obs_cols.py 逐字节同源(仅本注释块为新增),收进 repo 是为了
版本管控 + CPU 单测(tests/test_station_anchor_obs.py 的手术往返测试)。工具本身
完全通用:R10c 的 179->181 手术就是 `python pad_obs_cols.py src.pt dst.pt 179 181`,
不需要任何改动。按位插列(jiayi 177 在第 167 列)用兄弟脚本 pad_obs_cols_insert.py
(pod /workspace/shared)或 make_hitter_warmstart.py。
"""
import sys

import torch

import sys
SRC, DST, OLD, NEW = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
PAD = NEW - OLD

c = torch.load(SRC, map_location="cpu", weights_only=False)
padded = []


def pad_last(t, fill):
    assert t.shape[-1] == OLD, t.shape
    ext = torch.full((*t.shape[:-1], PAD), fill, dtype=t.dtype)
    return torch.cat([t, ext], dim=-1)


# 1) actor 首层
msd = c["model_state_dict"]
hit = [k for k, v in msd.items() if hasattr(v, "shape") and OLD in tuple(v.shape)]
assert hit == ["actor.0.weight"], f"意外的 175 张量: {hit}"
msd["actor.0.weight"] = pad_last(msd["actor.0.weight"], 0.0)
padded.append(("model:actor.0.weight", tuple(msd["actor.0.weight"].shape)))

# 2) 优化器动量(exp_avg / exp_avg_sq,新列 0)
osd = c.get("optimizer_state_dict") or {}
n_opt = 0
for idx, st in (osd.get("state") or {}).items():
    for kk, t in list(st.items()):
        if hasattr(t, "shape") and t.dim() >= 1 and OLD in tuple(t.shape):
            assert t.shape[-1] == OLD, (idx, kk, t.shape)
            st[kk] = pad_last(t, 0.0)
            padded.append((f"opt[{idx}].{kk}", tuple(st[kk].shape)))
            n_opt += 1
assert n_opt == 2, f"优化器 175 张量数量异常: {n_opt}"

# 3) actor obs 归一化器(mean/含 mean 的键补 0,var/std 补 1;count 不动)
ons = c["obs_norm_state_dict"]
n_mean = n_var = 0
for k, t in list(ons.items()):
    if hasattr(t, "shape") and t.dim() >= 1 and OLD in tuple(t.shape):
        low = k.lower()
        if "var" in low or "std" in low:
            ons[k] = pad_last(t, 1.0)
            n_var += 1
        else:
            ons[k] = pad_last(t, 0.0)
            n_mean += 1
        padded.append((f"obsnorm:{k}", tuple(ons[k].shape)))
print("obs_norm keys:", {k: (tuple(v.shape) if hasattr(v, "shape") else v) for k, v in ons.items()})
assert n_mean >= 1 and n_var >= 1, f"归一化器扩列不完整 mean={n_mean} var={n_var}"

# privileged(critic)归一化器必须原样
for k, t in (c.get("privileged_obs_norm_state_dict") or {}).items():
    assert not (hasattr(t, "shape") and t.dim() >= 1 and tuple(t.shape)[-1] == NEW), k

import os

os.makedirs(os.path.dirname(DST), exist_ok=True)
torch.save(c, DST)
print("已扩列并保存:", DST, "iter=", c.get("iter"))
for name, shape in padded:
    print("  padded:", name, shape)
