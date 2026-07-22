#!/usr/bin/env python3
"""三人分支追踪看板(Franco 第 6 条:防"修复长期没 track 到"再次发生)。

人话:打印一张 markdown 表——我们三个人(Franco / jiayi=dongc1 / yikang=Catrunaround)
现在各自有哪些远端分支、每条领先/落后 origin/main 多少个 commit、最后一次提交是啥时候。
"领先数 > 0" 的分支就是"上面有没进 main 的东西",要么选择性重做搬进 main、要么在
审计文档里记"不搬 + 原因",不允许第三种状态(失踪)。

只读:只跑 git for-each-ref / rev-list,不 fetch、不 checkout、不写任何东西。
想要新鲜数据先自己 git fetch --prune。

用法:
    python scripts/branch_dashboard.py             # 全部三人,按领先数排序
    python scripts/branch_dashboard.py --min-ahead 1   # 只看有未合并提交的分支
    python scripts/branch_dashboard.py --limit 15      # 每人最多列 15 条
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict

# 提交者名 -> 人。tip committer 不在表里的分支归入 "其他/共享"。
PERSON_BY_COMMITTER = {
    "FranardoHuang": "Franco",
    "Franco": "Franco",
    "dongc1": "jiayi",
    "Catrunaround": "yikang",
}
BASE = "origin/main"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout


def _branches() -> list[dict]:
    rows = []
    output = _git(
        "for-each-ref", "refs/remotes/origin",
        "--format=%(refname:short)\t%(committername)\t%(committerdate:short)",
    )
    for line in output.splitlines():
        ref, committer, date = line.split("\t")
        if ref in ("origin", "origin/HEAD", BASE):
            continue
        counts = _git("rev-list", "--left-right", "--count", f"{BASE}...{ref}").split()
        behind, ahead = int(counts[0]), int(counts[1])
        rows.append({
            "branch": ref,
            "person": PERSON_BY_COMMITTER.get(committer, f"其他/共享({committer})"),
            "date": date,
            "ahead": ahead,   # 该分支上 main 没有的提交数 = 可能失踪的修复
            "behind": behind, # main 上该分支没有的提交数 = 分支有多陈旧
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-ahead", type=int, default=0,
                        help="只列领先 main 至少 N 个 commit 的分支")
    parser.add_argument("--limit", type=int, default=20, help="每人最多列几条")
    args = parser.parse_args(argv)

    try:
        rows = _branches()
    except subprocess.CalledProcessError as error:
        print(f"git failed: {error.stderr or error}", file=sys.stderr)
        return 1

    by_person: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["ahead"] >= args.min_ahead:
            by_person[row["person"]].append(row)

    head = _git("log", "-1", "--format=%h %cd", "--date=short", BASE).strip()
    print(f"# 三人分支追踪看板(基准 {BASE} @ {head};先 git fetch --prune 才是新鲜数据)\n")
    order = ["Franco", "jiayi", "yikang"]
    people = order + sorted(set(by_person) - set(order))
    for person in people:
        branches = sorted(
            by_person.get(person, ()), key=lambda r: (-r["ahead"], r["date"]),
        )
        total = len(branches)
        unmerged = sum(1 for r in branches if r["ahead"] > 0)
        print(f"## {person} — {total} 条分支,{unmerged} 条有未进 main 的提交\n")
        if not branches:
            print("(无)\n")
            continue
        print("| 分支 | 领先 main | 落后 main | 最后提交 |")
        print("| --- | ---: | ---: | --- |")
        for row in branches[: args.limit]:
            print(
                f"| {row['branch']} | {row['ahead']} | {row['behind']} | {row['date']} |"
            )
        if total > args.limit:
            print(f"| …还有 {total - args.limit} 条(用 --limit 调大) | | | |")
        print()
    print(
        "纪律:领先数 > 0 的分支上的每个提交,要么搬进 main、要么在"
        " docs/research/branch_fix_audit_*.md 记『不搬 + 原因』,不允许失踪。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
