#!/usr/bin/env python3
"""One-time, auditable migration of the action-ball solver pin: v2 -> v3.

人话:现役 manifest 里那枚 ``solver_profile_sha256`` 是**旧口径**算出来的 ——
对五份源文件做整文件 SHA。新口径(v3)钉的是**逐符号语义面**。两个数一定不同,
所以必须迁移一次;但迁移**不许悄悄改数**。

这支脚本做三件事,缺一不可:

1. 算出迁移前后各是什么(旧 pin / 新 pin / 新语义面 SHA / 覆盖了多少符号)。
2. **证明**这次迁移没有改题:把语义面**覆盖到**的每一个符号,在旧 pin 被铸造的
   那个 revision 和新 revision 上逐个对拍。只允许 pin 自己的声明半边
   (``action_ball_solver_profile_contract`` 与 schema 版本常量)发生变化 ——
   任何别的覆盖符号动了,脚本**拒绝迁移**并把它打印出来。
3. 把 1 和 2 写成一份收据 JSON,**连同它自己看不见的那两块一起写**。重新物化这条
   谱系的离线产物时必须引用这份收据,否则没人看得懂为什么所有文件名都变了。

**这份收据看不见什么(必须先读这一段再引用它)**:

* 排除清单里的符号一个都没比 —— 它们**按设计**在语义面之外
  (``coverage_this_receipt_does_not_check``)。
* 造产物的那套工具在 solver pin 的**任何一部分之外**,上面每一条比对都对它结构性
  失明(``producer_lineage_outside_this_pin``)。实例:2026-08-06 和 2026-08-07 两条
  N1 tape 是在两个不同版本的 ``training_contract.py`` 上建的(``8c9eec4c`` vs
  ``c6608440``),这支脚本一个字都看不到 —— 而"只有 pin 自己的声明半边动了"这句话
  被读成了对整条生产血统的陈述,对那个 revision 它是假的。emit 出来的题目数字确实
  一个没变(逐 leaf 对拍过),结论不受影响,但话要说准。
  ``training_contract.py`` **不该**进 solver 语义面:它不命名题目也不计算答案,
  五份纯求解器源码一个字都没提它,``hope_commands.py`` 提到它(那是 task-first
  训练合同那一摊,26k 行里的另一个 command term)但**没有任何覆盖符号**引用得到 ——
  这一条是测试断言,不是判断。它已经在该在的 pin 里 —— fixed-tape producer contract
  的 ``implementation_source_sha256``,而那个 sha 确实跟着它动了。

它**不会**去改那 13 份内容寻址的产物(receipt / tape / manifest / bundle 的文件名
里带着自己的摘要)。那是离线物化流水线的活;这支脚本只负责把"可以重签"这件事
连同证据一起立字据。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = SCRIPTS_DIR.parents[2]
MDP_REL = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp"
)
SURFACE_SOURCE = "action_ball_solver_semantic_surface.py"
PINNER = SCRIPTS_DIR / "pin_action_ball_profile_contracts.py"

MIGRATION_SCHEMA_VERSION = 1
MIGRATION_KIND = "whole_body_tracking.action_ball.solver_pin_v2_to_v3_migration"

#: The only covered symbols that this migration is allowed to have moved.  They
#: are the pin's own declaration half: the contract builder that now binds a
#: semantic surface instead of a byte map, and the schema version that announces
#: it.  Anything else moving means the migration would be re-asking questions,
#: not re-signing them.
ALLOWED_MOVED_SYMBOLS = {
    "hope_commands.py": frozenset(
        {
            "action_ball_solver_profile_contract",
            "_ACTION_BALL_SOLVER_PROFILE_SCHEMA_VERSION",
        }
    )
}

#: Covered symbols that did not exist at ``from_rev`` and whose *introduction* is
#: allowed.  They are the declaration/actual bridge: a knob accessor, the single
#: mapping into ``ContinuousQuestionCfg``, the fail-closed cross-check and the
#: two named constants they read.  Between them they add a refusal; they compute
#: no question field and no answer field.  Anything else appearing out of
#: nowhere is new solver behaviour and this script refuses it, because "it did
#: not exist before" is not the same as "it changes nothing".
ALLOWED_INTRODUCED_SYMBOLS = {
    "hope_commands.py": frozenset(
        {
            "action_ball_declared_solver_knobs",
            "action_ball_solver_cfg_from_declaration",
            "action_ball_assert_solver_runtime_matches_declaration",
            "_ACTION_BALL_SOLVER_FIXED_DIRECTION",
            "_ACTION_BALL_VIRTUAL_BALL_PARAM_NAMES",
        }
    )
}

#: Symbols whose invariance *is* the "questions were not re-drawn" claim.
QUESTION_IDENTITY_SYMBOLS = {
    "hope_commands.py": (
        "_action_ball_exact_question_payload",
        "_action_ball_semantic_levels",
        "_action_ball_canonical_sha256",
    )
}

#: Sources that produced the offline artifacts but are in **no** part of the
#: solver pin, so this script is structurally blind to them.  They are digested
#: at both revisions and reported, because "only the pin's own declaration half
#: moved" was read as a statement about the whole producing lineage and it is
#: not one: the 2026-08-06 and 2026-08-07 N1 tapes were built on two different
#: ``training_contract.py`` (8c9eec4c vs c6608440) and this script could not see
#: it.  ``training_contract.py`` does not belong in the solver semantic surface
#: -- it names no question and computes no answer, and the solver never imports
#: it -- it belongs where it already is: the fixed-tape producer contract's
#: ``implementation_source_sha256``, which is what actually moved with it.
PRODUCER_LINEAGE_SOURCES = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/utils/training_contract.py",
    "hope_training/whole_body_tracking/scripts/"
    "materialize_action_ball_n1_fixed_tape_variants.py",
    "hope_training/whole_body_tracking/scripts/"
    "materialize_measured_action_ball_n1_bundle.py",
    "hope_training/whole_body_tracking/scripts/"
    "pin_action_ball_profile_contracts.py",
)


class MigrationRefused(SystemExit):
    pass


def _git_blob_bytes(repo_root: Path, rev: str, relative: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{rev}:{relative}"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise MigrationRefused(
            f"cannot read {relative} at {rev}: "
            + result.stderr.decode("utf-8", "replace").strip()
        )
    return result.stdout


def _resolve_commit(repo_root: Path, rev: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--verify", f"{rev}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_surface_module(repo_root: Path, rev: str | None):
    relative = f"{MDP_REL}/{SURFACE_SOURCE}"
    raw = (
        _git_blob_bytes(repo_root, rev, relative)
        if rev
        else (repo_root / relative).read_bytes()
    )
    module = types.ModuleType("_migrate_action_ball_solver_semantic_surface")
    module.__file__ = f"{rev or 'worktree'}:{relative}"
    sys.modules[module.__name__] = module
    exec(compile(raw, module.__file__, "exec"), module.__dict__)
    return module


def _reader(repo_root: Path, rev: str | None):
    cache: dict = {}

    def read(filename: str) -> str:
        if filename not in cache:
            relative = f"{MDP_REL}/{filename}"
            raw = (
                _git_blob_bytes(repo_root, rev, relative)
                if rev
                else (repo_root / relative).read_bytes()
            )
            cache[filename] = raw.decode("utf-8")
        return cache[filename]

    return read


def _all_digests(surface, read) -> dict:
    return {
        name: surface.symbol_digests(read(name), filename=name)
        for name in surface.PINNED_SOURCES
    }


def _run_pinner(repo_root: Path, rev: str | None, out: Path) -> dict:
    command = [
        sys.executable,
        str(PINNER),
        "--repo-root",
        str(repo_root),
        "--out",
        str(out),
    ]
    if rev:
        command.extend(["--source-rev", rev])
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise MigrationRefused(
            "the pinner refused to mint the v3 profile:\n" + result.stderr
        )
    return json.loads(out.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    parser.add_argument(
        "--pins",
        required=True,
        help="the live v2 profile-pins document this lineage was launched with",
    )
    parser.add_argument(
        "--from-rev",
        required=True,
        help="the revision the v2 pin was minted from",
    )
    parser.add_argument(
        "--to-rev",
        default=None,
        help="the revision the v3 pin is minted from (default: the worktree)",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="a manifest whose solver_profile_sha256 carries the stale pin",
    )
    parser.add_argument("--out", required=True, help="write the migration receipt here")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    from_rev = _resolve_commit(repo_root, args.from_rev)
    to_rev = _resolve_commit(repo_root, args.to_rev) if args.to_rev else None

    old_document = json.loads(Path(args.pins).read_text(encoding="utf-8"))
    old_payload = old_document.get("solver_payload")
    if not isinstance(old_payload, dict):
        raise MigrationRefused("profile pins carry no solver payload")
    if old_payload.get("schema_version") != 2:
        raise MigrationRefused(
            "this migration only converts solver profile schema v2; found "
            f"{old_payload.get('schema_version')!r}"
        )
    if "implementation_source_sha256" not in old_payload:
        raise MigrationRefused(
            "the v2 payload must carry the whole-file map it is being migrated "
            "away from"
        )
    old_pin = old_document.get("solver_profile_sha256")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_document = _run_pinner(
        repo_root, to_rev, out_path.with_suffix(".pins.json")
    )
    new_payload = new_document["solver_payload"]
    if new_payload.get("schema_version") != 3:
        raise MigrationRefused(
            "the re-minted profile is not schema v3; nothing to migrate to"
        )
    new_pin = new_document["solver_profile_sha256"]

    # ---- Evidence 1: the ball physics and the exact-face geometry did not move.
    if (
        new_document["physics_profile_sha256"]
        != old_document["physics_profile_sha256"]
    ):
        raise MigrationRefused(
            "physics profile moved; this is not a pin migration, it is a "
            "different simulation"
        )
    if (
        new_document["contact_geometry"]["sha256"]
        != old_document["contact_geometry"]["sha256"]
    ):
        raise MigrationRefused(
            "exact-face contact geometry moved; this is not a pin migration"
        )

    # ---- Evidence 2: every symbol that decides a question or an answer is
    # byte-identical between the revision the old pin was minted from and the
    # revision the new pin is minted from.
    surface = _load_surface_module(repo_root, to_rev)
    old_digests = _all_digests(surface, _reader(repo_root, from_rev))
    new_digests = _all_digests(surface, _reader(repo_root, to_rev))

    moved = {}
    absent = {}
    introduced = {}
    for filename, covered in surface.COVERED.items():
        allowed_new = ALLOWED_INTRODUCED_SYMBOLS.get(filename, frozenset())
        for symbol in covered:
            before = old_digests.get(filename, {}).get(symbol)
            after = new_digests.get(filename, {}).get(symbol)
            if after is None:
                absent.setdefault(filename, []).append(symbol)
            elif before is None:
                if symbol in allowed_new:
                    introduced.setdefault(filename, []).append(symbol)
                else:
                    absent.setdefault(filename, []).append(symbol)
            elif before != after:
                moved.setdefault(filename, []).append(symbol)
    if absent:
        raise MigrationRefused(
            "a covered symbol is missing at one of the two revisions and is not "
            "on the allowed-introduction list, so the invariance claim cannot be "
            f"checked: {absent}"
        )

    unexpected = {
        filename: sorted(
            set(symbols) - ALLOWED_MOVED_SYMBOLS.get(filename, frozenset())
        )
        for filename, symbols in moved.items()
    }
    unexpected = {name: rows for name, rows in unexpected.items() if rows}
    if unexpected:
        raise MigrationRefused(
            "this is not a re-signing: the following covered symbols changed "
            "between the two revisions, so the questions or the answers may "
            f"have changed too: {json.dumps(unexpected, indent=1, sort_keys=True)}"
        )

    for filename, symbols in QUESTION_IDENTITY_SYMBOLS.items():
        for symbol in symbols:
            if old_digests[filename][symbol] != new_digests[filename][symbol]:
                raise MigrationRefused(
                    f"the question identity function {filename}:{symbol} "
                    "changed; every question would be renamed"
                )

    # ---- Evidence 2b: what this comparison structurally could NOT see.
    # A receipt that reports only what it checked reads as if it checked
    # everything.  These two blocks are the negative space, measured rather than
    # asserted: the excluded symbols (deliberately outside the surface) and the
    # producing tools (outside every part of the solver pin).
    excluded_not_compared = {
        filename: sorted(symbols)
        for filename, symbols in sorted(surface.EXCLUDED.items())
    }
    producer_lineage = {}
    for relative in PRODUCER_LINEAGE_SOURCES:
        try:
            before = hashlib.sha256(
                _git_blob_bytes(repo_root, from_rev, relative)
            ).hexdigest()
        except MigrationRefused:
            before = None
        if to_rev is None:
            path = repo_root / relative
            after = (
                hashlib.sha256(path.read_bytes()).hexdigest()
                if path.is_file()
                else None
            )
        else:
            try:
                after = hashlib.sha256(
                    _git_blob_bytes(repo_root, to_rev, relative)
                ).hexdigest()
            except MigrationRefused:
                after = None
        producer_lineage[relative] = {
            "file_sha256_at_from_rev": before,
            "file_sha256_at_to_rev": after,
            "moved": before != after,
        }
    moved_producers = sorted(
        name for name, row in producer_lineage.items() if row["moved"]
    )

    # ---- Evidence 3: what actually has to be re-signed.
    stale_manifests = []
    for path in args.manifest:
        manifest_path = Path(path)
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        carried = document.get("solver_profile_sha256")
        if carried != old_pin:
            raise MigrationRefused(
                f"{manifest_path} carries {carried!r}, not the stale pin "
                f"{old_pin!r}; refusing to migrate a manifest this receipt does "
                "not describe"
            )
        try:
            relative = manifest_path.resolve().relative_to(repo_root).as_posix()
        except ValueError:
            relative = str(manifest_path)
        stale_manifests.append(
            {
                "path": relative,
                "file_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
                "solver_profile_sha256_before": carried,
                "solver_profile_sha256_after": new_pin,
            }
        )

    receipt = {
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "kind": MIGRATION_KIND,
        "from_rev": from_rev,
        "to_rev": to_rev or "worktree",
        "solver_profile_schema_version_before": 2,
        "solver_profile_schema_version_after": 3,
        "solver_profile_sha256_before": old_pin,
        "solver_profile_sha256_after": new_pin,
        "pin_definition_before": (
            "sha256 of a payload whose implementation_source_sha256 is the "
            "whole-file SHA-256 of five solver sources"
        ),
        "pin_definition_after": (
            "sha256 of a payload whose semantic_surface.sha256 is the digest of "
            "the docstring-stripped AST of every covered symbol across six "
            "solver sources, with an explicit reasoned exclusion list and two "
            "fail-closed coverage gates"
        ),
        "semantic_surface": {
            "sha256": new_document["solver_semantic_surface"]["sha256"],
            "covered_symbol_count": new_payload["semantic_surface"][
                "covered_symbol_count"
            ],
            "pinned_sources": new_payload["semantic_surface"]["pinned_sources"],
        },
        "invariance_evidence": {
            "physics_profile_sha256": new_document["physics_profile_sha256"],
            "contact_geometry_sha256": new_document["contact_geometry"]["sha256"],
            "covered_symbols_compared": sum(
                len(symbols) for symbols in surface.COVERED.values()
            ),
            "covered_symbols_moved": {
                name: sorted(symbols) for name, symbols in sorted(moved.items())
            },
            "covered_symbols_introduced": {
                name: sorted(symbols)
                for name, symbols in sorted(introduced.items())
            },
            "question_identity_symbols_unchanged": {
                filename: {
                    symbol: new_digests[filename][symbol] for symbol in symbols
                }
                for filename, symbols in QUESTION_IDENTITY_SYMBOLS.items()
            },
            "claim": (
                "Scoped to the symbols this surface COVERS: only the pin's own "
                "declaration half moved, every other covered symbol that names a "
                "question or computes an answer is identical at both revisions "
                "(covered_symbols_introduced lists the ones that exist only at "
                "to_rev and were allowed in by name), and the ball physics and "
                "exact-face geometry digests are unchanged. This migration "
                "re-signs the lineage; it does not re-draw it."
            ),
            "not_claimed": [
                "Nothing about the symbols listed under "
                "coverage_this_receipt_does_not_check.excluded_symbols_not_"
                "compared: they are outside the surface by design, so this "
                "receipt neither compared them nor certifies them.",
                "Nothing about the producing toolchain listed under "
                "producer_lineage_outside_this_pin: those files are in no part "
                "of the solver pin, so a change there is invisible to every "
                "check above. Read that block before treating this receipt as a "
                "statement that 'the code was the same'.",
                "Nothing about canonical_sha256 of any artifact: it seals the "
                "solver pin, so it MUST move. Quoting it as evidence of "
                "sameness quotes the wrong field.",
            ],
        },
        "coverage_this_receipt_does_not_check": {
            "excluded_symbols_not_compared": excluded_not_compared,
            "excluded_symbol_count": sum(
                len(symbols) for symbols in excluded_not_compared.values()
            ),
            "why": (
                "The surface deliberately excludes these by name and reason so "
                "that checkpoint/ledger/telemetry commits do not invalidate a "
                "running curriculum. That exemption is exactly why they are not "
                "evidence here."
            ),
        },
        "producer_lineage_outside_this_pin": {
            "sources": producer_lineage,
            "moved": moved_producers,
            "why": (
                "These files produced the offline artifacts but are in no part "
                "of the solver pin, so every comparison above is structurally "
                "blind to them. training_contract.py in particular is already "
                "pinned where it belongs -- the fixed-tape producer contract's "
                "implementation_source_sha256 -- and it is NOT a candidate for "
                "the solver semantic surface: it names no question, computes no "
                "answer, is not mentioned at all by the five pure solver "
                "sources, and is not reachable from any covered symbol of "
                "hope_commands.py (which does own an unrelated task-first "
                "training-contract builder, so the narrow claim is the true one)."
            ),
        },
        "stale_manifests": stale_manifests,
        "content_addressed_artifacts_still_to_re_sign": (
            "This receipt does not rewrite content-addressed artifacts "
            "(task receipts, immutable tape, prototype, manifest, bundle, "
            "lineage): their file names carry their own digests, so they must "
            "be re-materialised by the offline pipeline against the new pin. "
            "Their physical question identities do not move -- that is what the "
            "invariance evidence above certifies."
        ),
    }
    encoded = json.dumps(receipt, indent=1, sort_keys=True)
    out_path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if moved_producers:
        # Not a refusal: these files are not this pin's business.  But a
        # difference nobody reads is the same disease as a difference nobody
        # records, so it goes to stderr where a launch summary will pick it up.
        print(
            "NOTE: the producing toolchain is NOT identical at the two "
            "revisions, and no check in this receipt covers it: "
            + ", ".join(moved_producers),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
