#!/usr/bin/env bash
# =============================================================================
# judge.sh — 判卷链单命令入口(标准入口;runbook「判卷链」手动工序收编,2026-07-09)
#
# 人话:给一个训练 run 目录,一条命令出考卷 md 报告。
#   ①从 run_dir/params/env.yaml 自动解析该臂的动作对/相位/题库(推导同源 exam 卷:
#     train.npz -> exam.npz;解析不到就 fail-loud 要求手传,绝不静默用默认值——hopex 默认相位坑)
#   ②isaac venv 里 play.py 原生导出 ONNX(认 "Exported ONNX policy to:" 成功行;
#     02196d2 后导出成功仍有无害 tuple Traceback,不许拿它判死)+ make_std_sidecar 补两个 sidecar
#   ③mjeval venv 里 mujoco_eval_onnx.py 考卷,双侧 × 双噪声档(ns=0 / 0.05)一次跑完,
#     默认 --qdes-clamp 开 + --hold-ref stand(可用 --no-qdes-clamp / --hold-ref clip 覆盖)
#   ④统一 md 报告(报告头=旗标状态+分母报表连抄,2×2 表:侧×噪声档)落到 run_dir/judge/
#
# 用法:
#   bash scripts/judge.sh <run_dir> [checkpoint.pt] [选项]
#     checkpoint 缺省 = run_dir 里编号最大的 model_*.pt
#
# 选项:
#   --dry-run                 只打印将执行的命令链(机制检查用),不动 GPU 不写文件
#   --gpu N                   导出用哪张卡(缺省=自动挑显存余量最大的卡;错峰规矩见下)
#   --steps N                 考卷步数(缺省 3000)
#   --seed N                  考卷种子(缺省 0)
#   --noise-scales "A B"      噪声档(缺省 "0.0 0.05")
#   --no-qdes-clamp           关限位剪切(缺省开;报告头照抄状态)
#   --hold-ref clip|stand     等球段参考语义(缺省 stand=07-05 后代际;老代际传 clip)
#   --motion-files FH BH      手传动作对(env.yaml 解析不到时必传)
#   --strike-phases A B       手传相位对(env.yaml 解析不到时必传)
#   --exam-bank PATH          手传 exam 卷(env.yaml 解析不到 question_bank 时必传)
#   --task NAME               手传任务名(cfg/task/<NAME>.yaml;缺省按 experiment_name 反查)
#   --export-extra "..."      追加给 play.py 的 hydra 覆盖(特殊臂:R9 三件套/腿遮蔽等)
#   --exam-extra "..."        追加给 mujoco_eval_onnx.py 的参数
#
# 环境变量(pod 缺省;别的机器可覆盖):
#   JUDGE_ISAAC_ENV      isaac venv 入口脚本(缺省 /workspace/franco/env.sh)
#   JUDGE_MJEVAL_ACT     mjeval venv activate(缺省 /workspace/hope_mjeval_venv/bin/activate)
#
# 铁律(都付过学费,见 docs/runbook.md「判卷链」):
#   - 严禁 kill 训练进程:本脚本只杀自己 setsid 出来的 play.py 进程组,别的谁都不碰。
#   - 导出错峰由调用方保证:一次只跑一个 judge.sh,不与其他 Isaac 同秒点火(撞 CUDA 枚举)。
#   - 每 checkpoint 必须重导出 + 重做 sidecar(先清旧 exported/,勿复用陈旧工件)。
#   - isaac venv 没有 onnxruntime,考卷必须走 mjeval venv;反之 sidecar 需要 torch 走 isaac venv。
#   - 原型=pod /workspace/franco/s1_wave4/judge_final.sh(巡检班第 3 遍手搓,坑已踩平)。
# =============================================================================
set -u -o pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WBT_DIR=$(dirname "$SCRIPT_DIR")

JUDGE_ISAAC_ENV=${JUDGE_ISAAC_ENV:-/workspace/franco/env.sh}
JUDGE_MJEVAL_ACT=${JUDGE_MJEVAL_ACT:-/workspace/hope_mjeval_venv/bin/activate}

die() { echo "[judge][FATAL] $*" >&2; exit 1; }
note() { echo "[judge] $*"; }

# ---------------------------------------------------------------- 参数解析
RUN_DIR=""; CKPT=""
DRY_RUN=0; GPU=""; STEPS=3000; SEED=0; NOISE_SCALES="0.0 0.05"
QDES_CLAMP=1; HOLD_REF="stand"; TASK=""
CLI_FH=""; CLI_BH=""; CLI_PH0=""; CLI_PH1=""; CLI_EXAM_BANK=""
EXPORT_EXTRA=""; EXAM_EXTRA=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)        DRY_RUN=1; shift ;;
    --gpu)            GPU=${2:?--gpu 需要参数}; shift 2 ;;
    --steps)          STEPS=${2:?}; shift 2 ;;
    --seed)           SEED=${2:?}; shift 2 ;;
    --noise-scales)   NOISE_SCALES=${2:?}; shift 2 ;;
    --no-qdes-clamp)  QDES_CLAMP=0; shift ;;
    --hold-ref)       HOLD_REF=${2:?}; shift 2 ;;
    --motion-files)   CLI_FH=${2:?}; CLI_BH=${3:?--motion-files 需要 FH BH 两个路径}; shift 3 ;;
    --strike-phases)  CLI_PH0=${2:?}; CLI_PH1=${3:?--strike-phases 需要两个相位}; shift 3 ;;
    --exam-bank)      CLI_EXAM_BANK=${2:?}; shift 2 ;;
    --task)           TASK=${2:?}; shift 2 ;;
    --export-extra)   EXPORT_EXTRA=${2:?}; shift 2 ;;
    --exam-extra)     EXAM_EXTRA=${2:?}; shift 2 ;;
    -h|--help)        sed -n '2,45p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*)               die "未知选项 $1(--help 看用法)" ;;
    *)  if [ -z "$RUN_DIR" ]; then RUN_DIR=$1
        elif [ -z "$CKPT" ]; then CKPT=$1
        else die "多余的位置参数 $1"; fi; shift ;;
  esac
done

[ -n "$RUN_DIR" ] || die "用法: judge.sh <run_dir> [checkpoint.pt] [选项](--help 看全部)"
RUN_DIR=$(cd "$RUN_DIR" 2>/dev/null && pwd) || die "run_dir 不存在: $RUN_DIR"
case "$HOLD_REF" in clip|stand) ;; *) die "--hold-ref 只认 clip|stand" ;; esac

# checkpoint 缺省 = 编号最大的 model_*.pt(rsl_rl 末迭代 0 起数:17000 轮的终档是 model_16999)
if [ -z "$CKPT" ]; then
  CKPT=$(ls "$RUN_DIR"/model_*.pt 2>/dev/null \
        | awk -F'model_' '{n=$NF; sub(/\.pt$/,"",n); printf "%012d %s\n", n, $0}' \
        | sort | tail -1 | cut -d' ' -f2-)
  [ -n "$CKPT" ] || die "run_dir 里没有 model_*.pt: $RUN_DIR"
fi
[ -f "$CKPT" ] || die "checkpoint 不存在: $CKPT"
CKPT_TAG=$(basename "$CKPT" .pt)
RUN_NAME=$(basename "$RUN_DIR")

STAMP=$(date -u +%Y%m%d_%H%M%S)
JUDGE_DIR="$RUN_DIR/judge/${CKPT_TAG}_${STAMP}"
REPORT="$RUN_DIR/judge/judge_report_${CKPT_TAG}_${STAMP}.md"
ONNX="$RUN_DIR/exported/policy.onnx"
STD_NPY="$RUN_DIR/exported/learned_std.npy"
OBS_NORM="$RUN_DIR/exported/obs_norm.npz"

# ---------------------------------------------------------------- ① 解析 env.yaml(fail-loud)
ENV_YAML="$RUN_DIR/params/env.yaml"
[ -f "$ENV_YAML" ] || die "缺 $ENV_YAML —— 没有配置底稿无法保证导出与训练同配置,不判"

PARSED=$(mktemp "${TMPDIR:-/tmp}/judge_parsed.XXXXXX.sh") || die "mktemp 失败"
trap 'rm -f "$PARSED"' EXIT

python3 - "$ENV_YAML" "$PARSED" "$CLI_FH" "$CLI_BH" "$CLI_PH0" "$CLI_PH1" "$CLI_EXAM_BANK" <<'PYEOF' || die "env.yaml 解析失败(上面有缺什么、该手传哪个旗标)"
import os, shlex, sys
import yaml

env_yaml, out_path, cli_fh, cli_bh, cli_ph0, cli_ph1, cli_bank = sys.argv[1:8]

# env.yaml 是 env_cfg 的 dump,带 !!python/tuple / !!python/object 标签 —— 宽松加载,只取数据
class Loose(yaml.SafeLoader):
    pass
Loose.add_constructor("tag:yaml.org,2002:python/tuple",
                      lambda l, n: tuple(l.construct_sequence(n)))
def _any_py(loader, suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_scalar(node)
Loose.add_multi_constructor("tag:yaml.org,2002:python/", _any_py)

with open(env_yaml) as f:
    cfg = yaml.load(f, Loader=Loose)

cmds = cfg.get("commands") or {}
motion = cmds.get("motion") or {}
rt = cmds.get("racket_target") or {}
fatal = []

# --- 动作对(必须成对;单 clip 臂不是本考卷的形状) ---
if cli_fh and cli_bh:
    fh, bh, mf_src = cli_fh, cli_bh, "CLI 手传"
else:
    mf = motion.get("motion_file")
    if isinstance(mf, str):
        fatal.append(f"motion_file 只有单 clip({mf})—— 双侧考卷要动作对;确认后用 --motion-files FH BH 手传")
        fh = bh = None
    elif isinstance(mf, (list, tuple)) and len(mf) == 2:
        fh, bh, mf_src = str(mf[0]), str(mf[1]), "env.yaml 自动解析"
    else:
        fatal.append("env.yaml 里解析不到 commands.motion.motion_file 动作对 —— 用 --motion-files FH BH 手传")
        fh = bh = None

# --- 相位对(hopex 默认相位坑:绝不静默用默认值) ---
if cli_ph0 and cli_ph1:
    ph, ph_src = (float(cli_ph0), float(cli_ph1)), "CLI 手传"
else:
    spc = rt.get("strike_phase_per_clip")
    if isinstance(spc, (list, tuple)) and len(spc) == 2:
        ph, ph_src = (float(spc[0]), float(spc[1])), "env.yaml 自动解析"
    else:
        fatal.append("env.yaml 里解析不到 commands.racket_target.strike_phase_per_clip —— "
                     "用 --strike-phases A B 手传(题库只管目标不管时机,默认值是 hopex 的,错相位=分数静默塌 0)")
        ph = None

# --- 题库 -> 同源 exam 卷(train.npz -> exam.npz) ---
if cli_bank:
    exam_bank, bank_src, train_bank = cli_bank, "CLI 手传", ""
else:
    train_bank = rt.get("question_bank") or ""
    if not str(train_bank).strip():
        fatal.append("env.yaml 里解析不到 commands.racket_target.question_bank(老代际臂?)—— "
                     "用 --exam-bank <exam npz> 手传同源考卷")
        exam_bank, bank_src = None, None
    else:
        train_bank = str(train_bank)
        if "_train.npz" not in os.path.basename(train_bank):
            fatal.append(f"question_bank 文件名不含 _train.npz({train_bank}),推导不出同源 exam 卷 —— "
                         "用 --exam-bank 手传")
            exam_bank, bank_src = None, None
        else:
            exam_bank = os.path.join(os.path.dirname(train_bank),
                                     os.path.basename(train_bank).replace("_train.npz", "_exam.npz"))
            bank_src = "env.yaml 自动推导(train->exam)"

# --- 存在性检查(dry-run 也查:机制检查就该在这里翻车) ---
for p, what in ((fh, "正手动作"), (bh, "反手动作"), (exam_bank, "exam 卷")):
    if p and not os.path.isfile(p):
        fatal.append(f"{what}文件不存在: {p}")

if fatal:
    print("[judge][FATAL] env.yaml 自动解析不完整,拒绝静默用默认值:", file=sys.stderr)
    for m in fatal:
        print(f"  - {m}", file=sys.stderr)
    sys.exit(1)

# --- 导出用 hydra 覆盖:把训练时生效的关键旗标从 env.yaml 原值搬回 play.py ---
# 关键度分两档:
#   必需 = 影响 ONNX 观测维度/元数据(motion_file、strike_phase_per_clip、face_command_obs)
#   陪跑 = 让导出环境与训练同构(bank/噪声/延迟/自旋等;值取 env.yaml 实测,不抄任何波次的默认)
def fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    return repr(v) if isinstance(v, float) else str(v)

ov = []
ov.append("'motion_file=[%s,%s]'" % (fh, bh))
ov.append("'++task.racket.strike_phase_per_clip=[%s,%s]'" % (repr(ph[0]), repr(ph[1])))
for key, src in (
    ("question_bank",          rt.get("question_bank")),
    ("face_command",           rt.get("face_command")),
    ("vb_spin_mode",           rt.get("vb_spin_mode")),
    ("adaptive_sigma",         rt.get("adaptive_sigma")),
    ("target_delay_steps",     rt.get("target_delay_steps")),
    ("target_noise_white",     rt.get("target_noise_white")),
    ("target_noise_ar1_sigma", rt.get("target_noise_ar1_sigma")),
    ("target_noise_ar1_rho",   rt.get("target_noise_ar1_rho")),
    ("midswing_resample_prob", rt.get("midswing_resample_prob")),
):
    if src is not None:
        ov.append(f"++task.racket.{key}={fmt(src)}")
if motion.get("post_swing_start_prob") is not None:
    ov.append(f"++task.motion.post_swing_start_prob={fmt(motion['post_swing_start_prob'])}")
if cfg.get("face_command_obs"):          # 顶层旗标:+4 obs 维(175->179),导错=checkpoint 加载即死
    ov.append("++task.racket.face_command_obs=true")
if cfg.get("station_obs"):               # 顶层旗标:+2 obs 维(179->181,R10c 站位锚),导错同上即死
    ov.append("++task.racket.station_obs=true")
    sax = rt.get("station_anchor_offset_xy")
    if sax is not None and [float(sax[0]), float(sax[1])] != [0.0, 0.0]:
        # 陪跑档:锚点偏移只影响观测数值不影响维度,搬回去保持导出环境与训练同构
        ov.append("'++task.racket.station_anchor_offset_xy=[%s,%s]'"
                  % (repr(float(sax[0])), repr(float(sax[1]))))
if cfg.get("physical_ball"):
    ov.append("++task.physical_ball=true")
ov.append("task.actor_obs_contract=null")   # 扩列观测放行(179 臂;runbook 发射核对单第 5 条)

with open(out_path, "w") as f:
    def w(k, v):
        f.write(f"{k}={shlex.quote(str(v))}\n")
    w("MOTION_FH", fh); w("MOTION_BH", bh)
    w("PH0", repr(ph[0])); w("PH1", repr(ph[1]))
    w("EXAM_BANK", exam_bank); w("TRAIN_BANK", train_bank or "")
    w("SRC_MOTION", mf_src); w("SRC_PHASES", ph_src); w("SRC_BANK", bank_src)
    w("EXPORT_OVERRIDES", " ".join(ov))
print("[judge] env.yaml 解析 OK")
PYEOF

# shellcheck disable=SC1090
. "$PARSED"

# ---------------------------------------------------------------- 任务名反查(experiment_name -> cfg/task/*.yaml)
EXP_NAME=$(basename "$(dirname "$RUN_DIR")")
if [ -z "$TASK" ]; then
  for f in "$WBT_DIR"/cfg/task/*.yaml; do
    if grep -q "^experiment_name: ${EXP_NAME}\$" "$f"; then TASK=$(basename "$f" .yaml); break; fi
  done
  [ -n "$TASK" ] || die "experiment_name=${EXP_NAME} 在 cfg/task/*.yaml 里反查不到任务名 —— 用 --task 手传"
fi

# ---------------------------------------------------------------- 挑卡(显存余量最大;错峰规矩见头注)
if [ -z "$GPU" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
          | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
    note "自动挑卡:gpu$GPU(显存余量最大;$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr '\n' ' '))"
  else
    [ "$DRY_RUN" = 1 ] || die "没有 nvidia-smi,导出无卡可挑 —— 用 --gpu N 手传"
    GPU="<运行时自动挑显存余量最大的卡>"
  fi
fi

CLAMP_FLAG=""; [ "$QDES_CLAMP" = 1 ] && CLAMP_FLAG="--qdes-clamp"
QB_PY="$WBT_DIR/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/stage1_question_bank.py"
[ -f "$QB_PY" ] || die "缺 stage1_question_bank.py: $QB_PY"

EXPORT_CMD="cd '$WBT_DIR' && CUDA_VISIBLE_DEVICES=$GPU python scripts/play.py task=$TASK algo=ppo headless=true num_envs=2 checkpoint='$CKPT' $EXPORT_OVERRIDES $EXPORT_EXTRA"
SIDECAR_CMD="python scripts/make_std_sidecar.py --checkpoint '$CKPT'"
EXAM_CMD="python scripts/mujoco_eval_onnx.py \
  --onnx '$ONNX' --std '$STD_NPY' \
  --motion-files '$MOTION_FH' '$MOTION_BH' \
  --strike-phase-per-clip $PH0 $PH1 \
  --target-source bank --exam-bank '$EXAM_BANK' \
  $CLAMP_FLAG --hold-ref $HOLD_REF \
  --noise-scales $NOISE_SCALES --steps $STEPS --seed $SEED --out-dir '$JUDGE_DIR/exam' $EXAM_EXTRA"

note "run_dir     = $RUN_DIR"
note "checkpoint  = $CKPT"
note "task        = $TASK(experiment_name=$EXP_NAME 反查)"
note "动作对      = $MOTION_FH | $MOTION_BH($SRC_MOTION)"
note "相位对      = [$PH0, $PH1]($SRC_PHASES)"
note "exam 卷     = $EXAM_BANK($SRC_BANK)"
note "旗标        = qdes_clamp=$([ "$QDES_CLAMP" = 1 ] && echo ON || echo OFF) hold_ref=$HOLD_REF steps=$STEPS seed=$SEED ns=[$NOISE_SCALES]"

if [ "$DRY_RUN" = 1 ]; then
  echo
  echo "===== DRY-RUN:将执行的命令链(不动 GPU 不写文件)====="
  echo "# ② 导出(isaac venv,gpu 占槽 ~4 分钟;认 'Exported ONNX policy to:' 行)"
  echo "source $JUDGE_ISAAC_ENV && export HYDRA_FULL_ERROR=1"
  echo "rm -rf '$RUN_DIR/exported'   # 每 checkpoint 必重导,勿复用陈旧工件"
  echo "setsid bash -c \"$EXPORT_CMD\" > $JUDGE_DIR/export_play.log 2>&1 &"
  echo "#   (等 $ONNX 落盘 + 成功行,然后只杀该 setsid 进程组;tuple Traceback=已知无害)"
  echo "$SIDECAR_CMD   # -> learned_std.npy + obs_norm.npz"
  echo
  echo "# ③ 考卷(mjeval venv,纯 CPU)"
  echo "source $JUDGE_MJEVAL_ACT && export HOPE_STAGE1_QB=$QB_PY"
  echo "cd '$WBT_DIR' && $EXAM_CMD > $JUDGE_DIR/exam.log 2>&1"
  echo
  echo "# ④ 报告 -> $REPORT"
  exit 0
fi

mkdir -p "$JUDGE_DIR/exam"

# ---------------------------------------------------------------- ② 原生导出 + sidecar(isaac venv)
[ -f "$JUDGE_ISAAC_ENV" ] || die "isaac venv 入口不存在: $JUDGE_ISAAC_ENV(JUDGE_ISAAC_ENV 可覆盖)"
note "② play.py 原生导出(isaac venv,gpu$GPU)…"
rm -rf "$RUN_DIR/exported"
(
  # shellcheck disable=SC1090
  source "$JUDGE_ISAAC_ENV" >/dev/null 2>&1
  export HYDRA_FULL_ERROR=1
  setsid bash -c "$EXPORT_CMD" > "$JUDGE_DIR/export_play.log" 2>&1 &
  PG=$!
  echo "$PG" > "$JUDGE_DIR/export.pgid"
  for i in $(seq 1 110); do
    [ -f "$ONNX" ] && { echo "[judge] onnx 落盘 ~$((i*5))s"; sleep 8; break; }
    kill -0 "$PG" 2>/dev/null || { echo "[judge] play.py 提前退出"; break; }
    sleep 5
  done
  # 只杀自己 setsid 出来的进程组;严禁碰训练进程
  kill -TERM -"$PG" 2>/dev/null; sleep 3; kill -KILL -"$PG" 2>/dev/null
  exit 0
)
if ! grep -aq "Exported ONNX policy to:" "$JUDGE_DIR/export_play.log"; then
  echo "[judge][FATAL] 导出失败:没等到 'Exported ONNX policy to:' 成功行" >&2
  echo "----- export_play.log 异常摘要(WARN|Error|Traceback)-----" >&2
  grep -aE "WARN|Error|Traceback" "$JUDGE_DIR/export_play.log" | head -20 >&2
  tail -20 "$JUDGE_DIR/export_play.log" >&2
  exit 1
fi
[ -f "$ONNX" ] || die "有成功行但 $ONNX 不存在?检查 $JUDGE_DIR/export_play.log"
if grep -aq "AttributeError: 'tuple' object has no attribute 'to'" "$JUDGE_DIR/export_play.log"; then
  note "导出日志里的 tuple Traceback=已知无害(02196d2,发生在导出成功之后的 play 循环)"
fi
note "onnx OK: $ONNX"

note "make_std_sidecar(isaac venv)…"
(
  # shellcheck disable=SC1090
  source "$JUDGE_ISAAC_ENV" >/dev/null 2>&1
  cd "$WBT_DIR" && eval "$SIDECAR_CMD"
) > "$JUDGE_DIR/sidecar.log" 2>&1 || { cat "$JUDGE_DIR/sidecar.log" >&2; die "sidecar 失败"; }
grep -a "\[make_std_sidecar\]" "$JUDGE_DIR/sidecar.log"
[ -f "$STD_NPY" ] || die "缺 $STD_NPY"
[ -f "$OBS_NORM" ] || note "⚠ 没生成 obs_norm.npz(仅 empirical_normalization=false 的臂才正常;报告里带出)"

# ---------------------------------------------------------------- ③ 考卷(mjeval venv,纯 CPU)
[ -f "$JUDGE_MJEVAL_ACT" ] || die "mjeval venv 不存在: $JUDGE_MJEVAL_ACT(JUDGE_MJEVAL_ACT 可覆盖)"
note "③ mujoco_eval_onnx 考卷(mjeval venv,ns=[$NOISE_SCALES] × 双侧,steps=$STEPS)…"
(
  # shellcheck disable=SC1090
  source "$JUDGE_MJEVAL_ACT"
  export HOPE_STAGE1_QB="$QB_PY"
  cd "$WBT_DIR" && eval "$EXAM_CMD"
) > "$JUDGE_DIR/exam.log" 2>&1
EXAM_RC=$?
if [ "$EXAM_RC" != 0 ]; then
  echo "[judge][FATAL] 考卷 rc=$EXAM_RC" >&2
  grep -aE "FATAL|Error|Traceback" "$JUDGE_DIR/exam.log" | head -20 >&2
  tail -20 "$JUDGE_DIR/exam.log" >&2
  exit 1
fi
note "考卷 OK(rc=0)"

# ---------------------------------------------------------------- ④ md 报告
python3 - "$JUDGE_DIR/exam.log" "$REPORT" <<PYEOF || die "报告生成失败(exam.log 在 $JUDGE_DIR/exam.log)"
import re, sys

log_path, report_path = sys.argv[1], sys.argv[2]
lines = open(log_path, encoding="utf-8", errors="replace").read().splitlines()

def grab1(pat, required=True):
    for ln in lines:
        if re.search(pat, ln):
            return ln
    if required:
        raise SystemExit(f"[judge][FATAL] exam.log 里找不到 /{pat}/ —— 评估器输出格式变了?报告拒绝猜数")
    return None

# 列头(噪声档标签)取自评估器自己打的表头
hdr = grab1(r"^metric\s+ns=")
ns_labels = hdr.split()[1:]

def row_vals(label, section_start=None, section_end=None):
    """按行首标签取一行的各噪声档数值(评估器 28+16N 定宽表)。"""
    lo, hi = 0, len(lines)
    if section_start is not None:
        for i, ln in enumerate(lines):
            if ln.startswith(section_start):
                lo = i; break
        else:
            raise SystemExit(f"[judge][FATAL] exam.log 缺段落 {section_start!r}")
        if section_end is not None:
            for j in range(lo + 1, len(lines)):
                if lines[j].startswith(section_end):
                    hi = j; break
    for ln in lines[lo:hi]:
        if ln.startswith(label) and len(ln) > 28 and ln[len(label)] == " ":
            vals = ln[28:].split()
            if len(vals) == len(ns_labels):
                return vals
    raise SystemExit(f"[judge][FATAL] exam.log 段落 {section_start!r} 里找不到行 {label!r}")

flag_line   = grab1(r"MuJoCo sim-to-sim \|")
phase_line  = grab1(r"strike_phase_per_clip in effect|OVERRIDE strike_phase_per_clip")
obsnorm_line = grab1(r"obs normalization")
clamp_line  = grab1(r"q_des clamp", required=False)
reset_line  = grab1(r"reset mode:")
bank_line   = grab1(r"\[mj-sim2sim\]   bank: ")

# 分母报表(末尾 DENOMINATORS 块连抄)
den = []
for i, ln in enumerate(lines):
    if ln.startswith("DENOMINATORS"):
        for j in range(i, len(lines)):
            if lines[j].startswith("=" * 20):
                break
            den.append(lines[j])
        break
if not den:
    raise SystemExit("[judge][FATAL] exam.log 里没有 DENOMINATORS 分母报表(bank 考卷必有)")

per_clip = {}
for side, sec in (("fh", "FOREHAND per-clip breakdown"), ("bh", "BACKHAND per-clip breakdown")):
    per_clip[side] = {
        "normal_err": row_vals("racket_normal_err(deg)", sec, "=" * 20),
        "pos_err":    row_vals("racket_pos_err@exact(m)", sec, "=" * 20),
        "vel_err":    row_vals("racket_vel_err@exact(m/s)", sec, "=" * 20),
    }
VSEC, VEND = "MODE B — VENUE-BALL REALISM", "=" * 20
venue = {}
for side in ("fh", "bh"):
    venue[side] = {
        "n":       row_vals(f"  {side}: n_strikes", VSEC, VEND),
        "contact": row_vals(f"  {side}: contact_rate", VSEC, VEND),
        "ret":     row_vals(f"  {side}: return_success", VSEC, VEND),
        "cf_ret":  row_vals(f"  {side}: CF return_success", VSEC, VEND),
    }
overall = {
    "contact": row_vals("contact_rate", VSEC, VEND),
    "ret":     row_vals("RETURN SUCCESS (回球成功率)", VSEC, VEND),
    "cf_ret":  row_vals("CF RETURN SUCCESS", VSEC, VEND),
}

side_cn = {"fh": "正手", "bh": "反手"}
out = []
out.append("# 判卷报告:${RUN_NAME}(${CKPT_TAG})")
out.append("")
out.append("- 产出:judge.sh 单命令判卷链(标准入口)@ ${STAMP} UTC;工件目录 \`${JUDGE_DIR}\`")
out.append("- checkpoint:\`${CKPT}\`")
out.append("- 动作对:\`${MOTION_FH}\` | \`${MOTION_BH}\`(${SRC_MOTION})")
out.append("- 相位对:[${PH0}, ${PH1}](${SRC_PHASES})")
out.append("- exam 卷:\`${EXAM_BANK}\`(${SRC_BANK})")
out.append("")
out.append("## 报告头(旗标状态,评估器原话照抄)")
out.append("")
out.append("\`\`\`")
out.append(flag_line)
if clamp_line:
    out.append(clamp_line)
out.append(phase_line)
out.append(obsnorm_line)
out.append(reset_line)
out.append(bank_line)
out.append("\`\`\`")
out.append("")
out.append("## 分母报表(判卷分母法则,连抄)")
out.append("")
out.append("\`\`\`")
out.extend(den)
out.append("\`\`\`")
out.append("")
out.append("## 考卷读数(侧 × 噪声档)")
out.append("")
out.append("| 侧 × 噪声 | 击数 | 接触率 | 回球成功率 | 拍面误差@exact | 拍位误差 | 拍速误差 |")
out.append("|---|---|---|---|---|---|---|")
for side in ("fh", "bh"):
    for k, ns in enumerate(ns_labels):
        out.append("| {} {} | {} | {} | {} | {}° | {}m | {}m/s |".format(
            side_cn[side], ns, venue[side]["n"][k], venue[side]["contact"][k],
            venue[side]["ret"][k], per_clip[side]["normal_err"][k],
            per_clip[side]["pos_err"][k], per_clip[side]["vel_err"][k]))
out.append("")
out.append("- 全侧汇总:接触率 " + " / ".join(f"{ns}→{v}" for ns, v in zip(ns_labels, overall["contact"]))
           + ";回球成功率 " + " / ".join(f"{ns}→{v}" for ns, v in zip(ns_labels, overall["ret"])))
out.append("- CF 对照(换成题目要求拍面重判):全侧 "
           + " / ".join(f"{ns}→{v}" for ns, v in zip(ns_labels, overall["cf_ret"]))
           + ";分侧 fh " + "/".join(venue["fh"]["cf_ret"]) + ",bh " + "/".join(venue["bh"]["cf_ret"])
           + " —— CF 显著高于实测=失分在拍面通道")
out.append("")
out.append("## 工件")
out.append("")
out.append("- 导出日志:\`${JUDGE_DIR}/export_play.log\`(成功行已认;tuple Traceback=已知无害)")
out.append("- sidecar:\`${STD_NPY}\` + \`${OBS_NORM}\`(\`${JUDGE_DIR}/sidecar.log\`)")
out.append("- 考卷日志:\`${JUDGE_DIR}/exam.log\`;逐步/逐击 CSV 在 \`${JUDGE_DIR}/exam/\`")
out.append("")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print(f"[judge] 报告已生成")
PYEOF

echo
note "==== 判卷完成 ===="
note "报告: $REPORT"
