"""Guard the 2026-07-12 decision not to splice an old recipe into current main."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "hope_training/whole_body_tracking/cfg/task"
TRAIN = ROOT / "hope_training/whole_body_tracking/scripts/train.py"
ENV_CFG = (
    ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/"
    "tracking/config/agibot_a3/hope_env_cfg.py"
)


def test_current_main_has_no_final_v2_or_v2_plus_recipe_surface():
    assert not (TASKS / "HOPEPingPongHitterPureRallyFinalV2.yaml").exists()
    assert not (TASKS / "HOPEPingPongHitterPureRallyFinalV2Plus.yaml").exists()


def test_head_discipline_is_not_silently_exposed_or_enabled():
    task_text = "\n".join(path.read_text(encoding="utf-8") for path in TASKS.glob("*.yaml"))
    assert "head_discipline_weight" not in task_text
    assert "head_discipline_weight" not in TRAIN.read_text(encoding="utf-8")
    assert "head_discipline = RewTerm" not in ENV_CFG.read_text(encoding="utf-8")
