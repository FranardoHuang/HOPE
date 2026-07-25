# Venue Profile — 场地/环境模型接口

**一句话:** 把"这个场地长什么样"的三类标定(动捕噪声、通讯链路、物理随机化)收进一个
JSON 档案,换场地/换动捕/换机器人时只改这一个文件,不再满 yaml 找散落的覆写。

- 档案目录:`configs/venue_profiles/<name>.json`
- 加载器:`whole_body_tracking/utils/venue_profile.py` 的 `load_venue_profile(path_or_name)`
- 首个档案:`franco_rig_20260725`(现役 Franco rig 标定)
- schema 契约串:`venue_profile_v1`

## 三个 section 各建模什么(每个键都对应仓库里已存在的旋钮)

| section | 人话 | 键 → 现有旋钮 |
|---|---|---|
| `mocap_noise` | 动捕建模:actor 看到的目标位置上叠多大的测量噪声 | `target_noise_white` / `target_noise_ar1_sigma`(std,米)、`target_noise_ar1_rho`(50 Hz 每步相关系数)→ `task.racket.*`(hope_commands.RacketTargetCommandCfg) |
| `transport` | 通讯/运行建模:目标数据晚几步到、每步丢帧概率多大、time-to-strike 用哪种时戳约定 | `target_delay_steps`(50 Hz 整步)、`target_dropout_prob`、`target_delay_tts_mode`("live" / "source_timestamp_compensated" / "uncompensated")→ `task.racket.*` |
| `physics` | 物理建模:机器人 body 材质和连杆质量的域随机化区间 | `static_friction_range` / `dynamic_friction_range` / `restitution_range` → `EventCfg.physics_material`(tracking_env_cfg.py);`mass_distribution_params`(质量 scale 乘子)→ `HOPEEventCfg.randomize_link_mass`(hope_env_cfg.py) |

注意:这三类噪声/延迟只降级 **actor 可见** 的 planner 元组;奖励、真时序、metrics/gate、
特权 critic 走 live 值(见 hope_commands 的 A1 注释)。physics 区间是 startup 域随机化,
不是单点标定值。

## 严格性(fail-loud,没有静默兜底)

- 顶层必须且只能有:`schema_version`(逐字 = `venue_profile_v1`)、三个 section、可选
  `_comment`(字符串,纯文档,不进返回值)。
- 每个 section 的键集合是【精确】的:多一个未知键、少一个必需键、类型不对、NaN/Infinity、
  JSON 重复键、会被下游静默 clamp 的越界值(如 rho > 0.9999、dropout ≥ 1、质量乘子 ≤ 0)
  —— 全部当场 `VenueProfileError`。
- `load_venue_profile` 返回 `(profile, meta)`:`profile` 是规范化后的三 section dict
  (区间已转成 `(float, float)` 元组,可直接塞 EventCfg params);`meta` 是
  `{"name", "path", "sha256", "schema_version"}`,其中 **sha256 是档案文件字节的摘要,
  要落进 run 记录**,事后可逐字节对账"这次训练用的是哪份场地标定"。

## 什么时候要重新标定 / 新开一份档案

换了下面任何一样,就该新开一个 `<venue>_<date>.json`(不要原地改旧档案 —— sha 就是为了
让旧 run 可对账):

- **换场地 / 换动捕系统**:重测 `mocap_noise`(位置噪声拟合)和 `transport`(延迟/丢帧;
  franco rig 实测 ≤20 ms 端到端 → 保守 2 步,见
  docs/research/mocap_timing_2026-07-05/mocap_random_delay.md)。
- **换机器人 / 换脚垫材质 / 换地面**:重定 `physics` 的摩擦/弹性/质量区间。
- **标定方法升级**(比如拿到了带时戳的 live 链路日志):同样新开档案并在 `_comment` 里写
  清来源。

franco_rig 档案的已知坑:`target_delay_steps=0` 是 planner-revision guard 的历史遗留
(hope_commands 拒绝 planner revision 与 delay>0 同时开),不是"链路零延迟"的实测结论;
实测场地保守上界是 2 步(40 ms)。换 venue 或不开 planner revision 时应上 2。

## train.py 怎么消费(2026-07-25 已接线,B2)

用法:`task.venue_profile=<裸名或 .json 路径>`。train.py 的 `_apply_task_overrides` 开头调
`load_venue_profile(...)`:

- `mocap_noise` + `transport` **注入 `task.racket.*` 同名键**,由现有 racket 覆写翻译层落地
  (记 applied、类型校验、tts 非 live 换 actor 观测 func 等副作用一个不少);
- `physics` 直接写 `events.physics_material` / `events.randomize_link_mass` 的 params;
- 每条 applied 标记带 `venue_profile=<name>@<sha256 前 8 位>`,进 run 记录逐字节对账。

**优先级(Franco B2 任务书拍板,取代此前"冲突应 fail-loud"的暂定)**:档案先展开,显式
同名键后写后赢 —— `task.racket.*` 显式键、`task.plant.robot_material_*` 显式键都压过档案值,
且每次"用户赢"都会在 applied 里留一条 `user override wins` 标记,不是静默裁决。
不给 `task.venue_profile` 时一切照旧 —— 本接口对现有训练路径是逐字节 no-op。

## 单测

`hope_training/whole_body_tracking/tests/test_venue_profile.py` —— 现役档案可加载、裸名与
路径解析一致、sha 稳定、各类 schema 违规逐一拒载。host 上直接跑,不需要 isaaclab。
`hope_training/whole_body_tracking/tests/test_reward_flags_overrides.py`(JOB2 段)——
train.py 接线:三 section 全落地、显式键赢、未知档案拒载、events 旋钮缺失 fail-loud。
