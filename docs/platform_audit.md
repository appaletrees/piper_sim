# VLA 数据生产平台审计（进行中）

目标是物理可信、可复现、高吞吐的合格数据生产，不以单个成功视频作为完成证据。
本审计从 `798d398` 开始。原有 PPO 工作不属于本次修改；所有真机接口仅做代码/记录审查，不连接 CAN。
用户目标正文在第七部分第 10 项“失败、成功、未验证、”处截断，已请求补充；已完整收到的要求全部保留。

## 验收证据清单

下表是工作清单，不是完成声明。每项只有取得对应当前版本证据后才可关闭。

| 范围 | 必需证据 | 当前状态 |
|---|---|---|
| 代码/历史基线 | 源码版本、dirty 状态、依赖/硬件、历史报告和当前复测分开 | 已确认起始提交；复测进行中 |
| 六项历史现象 | Isaac 原分辨率墙钟 FPS；种子 1/7/43 运动及碰撞复测；完整时间；加速版独立审计；标定来源 | 待复测，历史失败不直接当当前缺陷 |
| 关节动力学 | URDF/CAD 质量惯量/限位/驱动与两引擎实际参数对照，来源与未辨识范围 | 待审查 |
| 抓取/接触 | 实际接触力/滑移/释放/非物理状态修改检查，双臂和自碰撞 | 待审查；当前审计主要依赖教师遥测，不能等同于独立物理复现 |
| 运动约束 | 目标与实测关节速度/加速度/jerk，TCP 速度，保持/释放阶段、随机种子回归 | 待复测；当前未有真机驱动辨识 |
| 引擎一致性 | 相同姿态几何/相机/关节开口；物体落体/接触的可解释差异 | 待复测 |
| 相机/尺寸 | CAD 保持不变；三路 K/外参/模式/畸变/腕部更新；同姿态实虚误差报告 | 已有对比工具；须核对当前输出和实机标定缺口 |
| 视觉域 | 材质/灯光/曝光/噪声/遮挡随随机种子变化；成像开销 | 待测量 |
| 时序 | obs_t、施加目标、obs_t+1、仿真/主机/图像时钟，全部帧连续性 | 已读 recorder/backend/auditor；负例及复测待补 |
| 数据存储 | 无损压缩、完整/失败隔离、共享帧一致性、崩溃恢复、磁盘限制 | 当前每步双份 RGB、flush；准备基准和安全优化 |
| 性能 | 物理/IK/碰撞/渲染/读回/写入/审核/导出分阶段；资源占用；合格回合/小时 | 正在建立可重复基准；不得用微基准声称整体 2 倍 |
| 批处理 | 预检重复成本、进程启动/资产加载、恢复调度、并行可行性、失败保留 | 已读串行脚本；待优化与验证 |
| 专家策略 | 停顿/绕路/任务顺序/双臂并行、抓取规划与错误恢复、独立质量审核 | 待在复测依据上优化 |
| VLA 语义 | 单位/顺序/夹爪映射/语言匹配，非特权训练字段与独立额外状态 | 已读导出器，需端到端训练读取验证 |
| 数据泄漏/多样性 | 稳定场景族拆分、重复轨迹/跨 split 检查、随机化覆盖和接受偏差 | 当前 seed 哈希 split 仅为起点；待审查 |
| 正式验收 | 同条件基准、完整物理 RGB 回合、多种子且含未调参种子、LeRobot 读取和失败注入 | 未完成 |

## 起始观察

- `EpisodeWriter` 当前为每个 transition 分别写 obs/next_obs 图像，每步 flush。
  相邻帧虽在 MuJoCo 渲染层缓存，写盘仍重复；已有离线 compaction，不能节省采集时压缩成本。
- 批采集默认为状态预检后重跑完整 RGB 专家。当前不记录各阶段墙钟时间，也无断点恢复。
- Isaac 的 `step()` 即使无采集相机也调用 render；`read()` 每次重新 GPU→CPU 拷贝。
  这是候选瓶颈，尚未在本轮实测，不能直接下性能结论。
- 质量审核检查全部 transition 和图像，但接触/穿透量来自记录的教师遥测；质量阈值从 HDF5
  内嵌配置读取。后续必须区分“按记录声明阈值通过”和“按生产验收策略通过”。
- LeRobot 普通字段为图像、状态、动作及索引/时钟；物体真值和教师信息在 extras sidecar。
  导出器单位是 degree/归一化夹爪，不是 HDF5 的 rad/m，需要训练入口确认。
- 起始磁盘只剩约 8 GiB；先使用短诊断和已有数据，不删除任何源数据或旧基线。

证据存放在 `outputs/platform_audit_v1/`。短诊断、状态回合、RGB 训练回合需明确分开标记。
任何吞吐改进都必须保持原时间步、全部样本、像素和质量阈值；无实测即标记待验证。

## 第一轮证据与改进

### 当前版本复测

- `baseline_seed0/`：同样的随机化种子 0，物理模式 10/10 入篮，226.0667 秒，
  6,782 transitions；状态审核通过。动作/状态轨迹哈希和物体运动哈希均与上一轮
  `natural_motion_v3/rgb_seed_0` 一致。此回合没有 RGB，不能作为训练回合接受。
- `baseline_cprofile.json`：带 cProfile 的整回合约 389.9 秒，物理积分累计 250.5 秒；
  `robot_penetration` 累计 46.4 秒，任务 metrics 累计 23.2 秒。这些是包含 profiler
  开销的定位数据，不是未插桩的产线吞吐。规划、两次观测和教师检查有重复读取成本。
- `alignment/alignment_measurements.json`：原数字标注同姿态重投影诊断最大垂直残差
  67.1003 px；无拟合、无标定、CAD 未改变。它是有人工关联不确定度的诊断，不是标定板
  RMSE，也不是本轮实机采集得到的新内外参。

### 在线共享 RGB

`EpisodeWriter(image_storage="shared")` 只存 N 张 obs 图像和一张终止图像。
`next_observation/images` 使用同文件 HDF5 virtual dataset 映射到下一张，所有原公开字段、
shape、像素、时间戳都保留。每次共享前要求与上一 transition 的 next_obs 像素、采集时间
完全相同；缓存拥有独立副本，避免调用者复用缓冲区悄悄改写。中断/错误只留下 `.partial`。
验证错误即使被调用者捕获，也不能把截短回合发布为完整文件。

源像素：`outputs/natural_motion_v3/rgb_seed_0/episode.h5` 第 2400–2459 帧，三路
640×480，实际状态/动作/时间戳。三个重复、轮换顺序，每轮重新打开生成文件并检查全部图像。
临时诊断复制品在检查后删除，只保留 `writer_benchmark.json`；没有更改原数据。

| 写入方式 | 中位写入秒数 | 文件字节 | 写入倍率（对第一行） |
|---|---:|---:|---:|
| 两份 RGB，GZIP1，每步 flush | 2.980 | 163,045,130 | 1.00 |
| 共享 RGB，GZIP1，每步 flush | 1.642 | 83,262,865 | 1.81 |
| 共享 RGB，GZIP1，每 30 步 flush | 1.634 | 83,262,865 | 1.82 |
| 共享 RGB，LZF，每 30 步 flush | 1.002 | 112,418,421 | 2.97 |

所有方式的三路 RGB、动作/状态哈希相同；GZIP 共享格式的审核时间也从 1.104 秒降为
约 0.615 秒。结果只是无损写盘微基准，包含 close/flush，不包含仿真、规划、渲染或 fsync；
不能换算成整条产线已达 2 倍。flush 频率收益很小，优先保留每步 flush。
实际 RGB 回合接口验证后，默认采集已切换为共享 GZIP、仍每步 flush。
外部 `EpisodeWriter` / `SimulationSession` 的默认参数保留原布局；可显式选择共享模式。

实际物理抓笔对照 `rgb_pipeline_comparison.json`：553 transitions、18.4333 秒仿真，
每路原生 640×480/30 Hz，原布局 126.97 秒墙钟、共享布局 114.31 秒（约 1.11 倍）。
文件 1,492,591,995 → 750,717,757 字节；写盘本身约 1.83 倍。
这是单次顺序对照、单物体诊断，整桌任务未执行，两个文件均明确 `training_ready=false`。
两份数据所有 transition 均通过完整性审核，动作/状态哈希、每帧物体和相机位姿完全一致。

两次独立渲染的 RGB 未达到逐位相同：对每路 554 张唯一图像的全部像素比较，左腕 132、
中央 5,301、右腕 15,029 个通道值不同（各路共 510,566,400 个值），最大差 1/2/2。
原输入重复执行传感器后处理 12 次逐位相同，跨进程图像差异的具体来源尚未定位；
`rgb_all_frame_difference.json` 保留全部计数，不能宣称像素级跨进程确定性。
这与写入无损性分开：同一输入的所有 writer 微基准和回归测试都逐位一致，未丢帧或改时间。

复现（新输出文件）：

```bash
PYTHONPATH=src .venv-mujoco/bin/python scripts/benchmark_recording.py \
  --source outputs/natural_motion_v3/rgb_seed_0/episode.h5 \
  --output outputs/writer_benchmark_new.json --start 2400 --frames 60 --repeats 3

PYTHONPATH=src MUJOCO_GL=egl .venv-mujoco/bin/python scripts/profile_collection.py \
  --profile-output outputs/profile_new.json \
  --record --rgb --seed 0 --domain-randomization configs/domain_randomization/clean_table.yaml \
  --output outputs/profile_episode_new
```

插桩按物理积分、forward、IK、规划碰撞检查、渲染场景更新、渲染/读回、RGB 压缩写入、
元数据、flush、审核、checksum 和导出分别统计。inclusive 时间存在嵌套，不可求和；
exclusive 才可求和。未调用的阶段不声称为零成本。GPU 样本是整卡（包含其他应用），
尚未等价于本进程显存峰值或无竞争 GPU 基准。

### 独立生产验收策略

`configs/collection/quality_policy.yaml` 固定当前生产阈值，数值没有放宽。新版审核
`piper.collection.v3` 从独立文件加载策略，HDF5 中的教师阈值只作为溯源记录。
审核结果保存策略文件 SHA256。训练导出必须核对当前策略、源文件 checksum，以及内嵌/外置
审核报告一致性。旧 v2 审核需用明确策略重新检查后方能新导出；原始历史文件保持不变。

默认只读复核：

```bash
PYTHONPATH=src .venv-mujoco/bin/python scripts/audit_collected_episode.py EPISODE.h5 \
  --output NEW_AUDIT.json
```

要更新某个待发布 HDF5 的 acceptance，可显式加 `--update-acceptance`，并把新 sidecar
放在该 HDF5 同目录。这会改变文件 checksum；历史不可变基线不要执行该选项。
`--policy FILE` 由操作者选择，不能由记录中的路径指定。训练导出也支持相同 `--policy`。

剩余工作包括：完整任务的共享 RGB 回合和完整导出验证、Isaac 新基准、种子 7/43 复测、接触力与
自碰撞证据、策略/规划修复、调度恢复和并行、多样性/泄漏统计、同条件合格回合/小时对照。
整个审计目标仍未完成。

### 当前问题复现与参数来源

种子 1 已重跑完整物理回合：11/11 入篮，但小碟释放时右夹爪加速度
1.0461066 m/s² > 1.0，按原门限拒收。`seed1_profile.json` 记录墙钟 511.19 秒，
其中积分 384.23 秒、规划碰撞检查 21.07 秒、IK 5.79 秒、状态写入 19.83 秒。
期间存在独立的只读审核和测试进程；这个定位结果不是无竞争性能认证。
进程 RSS 峰值约 2.29 GB，GPU 显存数值包含无关应用，不归因于状态采集器。

当前旧 RGB 完整回合经 v3 独立策略只读重新审核仍通过（`existing_rgb_policy_v3.json`），
未改写原 HDF5、旧 sidecar 或旧验收哈希。

`physics_model.json` 从当前编译模型读取：30 Hz 控制、120 Hz 对外物理 tick、960 Hz 内部
积分，每个 action 32 子步，732 个 geom；机器人质量和主惯量与源 URDF 一致。
这不能证明实机动力学正确：相机/支架质量未辨识，URDF 六轴 velocity=10 rad/s 未实测，
MuJoCo 阻尼/armature、接触摩擦与夹爪控制参数是仿真假设。教师使用 `qfrc_applied`
施加偏置补偿，因此 actuator 的 forcerange 不能单独充当总关节驱动力上限。
