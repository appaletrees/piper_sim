# 刚体收桌、叠杯与域随机化采集

当前演示入口使用 `configs/tasks/clean_table_rigid.yaml`：移除纸团、毛巾、餐垫和左侧纸篓，
保留六件餐具，加入 2–3 只纸杯、笔、充电头。杯子先在桌面套叠，再整摞入右篮；碗直接入篮。
纸杯沿用照片参考的 80 mm 杯口、56 mm 杯底、95 mm 高的空心模型。
笔暂估 150×10 mm，充电头暂估 65×65×28 mm，插脚折叠。
收纳篮为本任务独立估算的 360×310×140 mm，容纳杯摞和餐具；不是旧视频篮筐的精确复刻。
机器人、台面、双臂基座、D455/D435i 安装与光学基准保持原 CAD 配置。

旧 `configs/tasks/clean_table.yaml` 保留用于视频回放对比。旧辅助抓取报告是历史结果，
**不能用于证明当前物理抓取成功，也不能作为本版本采集验收结果**。

## 预览、控制和录制

```bash
# 随机刚体场景、原三摄像头、外部控制/读状态/录制 API
./scripts/run_mujoco.sh --config configs/clean_table_workcell.yaml \
  --task-scene configs/tasks/clean_table_rigid.yaml \
  --domain-randomization configs/domain_randomization/clean_table.yaml \
  --seed 42 --enable_cameras --api-port 8765

# 单条完整特权策略；输出目录必须不存在
PYTHONPATH=src .venv-mujoco/bin/python scripts/demo_clean_table.py \
  --domain-randomization configs/domain_randomization/clean_table.yaml \
  --seed 42 --record --rgb --video --output outputs/rigid_demo_001

# 只诊断叠杯；子任务不进入 accepted 清单
PYTHONPATH=src .venv-mujoco/bin/python scripts/demo_clean_table.py \
  --objects cup_stack --record --viewer --output outputs/stack_check_001

# 批量采集：失败样本保留拒收原因，只转换合格的完整 RGB 任务
PYTHONPATH=src .venv-mujoco/bin/python scripts/collect_clean_table.py \
  --output outputs/rigid_batch_001 --seeds 0 1 2 3 4 --lerobot
```

不传随机化配置时，教师使用固定两杯基准场景。`--nominal-fraction 0.2` 可在批次中按 seed
混入基准场景。先用 `--states-only` 审计运动，再录制 RGB；状态审计不输出视觉训练集。
`--objects` 只用于诊断，不能导出为完整任务合格样本。

## 当前随机化维度

| 维度 | 默认范围／实现 |
| --- | --- |
| 杯子数量 | 每次 reset 采样 2 或 3；稳定分配槽位，停用杯不出现在观测/任务指标中 |
| 整体摆放 | 平移 x ±18 mm、y ±25 mm；偏航 ±6° |
| 单件/支撑组摆放 | 独立平移各轴 ±12 mm、偏航 ±12°；盘碗勺保留支撑关系；紧邻筷子使用 0.1 倍扰动 |
| 收纳篮 | 平移 x ±15 mm、y ±20 mm；偏航 ±5° |
| 外观 | 杯身独立颜色，材质色调/亮度与粗糙度；白色细纹、灰色层压板、浅木纹桌面切换 |
| 光照 | 光强 0.75–1.25 倍，灯位各轴 ±120 mm，光色扰动 |
| 物理 | 每件物体质量与惯量同倍缩放 0.85–1.15，摩擦 0.8–1.2 倍 |
| 相机成像 | 每路独立增益 0.97–1.03、白平衡 0.98–1.02、gamma 0.92–1.08、模糊半径 0–0.45 px、噪声标准差 0–1 灰度级 |

杯身与蓝色标识分开建材质，配色不再被原深蓝纹理压暗。桌面在原 CAD 顶点上补充 UV，
没有缩放或挪动 CAD；木纹/层压板变化来自材质。光色按暖冷方向变化，材质亮度按同一 RGB
倍率变化，避免把白色餐具和照明随机成不自然的红绿蓝。

布局通过桌面边界、非堆叠物体间间隔、容器间隔检查后才接受。杯子、笔、充电头不是绑定在
固定轨迹上的装饰物，均有自由刚体、碰撞和质量。可见的材料颜色、纹理选择、质量、摩擦、
传感器参数、活动物体清单、seed 都写入 episode 元数据，支持同 seed 重建。
每次 reset 从初始参数恢复，避免随机化累积；同一仿真 tick 重复 read 返回相同噪声。
成像扰动是 RGB 后处理，并非完整的 RealSense 噪声标定模型。MuJoCo 用高光参数近似粗糙度，
Isaac 使用材质粗糙度；两者共享采样值，不保证像素或接触轨迹完全一致。

相机安装/内参、原始 CAD、重力和采样周期固定。执行延迟、控制增益扰动、相机标定误差和
物体尺寸变化尚未启用；这些需要独立验证和课程式扩大范围，不能仅在配置里写个参数就算完成。
两引擎共享场景和采样字典，当前自动叠杯/分拣教师只在 MuJoCo 实现。

## 策略与质量门槛

默认 **physical**，不创建抓取 weld，不修改物体状态来搬运。历史 `assisted` 模式不能进入
合格采集清单，刚体叠杯教师直接拒绝该模式。

1. 每次接近和放置根据实时物体/篮筐位姿、抓取后相对变换重新算 IK。
   筷子按实际长轴与中心生成向下抓取姿态，避免邻近物体接触造成的滚转变成不可达手腕姿态。
2. 必须双指接触才抬起；搬运中按控制周期（30 Hz）监控接触锚点滑移与双指接触。滑移超过 12 mm 或连续丢失双指
   接触超过 0.2 秒即拒收。该门槛不会把空夹爪带动物体视为成功。
3. 左臂逐杯套叠到中央底杯，退出后右臂夹底杯、抬起整摞、保持直立搬运；上层杯靠接触和重力随行。
   每次套叠检查轴线偏差 <6 mm、倾角 <8°、相邻杯底高差 5–18 mm。
   全部杯子须先在桌面稳定 0.75 秒，再整摞入篮；最终仍须套叠且全部完整在篮内。
4. 碗直接入篮，盘、方碟、筷子等使用篮内的分区放置点。规划检查使用抓取后实测相对姿态；
   整摞杯子的预测碰撞检查同时移动全部杯子，实际仿真仍是独立自由刚体。
5. 关节/夹爪使用七次时间曲线，段首尾速度、加速度、jerk 为零。限制关节 0.5 rad/s、
   1.2 rad/s²、8 rad/s³，夹爪 0.04 m/s、0.12 m/s²、1 m/s³。实际轨迹还需通过记录后的质检。
   每段等待实测误差和速度连续达标，超时停止。轻物体使用较小夹爪力，避免被过度夹紧弹出。
6. 最大机器人障碍穿透 4 mm；碰撞、不可达、滑脱、套叠散开、落错位置、时钟断续、相机异常、
   实测抖动超限均拒收。完整任务的全部物体还会在退让后重新检查。

接触力、材质、惯量与几何均有估计部分；这些门槛是仿真数据筛选，不是真机迁移认证。
短段 IK 与碰撞拒绝检查不等于任意布局的通用规划器，失败率必须保留报告。
薄盘夹持启用多点接触，并在负载搬运时使用 NoSlip 迭代抑制软接触的数值漂移；
它仍受接触与摩擦约束，不是固定物体的 weld。求解器和主要夹爪参数写入 teacher 元数据，
其余分物体接触配置可按报告的源码指纹定位。
相关机制见 [MuJoCo 接触求解器说明](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#option)。
MuJoCo 教师的指垫具有固定接触优先级；本轮摩擦扰动作用于物体表面及其桌面/物体间接触，
没有同时随机化指垫摩擦。夹爪力与控制增益按物体类型配置，也未作为本轮随机变量。

## 输出与外部接口

`SimulationSession` 和 HTTP 接口的控制、读取、reset、HDF5 录制不变。
reset 改变杯子数量时，`observation.extras.objects` 仅包含活动物体；不要硬编码物体数量。
`extras.task.metrics.cup_stack` 提供套叠状态，`summary.required_count` 随数量改变。
元数据 `task_scene.layout.content` 与 `domain_randomization` 保存实际随机样本。

```python
from piper_sim.client import SimulationClient
from piper_sim.agilex import AgilexAdapter

robot = AgilexAdapter(SimulationClient('http://127.0.0.1:8765', timeout=15))
obs = robot.reset(seed=42, randomize=True)
print(obs['extras']['objects'])                 # 活动物体实时位姿
print(obs['extras']['task']['metrics'])          # 入篮和套叠指标
print(robot.meta['task_scene']['layout'])        # 本轮采样参数
```

读关节角、发控制、录制与转换的完整例子仍是 `examples/external_agilex.py`；上层使用度数和
0–1 夹爪开度，底层 HDF5 保持 rad/m。批量教师可用 `--config`、`--task-scene` 和
`--domain-randomization` 分别指定工作站、物体场景和随机化范围。

| 输出 | 内容 |
| --- | --- |
| `episode.h5` | 30 Hz 状态/实际动作/下一状态、可选三路 RGB、物体真值和策略阶段 |
| `report.json` | 成功与拒收原因、套叠事件、物理抓取统计、运动质检、配置/源码/素材 SHA-256 |
| `audit.json` | 可用 `scripts/audit_rigid_sequence.py episode.h5 --output audit.json` 独立复核记录中的叠杯顺序；不改原数据 |
| `accepted.jsonl` / `rejected.jsonl` | 全部采集尝试的分类，失败不混入合格清单 |
| `lerobot_v2/` | 可选，只从合格完整 RGB episode 转换；继续保留质量与仿真来源信息 |

真值单独保存在 extras，视觉策略训练时应明确选择输入字段，避免把部署拿不到的真值混入观测。
训练/验证按 seed 和场景组合切分，同轨迹的外观变体不要跨训练/验证集合。

## 验证

多域场景对比见下图；数量、位置、外观变化均来自真实 reset 后的渲染。

可用 `PYTHONPATH=src .venv-mujoco/bin/python scripts/preview_rigid_randomization.py` 重建预览与参数清单。

![域随机化对比](../outputs/rigid_table_collection/domain_comparison.jpg)

旧目录 `outputs/domain_randomization_collection` 的五 seed 辅助抓取验收已被用户指出不足，
保留作历史审计，不计入当前纯物理教师通过率。当前验收结果以
[validation.json](../outputs/rigid_table_collection/validation.json) 及对应报告为准。

最终策略的两组完整物理复测如下；两者均通过运动质检，所有搬运抓取在监控采样时刻均为双指接触。

| 随机回合 | 完整入篮 | 最大抓取锚点滑移 | 最大机器人穿透 | 桌面先套叠、再整摞入篮 |
| --- | --- | --- | --- | --- |
| [seed 0，两杯](../outputs/rigid_table_collection/random_seed_0_roll_fix/report.json) | 10/10 | 3.11 mm | 0.50 mm | 位姿复核通过 |
| [seed 42，三杯](../outputs/rigid_table_collection/random_seed_42_roll_fix/report.json) | 11/11 | 3.32 mm | 0.92 mm | 位姿复核通过 |

![三杯回合最终记录位姿](../outputs/rigid_table_collection/random_seed_42_roll_fix/final.png)

回归测试 105 项通过，其中刚体场景覆盖 80 个 seed 的布局合法性、原 CAD 台面尺寸、
活动物体切换、接触套叠与滚转筷子的抓取姿态。MuJoCo/Isaac 分别完成 42→43→42 的
运行时属性复位、三路成像和 HDF5 检查；HTTP 接口完成 0→42→0、2→3→2 杯切换，
控制指令确认与三路 640×480 RGB 录制。这些接口检查不计作教师任务成功。

完整教师复测使用状态记录；尚未批量生成完整 RGB 训练集。`nominal/demo.mp4` 为此前通过的
固定两杯参考回合，4 倍速预览，早于充电头抓取和滚转筷子修复；报告保留当时源码指纹。
新回合的 `final.png` 由记录末帧位姿离线渲染，用于核对摆放，不计作额外物理回合。

开发中的充电头接近失败、滚转筷子 IK 失败都保留在报告中。不能把单件诊断或“10/11 件入篮”
记为完整成功，也不能将有限 seed 的复测结果解释为整个随机分布的成功率。
