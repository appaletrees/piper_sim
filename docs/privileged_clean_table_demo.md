# 特权信息餐桌分拣 Demo

目标：`paper` 纸团、`towel` 白色纸巾放进左盒；盘、碗、勺子、两根筷子、方碟放进右篮。
餐垫留在桌面。保留原双臂 Piper、三摄像头和工作台尺寸。
为避免大盘架在碗上，先将碗暂放在桌面空位，盘及其他餐具入篮后再取回碗。
暂放与取回均通过机械臂动作完成并记录。

这是一段 **MuJoCo 脚本专家演示**，直接读取仿真物体位姿和接触信息，不使用视觉识别。
默认是 **接触辅助抓取**：夹爪实际接触目标物体后，临时约束保持当前相对位姿；
张开夹爪、指尖离开物体后解除约束，由重力和碰撞决定最终落放。
程序不会在演示途中改写物体或机械臂的 `qpos/qvel`。

辅助抓取不是实际 Piper 的能力，也不能算作纯接触物理验收。毛巾仍是刚体褶皱网格，
没有布料变形。视频叠字、HDF5 元数据、每帧 `teacher` 字段和报告均标注模式；
`training_ready=false`，用于先验证任务流程、接口和录制链路。

固定布局 seed 42 的辅助演示已完成 8/8：仿真时长约 273.7 秒，记录 8212 帧状态，
并验证 LeRobot v2.1 状态数据导出。此示例未录制三相机 RGB；采集 RGB 使用下方 `--rgb` 选项。
见 [4 倍速视频](../outputs/clean_table_demo/preview_4x.mp4)、
[逐物体报告](../outputs/clean_table_demo/report.json)、[验证记录](../outputs/clean_table_demo/validation.json)。

## 运行

在仓库根目录执行，输出目录必须尚不存在：

```bash
PYTHONPATH=src .venv-mujoco/bin/python scripts/demo_clean_table.py \
  --video --record --output outputs/my_sort_demo
```

加 `--viewer` 可打开交互窗口；不加时在后台仿真。默认固定布局、seed 42。
`--objects paper` 可单独测试一个物体。`--randomize --seed 123` 接入现有布局随机化，
但当前脚本不是经过大批随机场景验证的通用专家；不可达、未接触、掉落、未入筐都会报失败。

录制三台相机并导出 LeRobot v2.1：

```bash
PYTHONPATH=src .venv-mujoco/bin/python scripts/demo_clean_table.py \
  --video --record --rgb --lerobot --output outputs/my_sort_rgb_demo
```

三相机 RGB 会明显增加时间和磁盘占用。不加 `--rgb` 时 HDF5 仍保存关节状态、动作、
时间戳、物体真值位姿、任务指标及专家阶段，但不包含相机图像。

纯接触实验可用 `--grasp-mode physical`；该模式完全不创建辅助抓取约束。
当前薄盘边缘、刚体纸巾等抓取尚不稳定，不能保证完整任务成功，失败会停止并保留报告。

## 输出与成功含义

| 文件 | 内容 |
| --- | --- |
| `demo.mp4` | 开启 `--video` 时的总览视频，叠加抓取模式、阶段、时间和入筐计数 |
| `final.png` | 开启 `--video` 时的最终画面 |
| `episode.h5` | 开启 `--record` 时的 30 Hz `(obs_t, action_t, obs_t+1)` 记录 |
| `report.json` | 逐物体结果、辅助启用事件、实际接触手指、阶段时间和最终位姿 |
| `lerobot_v2/` | 开启 `--lerobot` 后的可选转换；辅助模式信息仍保存在元数据/额外信息中 |

`summary.success` 要求八个物体的可见网格全部处于各自容器的内部，并持续至少 0.5 秒。
机器人松开、退让后再次检查，最终也重新检查全部八件物体。
这里的成功表示**当前抓取模式下任务完成**，辅助模式的成功不等于纯物理抓取成功。
命令退出码 0 表示请求的流程执行完毕，2 表示失败；诊断子集不要求八件全部完成。

## 解耦接口

- [`CleanTableTeacher`](../src/piper_sim/experts/clean_table.py)：读真值、选择抓取点、规划阶段、发送 14 维 SI 位置目标。
- [`PiperIK`](../src/piper_sim/experts/mujoco_ik.py)：在独立 `MjData` 上求有界 IK，不改动实时仿真状态。
- [`ContactGraspAssist`](../src/piper_sim/experts/grasp_assist.py)：可选约束插件；没有实际指尖接触时拒绝绑定。
- 原 `SimulationSession` 负责步进、读取和录制；场景的配置、随机化、网格和成功指标继续由 `tasks/clean_table*` 管理。

可在自己的 Python 程序里使用控制器：

```python
from piper_sim.config import load
from piper_sim.backends.mujoco import MujocoBackend
from piper_sim.simulation import SimulationSession
from piper_sim.experts.clean_table import CleanTableTeacher
from piper_sim.experts.grasp_assist import add_grasp_constraints

cfg = load("configs/clean_table_workcell.yaml")
cfg["task_scene"] = "configs/tasks/clean_table.yaml"
backend = MujocoBackend(cfg, model_xml_transform=add_grasp_constraints)
with SimulationSession(backend) as session:
    teacher = CleanTableTeacher(session, grasp_mode="assisted")
    result = teacher.run()
```

需要带完整专家标记的录制时，使用上述命令行入口；它将 `teacher` 信息注入每帧和文件元数据。
既有外部 API、真机接口和场景随机化接口保持原用法。此脚本不会连接或驱动真机。

## 接触与控制参数

专家启用机器人关节的仿真动力学偏置前馈，仍通过位置执行器驱动动作。
夹爪位置增益为 10000 N/m、执行器力限幅 ±25 N；指尖使用六维接触，滑动/扭转/滚动摩擦为
`[1, 0.015, 0.002]`，接触时间常数 0.0042 秒。这些是明确记录的仿真参数，尚未通过真机辨识。

餐具接触网格同时沿周向和径向分块，避免凸包填实内腔；纸团碰撞覆盖外形，
白色纸巾使用刚性褶皱分片。外观尺寸没有缩小来适配夹爪。
辅助约束遵循 [MuJoCo weld 约束语义](https://mujoco.readthedocs.io/en/3.3.4/XMLreference.html#equality-weld)，
它只是本演示的可选模块，不会写入默认场景资产。
