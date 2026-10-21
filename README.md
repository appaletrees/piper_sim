# 双臂 Piper 仿真工作站

本项目以 **Isaac Lab v2.3.0 + Isaac Sim 5.1.0 + Python 3.11** 为固定基线，
在本机独立环境中运行。两臂、场景、传感器、真机通信、指令入口、数据记录分别实现。
VLA / WAM 是后续用途，本阶段提供通用数据契约，不绑定某个模型的训练代码。

**已在本机通过带三路相机的物理仿真、120 帧采集和 GUI 回放验收；真机联调待接入 CAN。**
相机配置为 **中央 Intel RealSense D455 × 1、腕部 Intel RealSense D435i × 2**，
外壳、支架和紧固件按提供的 STEP 装配坐标恢复。
详见 [相机与支架还原说明](docs/camera_restore.md)、
[RealSense RGB 参数与来源](docs/realsense_rgb.md)、
[验证记录](docs/validation.md)。

**真机核对更新：** 已检查 `connector_plugin_lerobot_v2` 的 210 集、630 路视频，
修正度/弧度和夹爪字段转换，并替换为原 STEP 的宽型夹爪视觉网格。
数据三路均为640×480。当前仍有约39–46 px桌边、58–67 px腕部指尖的投影偏差，
**尚未通过真机一比一视角验收**；设备内参、安装标定及实体开口测量缺失。
见 [真机对照、已修正项与剩余差异](docs/real_alignment.md)。

## 双臂 Piper 场景

下图为本项目在 Isaac Lab / Isaac Sim 中的实际零位场景截图。

![双臂 Piper：原CAD夹爪网格、灰黑Piper、三台RealSense和原支架](outputs/real_alignment_summary/overview.png)

总览和近景采用原 CAD 夹爪视觉网格及参考照片配色。后面的全幅 RGB 保留此前公开光学规格验收版本，
不代表640×480真机标定已完成。

| 左腕 Intel RealSense D435i | 右腕 Intel RealSense D435i | 中央 Intel RealSense D455 |
|---|---|---|
| ![左腕相机与原始支架](outputs/real_alignment_summary/left_mount.png) | ![右腕相机与原始支架](outputs/real_alignment_summary/right_mount.png) | ![中央相机与立柱支架](outputs/real_alignment_summary/central_mount.png) |

RGB 已使用官方公开光学规格及标称镜头偏移：

| 相机 | 默认 RGB 分辨率 | 水平 × 垂直 FOV | 焦距 |
|---|---|---|---|
| 左、右腕部 D435i | 1920×1080 | 69.4° × 42.5° | 1.88 mm |
| 中央 D455 | 1280×800 | 90° × 65° | 1.93 mm |

三路仿真均以 30 Hz 采样；按公开 FOV 推导标称 K，并将 RGB 光心从原开孔中点移到官方
坐标定义的位置。公开值不等于每台设备的出厂标定，畸变、滚动曝光和景深的当前边界见上述参数说明。

| 左腕 RGB | 右腕 RGB | 中央 RGB |
|---|---|---|
| ![D435i 左腕实际 RGB](outputs/realsense_rgb_acceptance/left_wrist.png) | ![D435i 右腕实际 RGB](outputs/realsense_rgb_acceptance/right_wrist.png) | ![D455 中央实际 RGB](outputs/realsense_rgb_acceptance/overhead.png) |

## 项目结构

```text
configs/workcell.yaml          布局、时步、相机和硬件通道配置
configs/realsense_rgb.json     官方 RGB 规格、镜头偏移及来源
configs/connector_alignment.yaml  历史数据640×480诊断模式（K为未标定假设）
configs/appearance.json        独立外观材质配置
src/piper_sim/
  contracts.py                观测 / 动作统一契约，SI 单位
  robot_asset.py              Piper articulation、关节驱动配置
  scene.py                    桌面、立柱、任务物体
  sensors.py                  三路相机配置
  camera_models.py            独立 RGB 光学模型与等比例缩放
  appearance.py              CAD夹爪视觉替换和材质绑定
  lerobot_alignment.py        connector_plugin专用单位与帧索引核对
  backends/isaac.py            Isaac 仿真步进与状态接口
  backends/piper.py            Piper SDK 真机适配器
  backends/sdk_tracking.py     每组 CAN 帧独立的新鲜度检查
  backends/mock.py             离线测试后端
  control.py                  本机 HTTP 指令入口，与具体控制设备解耦
  safety.py                   交接对齐、限位、限速和时效检查
  recording.py                HDF5 episode 记录
  dataset.py                  数据校验与连续时序窗口读取
  kinematics.py               离线 URDF 正运动学
scripts/
  install_runtime.sh          独立 Isaac 环境安装
  run.sh / run_scene.py        场景、采集、回放、双向主从入口
  prepare_assets.py           从原描述派生可运动夹爪的仿真 URDF
  inspect_step.py             STEP 装配层级与变换提取
  convert_workcell_cad.py      STEP 桌面、立柱、三相机及腕部支架转换
  prepare_cad_gripper.py       原CAD夹爪按link6/7/8拆分，可随关节运动
  extract_lerobot_alignment.py 只读抽帧、状态转换与来源核对
  render_lerobot_alignment.py  观测姿态下的三路实虚同帧对照
  measure_real_alignment.py   从CAD与数字标注复现投影残差
  export_realsense_rgb_calibration.py 只读导出设备RGB模式内参
  verify_cad_alignment.py     重新读取 STEP，逐顶点和尺寸核对导出几何
  capture_cad_reference.py    Isaac 零位总览和相机支架近景
  preview_layout.py           离线几何布局预览
  check_episode.py            数据完整性检查
  verify_simulation.py        关节跟踪、落桌接触、三路图像验收
  verify_rgb_projection.py    三维标记与实际渲染像素的投影核对
  doctor.py                  无硬件副作用的环境检查
  download_runtime_wheels.py  网络中断时按哈希校验并续传安装包
assets/generated/             自动生成的 URDF、网格、USD、CAD 元数据
outputs/                      日志、预览、episode、环境版本记录
```

```mermaid
flowchart LR
  T[遥操作设备 / 策略 / 回放] --> C[统一 Command]
  C --> G[对齐、限位、限速、时效检查]
  G --> S[Isaac 后端]
  G --> R[Piper 后端]
  S --> O[统一 Observation]
  R --> O
  O --> D[Episode 记录器]
  D --> E[VLA 导出器 / WAM 时序窗口]
  S --- A[场景 / 机器人资产 / 相机配置]
```

## 布局与资产

- 原始资料 `demoa_asm_20261020.stp`、`zuomian.png` 保持原样。
- STEP 中桌板几何尺寸为 **1400 × 700 × 25 mm**；两台 Piper 装配原点的横向距离为 **600 mm**。
- 场景采用两臂同向并排、中央相机立柱、一个测试物块、一个俯视相机和两个腕部相机。
- 世界坐标 `+X` 向桌面前方，`+Y` 向机器人左侧，`+Z` 向上；左右以站在机器人后方看向工作区为准。
- 桌面、1 m 高立柱、三台相机外壳、支架和螺钉均来自 STEP 实体。导出 52 个静态网格
  （含一个只作 guide 的视场示意件）及每腕 10 个随 `link6` 运动的网格。重复相机及中央 D435 备选件已排除。
- 原尺寸只进行毫米到米的单位转换，网格化公差为 0.1 mm。默认双臂六关节为零，与 CAD 装配参考姿态对应。
- CAD 基座底面比桌面高 11 mm，当前保留原位置。源 URDF 与 CAD 法兰中心相差约 0.205 mm，
  腕部附件使用零位变换的逆矩阵转入 `link6`，使附件在参考姿态下保持原 CAD 世界坐标。
- RGB 投影位置由 CAD 前表面坐标与官方标称镜头偏移组合，内参按对应型号的 RGB FOV 推导。
  几何安装位姿、公开光学规格与每台真机的出厂标定分别记录。
- 从 `example_agilex/piper_description` 复制 URDF 和 STL。原 URDF 中 `joint7`、`joint8` 为 IK 而固定，派生版本恢复两个相反空间方向的滑动手指，每指行程 35 mm，两指目标为开口宽度的一半。
- 所有关节通过名称映射，不依赖 USD 的关节排列。两臂使用同一资产实例。
- 视觉夹爪采用原 STEP 的164.003 mm宽型组件；碰撞、惯量和70 mm行程仍为旧URDF，
  尚未完成真实夹爪接触模型标定。`appearance` 配置独立控制CAD视觉和照片参考材质。
- 源 URDF 中的 100 Nm 等参数未经真机辨识；默认驱动参数用于初步仿真，不代表实际 Piper 动力学。
- 旧控制器将夹爪字段叫作 degrees，SDK 实际定义为 **0.001 mm**。这里转换为米。旧代码允许 95 mm，而原 URDF 的行程是 70 mm；本项目采用后者，超出范围报错，待实机行程确认后统一校准。

## 安装与启动

版本库包含源码、配置、原始 STEP / 布局图和选定的验收截图、数值报告。
虚拟环境、运行缓存、生成的 USD / URDF、HDF5 采集数据和调试输出保留在本机，不上传。
在另一台机器使用时，先准备原始 `example_agilex/piper_description` 资料，
修改 `configs/workcell.yaml` 的 `robot.source`；需要连接真机时再配置 `hardware.sdk_path`
及 CAN 通道。当前配置中的绝对路径对应开发机器，原始 SDK 和机器人描述不在本仓库中。
安装脚本会据此重新生成机器人资产及 CAD 场景。文档中的 episode / 日志路径是本地验证记录，
需要在新机器重新采集。

使用独立 `.venv-isaac`，不修改已有 conda 环境。官方固定版本参考：
[Isaac Lab v2.3.0 安装说明](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html)、
[Isaac Sim 5.1 安装说明](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/index.html)。

```bash
cd /home/midea/Desktop/piper_sim
bash scripts/install_runtime.sh
.venv-isaac/bin/python scripts/doctor.py

# GUI 场景，默认不连接任何 CAN 通道
bash scripts/run.sh --enable_cameras

# 有限步数无窗口测试；--demo 仅在纯仿真模式可用
bash scripts/run.sh --headless --enable_cameras --steps 120 --demo

# 采集 10 秒、30 Hz 的三路 RGB 和状态动作
bash scripts/run.sh --headless --enable_cameras --steps 300 --demo \
  --record outputs/episode_0001.h5 --task "检查双臂关节跟踪"
.venv-isaac/bin/python scripts/check_episode.py outputs/episode_0001.h5

# 回放记录的动作目标到仿真
bash scripts/run.sh --mode replay --trajectory outputs/episode_0001.h5

# 回放已通过验收的双臂动作；退出前保存总览（输出文件须不存在）
bash scripts/run.sh --enable_cameras --mode replay \
  --trajectory outputs/cad_restore_acceptance/dual_piper.h5 --overview-image outputs/overview.png
```

仿真时钟中物理频率 120 Hz、控制和采集频率 30 Hz。真实图像渲染需 `--enable_cameras`。
默认采集两路 1920×1080 和一路 1280×800。`cameras.image_scale: 0.5` 可将完整画幅
等比例缩小，降低渲染和落盘开销。本机原始画幅同步采集实测约 8.53 帧/墙钟秒；
仿真采样频率不代表墙钟实时性能保证。
硬件模式会按控制周期限速，实际频率仍受渲染、记录和通信开销影响。
首次运行涉及 USD 转换、扩展缓存和着色器编译，可能明显慢于后续运行。
初次启动遵循 NVIDIA 的 EULA 提示。启动脚本隔离桌面 shell 的 ROS Python 3.12 路径，
避免污染 Isaac 的 Python 3.11。

完整版本固定在 `requirements/runtime-linux-py311.lock`。采用 Lab 2.3.0 是因为
2.3.1 / 2.3.2 的 Starlette 固定版本与 Sim 5.1 的 FastAPI 依赖范围冲突。
项目内相机适配器读取实时 Fabric 位姿；显式分离物理步进与渲染，并处理
Lab 2.3.0 的退出回调。没有修改 Isaac Lab 或原 Piper SDK 源码。

下载不稳定时，使用 `download_runtime_wheels.py` 读取带 index 注释的
`uv pip compile --emit-index-annotation` 输出，将安装包断点续传到 `.runtime/wheels`，
逐一验证源索引公布的 SHA256。安装器优先复用这些本地文件。

CAD 转换需要资产准备环境中的 `cadquery-ocp==7.7.2`、`usd-core==26.8`。
转换器从冻结的 Piper 子装配中单独提取腕部附件，生成可挂载的 USD；
机器人主体使用可动 URDF。默认 `scene.geometry: cad`，切换为 `primitives`
只简化桌面和立柱，仍保留腕部附件。缺少 CAD 产物会明确报错。

```bash
PYTHONPATH=src .venv/bin/python scripts/convert_workcell_cad.py
PYTHONPATH=src .venv/bin/python scripts/prepare_cad_gripper.py
PYTHONPATH=src .venv/bin/python scripts/verify_cad_alignment.py
# 输出目录使用新名称，避免覆盖已有截图
PYTHONPATH=src .venv-isaac/bin/python scripts/capture_cad_reference.py --output outputs/reference_new
```

`.venv` 用于轻量工具和 STEP 元数据检查，`.venv-isaac` 用于仿真。所有代码也可以通过
`PYTHONPATH=src` 在已有合适的 Python 环境中进行离线验证。

## 双向主从控制

每次只选择一个主从方向，防止互相追随形成反馈环路。软件桥运行在本机；
跨机器通信需要另外实现时钟同步和通信适配器，不应暴露当前本机 HTTP 端口。

### 真机 → 仿真

先在 `configs/workcell.yaml` 填写真实的左右 CAN 通道名称：

```bash
bash scripts/run.sh --mode real-to-sim --enable_cameras
```

此模式以真实关节反馈作为仿真位置目标，只读取真实机器人。不会使能电机、
改变固件主从模式或导入旧的自动控制线程。真实主臂的拖动/示教模式由原有设备流程配置。
SDK 通过配置中的 `sdk_path` 延迟加载，不需要把旧遥操作程序的依赖装入仿真环境。
独立环境已安装兼容 Sim 依赖的 `python-can 4.2.2`、`setuptools 80.9.0`，
并验证 SDK 可以导入；未打开 CAN 或发送硬件命令。

### 仿真 → 真机

```bash
bash scripts/run.sh --mode sim-to-real --enable_cameras \
  --command-port 8765 --allow-actuation
```

程序先读取真实姿态并将仿真对齐。外部控制源向本机 `/target` 连续提交仿真目标，
仿真的测量关节位置经过独立限速后发送给真机。两臂电机必须已由操作员使能；
程序不自动上电或回零。首个指令必须与当前状态对齐。无新鲜指令时不启动真机控制，
启动后断流、反馈缺帧、异常或退出均触发两臂停止请求。

**这些保护是软件层实现，尚未完成实机验收，且无法替代硬件急停。** CAN 发送故障时停止
报文也可能无法送达；Python 线程不是硬实时控制器。先完成仿真与现场校准，再开展真机联调。
当前机器检查时没有 CAN 接口，因此本次不会驱动真实机械臂。

### 控制入口

纯仿真可先用 `bash scripts/run.sh --command-port 8765` 验证控制源。

```python
import json, time, urllib.request

base = "http://127.0.0.1:8765"
state = json.load(urllib.request.urlopen(base + "/state"))
q = state["position"]  # 从当前姿态起步；14 维，顺序见下文
for sequence in range(300):
    body = json.dumps({"position": q, "sequence": sequence}).encode()
    request = urllib.request.Request(base + "/target", data=body,
                                     headers={"Content-Type": "application/json"})
    urllib.request.urlopen(request).close()
    time.sleep(1/30)
```

控制源退出或 150 ms 无更新会终止控制会话。每次会话 sequence 递增。
时间戳由接收端的 monotonic 时钟生成；该接口不是远程网络协议。
后续可把 Quest、实体主臂、键盘、IK 或策略输出接到同一契约，核心记录器无需改动。

## 数据契约

状态和绝对位置动作都是 14 维：

```text
[left.joint1 ... left.joint6, left.gripper_width,
 right.joint1 ... right.joint6, right.gripper_width]
```

关节角是 rad，夹爪开口是 m，RGB 图像是 H×W×3 uint8。一个 HDF5 文件对应一个 episode：

| 路径 | 含义 |
|---|---|
| `observation/state` | 动作前观测 |
| `action/position` | 本周期施加的绝对位置目标 |
| `next_observation/state` | 动作后观测 |
| `observation/images/{overhead,left_wrist,right_wrist}` | 动作前 RGB |
| `next_observation/images/...` | 动作后 RGB |
| `*/timestamp` | 本机 monotonic 观测 / 动作时间 |
| `*/sim_time` | 仿真时钟，缺失时为 NaN |
| `*/image_timestamps/...` | 仿真图像采样时刻，使用仿真时钟 |
| `*/extras_json` | 相机内参、世界位姿；硬件模式另含 SDK 原始时间戳 |
| 文件 metadata | 任务文本、配置、频率、模式和数据来源 |

仿真→真机模式记录真实状态和实际限速后的硬件动作，图像仍来自仿真，
metadata 明确标记这种混合来源，不能当作真实相机数据训练视觉策略。
原始真实相机采集需要后续接入独立传感器后端。

记录先写 `.h5.partial`，正常结束后原子重命名；异常时保留 partial 便于诊断。
不覆盖既有 episode。不自动把测试动作标为任务成功。
`dataset.sequence_windows()` 返回连续状态、动作、图像序列，窗口不跨 episode；
可作为后续 WAM 数据入口。当前格式是项目原始格式，**不是开箱即用的 LeRobot 数据集**。
接入具体 VLA 时，应独立实现其图像处理、动作空间与归一化导出，保留原始 episode。

## 后续扩展边界

1. 标定基座间变换、TCP、夹爪行程、相机内外参，辨识真实关节驱动参数。
2. 为具体任务添加物体、重置条件、成功判定和域随机化。
3. 将单工作站后端封装为 Isaac Lab 的向量化任务环境，再做批量训练；当前启动器是单工作站。
4. 选择具体 VLA / WAM 后增加对应数据导出器与策略客户端。
5. 如需 ROS 2 或跨主机控制，作为新通信适配器实现，不让 ROS / CAN 依赖渗入数据与场景层。
