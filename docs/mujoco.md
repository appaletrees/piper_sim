# MuJoCo 双臂 Piper

MuJoCo 3.3.7 / Python 3.11 运行在独立 `.venv-mujoco` 中，与 Isaac 版本共存。
用户提供的 STEP 是尺寸、机械安装位置与支架形状的固定基准。
两个引擎共享 `configs/workcell.yaml`、公开 RGB 光学参数、14维状态/动作契约、
Piper SDK 适配器、HTTP 控制接口、HDF5 记录器和回放逻辑。

![MuJoCo 实际场景](../outputs/mujoco_validation/overview.png)

## 安装与运行

在项目根目录执行，需已安装 `uv`。普通安装只读取随仓库分发的 MJCF 和网格，
不需要 Isaac、Torch、USD、原始机器人描述目录或 CAN 硬件。

```bash
bash scripts/install_mujoco.sh
bash scripts/run_mujoco.sh --enable_cameras

# 手动滑条控制：每臂六关节＋一个gripper开口，可同时使用 --record。
bash scripts/run_mujoco.sh --interactive --enable_cameras

# 纯仿真：双臂小幅运动，采集4秒仿真时间，共120个transition。
# 输出文件必须不存在，避免覆盖已有数据。
bash scripts/run_mujoco.sh --headless --enable_cameras --steps 120 --demo \
  --record outputs/mujoco_episode_0001.h5
.venv-mujoco/bin/python scripts/check_episode.py outputs/mujoco_episode_0001.h5

# 同一HDF5动作可用于MuJoCo或Isaac回放。
bash scripts/run_mujoco.sh --enable_cameras --mode replay \
  --trajectory outputs/mujoco_episode_0001.h5
```

Linux 无窗口渲染默认使用 EGL；GUI 使用 GLFW。两者均在本机 RTX 5090 D 上通过检查。
原生窗口显示总览场景，`--enable_cameras` 启用三路 RGB 读回和记录。
`--interactive` 下展开右侧 `Control`，使用 `left/gripper`、`right/gripper` 控制整只夹爪，单位m。
每个夹爪只有一个独立控制量。两根物理手指仍有各自的滑动坐标，由约束保持联动；
`Joint` 面板中的手指坐标是内部状态，不是两路独立控制目标。
普通启动模式保持初始目标；交互模式不能与 `--demo`、回放或HTTP控制源混用。
纯物理测试可省略该选项。仿真采集频率为30 Hz，MuJoCo 内部积分为480 Hz，
每个控制周期推进16步。原始三路画幅采集本次实测约21.6帧/墙钟秒，
同时存在其他GPU任务；这不是实时性能承诺。

MuJoCo 动力学运行在 CPU，RGB 使用 OpenGL 渲染。
其光照、材质及接触求解与 Isaac 不同，不把两引擎的像素颜色或运动轨迹视为逐值相等。
参考 [官方 Python 接口](https://mujoco.readthedocs.io/en/stable/python.html) 与
[MJCF 相机参数](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera)。

## 同源资产与相机

`assets/mujoco/workcell.xml` 使用相对路径，可直接由 MuJoCo 加载。
模型包约33 MB，包含7个共用 Piper STL，以及99个源 CAD 网格：
51个静态场景部件、20个腕部相机/支架部件、28个原 CAD 夹爪部件。
不导入视场示意 guide 或重叠的备用相机。

- 桌面1400×700×25 mm，两臂基座间距600 mm，立柱16×40×1000 mm。
- 双臂按源URDF关节树运动；相机与原支架随 `link6` 运动。
- 左右腕为 D435i RGB，1920×1080；中央为 D455 RGB，1280×800。
- ROS 光学坐标转换为 OpenGL 相机坐标时保留安装变换；支持独立 fx/fy、偏心主点和整数像素中心。
- 相机光心采用原先已记录的公开标称镜头偏移。每台设备 K 与畸变的实测需求仍见 [RGB参数说明](realsense_rgb.md)。

`manifest.json` 记录源 STEP / URDF 哈希、相机参数、各部件父节点、尺寸和文件哈希。
`kinematics.urdf` 去掉视觉及碰撞网格路径，仅用于独立FK检查。
运行时校验模型包；存在本地源资产时还会核对来源哈希，配置或源资产变化则重新生成。
对相机/场景配置进行修改后，若本机没有源资产，需要先重建资产。

```bash
# 仅重新构建时才需要原始 robot.source 目录及 CAD / USD 转换依赖。
bash scripts/install_mujoco.sh --rebuild-assets

# 源资产已准备好时，导出到可分发目录。
PYTHONPATH=src .venv-mujoco/bin/python scripts/prepare_mujoco.py --output assets/mujoco
```

普通安装与重建都不修改原始 STEP、Piper SDK 或 Isaac 环境。
MuJoCo 的地面和测试方块是仿真辅助物体。机器人、相机、原支架与桌面均保持原设计尺度。

## 采集与双向主从

状态/动作顺序为左臂6关节+夹爪开口，再右臂6关节+夹爪开口；关节rad，开口m。
MuJoCo内部由一个tendon驱动总开口，两手指等量反向运动；Isaac使用URDF mimic约束和一个主动指驱动。
三路RGB、K、世界光学位姿、仿真时间和动作前后状态
使用与 Isaac 相同的 HDF5 schema；metadata 的 `source` 和 `backend` 标明 `mujoco`。
可供后续VLA数据导出和WAM时序窗口读取，尚未绑定特定训练框架。

```bash
# 本机HTTP控制源 → MuJoCo
bash scripts/run_mujoco.sh --command-port 8765

# 真机只读反馈 → MuJoCo
bash scripts/run_mujoco.sh --mode real-to-sim --enable_cameras

# MuJoCo测量运动 → 真机；需要已使能且对齐的实体双臂和连续指令流
bash scripts/run_mujoco.sh --mode sim-to-real --enable_cameras \
  --command-port 8765 --allow-actuation
```

主从模式复用现有新鲜度、限位、限速和停止逻辑，操作步骤见 [README](../README.md#双向主从控制)。
本次已检查SDK导入和离线保护测试，未连接CAN、未驱动真机。
不需要仿真窗口时，可直接使用 [PiperConnection / connect_piper.py](piper_connection.md)。

## 已完成验证与边界

2026-10-21 本机验证：

| 检查 | 结果 |
|---|---|
| 项目测试 | 52项通过；包含FK、单侧负载下的夹爪联动、真机连接与CAN异常处理 |
| 单输入夹爪 | 两引擎10/40/60 mm开口检查通过；Isaac双指位置差最大约2.92 μm |
| CAD网格转换 | 99个编译后网格转换回父节点后，包围盒与源资产一致，容差2 μm |
| 三路实际渲染投影 | 全幅/偏心K最大约0.06 px；640×480中心裁剪模式约0.12 px |
| 动力学与采集 | 120个transition；全部关节/夹爪运动、三路RGB、30 Hz仿真时间通过 |
| 关节跟踪 | 最差关节误差P95约0.0147 rad；夹爪约0.089 mm |
| 接触 | 方块稳定在桌面上，中心高44.784 mm，几何接触目标45 mm |
| GUI / 回放 | 原生窗口回放通过；Isaac入口5步回归通过 |
| 独立包 | 无generated目录且禁止外部运行时导入，三路RGB采集成功；34项测试通过，7项源资产测试按条件跳过 |

详细记录：[动力学和采集](../outputs/mujoco_validation/report.json)、
[全幅投影](../outputs/mujoco_validation/projection_native/projection.json)、
[640×480投影](../outputs/mujoco_validation/projection_vga/projection.json)、
[偏心内参投影](../outputs/mujoco_validation/projection_asymmetric/projection.json)、
[独立包检查](../outputs/mujoco_validation/portable_smoke.json)。
夹爪改动后的新检查：[MuJoCo联动](../outputs/gripper_coupling/mujoco_gripper.json)、
[Isaac联动](../outputs/gripper_coupling/isaac_gripper.json)、
[MuJoCo采集](../outputs/gripper_coupling/mujoco/report.json)、
[Isaac采集](../outputs/gripper_coupling/isaac/report.json)。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-mujoco/bin/python -m pytest -q
.venv-mujoco/bin/python scripts/verify_simulation.py outputs/mujoco_episode_0001.h5 \
  --output outputs/mujoco_check_new
MUJOCO_GL=egl .venv-mujoco/bin/python scripts/verify_mujoco_projection.py \
  --output outputs/mujoco_projection_new
.venv-mujoco/bin/python scripts/verify_gripper.py --backend mujoco \
  --output outputs/mujoco_gripper_new.json
```

几何转换与渲染投影通过表示仿真实现忠于输入设计及K；并不等于已经拿到每台真机的出厂标定。
所有CAD碰撞按分部件凸包处理，桌面为精确尺寸盒体；凹孔和夹爪软垫接触尚非精细模型。
惯量沿用源URDF，额外相机/支架质量未辨识，驱动增益为仿真参数。
原URDF的70 mm夹爪限位仍保留；历史数据更大的SDK行程需先解决映射，不能直接发送。
这些边界不改变原设计尺寸。真实照片、视频和HDF5验收episode保留在本地。
