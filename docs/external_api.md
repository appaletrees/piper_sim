# 外部控制、采集与场景随机化

对照资料 `example_agilex/piper_controller.py`、`examples/common/data_manager.py`、
`multi_camera.py` 和 `piper_recorder.py`：外层保持“读取当前状态 → 更新双臂目标 → 发送 → 采集”的流程。
仿真核心、协议、单位转换、任务随机化和记录器分别独立。Isaac和MuJoCo使用相同接口。

## 夹爪0.07的来源

- 旧URDF两指各35 mm，本项目早期使用总开口70 mm，所以GUI最大是0.07 **米**。
- 原控制器 `GRIPPER_DEGREES_MAX=95`，`set_gripper_open_value(1)` 对应95 mm，
  `GripperCtrl(95000, ...)` 单位为0.001 mm；其中“degrees”只是旧变量命名。
- 随附SDK `piper_param/piper_param_manager.py` 默认 `gripper_range=[0,.07]`，
  控制器还开启 `start_sdk_gripper_limit=True`，因此两处定义原本就不一致。
- 当前仿真统一为0–0.095 m，两指各0–0.0475 m、一个夹爪控制量。原CAD尺寸和安装位姿不变。
  这是与控制器/数据表达范围对齐，不是对真实机械行程的标定。真机读取SDK实际范围并拒绝超限，见 [真机接口](piper_connection.md)。

关闭旧MuJoCo窗口后重新启动，Control滑条才会加载0.095上限。

## 独立进程接口

在项目根目录启动；默认双臂加两种插头，任务配置可换：

```bash
bash scripts/run_mujoco.sh --task-scene configs/tasks/connector.yaml \
  --enable_cameras --api-port 8765 --record-dir outputs/api_episodes
```

Isaac替换为 `bash scripts/run.sh`；无窗口加 `--headless`。`--command-port` 与 `--api-port` 等价。
需要鼠标控制时加 `--interactive`（仅MuJoCo），这时API仍可读图、采集和reset，目标写入返回403。
每次只选择一个目标来源。原生相机分辨率较高；匹配历史采集的640×480模式可加
`--config configs/connector_alignment.yaml`，其中K仍是名义估计，设备标定文件可另行加载。

另一个进程运行完整例子：

```bash
PYTHONPATH=src .venv-mujoco/bin/python examples/external_agilex.py --episodes 2 --seed 42
```

该例子只接受纯仿真服务：每集重置为不同seed，读取初始姿态，连续发送双夹爪0–1目标，
录制实际执行过程，再保存三路PNG及状态；它演示控制/采集流程，不执行自动插接。

### 与原工程类似的Python调用

```python
import time
from piper_sim.client import SimulationClient
from piper_sim.agilex import AgilexAdapter

client = SimulationClient("http://127.0.0.1:8765")
robot = AgilexAdapter(client)
robot.reset(seed=42, randomize=True)
robot.send()  # 首条目标与读取的姿态一致
robot.start_recording("episode_000001", task="charger insertion")
try:
    for _ in range(150):
        # setter只是更新目标缓存，send一次提交完整双臂目标。
        robot.set_target_joint_angles([0, 0, 0, 0, 0, 0], arm="left")  # 度
        robot.set_gripper_open_value(1.0, arm="left")                 # 0..1
        robot.send()
        time.sleep(1/30)
finally:
    print(robot.stop_recording(success=False))
robot.refresh()  # 一次原子读取后，下列getter共享同一帧
print(robot.get_current_joint_angles(arm="left"))
print(robot.get_current_gripper_open_value(arm="left"))
frames = robot.capture_frames()
policy_obs = robot.get_observation()
```

| 原工程用法 | 适配接口与含义 |
|---|---|
| `get_current_joint_angles` / `set_target_joint_angles` | 同名方法，6个度，`arm='left'/'right'` |
| `get_current_gripper_open_value` / `set_gripper_open_value` | 同名方法，一个归一化值，1=95 mm |
| DataManager目标读取 | `get_target_joint_angles`、`get_target_gripper_open_value` 为客户端请求目标 |
| 双臂策略动作 | `send_action(action)` 接受14维 `[左6度,左夹爪0..1,右6度,右夹爪0..1]` |
| MultiCameraManager读取 | `capture_frames()` 返回 `cam_external/cam_left_wrist/cam_right_wrist` |
| 原相机的 `frames[name]['rgb']` | 与原工程一致为 **BGR** uint8；`depth=None`，当前只提供RGB |
| 策略输入 | `get_observation()` 返回 `observation.state`（度/归一化）与三路 `observation.images.cam_*_rgb`（**RGB** HWC） |
| episode开始/停止 | `start_recording(name, task)` / `stop_recording(success=False)` |

`refresh()` 显式更新读缓存；getters不各自读取不同帧。适配层支持关节控制流程，
不导入旧工程，不模拟其CAN使能、SDK缓存或末端IK接口。需要末端控制时由外部IK产生6关节目标。
原始物理反馈保留微小限位穿透，不截断观测；新控制目标必须在端点公布范围内。

### HTTP契约

仅监听 `127.0.0.1`。跨机器可使用SSH端口转发；不直接暴露到公网。
原始接口统一使用SI，**不能把兼容层的度/0–1直接提交给 `/target`**。

| 请求 | 内容 |
|---|---|
| `GET /metadata` | 协议版本、14维顺序/单位/限位、相机配置、任务配置/布局、控制能力 |
| `GET /state` | 14维测量值、时间、frame_id、reset_id、控制应用状态、相机外参/K、物体位姿/任务指标 |
| `POST /target` | `{"position":[14个rad/m值],"sequence":0}`；204表示收到，不代表已执行 |
| `GET /observation.npz` | 同一帧的state_json、position和三路 `rgb_<name>`；numpy读取不需要pickle |
| `GET /cameras/overhead.png` | 单相机PNG；另有left_wrist、right_wrist；响应头X-Observation-Id标记帧号 |
| `POST /reset` | `{"seed":42,"randomize":true}`，202排队，`state.reset_id`确认执行 |
| `POST /recording/start` | `{"name":"episode_000001","task":"charger insertion"}`；写入服务端record-dir |
| `POST /recording/stop` | `{"success":false}`，结束当前集 |
| `GET /recording` | active、frames、path、error和已完成request_id |

`SimulationClient` 封装上述HTTP操作，reset及录制生命周期会等待执行确认。
`read()` 返回统一 `Observation`，`observation()` 返回字典形式的同帧RGB与状态。
多个独立PNG请求可能跨帧，训练读取请用原子NPZ或服务端记录器。

`control.applied_sequence` 是已应用输入序号，`applied_target` 是限速后目标；
`rejected_sequence/error` 说明交接不对齐、跟踪超差等拒绝原因。HTTP400表示格式、顺序或限位错误。
输入只保留最新一条，适合一个持续控制源，sequence在reset后也必须递增；
Python客户端自动维护序号。首次交接与测量值相差需不超过关节0.15 rad、夹爪10 mm。
连续输入建议30 Hz，限速0.5 rad/s、0.03 m/s；一次发送远目标不会自动持续插值到终点。
150 ms断流时纯仿真保持最后接受的目标，API可继续访问；真机模式执行已有停止流程。
HTTP工作线程只交换数据，步进、重置和写文件都由仿真线程负责。
30 Hz指仿真时间采样频率；高分辨率渲染可能使实际墙钟运行更慢。记录保留两种时间，
不会把收到的每条HTTP命令虚构成一个物理步；需要严格逐步交互时使用下面的同步接口。

## 同进程训练/数据生成接口

需要确定性的“一个action对应一个step”时使用 `SimulationSession`，避免用HTTP墙钟频率模拟训练步进：

```python
from piper_sim.simulation import SimulationSession
from piper_sim.agilex import AgilexAdapter

with SimulationSession.mujoco(task_scene="configs/tasks/connector.yaml",
                             cameras=True, record_dir="outputs/training") as sim:
    robot = AgilexAdapter(sim)
    for episode in range(2):
        robot.reset(seed=100+episode, randomize=True)
        robot.start_recording(f"demo_{episode:06d}", task="charger insertion")
        for step in range(90):
            robot.set_gripper_open_value(min(1., .4+step*.005), arm="right")
            robot.send()        # 缓存下一步双臂目标
            obs = sim.step()    # 每步1/30仿真秒；返回实际测量值、RGB和任务信息
        robot.stop_recording(success=False)
```

本地step使用绝对目标和模型限位，不套HTTP的墙钟限速。也可直接 `sim.step(action_si)`。
Isaac先按原启动流程创建AppLauncher与 `IsaacBackend(cfg, ...)`，再用
`SimulationSession(backend)` 包装；Session关闭backend，AppLauncher仍由创建者关闭。

## 场景随机化与记录

`configs/tasks/connector.yaml` 独立定义插座XY/yaw、两种插头XY/yaw和最小间距。
`reset(seed=42, randomize=True)`可复现同一布局；换seed产生不同布局；
`randomize=False`使用nominal固定摆放。省略seed会复用配置seed，不隐式生成新随机数。
完整物体轮廓必须在桌面内且互不重叠。可修改配置的随机范围并重启；机器人和相机安装尺寸不随之变化。
当前随机化是物体位置/朝向，不包含材质、灯光、动力学随机化。

每个episode必须先reset、再start、最后stop；录制中reset返回409/异常，避免一集混入多个布局。
HTTP录制只在纯仿真且未使用CLI `--record` 时开放；CLI固定录制期间不允许远程reset。
文件名重复会报错，防止覆盖已有数据。正常停止生成 `.h5`，异常中断/空集保留 `.h5.partial`。

记录器保留项目的 `piper.transitions.v1`：`observation/state`、实际 `action/position`、
`next_observation/state`，以及前后帧RGB、相机时间和extras_json。内部单位rad/m，
每个文件metadata保存原始任务配置、采样布局、seed、相机配置、控制频率与时钟定义。
它与旧PiperRecorder的分目录H5+MP4不是相同磁盘格式；两者的字段/单位映射已由适配层明确提供，
策略输入使用相同14维度/归一化顺序和相机名称。
可选添加 `examples/external_agilex.py --lerobot-output outputs/lerobot_v2/demo_001`，
采集结束后生成LeRobot v2.1；已有HDF5也支持独立转换，见 [导出说明](lerobot_export.md)。
时间戳：state.timestamp为本机monotonic，RGB及sim_time为仿真秒；reset后sim_time重新从0开始。

## 本次验证

- 66项自动测试通过，含95 mm双指联动、SDK70 mm拒绝超限、显式95 mm单位换算、HTTP数据及episode边界。
- [MuJoCo实际客户端测试](../outputs/external_api/report.json)：seed42/43两集各181个transition，三路640×480 RGB，夹爪实测约95 mm。
- [Isaac实际客户端测试](../outputs/external_api/isaac_report.json)：89个transition、三路RGB、95 mm控制、seed复现。
- [Isaac同步reset检查](../outputs/external_api/isaac_reset.json)：reset不推进物理，首帧状态、物体布局与RGB一并刷新。
- [MuJoCo夹爪](../outputs/gripper_coupling/mujoco_95mm.json) / [Isaac夹爪](../outputs/gripper_coupling/isaac_95mm.json)：10/40/70/85/95 mm均通过；未执行真实CAN运动测试。
