# 真实 Piper 双臂连接接口

接口位于 `src/piper_sim/piper_connection.py`，命令行入口为 `scripts/connect_piper.py`。
直接使用提供的 `piper_sdk.C_PiperInterface_V2`，不加载 Isaac、MuJoCo 或旧遥操作控制线程。
仅写接口和执行离线检查；开发机当前 `--list-can` 输出为空，尚未完成真机CAN联调。

## 与原真机控制的对应关系

核对资料为 `example_agilex/piper_controller.py` 的 `_send_gripper_command()`、
`piper_sdk/piper_sdk/interface/piper_interface_v2.py` 的 `GripperCtrl()`，以及SDK的
`demo/V2/piper_ctrl_gripper.py`。原代码每臂一次 `JointCtrl` 加一次 `GripperCtrl`，
没有分别控制左右手指的硬件指令。

| 项目API | SDK调用 | 换算 |
|---|---|---|
| 每臂6个 `joints_rad` | `JointCtrl(j1, ..., j6)` | rad → 0.001° |
| 每臂1个 `gripper_width_m` | `GripperCtrl(width, effort, 0x01, 0)` | m × 1,000,000 → 0.001 mm |
| `read().position` | 关节反馈及 `GetArmGripperMsgs()` | 相反方向换算，仍为14维 |

例如0.04 m开口发送 `GripperCtrl(40000, ...)`，只发送一次，不分别给手指发送20000。
原控制器把夹爪字段叫作degrees，但SDK定义是长度；旧归一化0–1对应0–95 mm。
仿真现在使用0–95 mm总开口。原控制器开启了SDK限位，但随附SDK的默认范围为0–70 mm，
因此旧控制器请求95 mm仍可能被SDK裁到70 mm。真机接口通过 `GetSDKGripperRangeParam()`
读取实际软件范围；`robot.limits()` 和HTTP `/metadata` 返回左右范围，超限目标明确拒绝，读反馈不截断。
`hardware.gripper_max_width_m: null` 保留SDK当前范围；设备可用行程确认后可显式配置 `0.095`，
通过 `SetSDKGripperRangeParam(0, .095)` 修改当前SDK实例的软件限位，不修改固件或机械行程。
仿真范围与实际指尖开口的标定仍需真机测量。

MuJoCo使用一个总开口驱动及双指等量约束，Control只提供每臂一个gripper滑条；
Isaac使用 `joint8` mimic `joint7`，只给主指设置驱动目标。
物理模型仍需要两根可移动手指来计算几何与接触，独立控制自由度为1。
对应的MuJoCo约束机制见 [官方MJCF说明](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#equality-joint)。

## 连接与读取

在项目根目录执行，`can0/can1` 必须替换为实际左右臂通道。接口不修改系统CAN配置。
先接入并按原设备流程配置USB-CAN，确认左右映射和设备要求的bitrate。

```bash
# 只列出现有CAN接口，既不连接机器人，也不配置网络。
.venv-mujoco/bin/python scripts/connect_piper.py --list-can

# 连接双臂，读取一次完整的新鲜反馈，然后断开。
.venv-mujoco/bin/python scripts/connect_piper.py \
  --can-left can0 --can-right can1

# 持续读取，Ctrl+C退出。
.venv-mujoco/bin/python scripts/connect_piper.py \
  --can-left can0 --can-right can1 --samples 0 --rate 10
```

SDK路径沿用 `configs/workcell.yaml` 中的 `hardware.sdk_path`。
连接检查通道存在且UP，再以 `ConnectPort(piper_init=False)` 接收数据，等待两臂全部
关节对及夹爪反馈新鲜后才返回；默认5秒超时。不会自动上电、回零或切换固件主从模式。

```python
from piper_sim.piper_connection import PiperConnection, state_dict

with PiperConnection(can_left="can0", can_right="can1") as robot:
    state = robot.read()
    print(state_dict(state))
    print(robot.limits())
    # state.position: 左6关节、左夹爪、右6关节、右夹爪
```

## 显式控制API

物理电机需已由现有设备流程使能。`allow_actuation=True` 加 `arm()` 才允许发指令；
首个目标需与反馈对齐，持续命令仍经过新鲜度、顺序、限位、跟踪误差和速度检查。
下面示例只保持连接时读到的位置。控制源通常以30 Hz更新目标。

```python
import time
from piper_sim.piper_connection import PiperConnection

with PiperConnection(can_left="can0", can_right="can1", allow_actuation=True) as robot:
    q = robot.read().position.copy()
    robot.arm()
    for _ in range(30):
        accepted = robot.send(
            left_joints_rad=q[:6], left_gripper_width_m=q[6],
            right_joints_rad=q[7:13], right_gripper_width_m=q[13],
        )
        time.sleep(1/30)
```

只改变夹爪时，持续调用 `robot.set_gripper_widths(left_m=..., right_m=...)`；
机械臂关节保持上一条已接受的目标。返回值是经过速度限制的实际发送目标。
已有 `Command` 可用 `robot.apply(command)`，其时间戳和sequence不会被接口刷新。
停止发送超过默认150 ms、发送失败或反馈过期都会进入故障停止流程，需要重新连接。
SDK原本只记录日志的发送错误也会转为异常，交给双臂停止逻辑处理。
执行过控制的上下文退出时会发送停止请求，纯读取连接退出仅断开。

## 本机HTTP接口与仿真联动

```bash
# 只读状态服务：GET /state；POST /target会返回403。
.venv-mujoco/bin/python scripts/connect_piper.py \
  --can-left can0 --can-right can1 --command-port 8765

# 显式启用目标接收：收到首个有效指令后才进入控制。
.venv-mujoco/bin/python scripts/connect_piper.py \
  --can-left can0 --can-right can1 --command-port 8765 --allow-actuation
```

服务仅监听127.0.0.1。请求格式与现有仿真控制入口一致：
`POST /target` 的JSON为 `{"position": [14个SI值], "sequence": 递增整数}`；
客户端先读 `/state` 对齐，然后持续更新目标。重复旧命令不会延长控制源的有效期。

需要真实主臂带动仿真，或仿真带动真机时，使用已有 `run_mujoco.sh --mode real-to-sim`
与 `--mode sim-to-real`，并在配置中填写相同CAN映射；Isaac使用 `run.sh`。
同一双臂应由一个控制进程负责，不同时启动独立真机服务和仿真主从服务。

离线测试覆盖只读连接、单位换算、每臂一次夹爪命令、超时清理、陈旧命令、CAN发送失败及只读HTTP拒绝。
这验证软件行为，不代表物理控制已在当前设备验收。
