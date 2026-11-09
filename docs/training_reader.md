# 从 HDF5 读取训练序列

`piper_sim.dataset.sequence_windows()` 默认只返回通过项目独立 RGB 审计的完整回合。
它与 LeRobot 的 `--require-quality` 使用同一验收函数：检查成功/验收标记、选定策略的
指纹、内嵌与同目录报告的一致性，以及实际打开的 HDF5 文件的 SHA-256。
不是只看 `success=true`，也不会自动给文件补上验收标签。

这是默认行为变更：原来直接调用会读取结构合法但未验收的回合，现在会拒绝。
只做原始状态诊断时，显式传 `require_quality=False`。`validate_episode()` 仍然只是
结构检查，检查通过不能代替训练验收。当前生产任务审计针对完整刚体收桌任务；其他
任务或硬件记录不会因可以读出数组就自动获得这一资格。

## 使用方式

```python
from contextlib import closing
from piper_sim.dataset import sequence_windows

episode = "/home/midea/Desktop/piper_sim/outputs/rigid_batch_001/seed_0/episode.h5"
with closing(sequence_windows(episode, horizon=16, stride=1)) as windows:
    for window in windows:
        # 根据模型契约显式选字段；不要把整个 window 当成策略输入。
        policy_inputs = {
            "state": window["state"],
            "images": window["images"],
            "task": window["task"],
        }
        action_targets = window["action"]
        # 在这里交给自己的训练器；不提供默认归一化或模型专用时间偏移。
```

`horizon` 与 `stride` 必须是正整数。窗口内部逐帧连续，`stride` 只决定窗口起点间隔。
不补齐尾窗，不跨 episode，不从别的回合补帧。`closing` 在提前退出循环时也会关闭
生成器、释放 HDF5 读锁。逐窗口消费即可，不要把整个 RGB 回合转成 `list(windows)`。

可通过 `policy="/absolute/path/quality_policy.yaml"` 显式选择验收策略。
策略指纹不匹配时需要重新独立审核；读取器不会采用录制元数据中的策略来放宽条件。
每次开始一个生成器只对完整文件做一次哈希，并使用同一只读句柄读取所有窗口。
因此每个训练 epoch 重新创建生成器会再次产生一次完整文件读取成本。

诊断读取示例：

```python
with closing(sequence_windows(episode, horizon=16, require_quality=False)) as windows:
    first = next(windows)
```

诊断模式不执行训练验收，不修改文件的资格，也不允许同时传入 `policy` 后静默忽略。
它适合状态预检、失败轨迹排查和原始记录查看，不能把其输出直接列入合格训练清单。

## 字段和时间语义

每个窗口只返回下列字段，不包含 `extras_json`、物体位姿、教师状态、奖励或完整 metadata。

| 字段 | 含义 |
| --- | --- |
| `state` | 动作前实测状态，形状 H×14；左 6 关节 rad、左夹爪开口 m、右 6 关节 rad、右夹爪开口 m |
| `action` | 同一 transition 实际施加的绝对位置目标，形状 H×14，单位和顺序与 `state` 相同 |
| `next_state` | 该动作作用后的实测状态；可用作世界模型预测目标，不能当作该时刻的因果策略输入 |
| `images` | `overhead`、`left_wrist`、`right_wrist` 对应动作前 RGB，H×高度×宽度×3，uint8 |
| `task` | 原记录的任务文本，不自动改写或补造指令 |
| `timestamp` | 动作前观测的主机 monotonic 秒数；保留原字段含义，不是 Unix 时间或仿真时间 |
| `sim_timestamp` | 动作前的仿真秒数，保留录制开始时的偏移 |
| `next_sim_timestamp` | 动作后的仿真秒数 |
| `action_timestamp` | 应用动作时的主机 monotonic 秒数 |
| `action_sequence` | 原记录的动作序号 |
| `image_timestamps` | 各相机的仿真采样秒数，对应动作前观测 |

训练动态模型或计算动作间隔应使用仿真时钟。渲染/写盘耗时会影响主机间隔，不能用它
推断机器人在仿真中的速度。诊断读取中没有仿真时钟的硬件记录可保留 NaN，不伪造
采样时刻；生产 RGB 审计会拒绝缺失或不连续的仿真时钟。

HDF5 窗口保留 **rad/m**，不执行归一化。LeRobot 导出保持已有的 **度/夹爪归一化**
约定，详见 [导出字段说明](lerobot_export.md#数据内容与单位)。训练器必须为所选入口
设置相应单位，不能在两个入口之间直接复用不同单位的统计量。两种入口都使用
`observation_t` 配 `action_t`，不会把 `next_state` 当作动作标签。

## 文件完整性范围

生产验收、严格导出、默认窗口读取和状态诊断恢复要求 HDF5 自包含：普通硬链接和
源文件标记为 `.` 的内部虚拟数据集均可使用，因此现有共享 RGB、压缩 JSON 格式兼容。
外部链接、外部原始存储、指向其他文件的 VDS 会被拒绝，否则外部像素变化可以绕过
主文件指纹。软链接也不在项目支持的存储格式内，即使目标仍在同一文件中。

窗口读取期间保留标准 HDF5 读锁，并在每个窗口前检查路径仍指向已打开的文件。
这些检查防止常规并发写入、误替换和陈旧报告；不是防止恶意同时重写全部数据与证据的
数字签名，也不是实机标定证明。需要复制数据时，HDF5 与同目录审计 JSON 一起复制。
