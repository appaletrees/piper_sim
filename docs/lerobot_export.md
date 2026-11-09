# 可选导出 LeRobot v2 数据

录制默认仍保存原始HDF5。需要训练数据时，可选导出 **LeRobot Dataset v2.1**，
与现有 `connector_plugin_lerobot_v2` 使用同一版本、14维关节/夹爪顺序和三路相机名称。
格式实现对照 [LeRobot v0.3.3读取器](https://github.com/huggingface/lerobot/blob/v0.3.3/src/lerobot/datasets/lerobot_dataset.py)
及其 [数据格式定义](https://github.com/huggingface/lerobot/blob/v0.3.3/src/lerobot/datasets/utils.py)。
这是本地离线转换，原HDF5保留，不自动上传任何数据。

## 依赖

本机MuJoCo环境已经安装导出所需的PyArrow，系统已有ffmpeg/ffprobe。
新环境安装可选依赖，并确保系统ffmpeg包含libx264编码器：

```bash
uv pip install --python .venv-mujoco/bin/python -e '.[lerobot-export]'
```

导出模块不依赖MuJoCo、Isaac、CAN、PyTorch或LeRobot训练包，可放在独立数据处理进程。

## 采集结束后自动转换

先正常启动仿真API，再在客户端添加 `--lerobot-output`：

```bash
PYTHONPATH=src .venv-mujoco/bin/python examples/external_agilex.py \
  --episodes 2 --seed 42 \
  --lerobot-output outputs/lerobot_v2/demo_001 \
  --lerobot-image-size 640 480
```

不加 `--lerobot-output` 时，行为与之前一致。输出目录必须不存在；换一次数据集名称即可再次采集。
转换在全部episode停止录制后执行，避免编码工作阻塞实时控制。
该例子读取服务端返回的本地HDF5路径；跨机器使用时先把HDF5复制到导出进程可访问的位置。

## 转换已有HDF5

支持一个文件、多个文件或包含已完成 `.h5` 的目录：

```bash
PYTHONPATH=src .venv-mujoco/bin/python scripts/export_lerobot.py \
  --input outputs/api_episodes/external_20261022_145002_000.h5 \
          outputs/api_episodes/external_20261022_145002_001.h5 \
  --output outputs/lerobot_v2/my_dataset \
  --image-size 640 480
```

`--input outputs/api_episodes` 会按文件名顺序读取目录下的 `.h5`，不递归、不读 `.h5.partial`。
只接受 `complete=true`、非空、时间连续的项目HDF5；同一数据集各episode的相机集合、原始尺寸与fps须一致。
任何失败都不会改写源HDF5；不覆盖既有数据集，不把半成品目录当成成功输出。

带 `teacher` 元数据的脚本回合必须具有通过的完整 RGB 独立审计，导出时核对当前操作员
验收策略、内嵌/同目录审计 JSON，以及原文件的完整校验和。发布审计前同样检查报告是否
属于当前文件且数据未被修改，不能跨回合复用报告。普通外部录制默认可作格式诊断转换，
不因此获得训练质量认证；需要对它执行同样的生产门槛时使用 `--require-quality`，并用
`--policy` 选择独立验收策略。详细证据要求见 [采集质量说明](domain_randomization_collection.md)。

Python调用：

```python
from piper_sim.lerobot_export import export_lerobot_v2

result = export_lerobot_v2(
    ["outputs/api_episodes/episode_000001.h5", "outputs/api_episodes/episode_000002.h5"],
    "outputs/lerobot_v2/my_dataset",
    image_size=(640, 480),  # 可省略，保留每路相机原始分辨率
)
print(result)
```

自定义程序可在 `client.stop_recording()` / `sim.stop_recording()` 后收集其返回的 `path`，
结束控制循环后调用这个函数；录制、仿真和数据格式转换各自独立。

## 数据内容与单位

```text
my_dataset/
  meta/info.json
  meta/tasks.jsonl
  meta/episodes.jsonl
  meta/episodes_stats.jsonl
  meta/stats.json
  meta/export.json
  meta/sim_episodes.jsonl
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.cam_external_rgb/episode_000000.mp4
  videos/chunk-000/observation.images.cam_left_wrist_rgb/episode_000000.mp4
  videos/chunk-000/observation.images.cam_right_wrist_rgb/episode_000000.mp4
  extras/chunk-000/episode_000000.jsonl
```

| LeRobot字段 | 来源及语义 |
|---|---|
| `observation.state` | 动作前实测状态，左6关节度＋左夹爪归一化＋右6关节度＋右夹爪归一化 |
| `action` | 同一行实际施加的绝对位置目标；单位与state一致 |
| `observation.images.cam_*_rgb` | 同一动作前帧的RGB，以H.264/yuv420p MP4保存 |
| `timestamp` | `frame_index/fps`，每集从0开始的仿真相对时间 |
| `source_timestamp` | 原记录的主机monotonic时间，**不是Unix时间**；不能直接跨机器比较 |
| `sim_timestamp` | 原HDF5中的仿真时间，保留录制开始时偏移 |
| `frame_index` / `episode_index` / `index` / `task_index` | 集内帧号、集号、全局帧号、任务索引 |

原HDF5内部rad/m转换为度/归一化后，`gripper=width_m/0.095`。
保留实测值的微小物理求解越界，不静默裁剪原始反馈。`action`来自已应用目标，
不使用下一帧测量状态或客户端尚未执行的请求代替。
未记录的SDK末端反馈字段不凭空生成；核心state/action与三路RGB可直接供对应策略读取。

每条HDF5 transition导出一行，视频使用 `observation`，不误用 `next_observation`。
108帧和111帧的两集得到219行、各自108/111帧的视频，不补帧成180帧。
fps依据仿真时间；即使墙钟运行约18步/秒，源仿真每步1/30秒仍保存为30 fps。
编码完成后逐路解码检查实际帧数和尺寸。统计包含每集及全局mean/std/min/max/count，
RGB统计使用编码前输出像素的0–1值，形状为3×1×1。

`meta/sim_episodes.jsonl` 保留来源路径、SHA256、任务文本、成功标记、seed、采样布局、完整源配置和图像变换。
`extras/*.jsonl` 保留每帧原始SI状态/动作/下一状态、物体位姿、任务指标、相机时间与内外参。
它们是LeRobot核心目录之外的附加数据；标准读取器会照常加载Parquet和视频。

## 分辨率与相机参数

默认不改变分辨率或视场。指定 `--image-size WIDTH HEIGHT` 时等比例缩放并补黑边，
不把16:9图像直接拉伸成4:3；宽高须为正偶数。
1920×1080→640×480时图像变为640×360，上下各补60像素；中央1280×800→640×480时上下各补40像素。
变换矩阵保存在 `image_transforms`；每帧 `output_camera_calibration` 保存相应更新的K，
原K仍保留在 `observation_extras` 中。

缩放补边保留源视场，不等同于真实相机的640×480硬件裁剪模式。
需要该模式时，采集阶段使用 `configs/connector_alignment.yaml` 或设备标定配置；
不要靠导出时改变宽高比冒充相机视角对齐。

## 读取验证

使用支持v2.1的LeRobot读取器，例如v0.3.3；新版v3数据集目录格式不同。

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "local/piper_sim",
    root="outputs/lerobot_v2/my_dataset",
    video_backend="pyav",
)
sample = dataset[0]
print(sample["observation.state"].shape, sample["action"].shape)  # [14], [14]
print(sample["observation.images.cam_external_rgb"].shape)       # [3, H, W]
```

本机已使用未修改的LeRobot v0.3.3读取器验证用户已有两集（108＋111＝219帧、6路视频），
覆盖跨集首尾图像读取、全部动作单位转换和动作序列在episode末尾的padding。
自动导出选项也完成实际仿真采集验证（181帧、3路视频）。
测试与读取结果见 [验证报告](../outputs/lerobot_validation/report.json)。

2026-11 又完成 6,782 帧完整物理收桌回合的官方 v0.3.3 读取测试：三路 640×480 RGB、
14 维 state、16×14 action chunk、首/中/末尾索引、单位转换和末尾 padding 均通过。
特权接触力与物体真值保留在 extras，不进入普通训练行。
优化后的直方图统计保持编码像素及视频参数不变；同源完整导出的 MP4、Parquet 和 extras
SHA256 与旧实现一致。具体数值与局限见 [平台审计](platform_audit.md)。

本机独立读取环境可重复检查单回合导出（输出报告必须是新路径）：

```bash
PYTHONPATH=src .venv-lerobot-audit/bin/python scripts/check_lerobot_reader.py \
  --dataset outputs/platform_audit_v1/lerobot_histogram_full \
  --source outputs/platform_audit_v1/full_shared_seed0/episode.h5 \
  --output outputs/reader_recheck.json
```

该环境的包版本记录在检查报告中；它不是项目基础运行依赖。脚本强制离线读取，属于消费者
兼容性抽查，不会单独给数据授予任务成功或训练资格。
