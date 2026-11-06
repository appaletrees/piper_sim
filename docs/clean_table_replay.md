# 真实清桌数据的 MuJoCo 回放与对齐检查

已对 `/home/midea/phoenix-mwa/datasets/piper_clean_table_lerobot` 的第 0、100、200 段做完整状态回放。
结论：**记录的双臂关节和夹爪能完整读入、逐帧重现，但当前仿真画面尚未与实拍完全对齐。**
数值回放一致不等于相机标定、物体布局或真实接触过程一致。

## 运行

从仓库根目录执行；每次指定新的输出目录：

```bash
PYTHONPATH=src MUJOCO_GL=egl .venv-mujoco/bin/python scripts/replay_clean_table.py \
  --dataset /home/midea/phoenix-mwa/datasets/piper_clean_table_lerobot \
  --episode 0 --output outputs/clean_table_replay/my_episode000
```

输出 `comparison.mp4`：上排实拍、下排仿真；从左到右为中央、左腕、右腕。
三路均使用源数据的 640×480 分辨率，默认逐帧保留 30 fps，不对实拍做裁剪、拉伸、翻转或时移。
`--episode` 可选择 0–200。`--max-frames 300` 只回放前十秒。
`--stride 2` 仍放置并检查每一条状态，但对照视频每两帧采样一次，以 15 fps 播放，时长保持原速。

同时打开 MuJoCo 观察窗，去掉 EGL 环境变量并加 `--viewer`：

```bash
PYTHONPATH=src .venv-mujoco/bin/python scripts/replay_clean_table.py \
  --dataset /home/midea/phoenix-mwa/datasets/piper_clean_table_lerobot \
  --episode 0 --viewer --output outputs/clean_table_replay/view_episode000
```

观察窗按记录时钟限速；渲染过慢时会降速而不跳过状态。此模式供观察，滑块不是控制入口。
头部标题中的时间为源记录时间；MuJoCo 物理时间保持不变。

依赖沿用 `.venv-mujoco` 中的 MuJoCo、numpy、Pillow、pyarrow，另需系统 `ffmpeg` / `ffprobe`。
脚本使用 CPU 解码 AV1、编码 H.264，MuJoCo 渲染使用 EGL 或桌面窗口。

## 数据契约

这份数据与之前 `connector_plugin_lerobot_v2` **不是同一种单位接口**：

| 字段 | 本次解释 |
|---|---|
| `state[0:6]`、`state[7:13]` | 左/右臂关节 1–6，rad |
| `state[6]`、`state[13]` | 左/右夹爪分数，乘 0.095 得到仿真完整开口 m |
| `actions` | 同样的数值映射，保留在输出中；本次不作为动力学控制目标 |
| 第 15 维 | 辅助字段，含义未确定；保留，不作为关节或终止信号 |
| `image` | `overhead` |
| `wrist_image_l` / `wrist_image_r` | `left_wrist` / `right_wrist` |

单位/顺序依据相邻项目 `phoenix-mwa/configs/data/piper_sources.json` 和 `docs/PIPER_DATA.md`
的显式声明，并检查全库数值范围；不声称由这份匿名特征名元数据独立证明了采集单位。
95 mm 采用现有 Piper 控制接口约定，不代表重新实测了指尖净开口。

加载器拒绝不匹配的维度、非有限数、越界夹爪分数、缺帧、错误 episode 标识和时间轴。
视频按帧顺序解码，与同编号状态对应，检查分辨率、帧率和帧数；完整回放还检查解码到结尾。
数据没有保存曝光级同步、相机内参、畸变或手眼标定，因此同编号对应不构成硬件同步证明。

全库共 201 段、388,506 帧。检查到左右第五关节的实测最大值为 `1.2200026512 rad`，
比当前模型 `1.22 rad` 上限高约 `2.65e-6 rad`（约 0.000152°）。回放保留原值并在报告里计数，
不裁剪、不放宽运行时控制限位，也不把这点边界数值差异当成画面偏差的原因。

## 本次验证结果

| episode | 完整回放帧数 | 原速视频时长 | 状态写入/读回最大差值 |
|---|---:|---:|---:|
| 0 | 1645 | 54.83 s | 0 |
| 100 | 1659 | 55.30 s | 0 |
| 200 | 2344 | 78.13 s | 0 |

三段共 5648 帧、16944 张仿真相机图像；原文件的 SHA-256 在运行前后保持一致。
本机结果分别位于 `outputs/clean_table_replay/episode000_full`、`episode100_full`、`episode200_full`。
原视频和含实拍画面的对照文件保留在本机，不纳入 Git。

检查首帧和全程均匀采样画面发现：

- 双臂动作顺序、朝向变化、左右腕流对应关系目视基本吻合。
- 中央画面中机器人轮廓有持续位置/比例偏差，叠加图可看到重影。当前 CAD + 标称 RGB
  投影尚不能复现原视频；只凭这项对照无法把原因唯一归结为内参、光学外参或真实安装误差。
- 腕部可见指尖轮廓也有偏差。这部分同时受内参、相机相对夹爪的光学坐标和指尖几何影响。
- 原数据不同段的物体摆放、堆叠和类别会变化。第 100/200 段出现不同布局及蓝色小包装、杯状物，
  不能把为第 0 段估算的固定场景当作全库统一初始布局。
- 原数据只给出双臂状态/动作及视频，没有餐具、纸巾、篮筐的三维轨迹。回放中的物体保持初始
  状态，不会随着真机搬运而移动；视频后半段差异不能直接算成相机误差。柔性纸巾/餐垫也未重建动态形变。
- 亮度、阴影、材质仍有渲染域差异，像素差或 SSIM 不能单独充当尺寸误差指标。

没有为让画面看起来重合而修改原 CAD、相机配置或物体尺寸。本次不是相机标定结果，
也不是将 `actions` 输入控制器后成功完成真实任务的动力学验证。
下一步应先核实实际 RGB 模式的内参/畸变和光学坐标，再按 episode 匹配初始物体布局；
之后另行评估动作驱动下的跟踪与接触误差。

## 输出与模块边界

- `comparison.mp4`：三相机同索引对照视频。
- `samples/`：七个时刻的原始 RGB、仿真 RGB、拼图和中央相机 50% 叠加图。
- `states.npz`：原始 15 维 state/actions、转换后的 14 维 SI 值、读回状态、帧号和源时间。
- `camera_poses.jsonl`：渲染帧对应的仿真相机位姿与 K。
- `report.json`：输入哈希、单位、限位偏差、帧数、相机配置、完整性和验证边界。

`src/piper_sim/recorded_replay.py` 负责这份 rad15 数据的读取、转换与纯运动学姿态放置；
`scripts/replay_clean_table.py` 负责视频对应、呈现和报告。不会导入真机后端或访问 CAN。
正常 `apply()` / `step()`、外部控制、录制和动力学限制不受影响。

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-mujoco/bin/python -m pytest -q tests/test_recorded_replay.py
```
