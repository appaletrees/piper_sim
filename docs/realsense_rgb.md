# RealSense RGB 成像配置

三台相机保留原 STEP 外壳及支架，RGB 渲染采用官方公开的型号参数。
标称配置、推导内参和真实设备出厂标定有明确区分；本次未连接相机读取出厂数据。

## 参数与出处

| RGB 参数 | 左、右腕部 D435i | 中央 D455 |
|---|---|---|
| 彩色传感器 | OV2740 | OV9782 |
| 默认完整画幅 | 1920×1080 | 1280×800 |
| 本场景采样 | 30 Hz | 30 Hz |
| 标称水平 × 垂直视场 | 69.4° × 42.5° | 90° × 65° |
| 标称焦距 | 1.88 mm | 1.93 mm |
| 标称光圈 | f/2.0 | f/2.0 |
| 对焦 / 滤光 | 固定焦点 / IR cut | 固定焦点 / IR cut |
| 硬件 RGB 快门 | 滚动快门 | 全局快门 |
| 深度双目基线（用于区分开孔） | 50 mm | 95 mm |

依据 [RealSense D400 Datasheet，Rev 023，2026-03](https://www.realsenseai.com/wp-content/uploads/2026/03/RealSense-D400-Series-Datasheet-Mar-2026.pdf)
的 RGB 传感器表、相机型号表、USB 3 模式表及机械图（PDF 页 43–44、65、74–75、139、142）。
D435 的较精确 RGB FOV 取自 [Intel 官方产品简介](https://simplecore.intel.com/realsensehub/wp-content/uploads/sites/63/D435_Series_ProductBrief_111207.pdf)，
D435/D435i 共用 RGB 几何的依据为 [官方 D435i 描述](https://github.com/realsenseai/realsense-ros/blob/7a87097c499e3b0b69518d54a8e7a9ac1924e87c/realsense2_description/urdf/_d435i.urdf.xacro)。
OV2740 彩色镜头参数另与 [RealSense 官方支持答复](https://support.realsenseai.com/hc/en-us/community/posts/42873111326611-Issues-regarding-the-RGB-sensor-of-the-D435?page=1)核对。
原始配置及来源固定在 `configs/realsense_rgb.json`；生成清单和 episode 保存其 SHA256。

## RGB 光学坐标与 CAD 对齐

先由原 CAD 两端开孔轴线确定前表面中心与方向，再应用官方 ROS 描述的标称 RGB 偏移。
使用 ROS 光学轴：`+X` 图像向右、`+Y` 图像向下、`+Z` 镜头向前。

| 型号 | 相对前表面中心的 RGB 偏移（mm，光学轴） | 依据 |
|---|---|---|
| D435i | (-32.5, 0, -4.3) | 左 IR 相对机身中心 +17.5 mm；RGB 再向机身左侧 +15 mm；前表面到光学平面 0.1+4.2 mm |
| D455 | (+11.5, 0, -4.65) | 左 IR 相对机身中心 +47.5 mm；RGB 向机身右侧 59 mm；前表面到光学平面 0.1+4.55 mm |

偏移依据固定版本的 [D435 描述](https://github.com/realsenseai/realsense-ros/blob/7a87097c499e3b0b69518d54a8e7a9ac1924e87c/realsense2_description/urdf/_d435.urdf.xacro)
和 [D455 描述](https://github.com/realsenseai/realsense-ros/blob/7a87097c499e3b0b69518d54a8e7a9ac1924e87c/realsense2_description/urdf/_d455.urdf.xacro)。
上游也说明这些外参是用于仿真的近似值，真机驱动应使用设备标定。
当前假定 RGB 与前面板法线平行，未伪造单台设备的微小轴线偏角。

**更正旧标注：D435i 原 CAD 最外侧两开孔相距 65 mm，对应右 IR 与 RGB；不是双目基线。**
D455 的两端开孔才是 95 mm 的左右 IR。旧版使用开孔中点作投影参考，本版已改为上述 RGB 坐标。
外壳、支架及法兰装配位置均保持源 CAD 坐标。相机近裁面为 10 mm，以越过不透明的 CAD 前玻璃，
避免从机壳内部渲染；这不是 RealSense 的最小工作距离参数。
腕部零位俯角约 15°，此时图像包含较多桌缘和桌外背景；任务视角应通过机械臂姿态调整。

## 渲染与记录

公开 FOV 推导标称针孔矩阵：

```text
fx = width  / (2 tan(horizontal_FOV / 2))
fy = height / (2 tan(vertical_FOV / 2))
cx = width / 2; cy = height / 2
```

本机固定的 Sim 5.1 / Lab 2.3 渲染路径在设置独立 fy 后仍按方形像素投影，
直接使用会让 D455 的验证标记纵向偏移约 4 px。因此先以 `min(fx, fy)` 渲染覆盖所需
视场的针孔画面（使用本版 Kit 的 `pinholeOpenCV` 接口），再在 GPU 上按目标 K 的光线坐标
进行双线性采样。光心、朝向和时间不变，
画面边界落在原始渲染范围内，不用补边凑视场。观测记录最终 RGB 对应的 K；元数据同时保存
内部渲染焦距和采样方式。该适配器目前只接受 RGB，深度需单独实现对应处理。
直接在 Isaac 视口选择 Camera prim 看到的是覆盖视场；传感器输出和 HDF5 中保存的是最终 RGB。
镜头毫米参数转换为 USD 所需的厘米单位。
由焦距与 FOV 推导的有效 aperture 是投影参数，不宣称是厂家公布的感光芯片物理尺寸。

`configs/workcell.yaml` 中 `cameras.image_scale: 1.0` 为原始画幅；需要减少渲染与存储开销时，
可改为 `0.5`，分别输出 960×540 和 640×400，K 同比缩放且视场不变。
这是完整图像的等比例缩放，不宣称复现 RealSense 的硬件 ROI / VGA 裁剪模式。
每路 HDF5 图像可有不同大小，清单保存分辨率、K、来源、畸变状态和快门类型。
修改型号参数文件后须重新运行 CAD 转换器；哈希检查避免旧安装坐标与新光学参数混用。

## 复现范围

本次实际生效的是 RGB 安装偏移、朝向、原始画幅、视场、投影内参和焦距。
公共 FOV 带制造公差，不能由型号规格还原每台设备的主点和畸变。
目前畸变系数设零，标记为理想针孔；未把规格中的畸变上限转换成虚构系数。
固定焦点距离没有确认值，因此景深关闭；f/2.0 保留为硬件元数据。
快门类型也记录为硬件特征，目前渲染为同一时刻曝光，没有模拟滚动读出、曝光积分、ISP、
白平衡、噪声、IR 投射或 IMU 数据。后续可由独立传感器模块增加这些功能。

## 验证

`verify_rgb_projection.py` 在三台已安装相机前放置已知坐标的彩色标记，逐个比较解析投影
与实际 RTX 输出像素，阈值 2 px；标记仅存在于验证场景，不写入训练 episode。
动态 episode 另检查每帧 K、分辨率、相机随动位姿、双臂跟踪和物体接触。
2026-10-21 三路共 9 个标记的最大误差为 **0.886 px**，
证据保存在 `outputs/realsense_rgb_acceptance/projection.json` 及对应 `projection_*.png`。

```bash
PYTHONPATH=src .venv/bin/python scripts/convert_workcell_cad.py
PYTHONPATH=src .venv-isaac/bin/python scripts/verify_rgb_projection.py \
  --headless --output outputs/rgb_projection_new
bash scripts/run.sh --headless --enable_cameras --steps 120 --demo \
  --record outputs/rgb_episode_new.h5
PYTHONPATH=src .venv-isaac/bin/python scripts/verify_simulation.py \
  outputs/rgb_episode_new.h5 --output outputs/rgb_report_new
```
