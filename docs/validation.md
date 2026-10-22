# 验证记录

最新验证：2026-10-21。本机独立环境，未连接或驱动真实机械臂。

型号说明更新于 2026-10-21：用户确认中央为 Intel RealSense D455，双腕为 Intel RealSense D435i。
此前 episode 的腕部型号标签沿用了 STEP 中的 D435 名称；原始采集记录保留不改，
当前导出清单和后续采集使用 D435i。此次型号更正不改变已验证的几何和安装坐标。

## 2026-10-21 RGB 光学更新验收

已替换通用针孔占位参数及开孔中点投影，采用官方 RGB 规格和标称光心偏移。
来源、推导及复现边界见 [RealSense RGB 配置](realsense_rgb.md)。

- 双腕 D435i：1920×1080，69.4°×42.5°；中央 D455：1280×800，90°×65°。
- 三台实际传感器前共放置 9 个三维标记，最大像素投影误差 **0.886 px**，通过 2 px 阈值。
  本机渲染路径未正确应用独立 fy，现通过覆盖视场的 OpenCV 针孔渲染和 GPU 光线采样校正。
- 新采集 `outputs/realsense_rgb_acceptance/dual_piper.h5`：120 个 transition，三路原始画幅 RGB；
  每帧分辨率及记录内参核对通过，完整性标记为 true。验证标记不出现在此动态 episode 中。
- 全部双臂关节、夹爪跟踪及桌面接触通过；最大关节误差的 95 分位 0.016312 rad，
  夹爪 0.0950 mm，物块中心高度 0.045000006 m。
- 三路每帧相机位姿与独立 URDF 正运动学核对：最大位置差 **0.000364 mm**，
  最大角度差 0.000481 rad。采样时钟为仿真 30 Hz，物理时钟为 120 Hz。
- 原始分辨率下同步渲染和 HDF5 落盘实测 **8.53 帧/墙钟秒**，95 分位帧间隔 129.9 ms。
  完整 episode 约 581 MiB，仅保留本机；这不是实时 30 Hz 保证。
- 72 个网格重新与 STEP 对照通过，最大逐顶点误差和包围盒差异分别保持
  0.00002986 mm、0.04944 mm。
- 资产环境离线测试 **24 项通过**；Isaac 独立解释器 **20 项通过、1 个 CAD 模块跳过**。

当前证据：`outputs/realsense_rgb_acceptance/{projection,report,cad_alignment}.json`，
三路 RGB 与 `projection_*.png`。旧的 `cad_restore_acceptance` 保留为几何验收历史，
其中 RGB 画面使用旧投影参数；当前 RGB 以本节的新记录为准。

## 环境与资产

- Ubuntu 24.04，RTX 5090 D / 32607 MiB，驱动 580.178.04。
- `.venv-isaac`：Python 3.11.15、Isaac Sim 5.1.0、Isaac Lab 源码标签
  v2.3.0（包内版本 0.47.2）、PyTorch 2.7.0+cu128。CUDA 张量计算通过。
- 192 个安装包通过 `uv pip check`。完整固定版本在
  `requirements/runtime-linux-py311.lock`，环境报告在 `outputs/acceptance/environment.json`。
- 已读取布局图、STEP 和原始 Piper URDF / SDK；原始文件未改动。
- STEP 元数据包含 172 个装配实例；OCP 读取到 168 个叶级实体出现项。
  按确认的“双腕 Intel RealSense D435i、中央 Intel RealSense D455”组合导出 52 个静态网格（含一个无碰撞 guide）
  和左右各 10 个腕部相机/支架/螺钉网格，排除重复和备选件。
- 重新读取 STEP，对全部 72 个导出网格逐顶点核对：变换与浮点存储最大误差
  0.00002986 mm；源 BRep 与导出网格的轴向包围盒最大差异 0.04944 mm。
  网格化公差设为 0.1 mm。桌板 1400×700×25 mm、两臂间距 600 mm、
  中央立柱高度 1 m；USD 单位、上方向、网格索引与碰撞 API 检查通过。
- 原 URDF 派生为每臂 6 个转动关节和 2 个滑动手指；初始每指行程35 mm；2026-10-22按原控制器映射改为47.5 mm，总开口95 mm。
- 原 Piper SDK 和 `python-can 4.2.2` 已在独立环境成功导入，没有实例化 CAN 控制器。

## 2026-10-20 几何还原阶段的引擎验收

本次相机/支架修正后的验收数据：`outputs/cad_restore_acceptance/dual_piper.h5`，120 个 transition，
覆盖双臂全部六关节和两个夹爪的低幅运动，三路 640×480 RGB。
文件正常结束，完整性标记为 true，任务成功标记仍为 false（不是抓取成功示范）。

- 引擎实际时钟与记录时钟一致：物理 120 Hz，每 4 个物理步采集一次，即仿真 30 Hz。
- 最大关节跟踪误差的 95 分位为 0.01631 rad（约 0.93°）。
- 夹爪宽度跟踪误差的 95 分位约 0.095 mm。
- 40 mm 测试物块最终中心高度 0.045000006 m，桌面接触预期为 0.045 m。
- 三路输出均为有效 RGB；固定相机可见红色测试物块，腕部相机可见夹爪与桌面。
- 每一帧三路相机的实时 Fabric 位姿均与 CAD 安装坐标及实测关节角的独立 URDF 正运动学核对：
  最大位置差 0.000348 mm，最大旋转矩阵角度差 0.000481 rad。
- GUI 在精确零位保存总览和三处支架近景；参考截图中相机与 CAD 参考点的最大位置差
  0.000175 mm。截图过程不推进物理积分，避免重力下的驱动跟踪偏差影响装配对照。
  GUI 截图和无窗口动态采集均退出码为 0。
- 本次同步渲染和 HDF5 落盘约 16.5 帧/墙钟秒，95 分位帧间隔约 69 ms。
  这是当前实测吞吐量，不是 30 Hz 实时硬件控制保证。

证据：

- `outputs/cad_restore_acceptance/cad_alignment.json`：72 个网格的 STEP 对照。
- `outputs/cad_restore_acceptance/report.json`：动态数值检查结果。
- `outputs/cad_restore_acceptance/overview.png`：真实 Isaac GUI 零位总览。
- `outputs/cad_restore_acceptance/{left_mount,right_mount,central_mount}.png`：相机及支架近景。
- `outputs/cad_restore_acceptance/{overhead,left_wrist,right_wrist}.png`：动态验收最后一帧的实际传感器画面。
- `outputs/cad_restore_acceptance/reference_poses.json`：精确零位截图的相机位姿校核。
- `outputs/cad_restore_acceptance/run.log`、`reference.log`：运行日志。

早期平台验收保留在 `outputs/acceptance`，其中相机几何和安装配置已被本次修正取代；
不应使用早期截图判断当前支架还原效果。排错记录位于 `outputs/cad_restore_debug`，
其中遮挡画面的 episode 不是验收数据。

## 2026-10-20 软件验证及兼容处理

资产环境 `.venv` 中 17 项离线测试通过，覆盖指令校验、限速、SDK 单位映射、只读初始化、
单组 CAN 帧丢失检测、断流停止、双臂停止请求、HDF5 完整性、
时序窗口、HTTP 入口、夹爪运动方向、三相机选择、腕部附件及 CAD 几何。
Isaac 环境单独运行 pytest 为 13 项通过、CAD 测试模块跳过（独立解释器未加载 Kit 的 pxr）；
CAD 检查已在资产环境完整执行，并在实际 Isaac 中完成加载和渲染验证。

使用 `.venv` 和 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，避免桌面 ROS Python
3.12 插件污染 Python 3.11。已有 conda 环境未改动。

- Lab 2.3.1 / 2.3.2 的 Starlette 固定版本与 Sim 5.1 的 FastAPI 依赖冲突，
  因而采用完整依赖相容的 Lab 2.3.0。
- 适配器使用渲染器实际的 Fabric 相机变换，并转换 Fabric 索引为 32 位。
  默认 USD 相机读取在此配置下会返回旧位姿，不能直接作为训练数据外参。
- 在物理启动前初始化 Fabric 相机世界变换，避免首次读取从过期 USD 坐标初始化缓存，
  造成约 2 mm 的安装偏移和自身外壳遮挡；相机复位保持其安装局部变换。
- 显式物理步进、每控制周期一次渲染，避免 Kit 的 render=True 更新路径改变步数。
- 显式关闭时抑制 Lab 2.3.0 的 STOP 渲染循环，异常构造路径也处理退出。
- 所有兼容处理均在本项目内；没有修改 Isaac Lab 和原 SDK 源码。

日志仍有引擎的 CPU 拓扑枚举错误、弃用及材质/Fabric 提示；此次未阻止
物理、图像、采集和正常退出。日志不应描述为“完全无警告”。

## 尚未验证

- 本机没有 CAN 接口，双向主从仅完成接口、保护逻辑和 SDK 替身测试，
  尚未完成真实硬件闭环验收。使用前需配置真实左右通道、标定姿态并联调。
- 相机、支架几何按 CAD 原尺寸及装配坐标还原；RGB 已采用官方标称偏移和光学规格。
  每台设备的出厂内外参、畸变、快门时序、景深和 ISP 尚未复刻。
  CAD 中 11 mm 安装间隙待现场核对。
- 源 CAD 与源 URDF 法兰中心相差约 0.205 mm，已记录；本次未重新辨识机器人运动学。
  腕部附件碰撞使用凸包近似，未实测其质量、惯量与支架变形。
- 驱动参数未经辨识，未验证抓取成功率、物体摩擦参数和真实机器人动力学一致性。
- VLA / WAM 尚未训练；HDF5 是通用原始格式，具体模型的数据导出由独立适配器实现。

## 复验命令

输出路径须不存在，程序不会覆盖已有 episode：

```bash
bash scripts/run.sh --headless --enable_cameras --steps 120 --demo \
  --record outputs/new_acceptance.h5
PYTHONPATH=src .venv-isaac/bin/python scripts/check_episode.py outputs/new_acceptance.h5
PYTHONPATH=src .venv-isaac/bin/python scripts/verify_simulation.py outputs/new_acceptance.h5 \
  --output outputs/new_acceptance_report
```
## 2026-10-21 真机数据核对补充

详见 [真机对照说明](real_alignment.md)。全部35项离线测试通过；
Isaac独立环境中28项通过、4项依赖独立USD/CAD环境的检查跳过，后者在轻量环境已验证。
实际静态渲染第0集0/100/200/300帧的三路640×480图像，未裁剪SDK行程。
120帧完整RGB仿真回归通过，最大相机位置/FK误差约3.64×10⁻⁷ m。
640×480投影标记测试最大误差0.539 px；非居中主点测试最大误差1.009 px。
这里验证的是渲染器和模型内的一致性；真实桌边/腕部指尖仍存在明显残差，
**真机一比一视角对齐未通过**。

可复现数字报告位于 `outputs/real_alignment_summary/`；
包含真实画面的对照位于本地 `outputs/real_alignment_audit/review.md`，不上传。
