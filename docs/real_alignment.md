# 真机照片与 connector_plugin 数据核对

2026-10-21：用户已确认原始资源设计正确，**CAD 尺寸和装配位置作为固定基准**。
下文保留历史数据的诊断结果，后续据此检查仿真实现、RGB 模式和数据映射，不反向修改资源设计。

核对结论：**已将旧 URDF 视觉夹爪替换为原 CAD 几何并建立同帧对照；真机视角尚未通过对齐验收。**
CAD 网格的毫米级一致性、渲染器对 K 的实现、真机相机的标定是三个不同检查。
不能把前两项通过写成第三项通过，也不能为使画面相似而缩放机械臂或桌面。

## 已确认的资料与修正

| 项目 | 证据与处理 |
|---|---|
| 数据规模 | 210 集、54,364 帧、630 个视频；全部视频头为 640×480、30 fps，帧数与对应 Parquet 一致 |
| 视频转换 | LeRobot 转换为原视频字节复制，无重新编码；索引仍是原始保存索引 |
| 三路映射 | `cam_external_rgb → overhead`；`cam_left_wrist_rgb → left_wrist`；`cam_right_wrist_rgb → right_wrist` |
| 关节单位 | 原记录为度，读入转换为弧度；不增加未经测量的关节零偏 |
| 夹爪字段 | 原控制器归一化值乘 0.095 得到 SDK 行程米数；SDK 行程与实际指尖开口是否相等尚未测量 |
| 夹爪范围 | 数据左侧约 75.2–75.3 mm、右侧约 45.6–93.6 mm；超过当前动力学模型的 70 mm 限位，导出时记录超限且不截断 |
| CAD 尺寸 | 桌面 1400×700×25 mm、臂距 600 mm、立柱 16×40×1000 mm，均直接来自 STEP；还没有独立真机尺量 |
| 夹爪几何 | 旧 URDF 固定底座宽约 145 mm，源 CAD 164.003 mm；指尖距 link6 轴向位置由 135.8 改为约 142.505 mm |
| 网格拆分 | 从原 STEP 按 link6、link7、link8 分组，共 28 个夹爪部件；扣除 CAD 58.3713 mm 参考开口后，视觉开口为 joint7+joint8 |
| 外观 | 已有 Piper STL、三台相机及支架 CAD mesh；添加灰黑 Piper、银色相机、黑支架、白桌面、铝立柱材质 |

新夹爪逐点还原误差约 **5.2×10⁻⁹ m**；0/40/70/95 mm 的视觉开口检查通过。
这不是95 mm真实行程认证。Isaac 中新网格替换**视觉**，碰撞、惯量、驱动与70 mm限位仍来自旧URDF；
因此还不能用这版夹爪声称完成精确接触/力学仿真。配置 `appearance.cad_gripper` 与
`appearance.photo_materials` 控制两项独立能力。
新增 [MuJoCo 版本](mujoco.md) 使用分部件 CAD 凸包碰撞，惯量及限位仍取自源 URDF。

材质来自照片观察，未经反射率或颜色测量。相机单体 CAD 尚未细分黑色前面板与镜片材质，
也未虚构贴纸、标志或线缆。照片中的长底座安装板和 C 形桌夹未在这版 STEP 中找到，仍缺尺寸来源。

## 实际视角残差

历史数据没有保存设备 K、畸变、相机安装外参、逐相机曝光时间戳或完整相机配置快照。
当前采集源码请求 640×480，但当前代码版本不能证明历史录制配置完全一致。
因此 `configs/connector_alignment.yaml` 使用**全传感器高度、中心裁剪的假设**，不是测量标定：

| RGB | 假设 fx / fy | cx / cy | 分辨率 |
|---|---|---|---|
| D435i 两腕 | 616.185 / 617.159 | 320 / 240 | 640×480 |
| D455 中央 | 384.000 / 376.725 | 320 / 240 | 640×480 |

RealSense 官方说明不同分辨率有不同 FOV 与 K，VGA 会改变横向覆盖，不能把全幅图像非等比压缩后当作设备模式。
参见 [RealSense 分辨率与裁剪说明](https://support.realsenseai.com/hc/en-us/community/posts/13718098480019-How-to-resample-D455-RGB-images-from-1280-720-to-640-480)。
这里保留裁剪假设标签，等待设备640×480 profile的实测K替换。

| 特征 | 真机像素 y | CAD + 假设 K | 纵向差异 |
|---|---|---|---|
| 中央画面远侧桌边 | 130.5 | 169.37 | 38.87 px |
| 中央画面近侧桌边局部 | 463.5 | 509.50 | 46.00 px |
| 左腕两指尖 | 约 312 / 303 | 约 370.1 | 约 58–67 px |

桌边在5个抽样帧中位置稳定，人工关联不确定度约2 px；指尖轮廓关联约±4 px。
这些是诊断特征，不是已知标定板角点。近桌边只在局部可见，真实桌角有圆角。
腕部相机和夹爪共享 link6，因此六轴姿态改变不能消除其相对投影残差。
替换 STEP 中重叠的备用腕部相机只引起约0.4 px变化，不能解释该偏差。
在原设计固定的前提下，需要检查导入后的指尖/转接件坐标、RGB光心坐标约定、内参与主点，
不能将这些残差直接归因于 CAD 设计。

后续交叉检查发现，AgileX 官方大夹爪模型的法兰至指尖为4.5+138=142.5 mm，
与原STEP约142.505 mm吻合；该模型标称总行程100 mm。它支持本地旧URDF与CAD使用不同夹爪配置的判断，
但没有代替这台实机的开口测量，也没有据此扩大真机控制限位。
来源：[固定版本的AgileX夹爪描述](https://github.com/agilexrobotics/agx_arm_urdf/blob/680fa925cd00db7e2628563d2edc5540963045de/piper/urdf/piper_with_gripper_description.xacro#L48)。

左臂反馈长期接近零；采集源码存在缺失反馈填零、SDK缓存无新鲜度验证的路径，数据没有保存有效位。
它可以是真实静止、缓存或未标定反馈，**不能作为真机零位标定证据**。
同一视频保存索引也不等于三相机曝光硬同步。

## 可复现对照

以下只读数据，不连接 CAN。真实视频、照片和包含真人的对照图留在本地 `outputs/real_alignment_audit/`。
版本库只保存代码、数字诊断和纯仿真截图。

```bash
# 生成原 CAD 夹爪资产
PYTHONPATH=src .venv/bin/python scripts/prepare_cad_gripper.py

# 轻量环境需要 pyarrow、opencv-python-headless；可安装项目 [alignment] 依赖。
# 本机已有这些依赖的环境：/home/midea/miniconda3/envs/lingbotvla/bin/python
PYTHONPATH=src /home/midea/miniconda3/envs/lingbotvla/bin/python \
  scripts/extract_lerobot_alignment.py \
  --dataset /home/midea/datasets/connector_plugin_lerobot_v2 \
  --output outputs/alignment_reference_new \
  --source-root /home/midea/Desktop/piper_tele/sink-bot-control_bk/example_agilex \
  --sample 0:0 --sample 0:100 --sample 0:200 --sample 0:300

PYTHONPATH=src .venv-isaac/bin/python scripts/render_lerobot_alignment.py \
  --manifest outputs/alignment_reference_new/manifest.json \
  --output outputs/alignment_render_new --headless
```

静态审查直接写入观测状态，再只渲染；采图期间不积分物理，不经过轨迹控制器。
这样可以显式显示超过模型限位的SDK行程，而不把这些值送到真机，也不会将其裁成70 mm。
它是画面核对工具，不能作为动力学回放或训练成功的证明。
输出包含图像与源文件SHA、单位转换、超限项、帧索引、相机位姿、同帧三路实虚对照。

仅复现数字残差，不读取真实照片：

```bash
PYTHONPATH=src .venv/bin/python scripts/measure_real_alignment.py \
  --output-dir outputs/alignment_measurement_new
```

`configs/alignment_landmarks.json` 保存数字标注、不确定度和可选本地参考图SHA。
报告中的 `passed=false` 表示这些关联尚不是物理标定验收。

接入原三台 RealSense 后，可只读导出对应 RGB profile：

```bash
python scripts/export_realsense_rgb_calibration.py \
  --camera overhead=中央序列号 \
  --camera left_wrist=左腕序列号 \
  --camera right_wrist=右腕序列号 \
  --output outputs/device_rgb_calibration.json
```

该脚本依赖 `pyrealsense2`，调用 `config.resolve()` 与 `get_intrinsics()`，不启动采流、不设置设备选项。
本机当前未接入 RealSense，所以该读取尚未通过设备实测。
导出包含 K、畸变、分辨率、序列号、固件，但不包含手眼/安装外参。
`calibrated_pinhole` 配置可接收有来源的 K；非零畸变必须先对真实图像做相应去畸变，不能忽略系数。
像素坐标使用 [RealSense 的整数像素中心定义](https://dev.realsenseai.com/docs/projection-in-realsense-sdk-2-0/)；
渲染采样已处理半像素转换。

## 完成一比一验收还缺什么

需要原三相机对应模式的设备内参、RGB光心标定信息或多姿态标定板数据；
同时确认 SDK 行程到开口的映射关系。照片中源文件未列出的附件可另行补充，不改变已有装配。
修正数据映射与光学参数后，应在未参与拟合的帧上重新检查桌边、夹爪和标定点的重投影误差。
当前记录明确为**未通过真机视角对齐**，没有用拟合画面反向更改已知CAD尺寸。
