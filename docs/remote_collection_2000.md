# 远端清桌数据采集：2000 条合格成功回合

任务保持为：桌面套叠全部纸杯，将整摞杯子、笔、充电头和全部刚体餐具放入右篮。
使用已验证的固定指垫阻尼和碗落点避让配置；独立质量门限保持不变。

## 数据与随机化

- 仿真时间频率 30 Hz，每帧保留三路 640×480 RGB、实测状态、实际绝对位置动作和下一帧。
- 14 维顺序：左臂 6 个绝对关节角（rad）、左夹爪完整开口（m）、右臂 6 个绝对关节角（rad）、右夹爪完整开口（m）。
- 状态是实测值，动作是后端在该区间执行的绝对目标；不是关节增量或末端增量。
- 每回合自动生成新的系统熵随机种子，并保存用于复现；不需要人工指定 seed。
- 随机化覆盖整体和单件摆放、篮筐位姿、杯子数量/配色、桌面材质、光照、质量、摩擦和成像扰动。
- CAD 尺寸、相机安装基准和独立质量门限保持原配置；随机化范围使用 `configs/domain_randomization/clean_table.yaml`。
- RGB 为 GZIP level 1 无损压缩，观测/下一观测图像共用自包含 HDF5 帧存储；不抽帧或改变动作时间。

## 启动

远端 Python 3.11 / MuJoCo 3.3.7 环境安装后，从项目根目录运行：

```bash
PYTHONPATH=src .venv-mujoco/bin/python scripts/collect_success_quota.py \
  --output /mnt/piper_sim_data/clean_table_2000_30hz_abs_20261112 \
  --target-success 2000 --gpus 0 1 2 3 4 5 6 7 --workers-per-gpu 2
```

入口默认先完整验证一条成功回合，再展开所有 GPU 槽位。每个槽位是独立进程、物理环境、
随机状态和文件；GPU 负责相机渲染，MuJoCo 物理和数据压缩也使用 CPU。
`--workers-per-gpu` 控制每卡环境数，实际吞吐还受 CPU 与磁盘影响，不能用显卡数直接推算。
无需设置 `--seed-start`；该参数仅供可重复的顺序 seed 调试。

程序冻结实际代码、配置和模型资产到数据目录下的 `inputs/project/`，保留文件指纹和依赖
版本。采集子进程只导入该快照。SSH 断开不应停止以独立 session 启动的采集管理进程。

## 目录和计数

```text
clean_table_2000_30hz_abs_20261112/
  run.json                     # 目标、输入快照指纹、随机化和动作契约
  status.json                  # 成功/失败计数、GPU、当前 PID 和进行中回合
  success.jsonl                # 通过独立审核且复核 HDF5 校验和的回合
  failed.jsonl                 # 拒收/中断/超时/基础设施错误，含原因
  success/episode_.../recording/episode.h5
  failed/episode_.../recording/episode.h5[.partial]
  running/episode_.../          # 仅保存尚未分类的回合
  inputs/project/              # 固定代码、配置和模型快照
```

每个回合目录另有 `collection.log`、`claim.json`、`process.json`、`result.json`，
`recording/` 中保留原始 `report.json` 和 `audit.json`。准备阶段即失败时可能没有 HDF5，
仍保留错误报告；不会补写虚构帧。完整但质量不合格的回合也只进入 `failed/`。
只有 `success/` 计入 2000；失败后自动采样新的 seed 补采。进行中的槽位也占用剩余额度，
因此接近目标时自动减少并发，避免成功回合超采。失败数据不用于合格训练集。

## 状态、停止与恢复

```bash
cat /mnt/piper_sim_data/clean_table_2000_30hz_abs_20261112/status.json
# 使用 status.json 中的 manager_pid；只停止本批管理进程及其拥有的工作进程。
kill -TERM MANAGER_PID

# 恢复会验证快照、依赖和已有成功文件，不覆盖旧回合。
PYTHONPATH=src .venv-mujoco/bin/python scripts/collect_success_quota.py \
  --output /mnt/piper_sim_data/clean_table_2000_30hz_abs_20261112 --resume
```

恢复时，未完成回合的部分文件会保留；仅有完整独立审核证据的回合可恢复为成功。
输入或依赖改变时拒绝继续，重复基础设施错误或剩余空间不足时明确停止并记录原因。
默认保留至少 512 GiB 空间，低于此值不启动新回合；不自动删除失败数据。

### Practical v2 revision (2026-11-12)

Early remote failures were mostly planner rejection: TCP errors of 3.5–5.7 mm
or 2.3–2.9 degrees exceeded the original 3 mm / 0.035 rad IK criterion. Some
full physical runs were still in flight while these early failures appeared.
A targeted rerun then exposed a further 7.2 mm waypoint rejection, so the
explicit fallback budget is 10 mm / 5 degrees. At the operator's request, the
alternative profiles are:

- `configs/collection/clean_table_practical_v2.yaml`: IK fallback 10 mm / 5 degrees,
  after trying all seeds at original precision; bounded spline approximation
  4 mm / 0.02 rad instead of 2 mm / 0.01 rad; duration up to 420 s.
- `configs/collection/quality_policy_practical_v2.yaml`: only the maximum
  duration changes from 300 to 420 s. Motion smoothness, contacts, camera,
  timestamps, physical completion and full containment gates remain unchanged.

This does not repair physically unreachable poses, missed grasps or failed
placements. Such attempts remain failures. Fallback errors are recorded in
teacher events; this is a simulation planning allowance, not a hardware accuracy
claim.

To change an active run, drain or stop/reconcile it first. Do not edit its frozen
inputs or re-audit failed episodes as successful. Start a new revision beneath
the same dataset root with `--prior-run /absolute/path/to/stopped/run`. The old
manifests and run identity are hashed, prior successful HDF5 files are verified
under their original policy, and only verified successes count toward the same
2000 total. Old failures retain their category. The new revision's manifests
include inherited records with `source_run` and `acceptance_policy` paths.

For new collectors, creating an empty `DRAIN` file in the run folder stops new
launches, allows active workers to finish, and exits with status `drained`.
Remove that file before resuming. `DRAIN` affects scheduling only and never
changes the configured success quota.

### Elastic scheduling without interrupting an active recording

`scripts/collect_elastic_quota.py` adds an independently frozen scheduling layer
above an existing run. It retains the original worker snapshot, run.json,
randomization, 30 Hz actions/images, acceptance policy, episode index/seed
registry and the shared 2000-success quota. It adopts existing worker PID/start
identities and records controller code, handoff and scaling decisions beneath
`<run>/schedulers/elastic_*/`. Existing HDF5 files are not edited by the handoff.

Example (run using the remote MuJoCo environment):

```bash
PYTHONPATH=src .venv-mujoco/bin/python scripts/collect_elastic_quota.py \
  --output /mnt/piper_sim_data/clean_table_2000_30hz_abs_20261112/current \
  --takeover --initial-workers-per-gpu 3 --max-workers-per-gpu 10 \
  --stage-seconds 180
```

A legacy handoff freezes the old supervisor and validates its atomic on-disk
claim/result catalog. An incomplete launch registration is resumed and retried;
terminal folders must already have a committed result. This avoids relying on
privileged process inspection. Only the verified supervisor PID is replaced;
its independently created worker sessions receive no signal. The same exclusive manager lock prevents two quota
owners. A failed precondition leaves the legacy supervisor running. PID reuse
and mismatched commands prohibit worker signalling.

Growth stages are 3, 4, 6, 8 and 10 workers per GPU (24 through 80 on this host).
Every stage samples CPU, GPU, available memory and I/O pressure. Growth stops
at 85% CPU busy, 10% CPU pressure, 95% average GPU utilization, 85% GPU memory,
less than 64 GiB available memory, or sustained I/O pressure. Existing workers
finish naturally when the desired cap is reduced; no new workers refill a GPU
already above that cap. The disk reserve stays unchanged. Wall-clock worker
watchdog is separately declared as six hours for oversubscribed execution;
simulation-time episode duration remains governed by the original policy.

Phase-log simulation-time advancement is a **rough aggregate progress proxy**,
not a measurement of accepted episodes per hour or proof of optimal throughput.
A substantial proxy drop under CPU load backs off to the previous stage. Real
accepted/failed episode counts remain in the ordinary manifests. Resource and
phase differences between episodes mean short scaling windows are approximate.

`status.json` reports actual active workers, desired capacity, original frozen
capacity, resource samples and scale history. `DRAIN` still stops admission and
finishes the active workers. To resume an elastic controller after interruption,
run its frozen `controller.py --output <run> --controller-dir <controller-dir>`
with the MuJoCo environment; never launch a second collector alongside it.

Historical successes keep their predecessor audit records. During an operational
handoff their full hashes are revalidated by a background process, rather than
blocking all admissions on a large sequential read. Pending historical
revalidation is explicit in status.json; the quota cannot become `completed`
until it passes. A mismatch stops the controller. Every newly accepted episode
still uses the ordinary synchronous independent closed-file verification.
I/O admission thresholds use whole-host **full** I/O stall pressure; `some`
pressure is also reported, since a single background sequential reader waiting
for a disk is not evidence that all simulation workers are starved.
