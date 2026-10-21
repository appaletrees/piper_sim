"""Read-only alignment export for the team's connector_plugin LeRobot v2.1.

This is deliberately not a generic LeRobot unit conversion. The producer stores
six joint angles in degrees and a normalized SDK gripper stroke per arm.
Neither camera calibration nor device capture timestamps exist in these files.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .contracts import FEATURE_NAMES, LOWER, UPPER, vector


CONNECTOR_PLUGIN_NAMES = tuple(
    name for arm in ("left_arm", "right_arm")
    for name in (*[f"{arm}.joint_{i}" for i in range(1, 7)], f"{arm}.gripper_open")
)
CAMERA_MAPPING = {
    "cam_external_rgb": "overhead",
    "cam_left_wrist_rgb": "left_wrist",
    "cam_right_wrist_rgb": "right_wrist",
}
CONNECTOR_PLUGIN_GRIPPER_SCALE_M = .095
SOURCE_EVIDENCE = {
    "piper_controller.py": {
        "lines": [103, 105, 1217, 1235, 1243, 1258],
        "meaning": "joint degrees; normalized gripper = SDK stroke / 1000 / 95",
    },
    "piper_sdk/piper_sdk/piper_msgs/msg_v2/feedback/arm_feedback_gripper.py": {
        "lines": [14, 20, 46, 53], "meaning": "SDK gripper stroke unit is 0.001 mm",
    },
    "examples/common/piper_recorder.py": {
        "lines": [129, 130, 162, 183, 241, 258, 313, 319],
        "meaning": "latest camera frames then state; saves no per-camera timestamps or K",
    },
    "examples/common/multi_camera.py": {
        "lines": [76, 84, 151, 156],
        "meaning": "native requested stream profile; optional configured rotation",
    },
    "examples/common/recording_config.yaml": {
        "lines": [17, 18, 25, 39],
        "meaning": "current source configuration: 640x480@30; rotation 0 for all cameras; not historical episode metadata",
    },
    "examples/common/threads/joint_state.py": {
        "lines": [27, 30],
        "meaning": "copies any non-None cached feedback; otherwise retains previous state; no freshness check",
    },
    "piper_sdk/piper_sdk/interface/piper_interface_v2.py": {
        "lines": [74, 85, 988, 1011],
        "meaning": "SDK joint cache initialized to zeros; getter returns cached message including separate Hz/timestamp",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_connector_plugin_info(info: dict) -> None:
    """Reject a different schema or reordered features before assigning units."""
    if info.get("codebase_version") != "v2.1" or info.get("fps") != 30:
        raise ValueError("connector_plugin contract requires LeRobot v2.1 at 30 fps")
    features = info.get("features", {})
    for key in ("observation.state", "action"):
        feature = features.get(key, {})
        if (feature.get("shape") != [14]
                or tuple(feature.get("names", [])) != CONNECTOR_PLUGIN_NAMES
                or feature.get("dtype") != "float32"):
            raise ValueError(f"Unexpected connector_plugin feature order or dtype: {key}")
    for camera in CAMERA_MAPPING:
        feature = features.get(f"observation.images.{camera}", {})
        if feature.get("dtype") != "video" or feature.get("shape") != [3, 480, 640]:
            raise ValueError(f"Expected RGB video with CHW shape [3,480,640]: {camera}")
    if features.get("source_timestamp", {}).get("dtype") != "float64":
        raise ValueError("source_timestamp must preserve float64 Unix time")


def connector_plugin_state_to_si(value) -> np.ndarray:
    """Convert degrees/normalized SDK stroke to radians/metres without clipping."""
    result = vector(value)
    if ((result[[6, 13]] < 0) | (result[[6, 13]] > 1)).any():
        raise ValueError("connector_plugin normalized gripper must be in [0,1]")
    joints = np.r_[0:6, 7:13]
    result[joints] = np.deg2rad(result[joints])
    result[[6, 13]] *= CONNECTOR_PLUGIN_GRIPPER_SCALE_M
    return result


def bounds_violations(value, gripper_limit_m: float = .07) -> list[dict]:
    """Report, rather than silently clamp, incompatibility with the sim model."""
    if not math.isfinite(gripper_limit_m) or gripper_limit_m <= 0:
        raise ValueError("gripper_limit_m must be finite and positive")
    q = vector(value)
    upper = UPPER.copy()
    upper[[6, 13]] = gripper_limit_m
    return [{"index": i, "feature": FEATURE_NAMES[i], "value": float(q[i]),
             "lower": float(LOWER[i]), "upper": float(upper[i]),
             "unit": "m" if i in (6, 13) else "rad"}
            for i in range(14) if q[i] < LOWER[i] - 1e-6 or q[i] > upper[i] + 1e-6]


def feedback_quality(raw_states) -> dict:
    """Identify suspicious zero feedback, without asserting it is fabricated.

    The producer falls back to zeros for missing joints and does not save SDK
    message age, motor enable state, or a validity bit. Kinematic consistency
    with a recorded end pose therefore does not verify the physical pose.
    """
    states = np.asarray(raw_states, dtype=float)
    if states.ndim != 2 or states.shape[1] != 14 or not np.isfinite(states).all():
        raise ValueError("Expected a finite Nx14 array of recorded states")
    result = {"feedback_freshness_saved": False, "motor_enable_status_saved": False,
              "physical_pose_verified": False, "arms": {}}
    for name, offset in (("left", 0), ("right", 7)):
        joints = states[:, offset:offset+6]
        near_zero = bool(np.max(np.abs(joints)) <= .05)
        result["arms"][name] = {
            "status": "suspicious_near_zero_feedback" if near_zero else "reported_feedback_unverified",
            "joint_min_degrees": joints.min(axis=0).tolist(),
            "joint_max_degrees": joints.max(axis=0).tolist(),
            "max_joint_span_degrees": float(np.ptp(joints, axis=0).max()),
            "near_zero_threshold_degrees": .05,
            "use_as_zero_pose_calibration": False,
            "reason": ("Could be actual static zero, zero fallback, or stale/uncalibrated SDK feedback; "
                       "the recording cannot distinguish these cases") if near_zero else
                      "Timestamp/enable/validity evidence is absent; moving samples still need visual cross-checks",
        }
    return result


def _load_episode(dataset: Path, episode: int, pq):
    path = dataset / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
    table = pq.read_table(path)
    n = table.num_rows
    if n == 0:
        raise ValueError(f"Empty episode {episode}")
    required = {"observation.state", "action", "frame_index", "episode_index",
                "timestamp", "source_timestamp"}
    if not required.issubset(table.column_names):
        raise ValueError(f"Missing episode columns: {required - set(table.column_names)}")
    frames = np.asarray(table["frame_index"])
    if not np.array_equal(frames, np.arange(n)):
        raise ValueError(f"Nonconsecutive frame_index in episode {episode}")
    if not np.all(np.asarray(table["episode_index"]) == episode):
        raise ValueError(f"episode_index mismatch in {path}")
    timestamps = np.asarray(table["timestamp"])
    if not np.allclose(timestamps, frames / 30, rtol=1e-7, atol=1e-5):
        raise ValueError(f"Video timestamp != frame_index/30 in episode {episode}")
    source_time = np.asarray(table["source_timestamp"])
    if not np.isfinite(source_time).all() or not np.all(np.diff(source_time) > 0):
        raise ValueError(f"Invalid source timestamps in episode {episode}")
    return path, table


def extract_connector_plugin_alignment(
    dataset_root, output_dir, samples, *, gripper_limit_m: float = .07, source_root=None,
) -> dict:
    """Export selected measured states and identically indexed decoded RGB frames.

    ``samples`` is a list of (zero-based episode, zero-based frame) pairs. The
    destination must not exist; real data remains local. This does not operate
    any robot, infer calibration, or guarantee synchronization at exposure time.
    pyarrow and OpenCV are optional dependencies imported only by this function.
    """
    try:
        import cv2
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Alignment export needs optional pyarrow and opencv-python-headless") from exc
    dataset = Path(dataset_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output == dataset or output.is_relative_to(dataset):
        raise ValueError("Alignment outputs must be outside the read-only dataset")
    requested = list(samples)
    if not requested or any(len(pair) != 2 or any(
        not isinstance(i, (int, np.integer)) or isinstance(i, bool) or i < 0 for i in pair
    ) for pair in requested):
        raise ValueError("Samples must contain nonnegative integer (episode, frame) pairs")
    requested = sorted(set((int(ep), int(frame)) for ep, frame in requested))
    info_path = dataset / "meta/info.json"
    info = json.loads(info_path.read_text())
    validate_connector_plugin_info(info)
    # Validate the configurable limit even if none of the chosen samples exceeds it.
    bounds_violations(np.zeros(14), gripper_limit_m)
    episodes = {}
    for episode, frame in requested:
        if episode >= info["total_episodes"]:
            raise ValueError(f"Episode {episode} outside dataset")
        if episode not in episodes:
            episodes[episode] = _load_episode(dataset, episode, pq)
        if frame >= episodes[episode][1].num_rows:
            raise ValueError(f"Frame {frame} outside episode {episode}")
    manifest = {
        "schema_version": 1, "unit_contract": "connector_plugin_v2_1_degrees_normalized95mm",
        "exporter_module_sha256": sha256(Path(__file__)),
        "dataset_root": str(dataset), "output_root": str(output),
        "source_feature_names": list(CONNECTOR_PLUGIN_NAMES),
        "si_feature_names": list(FEATURE_NAMES),
        "units": {"arm_joints": "rad", "gripper": "SDK stroke m",
                  "gripper_scale_m": CONNECTOR_PLUGIN_GRIPPER_SCALE_M,
                  "gripper_scale_is_mechanical_aperture_calibration": False},
        "sim_gripper_limit_m": gripper_limit_m, "clipping_applied": False,
        "camera_mapping": CAMERA_MAPPING,
        "image_contract": {"width": 640, "height": 480, "channels": 3, "fps": 30,
                           "color_order": "RGB", "dtype": "uint8"},
        "time_semantics": {
            "timestamp": "frame_index/30, encoded video playback seconds",
            "source_timestamp": "Unix seconds when proprioception was sampled",
            "camera_correspondence": "same saved recording index; camera latest frames sampled before proprioception",
            "device_capture_timestamps_saved": False, "hardware_synchronization_verified": False,
        },
        "calibration": {"K_saved": False, "distortion_saved": False,
                        "camera_extrinsics_saved": False,
                        "historical_crop_and_rotation_verified": False,
                        "note": "640x480 native capture request is evidenced by current producer code; no historical camera profile snapshot exists"},
        "files": [{"path": str(info_path), "sha256": sha256(info_path)}],
        "producer_evidence": [], "samples": [],
        "episode_feedback_quality": {},
    }
    for relative in ("provenance/conversion.json", "provenance/validation.json",
                     "provenance/episode_mapping.json", "provenance/video_manifest.json"):
        path = dataset / relative
        if path.exists():
            manifest["files"].append({"path": str(path), "sha256": sha256(path)})
    episode_mapping = dataset / "provenance/episode_mapping.json"
    if episode_mapping.exists():
        for entry in json.loads(episode_mapping.read_text()):
            if entry["episode_index"] not in episodes:
                continue
            source_h5 = Path(entry["source_h5"])
            digest = sha256(source_h5) if source_h5.is_file() else None
            if digest is not None and digest != entry["source_h5_sha256"]:
                raise ValueError(f"Original source H5 changed since conversion: {source_h5}")
            manifest["files"].append({"path": str(source_h5), "sha256": digest,
                                      "conversion_sha256": entry["source_h5_sha256"]})
    if source_root is not None:
        root = Path(source_root).expanduser().resolve()
        for relative, evidence in SOURCE_EVIDENCE.items():
            path = root / relative
            manifest["producer_evidence"].append({
                "path": str(path), "sha256": sha256(path) if path.is_file() else None,
                **evidence, "historical_revision_verified": False,
            })
    for episode, (path, table) in episodes.items():
        manifest["files"].append({"path": str(path), "sha256": sha256(path)})
        manifest["episode_feedback_quality"][str(episode)] = feedback_quality(
            table["observation.state"].to_pylist())
        for _, frame in (pair for pair in requested if pair[0] == episode):
            state = table["observation.state"][frame].as_py()
            action = table["action"][frame].as_py()
            state_si = connector_plugin_state_to_si(state)
            action_si = connector_plugin_state_to_si(action)
            manifest["samples"].append({
                "episode_index": episode, "frame_index": frame,
                "timestamp": float(table["timestamp"][frame].as_py()),
                "source_timestamp": float(table["source_timestamp"][frame].as_py()),
                "state_raw": state, "state_si": state_si.tolist(),
                "action_raw": action, "action_si": action_si.tolist(),
                "state_bounds_violations": bounds_violations(state_si, gripper_limit_m),
                "action_bounds_violations": bounds_violations(action_si, gripper_limit_m),
                "images": {},
            })
    video_manifest = dataset / "provenance/video_manifest.json"
    expected_hashes = {entry["output"]: entry["sha256"] for entry in
                       json.loads(video_manifest.read_text())} if video_manifest.exists() else {}
    output.mkdir(parents=True, exist_ok=False)
    for episode, (_, table) in episodes.items():
        rows = {row["frame_index"]: row for row in manifest["samples"]
                if row["episode_index"] == episode}
        for camera, sim_camera in CAMERA_MAPPING.items():
            relative = (f"videos/chunk-{episode // 1000:03d}/observation.images.{camera}/"
                        f"episode_{episode:06d}.mp4")
            path = dataset / relative
            digest = sha256(path)
            if relative in expected_hashes and digest != expected_hashes[relative]:
                raise ValueError(f"Video changed since conversion manifest: {path}")
            manifest["files"].append({"path": str(path), "sha256": digest})
            capture = cv2.VideoCapture(str(path))
            try:
                profile = [capture.get(key) for key in (
                    cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
                    cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT)]
                if not capture.isOpened() or not np.allclose(profile, [640, 480, 30, table.num_rows], rtol=0, atol=1e-6):
                    raise ValueError(f"Video/state profile or frame count mismatch: {path}: {profile}")
                # Sequential decode avoids approximate/keyframe seeking. Only the
                # prefix through the last requested frame is decoded, once.
                for frame in range(max(rows) + 1):
                    ok, bgr = capture.read()
                    if not ok or bgr.shape != (480, 640, 3):
                        raise ValueError(f"Missing or malformed RGB frame {frame}: {path}")
                    if abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - (frame + 1)) > .01:
                        raise ValueError(f"Decoder frame index mismatch at {frame}: {path}")
                    if frame in rows:
                        target = output / f"episode_{episode:06d}_frame_{frame:06d}_{sim_camera}.png"
                        if not cv2.imwrite(str(target), bgr):
                            raise OSError(f"Could not save {target}")
                        rows[frame]["images"][sim_camera] = {
                            "path": str(target), "sha256": sha256(target),
                            "source_camera": camera, "source_video": str(path),
                            "decoded_frame_index": frame, "shape_hwc": [480, 640, 3],
                        }
            finally:
                capture.release()
    manifest["status"] = "indexed_state_and_rgb_exported_calibration_unverified"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest
