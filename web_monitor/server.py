#!/usr/bin/env python3
"""Dependency-free LAN dashboard for CVIA on AGX Orin."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import struct
import subprocess
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from mocap_receiver import MocapReceiver
from offline_sync import (
    estimate_time_offset,
    generate_offline_reports,
    load_detection_poses,
    load_mocap_poses,
    stabilize_detection_poses,
)

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
CONFIG_PATH = Path(
    os.environ.get("CVIA_CONFIG_FILE", str(ROOT / "config.json"))
).expanduser().resolve()
_default_trt_binary = PROJECT_ROOT / "build" / "trt"
if not _default_trt_binary.is_file():
    _default_trt_binary = PROJECT_ROOT / "trt"
TRT_BINARY_PATH = Path(
    os.environ.get("CVIA_TRT_BINARY", str(_default_trt_binary))
).expanduser().resolve()
RUNTIME_PATH = ROOT / "runtime"
PREVIEW_PAIR_PATH = RUNTIME_PATH / "preview_pair.bin"
PID_PATH = RUNTIME_PATH / "trt.pid"
SESSION_CSV_PATH = RUNTIME_PATH / "pnp_session.csv"
MOCAP_SESSION_CSV_PATH = RUNTIME_PATH / "mocap_session.csv"
SYNCED_SESSION_CSV_PATH = RUNTIME_PATH / "synchronized_session.csv"
# The short historical basename remains the composite canvas; all primary
# report artifacts are now real RGB PNG files.
OFFLINE_REPORT_PATH = RUNTIME_PATH / "synchronized_report.png"
OFFLINE_CAMERA_REPORT_PATH = RUNTIME_PATH / "synchronized_camera_report.png"
OFFLINE_MOCAP_REPORT_PATH = RUNTIME_PATH / "synchronized_mocap_report.png"
LEGACY_OFFLINE_REPORT_PATHS = (
    RUNTIME_PATH / "synchronized_report.svg",
    RUNTIME_PATH / "synchronized_camera_report.svg",
    RUNTIME_PATH / "synchronized_mocap_report.svg",
)
OFFLINE_SUMMARY_PATH = RUNTIME_PATH / "synchronized_summary.json"
OFFSET_ESTIMATE_PATH = RUNTIME_PATH / "offset_estimate.json"
RUNTIME_CONFIG_STATUS_PATH = RUNTIME_PATH / "runtime_config_status.json"
CONFIG_APPLY_META_PATH = RUNTIME_PATH / "config_apply_meta.json"
CALIBRATION_HISTORY_PATH = ROOT / "calibration_history.json"
MAX_BODY = 256 * 1024
CALIBRATION_HISTORY_LIMIT = 50
MODEL_KEYPOINT_MIN_COUNT = 4
MODEL_KEYPOINT_MAX_COUNT = 256
DEFAULT_ONNX_PATH = "models/onnx/qdy0815.onnx"
CALIBRATION_HISTORY_LOCK = threading.Lock()
CONFIG_SAVE_LOCK = threading.Lock()
SERVER_STARTED_NS = time.time_ns()
PREVIEW_PAIR_HEADER = struct.Struct("!4sII")
PREVIEW_PAIR_MAGIC = b"CVP1"
MOCAP_RECEIVER: MocapReceiver | None = None
MOCAP_APPLY_LOCK = threading.Lock()
SYNC_REPORT_LOCK = threading.Lock()
INFERENCE_LOCK = threading.RLock()
INFERENCE_PROCESS: subprocess.Popen | None = None
INFERENCE_LAST_EXIT_CODE: int | None = None
INFERENCE_STARTED_NS: int | None = None
SESSION_ARCHIVED_MTIME_NS: int | None = None

SYNC_DEFAULTS = {
    "enabled": True,
    "offset_ms": 0.0,
    "max_error_ms": 12.0,
    "history_ms": 5000,
    "interpolate": True,
    "auto_offset_search_ms": 1000,
}

# Default geometry for the bundled model. Other models may use a different
# number of ordered P0..Pn points through calibration.model_keypoints_3d.
DEFAULT_MODEL_KEYPOINTS_3D = (
    (179.617252624, -40.337850737, 17.513833145),
    (178.579489770, -31.449488527, 32.085969252),
    (203.784911818, -26.404937192, 15.758290772),
    (178.104641601, -14.015844432, 45.035654805),
    (190.624677177, -13.346597405, 38.884062510),
    (178.614863843, 2.826463025, 47.693300195),
    (179.749162763, 29.809748516, 39.236784259),
    (197.050679032, 28.559032830, 28.822779855),
    (197.177402103, 35.495516327, 19.641749261),
    (178.457384656, 41.633121462, 26.852492195),
)


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_preview_pair() -> tuple[bytes, bytes, bytes, os.stat_result]:
    """Read one atomically published, generation-consistent dual-image frame."""
    with PREVIEW_PAIR_PATH.open("rb") as stream:
        payload = stream.read()
        stat = os.fstat(stream.fileno())
    if len(payload) < PREVIEW_PAIR_HEADER.size:
        raise ValueError("预览帧包头不完整")
    magic, primary_length, secondary_length = PREVIEW_PAIR_HEADER.unpack_from(payload)
    expected_length = PREVIEW_PAIR_HEADER.size + primary_length + secondary_length
    if (
        magic != PREVIEW_PAIR_MAGIC
        or primary_length == 0
        or secondary_length == 0
        or len(payload) != expected_length
    ):
        raise ValueError("预览帧包格式无效")
    primary_start = PREVIEW_PAIR_HEADER.size
    secondary_start = primary_start + primary_length
    return (
        payload,
        payload[primary_start:secondary_start],
        payload[secondary_start:],
        stat,
    )


def tail_text_lines(path: Path, limit: int) -> list[str]:
    """Read only enough data from the end of a growing CSV for the live plots."""
    block_size = 64 * 1024
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        position = stream.tell()
        chunks: list[bytes] = []
        line_count = 0
        while position > 0 and line_count <= limit:
            size = min(block_size, position)
            position -= size
            stream.seek(position)
            chunk = stream.read(size)
            chunks.append(chunk)
            line_count += chunk.count(b"\n")
    return b"".join(reversed(chunks)).decode("utf-8", errors="replace").splitlines()[-limit:]


def require_number(value, path: str, minimum=None, maximum=None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} 必须是数字")
    if not math.isfinite(float(value)):
        raise ValueError(f"{path} 必须是有限数字")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} 不能小于 {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{path} 不能大于 {maximum}")


def model_asset_paths(config: dict) -> tuple[Path, Path]:
    """Resolve the configured ONNX and its conventional sibling FP16 engine."""
    raw_onnx = config["model"]["onnx_path"]
    onnx_input = Path(raw_onnx).expanduser()
    # Derive the engine before anchoring a relative path. This exactly matches
    # C++ changePath(), including the valid bare-name case "model.onnx".
    engine_input = (
        onnx_input.parent.parent / "engine" / f"{onnx_input.stem}-fp16.engine"
    )
    onnx = (
        onnx_input if onnx_input.is_absolute() else PROJECT_ROOT / onnx_input
    ).resolve(strict=False)
    engine = (
        engine_input if engine_input.is_absolute() else PROJECT_ROOT / engine_input
    ).resolve(strict=False)
    return onnx, engine


def model_asset_status(config: dict | None = None) -> dict:
    try:
        config = validate(load_config()) if config is None else config
        onnx, engine = model_asset_paths(config)
        engine_exists = engine.is_file()
        onnx_exists = onnx.is_file()
        available = engine_exists or onnx_exists
        if engine_exists:
            message = "已找到 FP16 TensorRT engine"
        elif onnx_exists:
            message = "未找到 engine；启动时将从 ONNX 构建"
        else:
            message = "ONNX 和对应的 FP16 engine 均不存在"
        return {
            "available": available,
            "onnx_path": str(onnx),
            "engine_path": str(engine),
            "onnx_exists": onnx_exists,
            "engine_exists": engine_exists,
            "message": message,
        }
    except (OSError, ValueError, KeyError) as exc:
        return {
            "available": False,
            "onnx_path": None,
            "engine_path": None,
            "onnx_exists": False,
            "engine_exists": False,
            "message": f"模型配置无效: {exc}",
        }


def ensure_model_assets(config: dict) -> dict:
    status = model_asset_status(config)
    if not status["available"]:
        raise FileNotFoundError(
            f"{status['message']}；ONNX: {status['onnx_path']}；engine: {status['engine_path']}"
        )
    if not status["engine_exists"]:
        engine_parent = Path(status["engine_path"]).parent
        engine_parent.mkdir(parents=True, exist_ok=True)
        if not os.access(engine_parent, os.W_OK):
            raise PermissionError(f"engine 输出目录不可写: {engine_parent}")
    return status


def validate(config: dict) -> dict:
    required = {
        "input", "detect_camera", "photo_camera", "calibration", "network",
        "runtime", "mocap",
    }
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError("配置结构不完整")

    model = config.setdefault("model", {"onnx_path": DEFAULT_ONNX_PATH})
    if not isinstance(model, dict):
        raise ValueError("model 必须是对象")
    model.setdefault("onnx_path", DEFAULT_ONNX_PATH)
    if not isinstance(model.get("onnx_path"), str) or not model["onnx_path"].strip():
        raise ValueError("model.onnx_path 不能为空")
    model["onnx_path"] = model["onnx_path"].strip()
    if len(model["onnx_path"]) > 4096 or not model["onnx_path"].lower().endswith(".onnx"):
        raise ValueError("model.onnx_path 必须是长度不超过 4096 的 .onnx 文件路径")

    source = config["input"]
    if source.get("mode") not in ("camera", "folder"):
        raise ValueError("input.mode 只能是 camera 或 folder")
    if not isinstance(source.get("folder_path"), str) or not source["folder_path"].strip():
        raise ValueError("input.folder_path 不能为空")
    require_number(source.get("interval_ms"), "input.interval_ms", 1, 60000)

    for name in ("detect_camera", "photo_camera"):
        camera = config[name]
        require_number(camera.get("camera_id"), f"{name}.camera_id", 0, 128)
        require_number(camera.get("fps"), f"{name}.fps", 1, 240)
        if camera.get("resolution") not in ("HD1080", "HD720"):
            raise ValueError(f"{name}.resolution 只能是 HD1080 或 HD720")
        if not isinstance(camera.get("apply_image_controls"), bool):
            raise ValueError(f"{name}.apply_image_controls 必须是布尔值")
        for field in ("auto_exposure_mode", "exposure", "white_balance_temperature", "brightness", "contrast", "sharpness"):
            require_number(camera.get(field), f"{name}.{field}")

    calibration = config["calibration"]
    calibration.setdefault(
        "model_keypoints_3d",
        [list(point) for point in DEFAULT_MODEL_KEYPOINTS_3D],
    )
    for field, length in (("camera_matrix", 9), ("distortion", 5), ("extrinsic", 16)):
        values = calibration.get(field)
        if not isinstance(values, list) or len(values) != length:
            raise ValueError(f"calibration.{field} 必须包含 {length} 个数字")
        for index, value in enumerate(values):
            require_number(value, f"calibration.{field}[{index}]")
    model_keypoints = calibration.get("model_keypoints_3d")
    if not isinstance(model_keypoints, list) or not (
        MODEL_KEYPOINT_MIN_COUNT <= len(model_keypoints) <= MODEL_KEYPOINT_MAX_COUNT
    ):
        raise ValueError(
            "calibration.model_keypoints_3d 必须包含 "
            f"{MODEL_KEYPOINT_MIN_COUNT}～{MODEL_KEYPOINT_MAX_COUNT} 个点"
        )
    for point_index, point in enumerate(model_keypoints):
        if not isinstance(point, list) or len(point) != 3:
            raise ValueError(
                f"calibration.model_keypoints_3d[{point_index}] 必须包含 X、Y、Z 三个数字"
            )
        for coordinate_index, value in enumerate(point):
            require_number(
                value,
                f"calibration.model_keypoints_3d[{point_index}][{coordinate_index}]",
            )
    if abs(calibration["camera_matrix"][8] - 1.0) > 1e-6:
        raise ValueError("相机内参矩阵最后一个元素必须为 1")
    if calibration["camera_matrix"][0] <= 0 or calibration["camera_matrix"][4] <= 0:
        raise ValueError("焦距 fx、fy 必须大于 0")
    if calibration["extrinsic"][12:16] != [0, 0, 0, 1] and calibration["extrinsic"][12:16] != [0.0, 0.0, 0.0, 1.0]:
        raise ValueError("外参矩阵最后一行必须为 0, 0, 0, 1")

    network = config["network"]
    if not isinstance(network.get("tcp_enabled"), bool):
        raise ValueError("network.tcp_enabled 必须是布尔值")
    for field in ("tcp_ip", "udp_ip"):
        if not isinstance(network.get(field), str):
            raise ValueError(f"network.{field} 必须是字符串")
    require_number(network.get("tcp_port"), "network.tcp_port", 1, 65535)
    require_number(network.get("udp_port"), "network.udp_port", 1, 65535)
    if network.get("socket_mode") not in (0, 1, 2):
        raise ValueError("network.socket_mode 只能是 0、1 或 2")

    mocap = config["mocap"]
    mocap.setdefault("clock_mode", "sdk_affine")
    mocap.setdefault("clock_fit_window_seconds", 60.0)
    mocap.setdefault("clock_fit_min_seconds", 3.0)
    mocap.setdefault("clock_max_rate_ppm", 5000.0)
    if not isinstance(mocap.get("enabled"), bool):
        raise ValueError("mocap.enabled 必须是布尔值")
    for field in ("server", "tracker"):
        if not isinstance(mocap.get(field), str) or not mocap[field].strip():
            raise ValueError(f"mocap.{field} 不能为空")
    require_number(mocap.get("stale_ms"), "mocap.stale_ms", 20, 10000)
    require_number(mocap.get("retry_seconds"), "mocap.retry_seconds", 1, 60)
    if mocap.get("clock_mode") not in ("sdk_affine", "receive"):
        raise ValueError("mocap.clock_mode 只能是 sdk_affine 或 receive")
    require_number(
        mocap.get("clock_fit_window_seconds"),
        "mocap.clock_fit_window_seconds",
        5,
        3600,
    )
    require_number(
        mocap.get("clock_fit_min_seconds"),
        "mocap.clock_fit_min_seconds",
        1,
        60,
    )
    require_number(
        mocap.get("clock_max_rate_ppm"),
        "mocap.clock_max_rate_ppm",
        100,
        100000,
    )

    sync = config.setdefault("sync", dict(SYNC_DEFAULTS))
    sync.setdefault("auto_offset_search_ms", SYNC_DEFAULTS["auto_offset_search_ms"])
    for field in ("enabled", "interpolate"):
        if not isinstance(sync.get(field), bool):
            raise ValueError(f"sync.{field} 必须是布尔值")
    require_number(sync.get("offset_ms"), "sync.offset_ms", -5000, 5000)
    require_number(sync.get("max_error_ms"), "sync.max_error_ms", 0.1, 1000)
    require_number(sync.get("history_ms"), "sync.history_ms", 100, 60000)
    require_number(
        sync.get("auto_offset_search_ms"),
        "sync.auto_offset_search_ms",
        50,
        5000,
    )

    runtime = config["runtime"]
    require_number(runtime.get("frame_width"), "runtime.frame_width", 1, 16384)
    require_number(runtime.get("frame_height"), "runtime.frame_height", 1, 16384)
    require_number(runtime.get("preview_fps"), "runtime.preview_fps", 1, 30)
    if not isinstance(runtime.get("preview_dir"), str) or not runtime["preview_dir"]:
        raise ValueError("runtime.preview_dir 不能为空")
    if not isinstance(runtime.get("save_pnp_results"), bool):
        raise ValueError("runtime.save_pnp_results 必须是布尔值")
    config["version"] = 1
    return config


def atomic_save_json(path: Path, payload, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def atomic_save_text(path: Path, payload: str, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def atomic_save_bytes(path: Path, payload: bytes, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def atomic_save(config: dict) -> None:
    atomic_save_json(CONFIG_PATH, config, "config-")


def changed_config_paths(before, after, prefix: str = "") -> list[str]:
    """Return stable leaf paths changed by one validated configuration save."""
    if isinstance(before, dict) and isinstance(after, dict):
        changed: list[str] = []
        for key in sorted(set(before) | set(after)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in before or key not in after:
                changed.append(path)
            else:
                changed.extend(changed_config_paths(before[key], after[key], path))
        return changed
    # Matrix/list elements are applied as one coherent unit, so report the field.
    if isinstance(before, list) or isinstance(after, list):
        return [prefix] if before != after else []
    return [prefix] if before != after else []


def config_apply_metadata(before: dict, after: dict) -> dict:
    changed = changed_config_paths(before, after)
    before_points = before.get("calibration", {}).get("model_keypoints_3d", [])
    after_points = after.get("calibration", {}).get("model_keypoints_3d", [])
    point_count_changed = (
        isinstance(before_points, list)
        and isinstance(after_points, list)
        and len(before_points) != len(after_points)
    )
    hot_apply = [
        path for path in changed
        if path.startswith("calibration.")
        and not (path == "calibration.model_keypoints_3d" and point_count_changed)
    ]
    restart_required = [path for path in changed if path not in hot_apply]
    return {
        "config_mtime_ns": str(CONFIG_PATH.stat().st_mtime_ns),
        "saved_at_unix_ns": str(time.time_ns()),
        "changed_paths": changed,
        "hot_apply_paths": hot_apply,
        "restart_required_paths": restart_required,
    }


def load_json_object(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def runtime_config_snapshot() -> dict:
    current_mtime = str(CONFIG_PATH.stat().st_mtime_ns)
    runtime = load_json_object(RUNTIME_CONFIG_STATUS_PATH)
    apply_meta = load_json_object(CONFIG_APPLY_META_PATH)
    if runtime is not None:
        runtime["current"] = (
            str(runtime.get("observed_config_mtime_ns", "")) == current_mtime
        )
    return {
        "config_mtime_ns": current_mtime,
        "runtime": runtime,
        "last_save": apply_meta,
    }


def calibration_signature(calibration: dict) -> str:
    return json.dumps(calibration, sort_keys=True, separators=(",", ":"))


def load_calibration_history() -> list[dict]:
    if not CALIBRATION_HISTORY_PATH.exists():
        return []
    payload = json.loads(CALIBRATION_HISTORY_PATH.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def record_calibration_history(*calibrations: dict) -> list[dict]:
    with CALIBRATION_HISTORY_LOCK:
        history = load_calibration_history()
        known = {calibration_signature(entry["calibration"]) for entry in history if isinstance(entry.get("calibration"), dict)}
        for calibration in calibrations:
            signature = calibration_signature(calibration)
            if signature in known:
                continue
            now_ns = time.time_ns()
            history.append({
                "id": str(now_ns),
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "calibration": calibration,
            })
            known.add(signature)
        history = history[-CALIBRATION_HISTORY_LIMIT:]
        atomic_save_json(CALIBRATION_HISTORY_PATH, history, "calibration-history-")
        return history


def _inference_process_unlocked() -> tuple[bool, int | None]:
    global INFERENCE_PROCESS, INFERENCE_LAST_EXIT_CODE
    if INFERENCE_PROCESS is not None:
        exit_code = INFERENCE_PROCESS.poll()
        if exit_code is None:
            return True, INFERENCE_PROCESS.pid
        INFERENCE_LAST_EXIT_CODE = exit_code
        INFERENCE_PROCESS = None
        PID_PATH.unlink(missing_ok=True)

    try:
        pid = int(PID_PATH.read_text(encoding="utf-8").strip())
        if pid <= 1:
            return False, None
        cmdline_path = Path(f"/proc/{pid}/cmdline")
        cmdline = cmdline_path.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
        if "trt" not in cmdline:
            return False, None
        os.kill(pid, 0)
        return True, pid
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, OSError):
        return False, None


def inference_process() -> tuple[bool, int | None]:
    with INFERENCE_LOCK:
        return _inference_process_unlocked()


def inference_capability() -> dict:
    executable = TRT_BINARY_PATH.is_file() and os.access(TRT_BINARY_PATH, os.X_OK)
    model_status = model_asset_status() if CONFIG_PATH.is_file() else {
        "available": False,
        "message": "配置文件不存在",
    }
    return {
        "available": executable and CONFIG_PATH.is_file() and model_status["available"],
        "binary_available": executable,
        "config_available": CONFIG_PATH.is_file(),
        "model_available": model_status["available"],
        "model": model_status,
        "last_exit_code": INFERENCE_LAST_EXIT_CODE,
        "started_unix_ns": (
            str(INFERENCE_STARTED_NS) if INFERENCE_STARTED_NS is not None else None
        ),
    }


def start_inference(*, discard_previous_session: bool = False) -> dict:
    """Start one inference process from the latest validated saved config."""
    global INFERENCE_PROCESS, INFERENCE_LAST_EXIT_CODE, INFERENCE_STARTED_NS
    global SESSION_ARCHIVED_MTIME_NS
    with INFERENCE_LOCK:
        running, pid = _inference_process_unlocked()
        if running:
            raise RuntimeError(f"推理进程已经在运行（PID {pid}）")
        if not TRT_BINARY_PATH.is_file() or not os.access(TRT_BINARY_PATH, os.X_OK):
            raise FileNotFoundError(f"推理程序不存在或不可执行: {TRT_BINARY_PATH}")
        if not CONFIG_PATH.is_file():
            raise FileNotFoundError(f"配置文件不存在: {CONFIG_PATH}")

        previous_session = session_status()
        if (previous_session.get("available") and
                previous_session.get("current_run") and
                not previous_session.get("archived") and
                not discard_previous_session):
            raise FileExistsError("当前 PnP 临时会话尚未另存；开始新推理会覆盖它")

        config = validate(load_config())
        model_status = ensure_model_assets(config)
        restart_mocap_for_session(config)
        for stale_path in (
            SESSION_CSV_PATH,
            PREVIEW_PAIR_PATH,
            RUNTIME_PATH / "primary.jpg",
            RUNTIME_PATH / "secondary.jpg",
            RUNTIME_PATH / "reprojection.json",
            RUNTIME_CONFIG_STATUS_PATH,
            SYNCED_SESSION_CSV_PATH,
            OFFLINE_CAMERA_REPORT_PATH,
            OFFLINE_MOCAP_REPORT_PATH,
            OFFLINE_REPORT_PATH,
            *LEGACY_OFFLINE_REPORT_PATHS,
            OFFLINE_SUMMARY_PATH,
            OFFSET_ESTIMATE_PATH,
        ):
            stale_path.unlink(missing_ok=True)

        process = subprocess.Popen(
            [str(TRT_BINARY_PATH), str(CONFIG_PATH)],
            cwd=str(PROJECT_ROOT),
        )
        INFERENCE_PROCESS = process
        INFERENCE_LAST_EXIT_CODE = None
        INFERENCE_STARTED_NS = time.time_ns()
        SESSION_ARCHIVED_MTIME_NS = None
        atomic_save_text(PID_PATH, f"{process.pid}\n", "trt-pid-")
        atomic_save_json(CONFIG_APPLY_META_PATH, {
            "config_mtime_ns": str(CONFIG_PATH.stat().st_mtime_ns),
            "saved_at_unix_ns": str(time.time_ns()),
            "changed_paths": [],
            "hot_apply_paths": [],
            "restart_required_paths": [],
        }, "config-apply-")
        return {
            "running": True,
            "pid": process.pid,
            "config_mtime_ns": str(CONFIG_PATH.stat().st_mtime_ns),
            "model": model_status,
        }


def request_inference_stop() -> tuple[bool, int | None]:
    with INFERENCE_LOCK:
        running, pid = _inference_process_unlocked()
        if not running or pid is None:
            return False, None
        if INFERENCE_PROCESS is not None and INFERENCE_PROCESS.pid == pid:
            INFERENCE_PROCESS.terminate()
        else:
            os.kill(pid, signal.SIGTERM)
        return True, pid


def shutdown_inference() -> None:
    global INFERENCE_PROCESS, INFERENCE_LAST_EXIT_CODE
    with INFERENCE_LOCK:
        process = INFERENCE_PROCESS
        if process is None:
            PID_PATH.unlink(missing_ok=True)
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        INFERENCE_LAST_EXIT_CODE = process.returncode
        INFERENCE_PROCESS = None
        PID_PATH.unlink(missing_ok=True)


def session_status() -> dict:
    if not SESSION_CSV_PATH.exists():
        return {"available": False, "bytes": 0, "mtime_ns": None}
    stat = SESSION_CSV_PATH.stat()
    has_rows = len(tail_text_lines(SESSION_CSV_PATH, 2)) >= 2
    return {"available": has_rows, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "current_run": stat.st_mtime_ns >= SERVER_STARTED_NS,
            "archived": SESSION_ARCHIVED_MTIME_NS == stat.st_mtime_ns}


def mark_session_archived() -> None:
    global SESSION_ARCHIVED_MTIME_NS
    try:
        SESSION_ARCHIVED_MTIME_NS = SESSION_CSV_PATH.stat().st_mtime_ns
    except OSError:
        SESSION_ARCHIVED_MTIME_NS = None


def latest_detection_pose() -> dict:
    """Read the newest visual PnP pose without retaining an open CSV handle."""
    state_path = RUNTIME_PATH / "reprojection.json"
    if state_path.exists():
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            pose = payload.get("pose")
            if payload.get("valid") and isinstance(pose, list) and len(pose) == 6:
                values = [float(value) for value in pose]
                if all(math.isfinite(value) for value in values):
                    stat = state_path.stat()
                    capture_timestamp_ns = int(
                        payload.get("capture_timestamp_ns")
                        or int(payload.get("timestamp", 0)) * 1_000_000
                    )
                    publish_timestamp_ns = int(payload.get("publish_timestamp_ns") or 0)
                    return {
                        "available": True,
                        "source": "reprojection",
                        "timestamp": int(payload.get("timestamp", 0)),
                        "capture_timestamp_ns": capture_timestamp_ns,
                        "publish_timestamp_ns": publish_timestamp_ns,
                        "pipeline_latency_ms": (
                            round((publish_timestamp_ns - capture_timestamp_ns) / 1_000_000, 3)
                            if publish_timestamp_ns >= capture_timestamp_ns else None
                        ),
                        "timestamp_source": payload.get("timestamp_source", "unknown"),
                        "timestamp_point": payload.get("timestamp_point", "unknown"),
                        "coordinate_frame": payload.get("coordinate_frame", "mocap"),
                        "updated_unix_ns": stat.st_mtime_ns,
                        "age_ms": round(max(0, time.time_ns() - stat.st_mtime_ns) / 1_000_000, 3),
                        "pose": dict(zip(("x", "y", "z", "rx", "ry", "rz"), values)),
                    }
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    if SESSION_CSV_PATH.exists():
        try:
            with SESSION_CSV_PATH.open("r", encoding="utf-8", newline="") as stream:
                header = next(csv.reader(stream))
            for line in reversed(tail_text_lines(SESSION_CSV_PATH, 10)):
                row = next(csv.reader([line]))
                if row == header or len(row) != len(header):
                    continue
                record = dict(zip(header, row))
                values = [float(record[field]) for field in ("x", "y", "z", "rx", "ry", "rz")]
                timestamp = int(record["timestamp"])
                capture_timestamp_ns = int(
                    record.get("capture_timestamp_ns") or timestamp * 1_000_000
                )
                publish_timestamp_ns = int(record.get("publish_timestamp_ns") or 0)
                stat = SESSION_CSV_PATH.stat()
                return {
                    "available": True,
                    "source": "csv",
                    "timestamp": timestamp,
                    "capture_timestamp_ns": capture_timestamp_ns,
                    "publish_timestamp_ns": publish_timestamp_ns,
                    "pipeline_latency_ms": (
                        round((publish_timestamp_ns - capture_timestamp_ns) / 1_000_000, 3)
                        if publish_timestamp_ns >= capture_timestamp_ns else None
                    ),
                    "timestamp_source": "csv",
                    "coordinate_frame": record.get("coordinate_frame") or "mocap",
                    "updated_unix_ns": stat.st_mtime_ns,
                    "age_ms": round(max(0, time.time_ns() - stat.st_mtime_ns) / 1_000_000, 3),
                    "pose": dict(zip(("x", "y", "z", "rx", "ry", "rz"), values)),
                }
        except (KeyError, OSError, ValueError, StopIteration, csv.Error):
            pass
    return {"available": False, "pose": None}


def comparison_snapshot() -> dict:
    detection = latest_detection_pose()
    mocap = (
        MOCAP_RECEIVER.snapshot()
        if MOCAP_RECEIVER is not None
        else {
            "enabled": False,
            "connection": "unavailable",
            "status": "unavailable",
            "message": "动捕接收器尚未初始化",
            "pose": None,
        }
    )
    receive_delta_ms = None
    mocap_receive_ns = (mocap.get("pose") or {}).get("receive_unix_ns")
    detection_update_ns = detection.get("updated_unix_ns")
    if isinstance(mocap_receive_ns, int) and isinstance(detection_update_ns, int):
        receive_delta_ms = round(
            (detection_update_ns - mocap_receive_ns) / 1_000_000, 3
        )
    sync_config = load_config().get("sync", SYNC_DEFAULTS)
    synchronization = {
        "status": "disabled",
        "message": "实时软件同步已关闭",
        "pose": None,
        "enabled": bool(sync_config.get("enabled", True)),
    }
    if synchronization["enabled"]:
        if MOCAP_RECEIVER is None:
            synchronization.update(status="waiting", message="动捕接收器尚未初始化")
        elif not detection.get("available"):
            synchronization.update(status="waiting", message="等待视觉 PnP 结果")
        elif detection.get("source") != "reprojection":
            synchronization.update(status="waiting", message="当前视觉结果来自 CSV 回退，不参与实时配对")
        else:
            synchronization = MOCAP_RECEIVER.match_at_unix_ns(
                int(detection.get("capture_timestamp_ns")
                    or int(detection["timestamp"]) * 1_000_000),
                offset_ms=float(sync_config.get("offset_ms", 0.0)),
                max_error_ms=float(sync_config.get("max_error_ms", 12.0)),
                interpolate=bool(sync_config.get("interpolate", True)),
            )
            synchronization["enabled"] = True
    synchronization["visual_timestamp_ms"] = detection.get("timestamp")
    synchronization["visual_capture_timestamp_ns"] = detection.get("capture_timestamp_ns")
    return {
        "server_unix_ns": time.time_ns(),
        "detection": detection,
        "mocap": mocap,
        # 正值表示视觉状态文件晚于最新动捕回调到达本机。
        "receive_delta_ms": receive_delta_ms,
        "synchronization": synchronization,
    }


def offline_report_status() -> dict:
    offset_estimation = None
    if OFFSET_ESTIMATE_PATH.exists():
        try:
            offset_estimation = json.loads(
                OFFSET_ESTIMATE_PATH.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            offset_estimation = None
    if not OFFLINE_SUMMARY_PATH.exists():
        return {
            "available": False,
            "detection_available": SESSION_CSV_PATH.exists(),
            "mocap_available": MOCAP_SESSION_CSV_PATH.exists(),
            "offset_estimation": offset_estimation,
        }
    try:
        summary = json.loads(OFFLINE_SUMMARY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {
            "available": False,
            "message": "离线同步摘要无法读取",
            "offset_estimation": offset_estimation,
        }
    camera_ready = OFFLINE_CAMERA_REPORT_PATH.is_file()
    matched_samples = int(summary.get("matched_samples") or 0)
    synchronized = (
        summary.get("report_mode") != "camera_only"
        and matched_samples > 0
        and OFFLINE_MOCAP_REPORT_PATH.is_file()
        and OFFLINE_REPORT_PATH.is_file()
    )
    report_urls = {}
    if camera_ready:
        report_urls["camera"] = "/api/sync/offline/report/camera.png"
    if synchronized:
        report_urls.update({
            "mocap": "/api/sync/offline/report/mocap.png",
            "composite": "/api/sync/offline/report/composite.png",
        })
    return {
        "available": SYNCED_SESSION_CSV_PATH.is_file() and camera_ready,
        "detection_available": int(summary.get("detection_samples") or 0) > 0,
        "mocap_available": int(summary.get("mocap_samples") or 0) > 0,
        "camera_only": not synchronized,
        "summary": summary,
        "offset_estimation": offset_estimation,
        "csv_url": "/api/sync/offline/download.csv",
        "report_urls": report_urls,
        "report_url": (
            "/api/sync/offline/report.png"
            if synchronized
            else "/api/sync/offline/report/camera.png"
        ),
    }


def estimate_offline_offset() -> dict:
    running, _ = inference_process()
    if running:
        raise RuntimeError("请先停止推理，确保两路会话完成刷新")
    if not SESSION_CSV_PATH.exists():
        raise FileNotFoundError("本次运行没有视觉 PnP CSV")
    if MOCAP_RECEIVER is not None:
        MOCAP_RECEIVER.finish_recording()
    if not MOCAP_SESSION_CSV_PATH.exists():
        raise FileNotFoundError("本次运行没有动捕会话 CSV")
    with SYNC_REPORT_LOCK:
        config = validate(load_config())
        stable_detections, _ = stabilize_detection_poses(
            load_detection_poses(SESSION_CSV_PATH)
        )
        estimation = estimate_time_offset(
            stable_detections,
            load_mocap_poses(MOCAP_SESSION_CSV_PATH),
            max_offset_ms=float(config["sync"]["auto_offset_search_ms"]),
        )
        estimation["estimated_unix_ns"] = time.time_ns()
        estimation["estimated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        estimation["applied"] = bool(estimation["reliable"])
        if estimation["applied"]:
            config["sync"]["offset_ms"] = estimation["offset_ms"]
            atomic_save(config)
            for stale_report in (
                SYNCED_SESSION_CSV_PATH,
                OFFLINE_CAMERA_REPORT_PATH,
                OFFLINE_MOCAP_REPORT_PATH,
                OFFLINE_REPORT_PATH,
                *LEGACY_OFFLINE_REPORT_PATHS,
                OFFLINE_SUMMARY_PATH,
            ):
                stale_report.unlink(missing_ok=True)
        atomic_save_json(OFFSET_ESTIMATE_PATH, estimation, "offset-estimate-")
    return {
        "ok": True,
        "offset_estimation": estimation,
        "config_offset_ms": config["sync"]["offset_ms"],
    }


def build_offline_report() -> dict:
    running, _ = inference_process()
    if running:
        raise RuntimeError("请先停止推理，确保视觉 CSV 完成刷新")
    if not SESSION_CSV_PATH.exists():
        raise FileNotFoundError("本次运行没有视觉 PnP CSV")
    if MOCAP_RECEIVER is not None:
        MOCAP_RECEIVER.finish_recording()
    sync_config = load_config().get("sync", SYNC_DEFAULTS)
    with SYNC_REPORT_LOCK:
        offset_estimation = None
        if OFFSET_ESTIMATE_PATH.exists():
            try:
                offset_estimation = json.loads(
                    OFFSET_ESTIMATE_PATH.read_text(encoding="utf-8")
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        summary, csv_payload, reports = generate_offline_reports(
            SESSION_CSV_PATH,
            MOCAP_SESSION_CSV_PATH if MOCAP_SESSION_CSV_PATH.is_file() else None,
            offset_ms=float(sync_config.get("offset_ms", 0.0)),
            max_error_ms=float(sync_config.get("max_error_ms", 12.0)),
            interpolate=bool(sync_config.get("interpolate", True)),
            offset_estimation=offset_estimation,
            report_format="png",
        )
        matched_mocap = int(summary.get("matched_samples") or 0) > 0
        summary["report_mode"] = "synchronized" if matched_mocap else "camera_only"
        if matched_mocap:
            summary["report_message"] = "相机、动捕与合成报告已生成"
        elif int(summary.get("mocap_samples") or 0) > 0:
            summary["report_message"] = "动捕样本未与相机成功配对；已生成相机检测报告"
        else:
            summary["report_message"] = "未记录到动捕数据；已生成相机检测报告"
        summary["generated_unix_ns"] = time.time_ns()
        summary["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        atomic_save_text(SYNCED_SESSION_CSV_PATH, csv_payload, "sync-csv-")
        atomic_save_bytes(
            OFFLINE_CAMERA_REPORT_PATH, reports["camera"], "sync-camera-png-"
        )
        if matched_mocap:
            atomic_save_bytes(
                OFFLINE_MOCAP_REPORT_PATH, reports["mocap"], "sync-mocap-png-"
            )
            atomic_save_bytes(
                OFFLINE_REPORT_PATH, reports["composite"], "sync-composite-png-"
            )
        else:
            OFFLINE_MOCAP_REPORT_PATH.unlink(missing_ok=True)
            OFFLINE_REPORT_PATH.unlink(missing_ok=True)
        atomic_save_json(OFFLINE_SUMMARY_PATH, summary, "sync-summary-")
    return offline_report_status()


def tracker_selector(mode, value) -> str:
    """Build an unambiguous bridge selector from the web form."""
    if mode == "name":
        if not isinstance(value, str):
            raise ValueError("刚体名称必须是字符串")
        name = value.strip()
        if not name:
            raise ValueError("刚体名称不能为空")
        if len(name) > 200 or any(ord(character) < 32 for character in name):
            raise ValueError("刚体名称不能超过 200 个字符或包含控制字符")
        return f"name:{name}"
    if mode == "id":
        if isinstance(value, bool):
            raise ValueError("刚体 ID 必须是非负整数")
        try:
            tracker_id = int(value)
        except (TypeError, ValueError):
            raise ValueError("刚体 ID 必须是非负整数") from None
        if str(value).strip() != str(tracker_id) or not 0 <= tracker_id <= 2_147_483_647:
            raise ValueError("刚体 ID 必须是非负整数")
        return f"id:{tracker_id}"
    raise ValueError("目标选择方式只能是 name 或 id")


def replace_mocap_receiver(config: dict, *, clear_session: bool = False) -> dict:
    global MOCAP_RECEIVER
    with MOCAP_APPLY_LOCK:
        previous = MOCAP_RECEIVER
        if previous is not None:
            previous.stop()
        if clear_session:
            MOCAP_SESSION_CSV_PATH.unlink(missing_ok=True)
        replacement = MocapReceiver(
            config["mocap"],
            PROJECT_ROOT,
            history_ms=config["sync"]["history_ms"],
            session_csv_path=MOCAP_SESSION_CSV_PATH,
        )
        MOCAP_RECEIVER = replacement
        replacement.start()
        return replacement.snapshot()


def restart_mocap_for_session(config: dict) -> dict:
    """Apply the latest saved mocap settings and begin a fresh session."""
    return replace_mocap_receiver(config, clear_session=True)


def apply_mocap_target(mode, value) -> tuple[str, dict]:
    """Persist one target selector and reconnect only the mocap receiver."""
    selector = tracker_selector(mode, value)
    with CONFIG_SAVE_LOCK:
        config = load_config()
        current = json.loads(json.dumps(config))
        config["mocap"]["tracker"] = selector
        config = validate(config)
        atomic_save(config)
        apply_meta = config_apply_metadata(current, config)
        atomic_save_json(CONFIG_APPLY_META_PATH, apply_meta, "config-apply-")
    return selector, replace_mocap_receiver(config)


class DashboardHandler(SimpleHTTPRequestHandler):
    server_version = "CVIA-Dashboard/1.2"
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "static"), **kwargs)

    def guess_type(self, path):
        """Declare UTF-8 explicitly for static text on legacy Windows browsers."""
        content_type = super().guess_type(path)
        media_type = content_type.split(";", 1)[0].lower()
        if media_type in {
            "text/html", "text/css", "text/javascript", "application/javascript",
            "application/json", "image/svg+xml",
        }:
            return f"{media_type}; charset=utf-8"
        return content_type

    def end_headers(self):
        # Win7 上的旧版 Chromium/兼容配置不能总是可靠推断外部 CSS/JS 编码。
        self.send_header("Content-Language", "zh-CN")
        self.send_header("X-UA-Compatible", "IE=edge")
        static_path = urlparse(self.path).path
        if static_path == "/" or Path(static_path).suffix.lower() in {".html", ".css", ".js"}:
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        # 预览轮询属于正常高频流量。逐请求打印会在 15–30 FPS 双图模式下
        # 产生每秒几十行终端 I/O，反过来拖慢网页与推理进程。
        path = urlparse(self.path).path
        status = str(args[1]) if len(args) > 1 else ""
        hot_path = path.startswith("/preview/") or path in {
            "/api/reprojection", "/api/pnp", "/api/pose-comparison",
            "/api/status",
        }
        if hot_path and status.startswith(("2", "3")):
            return
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}")

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        if path == "/api/config":
            try:
                # Populate backward-compatible defaults for older config files without
                # writing to disk until the user explicitly saves the form.
                self.send_json(validate(load_config()))
            except Exception as exc:
                self.send_json({"error": str(exc)}, 500)
            return
        if path == "/api/status":
            frames = {}
            now = time.time()
            try:
                _, primary, secondary, stat = read_preview_pair()
                age_ms = round((now - stat.st_mtime) * 1000)
                frames["primary"] = {
                    "available": True,
                    "age_ms": age_ms,
                    "bytes": len(primary),
                }
                frames["secondary"] = {
                    "available": True,
                    "age_ms": age_ms,
                    "bytes": len(secondary),
                }
            except (OSError, ValueError):
                # 兼容尚未更新的推理程序；新版本只写 preview_pair.bin。
                for name in ("primary", "secondary"):
                    image = RUNTIME_PATH / f"{name}.jpg"
                    try:
                        stat = image.stat()
                        frames[name] = {
                            "available": True,
                            "age_ms": round((now - stat.st_mtime) * 1000),
                            "bytes": stat.st_size,
                        }
                    except OSError:
                        frames[name] = {
                            "available": False,
                            "age_ms": None,
                            "bytes": 0,
                        }
            inference_running, inference_pid = inference_process()
            self.send_json({"frames": frames, "config_mtime": CONFIG_PATH.stat().st_mtime_ns,
                            "runtime_config": runtime_config_snapshot(),
                            "inference": {
                                "running": inference_running,
                                "pid": inference_pid,
                                **inference_capability(),
                            },
                            "pnp_session": session_status(),
                            "mocap": MOCAP_RECEIVER.snapshot() if MOCAP_RECEIVER else None})
            return
        if path == "/api/pose-comparison":
            try:
                self.send_json(comparison_snapshot())
            except Exception as exc:
                self.send_json({"error": f"读取检测/动捕位姿失败: {exc}"}, 500)
            return
        if path == "/api/sync/offline":
            self.send_json(offline_report_status())
            return
        report_routes = {
            "/api/sync/offline/report.png": (OFFLINE_REPORT_PATH, "synchronized_report.png"),
            "/api/sync/offline/report/camera.png": (OFFLINE_CAMERA_REPORT_PATH, "synchronized_camera_report.png"),
            "/api/sync/offline/report/mocap.png": (OFFLINE_MOCAP_REPORT_PATH, "synchronized_mocap_report.png"),
            "/api/sync/offline/report/composite.png": (OFFLINE_REPORT_PATH, "synchronized_report.png"),
        }
        if path in report_routes:
            report_path, filename = report_routes[path]
            if not report_path.exists():
                self.send_error(404, "尚未生成对应的离线同步图")
                return
            data = report_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Disposition", f'inline; filename="{filename}"')
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)
            mark_session_archived()
            return
        if path == "/api/sync/offline/download.csv":
            if not SYNCED_SESSION_CSV_PATH.exists():
                self.send_json({"error": "尚未生成离线同步 CSV"}, 404)
                return
            data = SYNCED_SESSION_CSV_PATH.read_bytes()
            filename = time.strftime("cvia_synchronized_%Y%m%d_%H%M%S.csv")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/api/calibration-history":
            try:
                history = load_calibration_history()
                self.send_json({"entries": list(reversed(history)), "limit": CALIBRATION_HISTORY_LIMIT})
            except Exception as exc:
                self.send_json({"error": f"读取标定历史失败: {exc}"}, 500)
            return
        if path == "/api/reprojection":
            state_path = RUNTIME_PATH / "reprojection.json"
            try:
                if not state_path.exists():
                    self.send_json({"available": False})
                    return
                payload = json.loads(state_path.read_text(encoding="utf-8"))
                payload["available"] = True
                payload["mtime_ns"] = state_path.stat().st_mtime_ns
                self.send_json(payload)
            except Exception as exc:
                self.send_json({"error": f"读取同步重投影状态失败: {exc}"}, 500)
            return
        if path == "/api/pnp":
            try:
                limit = int(parse_qs(parsed_url.query).get("limit", ["240"])[0])
                limit = max(10, min(1000, limit))
                csv_path = SESSION_CSV_PATH
                if not csv_path.exists():
                    self.send_json({"available": False, "columns": ["timestamp", "x", "y", "z", "rx", "ry", "rz"], "rows": []})
                    return
                rows = []
                with csv_path.open("r", encoding="utf-8", newline="") as stream:
                    header = next(csv.reader(stream))
                for row in csv.reader(tail_text_lines(csv_path, limit + 1)):
                    if row == header or len(row) != len(header):
                        continue
                    try:
                        record = dict(zip(header, row))
                        rows.append([
                            int(record["timestamp"]),
                            *[float(record[field]) for field in ("x", "y", "z", "rx", "ry", "rz")],
                        ])
                    except (KeyError, ValueError):
                        # 跳过程序正在追加但尚未完整刷新的末行。
                        continue
                rows = rows[-limit:]
                stat = csv_path.stat()
                self.send_json({"available": True, "columns": ["timestamp", "x", "y", "z", "rx", "ry", "rz"], "rows": rows, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
            except Exception as exc:
                self.send_json({"error": f"读取 PnP CSV 失败: {exc}"}, 500)
            return
        if path == "/api/pnp/download":
            running, _ = inference_process()
            if running:
                self.send_json({"error": "请先停止推理，等待临时 PnP 会话完成刷新"}, 409)
                return
            if not session_status()["available"]:
                self.send_json({"error": "本次运行没有有效的 PnP 位姿可保存"}, 404)
                return
            data = SESSION_CSV_PATH.read_bytes()
            filename = time.strftime("cvia_pnp_%Y%m%d_%H%M%S.csv")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/preview/pair.bin":
            try:
                payload, _, _, stat = read_preview_pair()
            except (OSError, ValueError) as exc:
                self.send_error(404, f"等待推理程序输出双图帧: {exc}")
                return
            etag = f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
            if self.headers.get("If-None-Match") == etag:
                self.send_response(304)
                self.send_header("ETag", etag)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("ETag", etag)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)
            return
        if path.startswith("/preview/"):
            name = Path(path).name
            if name not in ("primary.jpg", "secondary.jpg"):
                self.send_error(404)
                return
            try:
                _, primary, secondary, _ = read_preview_pair()
                data = primary if name == "primary.jpg" else secondary
            except (OSError, ValueError):
                image = RUNTIME_PATH / name
                if not image.exists():
                    self.send_error(404, "等待推理程序输出图像")
                    return
                data = image.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_PUT(self):
        if urlparse(self.path).path != "/api/config":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY:
                raise ValueError("请求体大小无效")
            payload = validate(json.loads(self.rfile.read(length).decode("utf-8")))
            with CONFIG_SAVE_LOCK:
                current = validate(load_config())
                record_calibration_history(current["calibration"], payload["calibration"])
                atomic_save(payload)
                apply_meta = config_apply_metadata(current, payload)
                atomic_save_json(CONFIG_APPLY_META_PATH, apply_meta, "config-apply-")
            hot_apply = apply_meta["hot_apply_paths"]
            restart_required = apply_meta["restart_required_paths"]
            if hot_apply and restart_required:
                message = "配置已保存；标定参数正在热应用，其他修改将在下次开始推理时生效"
            elif hot_apply:
                message = "配置已保存；标定参数正在热应用"
            elif restart_required:
                message = "配置已保存；这些修改将在下次开始推理时生效"
            else:
                message = "配置没有变化"
            self.send_json({
                "ok": True,
                "message": message,
                "apply": apply_meta,
            })
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, 400)
        except Exception as exc:
            self.send_json({"error": f"保存失败: {exc}"}, 500)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/mocap/target":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_BODY:
                    raise ValueError("请求体大小无效")
                payload = json.loads(
                    self.rfile.read(length).decode("utf-8")
                )
                if not isinstance(payload, dict):
                    raise ValueError("请求体必须是对象")
                selector, mocap = apply_mocap_target(
                    payload.get("mode"), payload.get("value")
                )
                self.send_json({
                    "ok": True,
                    "tracker": selector,
                    "mocap": mocap,
                    "message": f"目标已切换为 {selector}，正在重新连接动捕",
                })
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, 400)
            except Exception as exc:
                self.send_json({"error": f"应用目标刚体失败: {exc}"}, 500)
            return
        if path == "/api/inference/start":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > MAX_BODY:
                    raise ValueError("请求体大小无效")
                payload = (
                    json.loads(self.rfile.read(length).decode("utf-8"))
                    if length else {}
                )
                if not isinstance(payload, dict):
                    raise ValueError("请求体必须是对象")
                discard = payload.get("discard_previous_session", False)
                if not isinstance(discard, bool):
                    raise ValueError("discard_previous_session 必须是布尔值")
                result = start_inference(discard_previous_session=discard)
                self.send_json({
                    "ok": True,
                    "message": "推理进程已启动，正在初始化模型和输入源",
                    **result,
                }, 201)
            except FileExistsError as exc:
                self.send_json({
                    "error": str(exc),
                    "requires_confirmation": True,
                }, 409)
            except RuntimeError as exc:
                self.send_json({"error": str(exc)}, 409)
            except FileNotFoundError as exc:
                self.send_json({"error": str(exc)}, 503)
            except PermissionError as exc:
                self.send_json({"error": str(exc)}, 503)
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, 400)
            except Exception as exc:
                self.send_json({"error": f"启动推理失败: {exc}"}, 500)
            return
        if path == "/api/inference/stop":
            try:
                stopped, pid = request_inference_stop()
            except ProcessLookupError:
                stopped, pid = False, None
            except Exception as exc:
                self.send_json({"error": f"停止推理失败: {exc}"}, 500)
                return
            if not stopped or pid is None:
                self.send_json({"ok": True, "message": "推理进程已经停止", "running": False})
                return
            self.send_json({
                "ok": True,
                "message": "已请求推理进程优雅停止",
                "running": True,
                "pid": pid,
            })
            return
        if path == "/api/sync/offline/generate":
            try:
                self.send_json({"ok": True, **build_offline_report()})
            except RuntimeError as exc:
                self.send_json({"error": str(exc)}, 409)
            except FileNotFoundError as exc:
                self.send_json({"error": str(exc)}, 404)
            except Exception as exc:
                self.send_json({"error": f"生成离线同步报告失败: {exc}"}, 500)
            return
        if path == "/api/sync/offline/estimate-offset":
            try:
                self.send_json(estimate_offline_offset())
            except RuntimeError as exc:
                self.send_json({"error": str(exc)}, 409)
            except FileNotFoundError as exc:
                self.send_json({"error": str(exc)}, 404)
            except ValueError as exc:
                self.send_json({"error": str(exc)}, 422)
            except Exception as exc:
                self.send_json({"error": f"自动估计时间补偿失败: {exc}"}, 500)
            return
        self.send_error(404)


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    request_queue_size = 64


def main() -> None:
    global MOCAP_RECEIVER
    parser = argparse.ArgumentParser(description="CVIA AGX Orin web dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="listen address; default exposes the dashboard to the local network")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    RUNTIME_PATH.mkdir(parents=True, exist_ok=True)
    config = validate(load_config())
    atomic_save_json(CONFIG_APPLY_META_PATH, {
        "config_mtime_ns": str(CONFIG_PATH.stat().st_mtime_ns),
        "saved_at_unix_ns": str(time.time_ns()),
        "changed_paths": [],
        "hot_apply_paths": [],
        "restart_required_paths": [],
    }, "config-apply-")
    for session_path in (
        PREVIEW_PAIR_PATH,
        RUNTIME_PATH / "primary.jpg",
        RUNTIME_PATH / "secondary.jpg",
        RUNTIME_PATH / "reprojection.json",
        RUNTIME_CONFIG_STATUS_PATH,
        MOCAP_SESSION_CSV_PATH,
        SYNCED_SESSION_CSV_PATH,
        OFFLINE_CAMERA_REPORT_PATH,
        OFFLINE_MOCAP_REPORT_PATH,
        OFFLINE_REPORT_PATH,
        *LEGACY_OFFLINE_REPORT_PATHS,
        OFFLINE_SUMMARY_PATH,
        OFFSET_ESTIMATE_PATH,
    ):
        session_path.unlink(missing_ok=True)
    record_calibration_history(config["calibration"])
    server = DashboardServer((args.host, args.port), DashboardHandler)
    MOCAP_RECEIVER = MocapReceiver(
        config["mocap"],
        PROJECT_ROOT,
        history_ms=config["sync"]["history_ms"],
        session_csv_path=MOCAP_SESSION_CSV_PATH,
    )
    MOCAP_RECEIVER.start()
    print(f"CVIA dashboard: http://{args.host}:{args.port}")
    if config["mocap"]["enabled"]:
        print(
            "NOKOV mocap: "
            f"{config['mocap']['tracker']} @ {config['mocap']['server']}"
        )
    print(
        "Inference is stopped initially. Save the configuration in the dashboard, "
        "then click Start Inference to launch the configured runtime."
    )
    def request_dashboard_shutdown(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_dashboard_shutdown)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        shutdown_inference()
        MOCAP_RECEIVER.stop()


if __name__ == "__main__":
    main()
