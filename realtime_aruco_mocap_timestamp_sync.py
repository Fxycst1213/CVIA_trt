#!/usr/bin/env python3
"""实时查看并拼接 ArUco 相机位姿与 NOKOV/XING 动捕位姿。

时间同步遵循 ``README_SDK_AFFINE_TIME_SYNC.md``：

1. ``MocapBridge`` 每帧输出 SDK 独立时钟及 Orin 收包时间；
2. ``MocapClockMapper`` 把 SDK 时钟动态映射到 Orin Unix 时间域；
3. 固定业务延迟 ``offset_ms`` 加到校正后的动捕时间轴；
4. 每个 ArUco 相机帧按采集时间查找两侧动捕帧，位置线性插值、姿态 SLERP；
5. CSV 同时保存原始 SDK、收包、仿射校正和最终配对时间，便于诊断。

坐标约定与 ``camera_mocap_extrinsic_batch.json`` 一致：``T_X_Y`` 表示坐标系 Y
在坐标系 X 中的位姿。ArUco 刚体预测为::

    T_M_B_pred = T_M_C @ T_C_A @ T_A_B

相机由 OpenCV 读取时，程序只能把 ``VideoCapture.read()`` 返回时刻作为相机时间戳，
它不是严格的曝光时刻。这个固定差可以通过 ``offset_ms`` 标定，但如果需要驱动 SOE/EOF
时间戳，应使用 ``/home/wts/CVIA_trt`` 的直接 V4L2 采集链路。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, TextIO

import cv2
import numpy as np

from aruco_mocap_curve import generate_curve_report
from capture_ir_camera import (
    DEFAULT_VALUES,
    SETTINGS,
    available_v4l2_controls,
    configure_camera,
    desktop_display_available,
    find_camera_on_usb_port,
    open_camera,
    print_camera_status,
)
from realtime_camera_error import (
    Calibration,
    detect_marker_pose,
    draw_text_lines,
    load_calibration,
    make_aruco_detector,
    pose_error,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION = SCRIPT_DIR / "output" / "camera_mocap_extrinsic_batch.json"
DEFAULT_CVIA_ROOT = Path("/home/wts/CVIA_trt")
DEFAULT_CVIA_CONFIG = DEFAULT_CVIA_ROOT / "web_monitor" / "config.json"
DEFAULT_BRIDGE = DEFAULT_CVIA_ROOT / "web_monitor" / "mocap" / "bin" / "MocapBridge"
CAMERA_TIMESTAMP_SOURCE = "opencv_read_return_clock_realtime"


CSV_COLUMNS = (
    "camera_frame_index",
    "camera_time_iso8601",
    "camera_capture_timestamp_ns",
    "camera_monotonic_ns",
    "camera_timestamp_source",
    "camera_publish_timestamp_ns",
    "pipeline_latency_ms",
    "status",
    "marker_detected",
    "marker_id",
    "reprojection_error_px",
    "aruco_x_mm",
    "aruco_y_mm",
    "aruco_z_mm",
    "aruco_qx",
    "aruco_qy",
    "aruco_qz",
    "aruco_qw",
    "clock_model_ready",
    "clock_model_mode",
    "clock_samples",
    "clock_span_seconds",
    "clock_rate_ppm",
    "clock_residual_p95_ms",
    "sync_offset_ms",
    "sync_method",
    "sync_error_ms",
    "interpolation_alpha",
    "bracket_span_ms",
    "mocap_frames",
    "mocap_bracket_receive_unix_ns",
    "mocap_bracket_aligned_unix_ns",
    "mocap_sdk_timestamp_ms",
    "mocap_receive_unix_ns",
    "mocap_reference_aligned_unix_ns",
    "mocap_aligned_unix_ns",
    "mocap_corrected_unix_ns",
    "mocap_x_mm",
    "mocap_y_mm",
    "mocap_z_mm",
    "mocap_qx",
    "mocap_qy",
    "mocap_qz",
    "mocap_qw",
    "pred_x_mm",
    "pred_y_mm",
    "pred_z_mm",
    "translation_error_mm",
    "rotation_error_deg",
)


def _load_cvia_mocap_api(cvia_root: Path) -> Any:
    """Load the tested affine mapper from CVIA_trt without copying its logic."""
    module_directory = cvia_root / "web_monitor"
    module_path = module_directory / "mocap_receiver.py"
    if not module_path.is_file():
        raise RuntimeError(f"找不到仿射时钟实现：{module_path}")
    directory_text = str(module_directory)
    if directory_text not in sys.path:
        sys.path.insert(0, directory_text)
    try:
        import mocap_receiver  # type: ignore
    except ImportError as exc:
        raise RuntimeError(f"无法加载 {module_path}：{exc}") from exc
    return mocap_receiver


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取配置 {path}：{exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"配置根节点必须是对象：{path}")
    return value


def _nested(config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    section_value = config.get(section)
    if not isinstance(section_value, dict):
        return default
    return section_value.get(key, default)


def _elf_architecture(path: Path) -> str | None:
    try:
        with path.open("rb") as stream:
            header = stream.read(20)
    except OSError:
        return None
    if len(header) < 20 or header[:4] != b"\x7fELF":
        return None
    machine = int.from_bytes(header[18:20], "little")
    return {0x3E: "x86_64", 0xB7: "aarch64"}.get(machine, f"e_machine={machine}")


def _native_architecture() -> str:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "aarch64"
    if machine in {"amd64", "x86_64"}:
        return "x86_64"
    return machine


def _iso8601_from_ns(timestamp_ns: int) -> str:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    value = dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc).astimezone()
    return f"{value.strftime('%Y-%m-%dT%H:%M:%S')}.{nanoseconds:09d}{value.strftime('%z')}"


def _rotation_to_quaternion_xyzw(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a normalized x/y/z/w quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ),
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]))
            quaternion = np.array(
                (
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                )
            )
        elif axis == 1:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]))
            quaternion = np.array(
                (
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                )
            )
        else:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]))
            quaternion = np.array(
                (
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                )
            )
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("旋转矩阵无法转换为有效四元数")
    normalized = quaternion / norm
    return tuple(float(value) for value in normalized)


def _payload_pose_matrix(payload: dict[str, Any]) -> np.ndarray:
    """Convert ``MocapTimeline.match`` payload to a 4x4 pose matrix."""
    position = payload["position"]
    quaternion = payload["quaternion"]
    x = float(quaternion["qx"])
    y = float(quaternion["qy"])
    z = float(quaternion["qz"])
    w = float(quaternion["qw"])
    norm = float(np.linalg.norm((x, y, z, w)))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("匹配后的动捕四元数无效")
    x, y, z, w = (value / norm for value in (x, y, z, w))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    matrix[:3, 3] = (
        float(position["x"]),
        float(position["y"]),
        float(position["z"]),
    )
    return matrix


class AffineMocapClient:
    """Run the CLOCK-capable bridge and maintain a searchable aligned timeline."""

    def __init__(
        self,
        *,
        api: Any,
        server: str,
        tracker: str,
        bridge_path: Path,
        start_timeout: float,
        history_ms: float,
        clock_window_seconds: float,
        clock_minimum_span_seconds: float,
        clock_maximum_rate_ppm: float,
        require_tracked: bool,
    ) -> None:
        self.api = api
        self.server = server
        self.tracker = tracker
        self.bridge_path = bridge_path
        self.start_timeout = start_timeout
        self.history_ns = int(history_ms * 1_000_000)
        self.require_tracked = require_tracked
        self.mapper = api.MocapClockMapper(
            window_seconds=clock_window_seconds,
            minimum_span_seconds=clock_minimum_span_seconds,
            maximum_rate_ppm=clock_maximum_rate_ppm,
        )
        self._condition = threading.Condition()
        self._history: deque[Any] = deque()
        self._process: subprocess.Popen[str] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._errors: deque[str] = deque(maxlen=20)
        self.sdk_version: str | None = None

    def start(self) -> None:
        if not self.bridge_path.is_file():
            raise RuntimeError(
                f"找不到 CLOCK 版 MocapBridge：{self.bridge_path}。"
                "请按 README_SDK_AFFINE_TIME_SYNC.md 第十一节构建"
            )
        executable_arch = _elf_architecture(self.bridge_path)
        native_arch = _native_architecture()
        if executable_arch and executable_arch != native_arch:
            raise RuntimeError(
                f"MocapBridge 架构为 {executable_arch}，当前机器为 {native_arch}"
            )
        command = (
            str(self.bridge_path),
            "--server",
            self.server,
            "--tracker",
            self.tracker,
        )
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise RuntimeError(f"无法启动 {self.bridge_path}：{exc}") from exc
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(self._process.stdout,),
            daemon=True,
            name="aruco-affine-mocap-stdout",
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(self._process.stderr,),
            daemon=True,
            name="aruco-affine-mocap-stderr",
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        if not self._ready.wait(self.start_timeout):
            detail = f"；最近错误：{self._errors[-1]}" if self._errors else ""
            self.stop()
            raise RuntimeError(f"等待动捕 SDK READY 超时（{self.start_timeout:g} 秒）{detail}")

    def _read_stdout(self, stream: TextIO) -> None:
        try:
            for raw_line in stream:
                if self._stop.is_set():
                    break
                line = raw_line.rstrip("\r\n")
                fields = line.split("\t")
                if not fields:
                    continue
                try:
                    if fields[0] == "READY" and len(fields) >= 2:
                        with self._condition:
                            self.mapper.reset()
                            self._history.clear()
                            self.sdk_version = fields[1]
                            self._ready.set()
                            self._condition.notify_all()
                    elif fields[0] == "CLOCK":
                        if len(fields) != 5:
                            raise ValueError(f"CLOCK 应为 5 列，实际 {len(fields)} 列")
                        with self._condition:
                            self.mapper.observe_timestamp(
                                int(fields[2]), int(fields[4]), int(fields[3])
                            )
                            self._condition.notify_all()
                    elif fields[0] == "POSE":
                        pose = self.api.parse_pose_line(line)
                        with self._condition:
                            model = self.mapper.observe(pose)
                            pose = self.mapper.map_pose(pose, model)
                            valid, issue = self.api.pose_validity(pose)
                            if self.require_tracked and not (pose.tracking_params & 0x01):
                                valid, issue = False, "tracking_params 未置有效位"
                            if valid:
                                self._history.append(pose)
                                cutoff = pose.receive_monotonic_ns - self.history_ns
                                while (
                                    self._history
                                    and self._history[0].receive_monotonic_ns < cutoff
                                ):
                                    self._history.popleft()
                            elif issue:
                                self._errors.append(issue)
                            self._condition.notify_all()
                except (ValueError, OverflowError) as exc:
                    self._errors.append(f"无法解析桥接数据：{exc}")
        finally:
            with self._condition:
                self._condition.notify_all()

    def _read_stderr(self, stream: TextIO) -> None:
        for raw_line in stream:
            line = raw_line.rstrip("\r\n")
            if line:
                self._errors.append(line)
                print(f"动捕 SDK：{line}", file=sys.stderr)

    def wait_clock_ready(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self.mapper.model.ready:
                process = self._process
                stdout_thread = self._stdout_thread
                if (
                    process is not None
                    and process.poll() is not None
                    and (stdout_thread is None or not stdout_thread.is_alive())
                ):
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(min(remaining, 0.2))
            return True

    def clock_model_dict(self) -> dict[str, Any]:
        with self._condition:
            return self.mapper.model.as_dict()

    def match(
        self,
        target_unix_ns: int,
        *,
        offset_ms: float,
        max_error_ms: float,
        interpolate: bool,
        wait_ms: float,
    ) -> dict[str, Any]:
        """Wait briefly for the right bracket, then match at camera capture time."""
        deadline = time.monotonic() + wait_ms / 1000.0
        raw_target_ns = target_unix_ns - int(round(offset_ms * 1_000_000))
        with self._condition:
            while True:
                model = self.mapper.model
                poses = list(self._history)
                if model.ready and poses:
                    newest = self.mapper.map_pose(poses[-1], model).timeline_unix_ns
                    if newest >= raw_target_ns:
                        break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(min(remaining, 0.01))
            model = self.mapper.model
            poses = list(self._history)
        if not model.ready:
            return {
                "status": "clock_warming",
                "message": "仿射时钟模型尚未就绪",
                "pose": None,
                "clock_model": model.as_dict(),
            }
        # Remap retained poses with the latest mature model so the whole window
        # uses one rate and anchor rather than the model version at arrival time.
        aligned = [self.mapper.map_pose(pose, model) for pose in poses]
        result = self.api.MocapTimeline(aligned).match(
            target_unix_ns,
            offset_ms=offset_ms,
            max_error_ms=max_error_ms,
            interpolate=interpolate,
        )
        result["clock_model"] = model.as_dict()
        return result

    def stop(self) -> None:
        self._stop.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        for thread in (self._stdout_thread, self._stderr_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=1)
        if process is not None:
            for stream in (process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
        self._process = None
        self._stdout_thread = None
        self._stderr_thread = None


def _write_row(writer: csv.DictWriter, row: dict[str, Any]) -> None:
    writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def _put_matrix_pose(row: dict[str, Any], prefix: str, matrix: np.ndarray) -> None:
    translation = matrix[:3, 3]
    quaternion = _rotation_to_quaternion_xyzw(matrix[:3, :3])
    row.update(
        {
            f"{prefix}_x_mm": f"{translation[0]:.9g}",
            f"{prefix}_y_mm": f"{translation[1]:.9g}",
            f"{prefix}_z_mm": f"{translation[2]:.9g}",
            f"{prefix}_qx": f"{quaternion[0]:.12g}",
            f"{prefix}_qy": f"{quaternion[1]:.12g}",
            f"{prefix}_qz": f"{quaternion[2]:.12g}",
            f"{prefix}_qw": f"{quaternion[3]:.12g}",
        }
    )


def _put_match(
    row: dict[str, Any],
    match: dict[str, Any],
    offset_ms: float,
    target_unix_ns: int,
) -> None:
    model = match.get("clock_model") or {}
    row.update(
        {
            "clock_model_ready": int(bool(model.get("ready"))),
            "clock_model_mode": model.get("mode", ""),
            "clock_samples": model.get("samples", ""),
            "clock_span_seconds": model.get("span_seconds", ""),
            "clock_rate_ppm": model.get("rate_ppm", ""),
            "clock_residual_p95_ms": model.get("residual_p95_ms", ""),
            "sync_offset_ms": offset_ms,
            "sync_method": match.get("status", ""),
            "sync_error_ms": match.get("sync_error_ms", ""),
            "interpolation_alpha": match.get("alpha", ""),
            "bracket_span_ms": match.get("bracket_span_ms", ""),
            "mocap_frames": ":".join(
                str(frame) for frame in match.get("mocap_frames", ())
            ),
            "mocap_bracket_receive_unix_ns": ":".join(
                str(value) for value in match.get("mocap_receive_unix_ns", ())
            ),
            "mocap_bracket_aligned_unix_ns": ":".join(
                str(value) for value in match.get("mocap_aligned_unix_ns", ())
            ),
        }
    )
    payload = match.get("pose")
    if not payload:
        return
    position = payload["position"]
    quaternion = payload["quaternion"]
    reference_aligned_ns = int(payload["aligned_unix_ns"])
    offset_ns = int(round(offset_ms * 1_000_000))
    # MocapTimeline keeps the nearest frame's metadata in an interpolated pose.
    # The returned position/quaternion, however, represents raw_target_ns.
    # Record both so the CSV never labels a nearest-frame timestamp as the
    # interpolation timestamp.
    aligned_ns = (
        target_unix_ns - offset_ns
        if match.get("status") == "interpolated"
        else reference_aligned_ns
    )
    row.update(
        {
            "mocap_sdk_timestamp_ms": payload["timestamp_ms"],
            "mocap_receive_unix_ns": payload["receive_unix_ns"],
            "mocap_reference_aligned_unix_ns": reference_aligned_ns,
            "mocap_aligned_unix_ns": aligned_ns,
            "mocap_corrected_unix_ns": aligned_ns + offset_ns,
            "mocap_x_mm": position["x"],
            "mocap_y_mm": position["y"],
            "mocap_z_mm": position["z"],
            "mocap_qx": quaternion["qx"],
            "mocap_qy": quaternion["qy"],
            "mocap_qz": quaternion["qz"],
            "mocap_qw": quaternion["qw"],
        }
    )


def _timestamp_console_line(row: dict[str, Any]) -> str:
    capture_ns = row["camera_capture_timestamp_ns"]
    aligned_ns = row.get("mocap_aligned_unix_ns", "-")
    corrected_ns = row.get("mocap_corrected_unix_ns", "-")
    sdk_ms = row.get("mocap_sdk_timestamp_ms", "-")
    error_ms = row.get("sync_error_ms", "-")
    rate_ppm = row.get("clock_rate_ppm", "-")
    return (
        f"frame={row['camera_frame_index']} status={row['status']} "
        f"camera_ns={capture_ns} sdk_ms={sdk_ms} aligned_ns={aligned_ns} "
        f"corrected_ns={corrected_ns} dt_ms={error_ms} rate_ppm={rate_ppm}"
    )


def run(args: argparse.Namespace, calibration: Calibration) -> None:
    api = _load_cvia_mocap_api(args.cvia_root)
    if platform.system() == "Linux" and args.device is None:
        args.camera, device = find_camera_on_usb_port(args.ir_usb_port)
        print(
            f"已按 USB 物理端口 {args.ir_usb_port} 选择相机："
            f"{device}（OpenCV 索引 {args.camera}）"
        )
    else:
        device = args.device or f"/dev/video{args.camera}"

    controls = available_v4l2_controls(device)
    capture: cv2.VideoCapture | None = None
    client: AffineMocapClient | None = None
    csv_stream: TextIO | None = None
    last_frame: np.ndarray | None = None
    statuses: Counter[str] = Counter()
    rows_written = 0
    window_enabled = not args.headless and desktop_display_available()
    try:
        client = AffineMocapClient(
            api=api,
            server=args.mocap_server,
            tracker=args.mocap_tracker,
            bridge_path=args.mocap_bridge,
            start_timeout=args.mocap_start_timeout,
            history_ms=args.history_ms,
            clock_window_seconds=args.clock_window_seconds,
            clock_minimum_span_seconds=args.clock_minimum_span_seconds,
            clock_maximum_rate_ppm=args.clock_maximum_rate_ppm,
            require_tracked=args.require_tracked,
        )
        print(
            f"正在连接 XING {args.mocap_server}，目标 {args.mocap_tracker}；"
            "刚体暂时不可见也不影响 CLOCK 预热……"
        )
        client.start()
        if not client.wait_clock_ready(args.clock_warmup_timeout):
            model = client.clock_model_dict()
            raise RuntimeError(
                "仿射时钟模型预热超时；请确认使用 CLOCK 版桥接且 FrameGroup 持续输出。"
                f" 当前模型：{json.dumps(model, ensure_ascii=False)}"
            )
        print(
            "仿射时钟已就绪："
            + json.dumps(client.clock_model_dict(), ensure_ascii=False)
        )

        capture = open_camera(args)
        configure_camera(capture, args, device, controls)
        print_camera_status(capture)
        print(
            "警告：本工具的相机时间戳是 OpenCV read() 返回时刻，不是驱动曝光时刻；"
            "最终物理相位请用可靠的 offset_ms 标定。",
            file=sys.stderr,
        )

        args.csv.parent.mkdir(parents=True, exist_ok=True)
        csv_stream = args.csv.open("x", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        detector, dictionary = make_aruco_detector()
        print(
            f"开始拼接：Marker ID={calibration.marker_id}，"
            f"offset={args.offset_ms:+.3f} ms，阈值={args.max_error_ms:g} ms，"
            f"结果={args.csv}"
        )
        print("预览窗口按 Q/Esc 退出；无界面模式按 Ctrl+C 退出。")

        frame_index = 0
        failed_reads = 0
        rows_since_flush = 0
        while args.max_frames == 0 or frame_index < args.max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                failed_reads += 1
                if failed_reads >= 30:
                    raise RuntimeError("连续 30 帧读取相机失败")
                continue
            # This is deliberately captured before ArUco/PnP.  It must never be
            # replaced by the later result-publish timestamp for synchronization.
            capture_unix_ns = time.time_ns()
            capture_monotonic_ns = time.monotonic_ns()
            failed_reads = 0
            last_frame = frame
            measurement, corners, ids = detect_marker_pose(
                frame, calibration, detector, dictionary
            )
            match = client.match(
                capture_unix_ns,
                offset_ms=args.offset_ms,
                max_error_ms=args.max_error_ms,
                interpolate=not args.nearest_only,
                wait_ms=args.match_wait_ms,
            )
            row: dict[str, Any] = {
                "camera_frame_index": frame_index,
                "camera_time_iso8601": _iso8601_from_ns(capture_unix_ns),
                "camera_capture_timestamp_ns": capture_unix_ns,
                "camera_monotonic_ns": capture_monotonic_ns,
                "camera_timestamp_source": CAMERA_TIMESTAMP_SOURCE,
                "marker_detected": int(measurement is not None),
                "marker_id": calibration.marker_id,
            }
            _put_match(row, match, args.offset_ms, capture_unix_ns)
            sync_ok = match.get("status") in {"matched", "interpolated"} and match.get("pose")
            predicted: np.ndarray | None = None
            error = None
            if measurement is None:
                status = "marker_not_detected"
            else:
                row["reprojection_error_px"] = f"{measurement.reprojection_error_px:.9g}"
                _put_matrix_pose(row, "aruco", measurement.T_C_A)
                predicted = calibration.T_M_C @ measurement.T_C_A @ calibration.T_A_B
                pred_translation = predicted[:3, 3]
                row.update(
                    {
                        "pred_x_mm": f"{pred_translation[0]:.9g}",
                        "pred_y_mm": f"{pred_translation[1]:.9g}",
                        "pred_z_mm": f"{pred_translation[2]:.9g}",
                    }
                )
                if not sync_ok:
                    status = str(match.get("status") or "mocap_unmatched")
                else:
                    actual = _payload_pose_matrix(match["pose"])
                    error = pose_error(predicted, actual)
                    row["translation_error_mm"] = f"{error.translation_mm:.9g}"
                    row["rotation_error_deg"] = f"{error.rotation_deg:.9g}"
                    status = "ok"

            publish_ns = time.time_ns()
            row["camera_publish_timestamp_ns"] = publish_ns
            row["pipeline_latency_ms"] = f"{(publish_ns - capture_unix_ns) / 1_000_000:.6f}"
            row["status"] = status
            statuses[status] += 1
            _write_row(writer, row)
            rows_written += 1
            rows_since_flush += 1
            if rows_since_flush >= 30:
                csv_stream.flush()
                rows_since_flush = 0

            if frame_index % args.print_every == 0:
                print(_timestamp_console_line(row))

            if window_enabled:
                preview = frame.copy()
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(preview, corners, ids)
                if measurement is not None:
                    cv2.drawFrameAxes(
                        preview,
                        calibration.camera_matrix,
                        calibration.dist_coeffs,
                        measurement.rvec,
                        measurement.tvec,
                        calibration.marker_length_mm * 0.4,
                    )
                lines = [
                    f"status: {status}",
                    f"camera: {capture_unix_ns}",
                    f"mocap: {row.get('mocap_corrected_unix_ns', '-')}",
                    f"dt: {row.get('sync_error_ms', '-')} ms  rate: {row.get('clock_rate_ppm', '-')} ppm",
                ]
                if error is not None:
                    lines.append(
                        f"pose error: {error.translation_mm:.2f} mm / {error.rotation_deg:.2f} deg"
                    )
                draw_text_lines(preview, lines)
                try:
                    cv2.imshow("ArUco / affine mocap timestamp sync", preview)
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error as exc:
                    if "GTK" not in str(exc) and "window" not in str(exc).lower():
                        raise
                    window_enabled = False
                    print("警告：预览窗口不可用，继续无界面记录。", file=sys.stderr)
                    key = -1
                if key in (ord("q"), ord("Q"), 27):
                    break
            frame_index += 1

        csv_stream.flush()
        print("拼接结束：" + ", ".join(f"{key}={value}" for key, value in statuses.items()))
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，停止记录。")
        if statuses:
            print("状态统计：" + ", ".join(f"{key}={value}" for key, value in statuses.items()))
    finally:
        if args.debug_image is not None and last_frame is not None:
            args.debug_image.parent.mkdir(parents=True, exist_ok=True)
            if cv2.imwrite(str(args.debug_image), last_frame):
                print(f"最后一帧已保存：{args.debug_image}")
        if csv_stream is not None:
            csv_stream.close()
        if capture is not None:
            capture.release()
        if client is not None:
            client.stop()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    if rows_written > 0:
        try:
            report = generate_curve_report(
                args.csv,
                args.calibration,
                args.curve_svg,
                trajectory_csv=args.trajectory_csv,
                fusion_mode=args.fusion_mode,
            )
        except (OSError, ValueError) as exc:
            print(f"警告：轨迹曲线生成失败：{exc}", file=sys.stderr)
        else:
            print(f"板轨迹CSV已保存：{report['trajectory_csv']}")
            print(f"动捕六轴曲线已保存：{report['mocap_curve_svg']}")
            print(f"相机坐标系六轴曲线已保存：{report['camera_curve_svg']}")
            print(f"拼接六轴曲线已保存：{report['fused_curve_svg']}")


def build_parser(config: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    config = config or {}
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="查看时间戳，并按仿射校正时间拼接 ArUco 与 NOKOV 刚体位姿"
    )
    parser.add_argument(
        "--cvia-root",
        type=Path,
        default=Path(os.environ.get("CVIA_TRT_ROOT", str(DEFAULT_CVIA_ROOT))),
        help="包含 web_monitor/mocap_receiver.py 的 CVIA_trt 根目录",
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
        "--mocap-server",
        default=str(_nested(config, "mocap", "server", "10.1.1.198")),
    )
    parser.add_argument(
        "--mocap-tracker",
        default=str(_nested(config, "mocap", "tracker", "name:Tracker0")),
    )
    parser.add_argument("--mocap-bridge", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--mocap-start-timeout", type=float, default=10.0)
    parser.add_argument(
        "--clock-warmup-timeout",
        type=float,
        default=10.0,
        help="等待 CLOCK 仿射模型 ready 的最长秒数",
    )
    parser.add_argument(
        "--clock-window-seconds",
        type=float,
        default=float(_nested(config, "mocap", "clock_fit_window_seconds", 60.0)),
    )
    parser.add_argument(
        "--clock-minimum-span-seconds",
        type=float,
        default=float(_nested(config, "mocap", "clock_fit_min_seconds", 3.0)),
    )
    parser.add_argument(
        "--clock-maximum-rate-ppm",
        type=float,
        default=float(_nested(config, "mocap", "clock_max_rate_ppm", 5000.0)),
    )
    parser.add_argument(
        "--offset-ms",
        type=float,
        default=float(_nested(config, "sync", "offset_ms", 0.0)),
        help="加到仿射校正动捕时间轴的固定物理延迟补偿",
    )
    parser.add_argument(
        "--max-error-ms",
        type=float,
        default=float(_nested(config, "sync", "max_error_ms", 10.0)),
    )
    parser.add_argument(
        "--history-ms",
        type=float,
        default=float(_nested(config, "sync", "history_ms", 5000.0)),
    )
    parser.add_argument(
        "--match-wait-ms",
        type=float,
        default=60.0,
        help="PnP 后最多等待多久以获得采集时刻之后的动捕包围帧",
    )
    parser.add_argument("--nearest-only", action="store_true", help="禁用双侧插值")
    parser.add_argument(
        "--require-tracked",
        action="store_true",
        help="只使用 tracking_params bit0 有效的动捕帧",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("sync_results") / f"aruco_mocap_sync_{timestamp}.csv",
    )
    parser.add_argument(
        "--curve-svg",
        type=Path,
        default=None,
        help="三个独立六轴SVG的基础路径；自动增加 mocap/camera/fused 后缀",
    )
    parser.add_argument(
        "--trajectory-csv",
        type=Path,
        default=None,
        help="统一到动捕世界坐标系的板轨迹CSV；默认增加 _trajectory.csv",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=("prefer_mocap", "prefer_camera", "blend"),
        default="prefer_camera",
        help="两路同时有效时的拼接策略；默认优先相机，未检出时回退动捕",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 表示持续运行")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--debug-image", type=Path, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--ir-usb-port",
        default="USB0",
        help="红外相机机箱端口或 sysfs 路径，默认 USB0（1-4.3.2）",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--fourcc", default="MJPG")
    exposure = parser.add_mutually_exclusive_group()
    exposure.add_argument("--auto-exposure", dest="auto_exposure", action="store_true")
    exposure.add_argument("--manual-exposure", dest="auto_exposure", action="store_false")
    exposure.set_defaults(auto_exposure=True)
    for name in SETTINGS:
        default = None if name == "exposure" else DEFAULT_VALUES.get(name)
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive = (
        ("--mocap-start-timeout", args.mocap_start_timeout),
        ("--clock-warmup-timeout", args.clock_warmup_timeout),
        ("--clock-window-seconds", args.clock_window_seconds),
        ("--clock-minimum-span-seconds", args.clock_minimum_span_seconds),
        ("--clock-maximum-rate-ppm", args.clock_maximum_rate_ppm),
        ("--max-error-ms", args.max_error_ms),
        ("--history-ms", args.history_ms),
        ("--width", args.width),
        ("--height", args.height),
        ("--fps", args.fps),
        ("--print-every", args.print_every),
    )
    for name, value in positive:
        if value <= 0:
            parser.error(f"{name} 必须大于 0")
    if args.match_wait_ms < 0:
        parser.error("--match-wait-ms 不能小于 0")
    if args.max_frames < 0:
        parser.error("--max-frames 不能小于 0")
    if len(args.fourcc) != 4:
        parser.error("--fourcc 必须是 4 个字符")
    if args.auto_exposure and args.exposure is not None:
        parser.error("--auto-exposure 不能和 --exposure 同时使用")
    if not args.auto_exposure and args.exposure is None:
        args.exposure = DEFAULT_VALUES["exposure"]


def main() -> int:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--cvia-config", type=Path, default=DEFAULT_CVIA_CONFIG)
    known, _unknown = bootstrap.parse_known_args()
    try:
        cvia_config = _read_json_object(known.cvia_config.expanduser())
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    parser = build_parser(cvia_config)
    parser.add_argument(
        "--cvia-config",
        type=Path,
        default=known.cvia_config,
        help="读取 mocap/sync 默认值的 web_monitor/config.json",
    )
    args = parser.parse_args()
    _validate_args(parser, args)
    for attribute in ("cvia_root", "calibration", "mocap_bridge", "csv"):
        value = getattr(args, attribute)
        setattr(args, attribute, value.expanduser().resolve())
    if args.debug_image is not None:
        args.debug_image = args.debug_image.expanduser().resolve()
    if args.curve_svg is None:
        args.curve_svg = args.csv.with_name(args.csv.stem + "_curves.svg")
    else:
        args.curve_svg = args.curve_svg.expanduser().resolve()
    if args.trajectory_csv is None:
        args.trajectory_csv = args.csv.with_name(args.csv.stem + "_trajectory.csv")
    else:
        args.trajectory_csv = args.trajectory_csv.expanduser().resolve()
    try:
        calibration = load_calibration(args.calibration)
        run(args, calibration)
        return 0
    except (ValueError, RuntimeError, OSError, cv2.error) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
