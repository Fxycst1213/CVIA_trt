"""Supervise the NOKOV SDK bridge and expose its latest rigid-body pose."""

from __future__ import annotations

import csv
import math
import os
import subprocess
import threading
import time
import urllib.parse
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from typing import Iterable, TextIO


SENTINEL_LIMIT = 9_000_000.0
MOCAP_SESSION_COLUMNS = (
    "receive_unix_ns",
    "receive_monotonic_ns",
    "mocap_timestamp_ms",
    "aligned_unix_ns",
    "clock_rate_ppm",
    "clock_model_ready",
    "frame",
    "tracker_id",
    "tracker_name",
    "valid",
    "x",
    "y",
    "z",
    "qx",
    "qy",
    "qz",
    "qw",
    "mean_error",
    "tracking_params",
)


@dataclass(frozen=True)
class MocapPose:
    selector: str
    tracker_id: int
    tracker_name: str
    mocap_frame: int
    mocap_timestamp_ms: int
    receive_unix_ns: int
    receive_monotonic_ns: int
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    mean_error: float
    tracking_params: int
    aligned_unix_ns: int = 0

    @property
    def timeline_unix_ns(self) -> int:
        """Timestamp used for fusion, falling back for old CSV/test data."""
        return self.aligned_unix_ns or self.receive_unix_ns


@dataclass(frozen=True)
class MocapClockModel:
    """Affine mapping from the independent SDK clock to the Orin clock."""

    ready: bool = False
    anchor_sdk_ms: int = 0
    anchor_monotonic_ns: int = 0
    realtime_minus_monotonic_ns: int = 0
    ns_per_sdk_ms: float = 1_000_000.0
    rate_ppm: float = 0.0
    sample_count: int = 0
    span_seconds: float = 0.0
    residual_p95_ms: float = 0.0

    def map_unix_ns(self, sdk_timestamp_ms: int, fallback_unix_ns: int) -> int:
        if not self.ready:
            return fallback_unix_ns
        delta_ms = sdk_timestamp_ms - self.anchor_sdk_ms
        mapped_monotonic_ns = self.anchor_monotonic_ns + int(
            round(delta_ms * self.ns_per_sdk_ms)
        )
        return mapped_monotonic_ns + self.realtime_minus_monotonic_ns

    def as_dict(self) -> dict:
        return {
            "mode": "sdk_affine" if self.ready else "receive_fallback",
            "ready": self.ready,
            "rate_ppm": round(self.rate_ppm, 3),
            "ns_per_sdk_ms": round(self.ns_per_sdk_ms, 3),
            "samples": self.sample_count,
            "span_seconds": round(self.span_seconds, 3),
            "residual_p95_ms": round(self.residual_p95_ms, 3),
        }


class MocapClockMapper:
    """Estimate SDK-clock rate without putting work on the inference thread.

    The SDK clock has an independent epoch and rate.  Receive MONOTONIC time is
    used to estimate its rate; a low residual quantile rejects callback queueing.
    The remaining fixed capture/solve/transport delay is intentionally left to
    the existing motion cross-correlation ``offset_ms`` calibration.
    """

    def __init__(
        self,
        *,
        window_seconds: float = 60.0,
        minimum_span_seconds: float = 3.0,
        update_every: int = 180,
        maximum_rate_ppm: float = 5000.0,
    ) -> None:
        self.window_ns = int(max(5.0, window_seconds) * 1_000_000_000)
        self.minimum_span_ns = int(max(1.0, minimum_span_seconds) * 1_000_000_000)
        self.update_every = max(1, int(update_every))
        self.maximum_rate_ppm = max(100.0, float(maximum_rate_ppm))
        self._samples: deque[tuple[int, int, int]] = deque()
        self._observations_since_fit = 0
        self._last_sdk_ms: int | None = None
        self._model = MocapClockModel()

    @property
    def model(self) -> MocapClockModel:
        return self._model

    def reset(self) -> None:
        self._samples.clear()
        self._observations_since_fit = 0
        self._last_sdk_ms = None
        self._model = MocapClockModel()

    def observe(self, pose: MocapPose) -> MocapClockModel:
        return self.observe_timestamp(
            pose.mocap_timestamp_ms,
            pose.receive_monotonic_ns,
            pose.receive_unix_ns,
        )

    def observe_timestamp(
        self,
        sdk_ms: int,
        receive_monotonic_ns: int,
        receive_unix_ns: int,
    ) -> MocapClockModel:
        """Add one FrameGroup timestamp, independent of rigid-body visibility."""
        if self._last_sdk_ms == sdk_ms:
            return self._model
        if self._last_sdk_ms is not None and sdk_ms <= self._last_sdk_ms:
            self.reset()
        self._last_sdk_ms = sdk_ms
        self._samples.append((
            sdk_ms,
            receive_monotonic_ns,
            receive_unix_ns - receive_monotonic_ns,
        ))
        cutoff_ns = receive_monotonic_ns - self.window_ns
        while self._samples and self._samples[0][1] < cutoff_ns:
            self._samples.popleft()
        self._observations_since_fit += 1
        if self._observations_since_fit >= self.update_every:
            self._fit()
            self._observations_since_fit = 0
        return self._model

    def map_pose(
        self,
        pose: MocapPose,
        model: MocapClockModel | None = None,
    ) -> MocapPose:
        selected = model or self._model
        aligned_ns = selected.map_unix_ns(
            pose.mocap_timestamp_ms,
            pose.receive_unix_ns,
        )
        return replace(pose, aligned_unix_ns=aligned_ns)

    def _fit(self) -> None:
        samples = list(self._samples)
        if len(samples) < 60:
            return
        first_sdk_ms, first_monotonic_ns, _ = samples[0]
        span_ns = samples[-1][1] - first_monotonic_ns
        if span_ns < self.minimum_span_ns:
            return

        xs = [float(sdk_ms - first_sdk_ms) for sdk_ms, _, _ in samples]
        ys = [float(monotonic_ns - first_monotonic_ns) for _, monotonic_ns, _ in samples]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        variance_x = sum((value - mean_x) ** 2 for value in xs)
        if variance_x <= 0:
            return
        ns_per_sdk_ms = sum(
            (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)
        ) / variance_x
        rate_ppm = (ns_per_sdk_ms / 1_000_000.0 - 1.0) * 1_000_000.0
        if not math.isfinite(rate_ppm) or abs(rate_ppm) > self.maximum_rate_ppm:
            return

        # The 10th percentile follows the least-queued callbacks while remaining
        # robust to a single early arrival.  Absolute transport latency is later
        # calibrated by motion cross-correlation.
        residuals = sorted(y - x * ns_per_sdk_ms for x, y in zip(xs, ys))
        baseline_index = int(round((len(residuals) - 1) * 0.10))
        anchor_monotonic_ns = first_monotonic_ns + int(round(residuals[baseline_index]))
        absolute_residuals = sorted(
            abs(value - residuals[baseline_index]) for value in residuals
        )
        p95_index = int(round((len(absolute_residuals) - 1) * 0.95))
        realtime_minus_monotonic_ns = int(median(
            sample[2] for sample in samples[-min(120, len(samples)):]
        ))
        self._model = MocapClockModel(
            ready=True,
            anchor_sdk_ms=first_sdk_ms,
            anchor_monotonic_ns=anchor_monotonic_ns,
            realtime_minus_monotonic_ns=realtime_minus_monotonic_ns,
            ns_per_sdk_ms=ns_per_sdk_ms,
            rate_ppm=rate_ppm,
            sample_count=len(samples),
            span_seconds=span_ns / 1_000_000_000,
            residual_p95_ms=absolute_residuals[p95_index] / 1_000_000,
        )


def align_mocap_poses(poses: Iterable[MocapPose]) -> tuple[list[MocapPose], MocapClockModel]:
    """Fit one independent-clock model and remap a recorded mocap session."""
    ordered = sorted(poses, key=lambda pose: pose.receive_monotonic_ns)
    if not ordered:
        return [], MocapClockModel()
    span_seconds = max(
        3.0,
        (ordered[-1].receive_monotonic_ns - ordered[0].receive_monotonic_ns)
        / 1_000_000_000,
    )
    mapper = MocapClockMapper(
        window_seconds=span_seconds + 1.0,
        minimum_span_seconds=min(3.0, max(1.0, span_seconds / 2)),
        update_every=max(1, len(ordered)),
    )
    for pose in ordered:
        mapper.observe(pose)
    # ``observe`` fits on the final sample because update_every == sample count.
    model = mapper.model
    if not model.ready:
        return ordered, model
    return [mapper.map_pose(pose, model) for pose in ordered], model


def parse_pose_line(line: str) -> MocapPose:
    """Parse the 17-column, tab-separated line emitted by MocapBridge."""
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) != 17 or fields[0] != "POSE":
        raise ValueError(f"期望 17 列 POSE 数据，实际 {len(fields)} 列")
    return MocapPose(
        selector=urllib.parse.unquote(fields[1]),
        tracker_id=int(fields[2]),
        tracker_name=urllib.parse.unquote(fields[3]),
        mocap_frame=int(fields[4]),
        mocap_timestamp_ms=int(fields[5]),
        receive_unix_ns=int(fields[6]),
        receive_monotonic_ns=int(fields[7]),
        x=float(fields[8]),
        y=float(fields[9]),
        z=float(fields[10]),
        qx=float(fields[11]),
        qy=float(fields[12]),
        qz=float(fields[13]),
        qw=float(fields[14]),
        mean_error=float(fields[15]),
        tracking_params=int(fields[16]),
    )


def pose_validity(pose: MocapPose) -> tuple[bool, str]:
    values = (pose.x, pose.y, pose.z, pose.qx, pose.qy, pose.qz, pose.qw)
    if not all(math.isfinite(value) for value in values):
        return False, "动捕帧包含非有限数值"
    if any(abs(value) >= SENTINEL_LIMIT for value in values):
        return False, "刚体在当前帧未被解算（9999999 哨兵值）"
    norm_sq = pose.qx**2 + pose.qy**2 + pose.qz**2 + pose.qw**2
    if norm_sq < 1e-12:
        return False, "动捕四元数长度为零"
    return True, ""


def quaternion_to_euler_xyz_degrees(
    qx: float, qy: float, qz: float, qw: float
) -> tuple[float, float, float]:
    """Return roll-X, pitch-Y and yaw-Z in degrees from a quaternion."""
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        raise ValueError("四元数长度为零")
    x, y, z, w = qx / norm, qy / norm, qz / norm, qw / norm

    roll_x = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch_term = 2 * (w * y - z * x)
    pitch_y = math.copysign(math.pi / 2, pitch_term) if abs(pitch_term) >= 1 else math.asin(pitch_term)
    yaw_z = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return tuple(math.degrees(value) for value in (roll_x, pitch_y, yaw_z))


def _normalized_quaternion(pose: MocapPose) -> tuple[float, float, float, float]:
    norm = math.sqrt(pose.qx**2 + pose.qy**2 + pose.qz**2 + pose.qw**2)
    if norm < 1e-12:
        raise ValueError("四元数长度为零")
    return pose.qx / norm, pose.qy / norm, pose.qz / norm, pose.qw / norm


def _slerp(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    """Interpolate two normalized quaternions along the shortest arc."""
    dot = sum(a * b for a, b in zip(first, second))
    if dot < 0:
        second = tuple(-value for value in second)
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        result = tuple(a + alpha * (b - a) for a, b in zip(first, second))
        norm = math.sqrt(sum(value * value for value in result))
        return tuple(value / norm for value in result)
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    scale_first = math.sin(theta_0 - theta) / sin_theta_0
    scale_second = math.sin(theta) / sin_theta_0
    return tuple(
        scale_first * a + scale_second * b
        for a, b in zip(first, second)
    )


def _pose_payload(
    pose: MocapPose,
    quaternion: tuple[float, float, float, float] | None = None,
    position: tuple[float, float, float] | None = None,
) -> dict:
    qx, qy, qz, qw = quaternion or _normalized_quaternion(pose)
    x, y, z = position or (pose.x, pose.y, pose.z)
    euler = quaternion_to_euler_xyz_degrees(qx, qy, qz, qw)
    return {
        "tracker_id": pose.tracker_id,
        "tracker_name": pose.tracker_name,
        "frame": pose.mocap_frame,
        "timestamp_ms": pose.mocap_timestamp_ms,
        "receive_unix_ns": pose.receive_unix_ns,
        "aligned_unix_ns": pose.timeline_unix_ns,
        "position": {"x": x, "y": y, "z": z},
        "euler_deg": {"rx": euler[0], "ry": euler[1], "rz": euler[2]},
        "quaternion": {"qx": qx, "qy": qy, "qz": qz, "qw": qw},
        "mean_error": pose.mean_error,
        "tracking_params": pose.tracking_params,
    }


class MocapTimeline:
    """Immutable, searchable view over valid mocap poses."""

    def __init__(self, poses: Iterable[MocapPose]) -> None:
        self.poses = sorted(poses, key=lambda pose: pose.timeline_unix_ns)
        self.times = [pose.timeline_unix_ns for pose in self.poses]

    def match(
        self,
        target_unix_ns: int,
        *,
        offset_ms: float = 0.0,
        max_error_ms: float = 12.0,
        interpolate: bool = True,
    ) -> dict:
        """Match one visual acquisition time against the buffered mocap time line.

        offset_ms is added to the rate-corrected mocap timeline before
        comparison.  Matching is performed in that raw timeline by subtracting
        the offset from the visual target.
        """
        if not self.poses:
            return {"status": "waiting", "message": "动捕历史缓存为空", "pose": None}
        offset_ns = int(round(offset_ms * 1_000_000))
        raw_target_ns = target_unix_ns - offset_ns
        maximum_ns = int(round(max_error_ms * 1_000_000))
        index = bisect_left(self.times, raw_target_ns)
        candidates = []
        if index > 0:
            candidates.append(self.poses[index - 1])
        if index < len(self.poses):
            candidates.append(self.poses[index])
        nearest = min(
            candidates,
            key=lambda pose: abs(pose.timeline_unix_ns - raw_target_ns),
        )
        signed_delta_ns = nearest.timeline_unix_ns + offset_ns - target_unix_ns
        if abs(signed_delta_ns) > maximum_ns:
            return {
                "status": "unmatched",
                "message": f"最近动捕帧相差 {abs(signed_delta_ns) / 1_000_000:.3f} ms，超过阈值",
                "pose": None,
                "sync_error_ms": round(signed_delta_ns / 1_000_000, 6),
                "max_error_ms": max_error_ms,
            }

        before = self.poses[index - 1] if index > 0 else None
        after = self.poses[index] if index < len(self.poses) else None
        if interpolate and before is not None and after is not None:
            before_distance = raw_target_ns - before.timeline_unix_ns
            after_distance = after.timeline_unix_ns - raw_target_ns
            span = after.timeline_unix_ns - before.timeline_unix_ns
            if (
                span > 0
                and before_distance >= 0
                and after_distance >= 0
                and max(before_distance, after_distance) <= maximum_ns * 2
            ):
                alpha = before_distance / span
                position = tuple(
                    first + alpha * (second - first)
                    for first, second in zip(
                        (before.x, before.y, before.z),
                        (after.x, after.y, after.z),
                    )
                )
                quaternion = _slerp(
                    _normalized_quaternion(before),
                    _normalized_quaternion(after),
                    alpha,
                )
                payload = _pose_payload(nearest, quaternion, position)
                payload["frame"] = nearest.mocap_frame
                return {
                    "status": "interpolated",
                    "message": "已用视觉采集时刻两侧的动捕帧插值",
                    "pose": payload,
                    "sync_error_ms": round(signed_delta_ns / 1_000_000, 6),
                    "max_error_ms": max_error_ms,
                    "offset_ms": offset_ms,
                    "alpha": round(alpha, 9),
                    "bracket_span_ms": round(span / 1_000_000, 6),
                    "mocap_frames": [before.mocap_frame, after.mocap_frame],
                    "mocap_receive_unix_ns": [before.receive_unix_ns, after.receive_unix_ns],
                    "mocap_aligned_unix_ns": [before.timeline_unix_ns, after.timeline_unix_ns],
                }

        return {
            "status": "matched",
            "message": "已匹配最近动捕帧",
            "pose": _pose_payload(nearest),
            "sync_error_ms": round(signed_delta_ns / 1_000_000, 6),
            "max_error_ms": max_error_ms,
            "offset_ms": offset_ms,
            "alpha": None,
            "bracket_span_ms": None,
            "mocap_frames": [nearest.mocap_frame],
            "mocap_receive_unix_ns": [nearest.receive_unix_ns],
            "mocap_aligned_unix_ns": [nearest.timeline_unix_ns],
        }


class MocapReceiver:
    """Keep the SDK subprocess alive and retain the newest selected pose."""

    def __init__(
        self,
        config: dict,
        project_root: Path,
        *,
        history_ms: float = 5000,
        session_csv_path: Path | None = None,
    ) -> None:
        self.enabled = bool(config.get("enabled", False))
        self.server = str(config.get("server", "")).strip()
        self.tracker = str(config.get("tracker", "")).strip()
        self.stale_ms = float(config.get("stale_ms", 500))
        self.retry_seconds = float(config.get("retry_seconds", 5))
        self.clock_mode = str(config.get("clock_mode", "sdk_affine")).strip()
        self.history_ms = max(100.0, float(history_ms))
        self.session_csv_path = session_csv_path
        workspace_bridge = (
            project_root / "web_monitor" / "mocap" / "bin" / "MocapBridge"
        )
        environment_bridge = os.environ.get("CVIA_MOCAP_BRIDGE", "").strip()
        if environment_bridge:
            self.bridge_path = Path(environment_bridge).expanduser()
        elif workspace_bridge.is_file():
            self.bridge_path = workspace_bridge
        else:
            self.bridge_path = workspace_bridge

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None
        self._connection = "disabled" if not self.enabled else "starting"
        self._sdk_version: str | None = None
        self._descriptions: dict[int, str] = {}
        self._latest_pose: MocapPose | None = None
        self._latest_valid = False
        self._latest_issue = ""
        self._last_error = ""
        self._pose_history: deque[tuple[MocapPose, bool]] = deque()
        self._clock_mapper = MocapClockMapper(
            window_seconds=float(config.get("clock_fit_window_seconds", 60.0)),
            minimum_span_seconds=float(config.get("clock_fit_min_seconds", 3.0)),
            maximum_rate_ppm=float(config.get("clock_max_rate_ppm", 5000.0)),
        )
        self._record_lock = threading.Lock()
        self._record_stream: TextIO | None = None
        self._record_writer = None
        self._record_rows_since_flush = 0
        self._recording_finished = False

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._supervise, daemon=True, name="nokov-supervisor"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3)
        process = self._process
        if process is not None and process.poll() is None:
            process.kill()
        self._thread = None
        self._process = None
        self.close_recording()

    def _record_pose(self, pose: MocapPose, valid: bool) -> None:
        if self.session_csv_path is None:
            return
        with self._record_lock:
            if self._recording_finished:
                return
            if self._record_stream is None:
                self.session_csv_path.parent.mkdir(parents=True, exist_ok=True)
                needs_header = not self.session_csv_path.exists() or self.session_csv_path.stat().st_size == 0
                self._record_stream = self.session_csv_path.open(
                    "a", encoding="utf-8", newline=""
                )
                self._record_writer = csv.writer(self._record_stream)
                if needs_header:
                    self._record_writer.writerow(MOCAP_SESSION_COLUMNS)
            self._record_writer.writerow((
                pose.receive_unix_ns,
                pose.receive_monotonic_ns,
                pose.mocap_timestamp_ms,
                pose.timeline_unix_ns,
                round(self._clock_mapper.model.rate_ppm, 6),
                int(self._clock_mapper.model.ready),
                pose.mocap_frame,
                pose.tracker_id,
                pose.tracker_name,
                int(valid),
                pose.x,
                pose.y,
                pose.z,
                pose.qx,
                pose.qy,
                pose.qz,
                pose.qw,
                pose.mean_error,
                pose.tracking_params,
            ))
            self._record_rows_since_flush += 1
            if self._record_rows_since_flush >= 30:
                self._record_stream.flush()
                self._record_rows_since_flush = 0

    def flush_recording(self) -> None:
        with self._record_lock:
            if self._record_stream is not None:
                self._record_stream.flush()
                self._record_rows_since_flush = 0

    def close_recording(self) -> None:
        with self._record_lock:
            if self._record_stream is not None:
                self._record_stream.flush()
                self._record_stream.close()
                self._record_stream = None
                self._record_writer = None
                self._record_rows_since_flush = 0

    def finish_recording(self) -> None:
        """Freeze the current session so offline analysis reads a stable CSV."""
        with self._record_lock:
            self._recording_finished = True
            if self._record_stream is not None:
                self._record_stream.flush()
                self._record_stream.close()
                self._record_stream = None
                self._record_writer = None
                self._record_rows_since_flush = 0

    def _set_connection(self, state: str, error: str | None = None) -> None:
        with self._lock:
            self._connection = state
            if error is not None:
                self._last_error = error[-500:]

    def _supervise(self) -> None:
        while not self._stop_event.is_set():
            if not self.bridge_path.is_file():
                self._set_connection(
                    "missing",
                    f"未找到动捕桥接程序：{self.bridge_path}",
                )
                if self._stop_event.wait(self.retry_seconds):
                    break
                continue

            command = [
                str(self.bridge_path),
                "--server",
                self.server,
                "--tracker",
                self.tracker,
            ]
            self._set_connection("connecting", "")
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
            except OSError as exc:
                self._set_connection("error", f"无法启动动捕桥接程序：{exc}")
                if self._stop_event.wait(self.retry_seconds):
                    break
                continue

            self._process = process
            assert process.stdout is not None
            assert process.stderr is not None
            stderr_thread = threading.Thread(
                target=self._read_stderr,
                args=(process.stderr,),
                daemon=True,
                name="nokov-stderr",
            )
            stderr_thread.start()
            try:
                self._read_stdout(process.stdout)
            finally:
                return_code = process.wait()
                self._process = None
                if not self._stop_event.is_set():
                    self._set_connection(
                        "disconnected",
                        self._last_error
                        or f"动捕桥接程序已退出（退出码 {return_code}）",
                    )
            if self._stop_event.wait(self.retry_seconds):
                break

    def _read_stdout(self, stream: TextIO) -> None:
        for raw_line in stream:
            if self._stop_event.is_set():
                break
            line = raw_line.rstrip("\r\n")
            fields = line.split("\t")
            if not fields:
                continue
            if fields[0] == "DESC" and len(fields) >= 3:
                try:
                    tracker_id = int(fields[1])
                except ValueError:
                    continue
                with self._lock:
                    self._descriptions[tracker_id] = urllib.parse.unquote(fields[2])
            elif fields[0] == "READY" and len(fields) >= 2:
                with self._lock:
                    self._clock_mapper.reset()
                    self._pose_history.clear()
                    self._sdk_version = fields[1]
                    self._connection = "ready"
                    self._last_error = ""
            elif fields[0] == "CLOCK":
                if len(fields) != 5:
                    self._set_connection(
                        "ready",
                        f"期望 5 列 CLOCK 数据，实际 {len(fields)} 列",
                    )
                    continue
                try:
                    self._clock_mapper.observe_timestamp(
                        int(fields[2]),
                        int(fields[4]),
                        int(fields[3]),
                    )
                except (ValueError, OverflowError) as exc:
                    self._set_connection("ready", f"无法解析 CLOCK 数据：{exc}")
            elif fields[0] == "POSE":
                try:
                    pose = parse_pose_line(line)
                    valid, issue = pose_validity(pose)
                except (ValueError, OverflowError) as exc:
                    self._set_connection("ready", f"无法解析动捕数据：{exc}")
                    continue
                if self.clock_mode == "sdk_affine":
                    model = self._clock_mapper.observe(pose)
                    pose = self._clock_mapper.map_pose(pose, model)
                else:
                    pose = replace(pose, aligned_unix_ns=pose.receive_unix_ns)
                self._record_pose(pose, valid)
                with self._lock:
                    self._latest_pose = pose
                    self._latest_valid = valid
                    self._latest_issue = issue
                    self._pose_history.append((pose, valid))
                    cutoff_ns = pose.receive_monotonic_ns - int(self.history_ms * 1_000_000)
                    while (
                        self._pose_history
                        and self._pose_history[0][0].receive_monotonic_ns < cutoff_ns
                    ):
                        self._pose_history.popleft()

    def match_at_unix_ns(
        self,
        target_unix_ns: int,
        *,
        offset_ms: float = 0.0,
        max_error_ms: float = 12.0,
        interpolate: bool = True,
    ) -> dict:
        with self._lock:
            poses = [pose for pose, valid in self._pose_history if valid]
            model = self._clock_mapper.model
        if self.clock_mode == "sdk_affine" and model.ready:
            poses = [self._clock_mapper.map_pose(pose, model) for pose in poses]
        result = MocapTimeline(poses).match(
            target_unix_ns,
            offset_ms=offset_ms,
            max_error_ms=max_error_ms,
            interpolate=interpolate,
        )
        result["target_unix_ns"] = target_unix_ns
        result["history_samples"] = len(poses)
        result["clock_model"] = model.as_dict()
        return result

    def _read_stderr(self, stream: TextIO) -> None:
        for raw_line in stream:
            line = raw_line.rstrip("\r\n")
            if line:
                with self._lock:
                    self._last_error = line[-500:]

    def snapshot(self) -> dict:
        # Explicit initialization also keeps ahead-of-time native compilers from
        # conservatively treating this local as possibly uninitialized.
        history_samples = 0
        with self._lock:
            connection = self._connection
            sdk_version = self._sdk_version
            descriptions = dict(self._descriptions)
            pose = self._latest_pose
            valid = self._latest_valid
            issue = self._latest_issue
            last_error = self._last_error
            history_samples = len(self._pose_history)
            clock_model = self._clock_mapper.model

        result = {
            "enabled": self.enabled,
            "connection": connection,
            "server": self.server,
            "selector": self.tracker,
            "bridge_path": str(self.bridge_path),
            "sdk_version": sdk_version,
            "descriptions": [
                {"id": key, "name": value}
                for key, value in sorted(descriptions.items())
            ],
            "status": connection,
            "message": last_error,
            "pose": None,
            "clock_model": clock_model.as_dict(),
        }
        if not self.enabled:
            result["message"] = "动捕接收已在配置中关闭"
            return result
        if pose is None:
            if connection == "ready":
                result["status"] = "waiting"
                discovered = "、".join(
                    f"{tracker_id}:{name}"
                    for tracker_id, name in sorted(descriptions.items())
                )
                discovery = f"；已发现 {discovered}" if discovered else ""
                result["message"] = (
                    f"SDK 已连接{discovery}；等待刚体 {self.tracker} 的数据"
                )
            return result

        age_ms = max(0.0, (time.time_ns() - pose.receive_unix_ns) / 1_000_000)
        if not valid:
            result["status"] = "invalid"
            result["message"] = issue
            result["age_ms"] = round(age_ms, 3)
            return result

        if age_ms > self.stale_ms:
            status = "stale"
            message = f"最新动捕数据已超过 {self.stale_ms:g} ms 未更新"
        else:
            status = "live"
            message = "动捕数据实时接收中"
        result.update({
            "status": status,
            "message": message,
            "age_ms": round(age_ms, 3),
            "pose": _pose_payload(pose),
            "history_samples": history_samples,
        })
        return result
