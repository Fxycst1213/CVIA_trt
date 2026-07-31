"""Supervise the NOKOV SDK bridge and expose its latest rigid-body pose."""

from __future__ import annotations

import math
import os
import subprocess
import threading
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


SENTINEL_LIMIT = 9_000_000.0


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


class MocapReceiver:
    """Keep the SDK subprocess alive and retain the newest selected pose."""

    def __init__(self, config: dict, project_root: Path) -> None:
        self.enabled = bool(config.get("enabled", False))
        self.server = str(config.get("server", "")).strip()
        self.tracker = str(config.get("tracker", "")).strip()
        self.stale_ms = float(config.get("stale_ms", 500))
        self.retry_seconds = float(config.get("retry_seconds", 5))
        workspace_bridge = (
            project_root / "web_monitor" / "mocap" / "bin" / "MocapBridge"
        )
        external_bridge = Path(
            "/home/wts/getViedo/XING_Linux/bin/MocapBridge"
        )
        environment_bridge = os.environ.get("CVIA_MOCAP_BRIDGE", "").strip()
        if environment_bridge:
            self.bridge_path = Path(environment_bridge).expanduser()
        elif workspace_bridge.is_file():
            self.bridge_path = workspace_bridge
        elif external_bridge.is_file():
            self.bridge_path = external_bridge
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
                    self._sdk_version = fields[1]
                    self._connection = "ready"
                    self._last_error = ""
            elif fields[0] == "POSE":
                try:
                    pose = parse_pose_line(line)
                    valid, issue = pose_validity(pose)
                except (ValueError, OverflowError) as exc:
                    self._set_connection("ready", f"无法解析动捕数据：{exc}")
                    continue
                with self._lock:
                    self._latest_pose = pose
                    self._latest_valid = valid
                    self._latest_issue = issue

    def _read_stderr(self, stream: TextIO) -> None:
        for raw_line in stream:
            line = raw_line.rstrip("\r\n")
            if line:
                with self._lock:
                    self._last_error = line[-500:]

    def snapshot(self) -> dict:
        with self._lock:
            connection = self._connection
            sdk_version = self._sdk_version
            descriptions = dict(self._descriptions)
            pose = self._latest_pose
            valid = self._latest_valid
            issue = self._latest_issue
            last_error = self._last_error

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

        euler = quaternion_to_euler_xyz_degrees(
            pose.qx, pose.qy, pose.qz, pose.qw
        )
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
            "pose": {
                "tracker_id": pose.tracker_id,
                "tracker_name": pose.tracker_name,
                "frame": pose.mocap_frame,
                "timestamp_ms": pose.mocap_timestamp_ms,
                "receive_unix_ns": pose.receive_unix_ns,
                "position": {"x": pose.x, "y": pose.y, "z": pose.z},
                "euler_deg": {"rx": euler[0], "ry": euler[1], "rz": euler[2]},
                "quaternion": {
                    "qx": pose.qx,
                    "qy": pose.qy,
                    "qz": pose.qz,
                    "qw": pose.qw,
                },
                "mean_error": pose.mean_error,
                "tracking_params": pose.tracking_params,
            },
        })
        return result
