"""Nonblocking UDP telemetry and latest-frame H.264/RTSP publishing for server.py."""

import ipaddress
import json
import os
import re
import select
import shutil
import socket
import subprocess
import threading
import time
from urllib.parse import urlsplit

import cv2


def pose_message(frame):
    return {"version": 1, "timestamp_ms": frame.timestamp, "units": "mm",
            "detected": bool(frame.keypoints[-1] > 0) if frame.keypoints else None,
            "predicted_xyz_mm": list(frame.pose[:3]), "filtered_xyz_mm": list(frame.pose[3:]),
            "pnp_valid": frame.pnp_valid,
            "pnp_rvec_rad": list(frame.pnp_rvec) if frame.pnp_valid else None,
            "pnp_tvec_mm": list(frame.pnp_tvec) if frame.pnp_valid else None,
            "camera": frame.camera, **frame.coordinate_status()}


class PoseSender:
    def __init__(self, host, port=9000):
        ipaddress.IPv4Address(host)
        if not 1 <= port <= 65535:
            raise ValueError("UDP pose port must be between 1 and 65535")
        self.target = (host, port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.last_warning = 0.0
        print(f"[pose] UDP JSON -> {host}:{port}", flush=True)

    def send(self, frame):
        # One small datagram per complete source frame, independent of video encoding.
        data = json.dumps(pose_message(frame), allow_nan=False, separators=(",", ":")).encode()
        try:
            self.socket.sendto(data, self.target)
        except OSError as error:
            if time.monotonic() - self.last_warning > 5:
                print(f"[pose] Datagram skipped: {error}", flush=True)
                self.last_warning = time.monotonic()

    def close(self):
        self.socket.close()


class RtspPublisher:
    def __init__(self, url, fps=20, bitrate="2M", ffmpeg="ffmpeg", write_timeout=0.5):
        parsed = urlsplit(url)
        if parsed.scheme != "rtsp" or not parsed.hostname or parsed.path in ("", "/"):
            raise ValueError("RTSP URL must include a server and stream path, e.g. rtsp://127.0.0.1:8554/live")
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            raise ValueError("Invalid RTSP port")
        if not 1 <= fps <= 120 or write_timeout <= 0:
            raise ValueError("Invalid RTSP FPS or pipe timeout")
        match = re.fullmatch(r"(\d+(?:\.\d+)?)([kKmM]?)", bitrate)
        if not match:
            raise ValueError("Bitrate must be 2M, 1500k or a positive integer")
        self.bitrate = int(float(match[1]) * {"": 1, "k": 1000, "m": 1000000}[match[2].lower()])
        if self.bitrate <= 0:
            raise ValueError("Bitrate must be positive")
        self.ffmpeg = shutil.which(ffmpeg)
        if self.ffmpeg is None:
            raise ValueError(f"FFmpeg executable not found: {ffmpeg}")
        self.url, self.fps, self.write_timeout = url, fps, write_timeout
        self.condition = threading.Condition()
        self.pending = None
        self.stop = threading.Event()
        self.thread = None
        self.sent = 0
        self.restarts = 0
        self.last_error = None
        self.connected = False
        self.width = None
        self.height = None

    def start(self):
        self.thread = threading.Thread(target=self._run, name="rtsp-publisher", daemon=True)
        self.thread.start()

    def submit(self, image, timestamp):
        # Images are new, owned decoder outputs and are read-only after submission.
        with self.condition:
            self.pending = (image, timestamp, time.monotonic())
            self.condition.notify()

    def status(self):
        with self.condition:
            return {"connected": self.connected, "frames_written": self.sent,
                    "restarts": self.restarts, "last_error": self.last_error,
                    "width": self.width, "height": self.height, "fps": self.fps}

    def command(self, width, height):
        return [self.ffmpeg, "-hide_banner", "-loglevel", "warning",
                "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
                "-framerate", str(self.fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                "-threads", "2", "-preset", "ultrafast", "-tune", "zerolatency",
                "-profile:v", "baseline", "-pix_fmt", "yuv420p", "-b:v", str(self.bitrate),
                "-maxrate", str(self.bitrate), "-bufsize", str(max(1, self.bitrate // 10)),
                "-g", str(self.fps), "-keyint_min", str(self.fps), "-sc_threshold", "0", "-bf", "0",
                "-flush_packets", "1", "-rw_timeout", "3000000",
                "-f", "rtsp", "-rtsp_transport", "tcp", self.url]

    def _write_frame(self, process, data):
        remaining = memoryview(data)
        deadline = time.monotonic() + self.write_timeout
        descriptor = process.stdin.fileno()
        while remaining and not self.stop.is_set():
            if process.poll() is not None:
                raise BrokenPipeError(f"FFmpeg exited with {process.returncode}")
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("FFmpeg input stalled; discarding partial frame and reconnecting")
            if not select.select([], [descriptor], [], min(left, 0.05))[1]:
                continue
            try:
                count = os.write(descriptor, remaining)
            except BlockingIOError:
                continue
            if count <= 0:
                raise BrokenPipeError("FFmpeg input closed")
            remaining = remaining[count:]

    @staticmethod
    def _stop_encoder(process):
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        if process.stdin:
            process.stdin.close()

    def _run(self):
        process, current, dimensions = None, None, None
        next_tick, retry_at = time.monotonic(), 0.0
        try:
            while not self.stop.is_set():
                if self.stop.wait(max(0, next_tick - time.monotonic())):
                    break
                with self.condition:
                    if self.pending is not None:
                        current, self.pending = self.pending, None
                next_tick = time.monotonic() + 1 / self.fps
                if current is None or time.monotonic() < retry_at:
                    continue
                image, timestamp, arrived = current
                height, width = image.shape[:2]
                width, height = width // 2 * 2, height // 2 * 2
                if width < 2 or height < 2:
                    continue
                try:
                    if process is None or dimensions != (width, height):
                        self._stop_encoder(process)
                        process = subprocess.Popen(self.command(width, height), stdin=subprocess.PIPE,
                                                   stdout=subprocess.DEVNULL, bufsize=0)
                        os.set_blocking(process.stdin.fileno(), False)
                        dimensions = (width, height)
                        with self.condition:
                            self.width, self.height = width, height
                    # Timestamp is burnt into the same frame as its XYZ/PnP overlay.
                    video = image[:height, :width].copy()
                    cv2.putText(video, f"Source timestamp ms: {timestamp}", (12, height - 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(video, f"Source timestamp ms: {timestamp}", (12, height - 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    if time.monotonic() - arrived > 1:
                        cv2.putText(video, "STALE: no fresh camera data", (12, height // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                    self._write_frame(process, video.tobytes())
                    with self.condition:
                        self.sent += 1
                        self.connected = True
                        self.last_error = None
                except (OSError, ValueError, TimeoutError) as error:
                    self._stop_encoder(process)
                    process = None
                    retry_at = time.monotonic() + 1
                    with self.condition:
                        self.connected = False
                        self.restarts += 1
                        self.last_error = str(error)
                    print(f"[rtsp] {error}; retrying in 1 second", flush=True)
        finally:
            self._stop_encoder(process)
            with self.condition:
                self.connected = False

    def close(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread:
            self.thread.join(timeout=3)
