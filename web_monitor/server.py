#!/usr/bin/env python3
"""Dependency-free LAN dashboard for CVIA on AGX Orin."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import struct
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
RUNTIME_PATH = ROOT / "runtime"
PREVIEW_PAIR_PATH = RUNTIME_PATH / "preview_pair.bin"
PID_PATH = RUNTIME_PATH / "trt.pid"
SESSION_CSV_PATH = RUNTIME_PATH / "pnp_session.csv"
CALIBRATION_HISTORY_PATH = ROOT / "calibration_history.json"
MAX_BODY = 256 * 1024
CALIBRATION_HISTORY_LIMIT = 50
CALIBRATION_HISTORY_LOCK = threading.Lock()
SERVER_STARTED_NS = time.time_ns()
PREVIEW_PAIR_HEADER = struct.Struct("!4sII")
PREVIEW_PAIR_MAGIC = b"CVP1"


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
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} 不能小于 {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{path} 不能大于 {maximum}")


def validate(config: dict) -> dict:
    required = {"input", "detect_camera", "photo_camera", "calibration", "network", "runtime"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError("配置结构不完整")

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
    for field, length in (("camera_matrix", 9), ("distortion", 5), ("extrinsic", 16)):
        values = calibration.get(field)
        if not isinstance(values, list) or len(values) != length:
            raise ValueError(f"calibration.{field} 必须包含 {length} 个数字")
        for index, value in enumerate(values):
            require_number(value, f"calibration.{field}[{index}]")
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


def atomic_save(config: dict) -> None:
    atomic_save_json(CONFIG_PATH, config, "config-")


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


def inference_process() -> tuple[bool, int | None]:
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


def session_status() -> dict:
    if not SESSION_CSV_PATH.exists():
        return {"available": False, "bytes": 0, "mtime_ns": None}
    stat = SESSION_CSV_PATH.stat()
    has_rows = stat.st_size > len("timestamp,x,y,z,rx,ry,rz\n")
    return {"available": has_rows, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "current_run": stat.st_mtime_ns >= SERVER_STARTED_NS}


class DashboardHandler(SimpleHTTPRequestHandler):
    server_version = "CVIA-Dashboard/1.1"
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "static"), **kwargs)

    def log_message(self, fmt, *args):
        # 预览轮询属于正常高频流量。逐请求打印会在 15–30 FPS 双图模式下
        # 产生每秒几十行终端 I/O，反过来拖慢网页与推理进程。
        path = urlparse(self.path).path
        status = str(args[1]) if len(args) > 1 else ""
        hot_path = path.startswith("/preview/") or path in {
            "/api/reprojection", "/api/pnp", "/api/status"
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
                self.send_json(load_config())
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
                            "inference": {"running": inference_running, "pid": inference_pid},
                            "pnp_session": session_status()})
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
                for row in csv.reader(tail_text_lines(csv_path, limit + 1)):
                    if len(row) != 7 or row[0] == "timestamp":
                        continue
                    try:
                        rows.append([int(row[0]), *[float(value) for value in row[1:]]])
                    except ValueError:
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
            current = load_config()
            record_calibration_history(current["calibration"], payload["calibration"])
            atomic_save(payload)
            self.send_json({"ok": True, "message": "配置已保存；重启推理进程后生效"})
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, 400)
        except Exception as exc:
            self.send_json({"error": f"保存失败: {exc}"}, 500)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/inference/stop":
            running, pid = inference_process()
            if not running or pid is None:
                self.send_json({"ok": True, "message": "推理进程已经停止", "running": False})
                return
            try:
                os.kill(pid, signal.SIGTERM)
                self.send_json({"ok": True, "message": "已请求推理进程优雅停止", "running": True})
            except ProcessLookupError:
                self.send_json({"ok": True, "message": "推理进程已经停止", "running": False})
            except Exception as exc:
                self.send_json({"error": f"停止推理失败: {exc}"}, 500)
            return
        self.send_error(404)


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    request_queue_size = 64


def main() -> None:
    parser = argparse.ArgumentParser(description="CVIA AGX Orin web dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="listen address; default exposes the dashboard to the local network")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    RUNTIME_PATH.mkdir(parents=True, exist_ok=True)
    config = validate(load_config())
    record_calibration_history(config["calibration"])
    server = DashboardServer((args.host, args.port), DashboardHandler)
    print(f"CVIA dashboard: http://{args.host}:{args.port}")
    print("Config changes are atomic and take effect after restarting trt.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
