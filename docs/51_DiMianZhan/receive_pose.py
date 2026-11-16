"""Ground station: receive Jetson UDP pose JSON and expose http://127.0.0.1:8081.

Uses only Python's standard library. Run alongside MediaMTX; no SSH is involved.
"""
import argparse
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import socket
import threading
import time


def validate_message(data):
    message = json.loads(data)
    if not isinstance(message, dict) or message.get("version") != 1 or message.get("units") != "mm":
        raise ValueError("Unknown pose schema")
    timestamp = message.get("timestamp_ms")
    if type(timestamp) is not int or timestamp < 0:
        raise ValueError("Invalid timestamp_ms")
    for flag in ("detected", "pnp_valid"):
        if message.get(flag) is not None and type(message[flag]) is not bool:
            raise ValueError(f"Invalid {flag}")
    for key in ("predicted_xyz_mm", "filtered_xyz_mm", "pnp_rvec_rad", "pnp_tvec_mm"):
        value = message.get(key)
        if key.startswith("pnp_") and message.get("pnp_valid") is not True:
            if value is not None:
                raise ValueError("Invalid PnP must not carry stale vectors")
            continue
        if not isinstance(value, list) or len(value) != 3 or not all(
                type(number) in (int, float) and math.isfinite(number) for number in value):
            raise ValueError(f"Invalid {key}")
    return message


class PoseStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.message = None
        self.arrived = 0
        self.sender = None
        self.received = 0
        self.rejected = 0

    def publish(self, message, sender):
        with self.lock:
            if (self.message and self.sender == sender and time.monotonic() - self.arrived < 3
                    and message["timestamp_ms"] <= self.message["timestamp_ms"]):
                self.rejected += 1
                return False
            self.message = message
            self.sender = sender
            self.arrived = time.monotonic()
            self.received += 1
            return True

    def status(self):
        with self.lock:
            age = round((time.monotonic() - self.arrived) * 1000) if self.message else None
            return {"live": age is not None and age < 1000, "age_ms": age,
                    "received": self.received, "rejected": self.rejected,
                    "sender": self.sender, "pose": self.message}


PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>地面站位姿结果</title>
<style>body{font:17px system-ui;background:#15191f;color:#edf2f7;margin:24px}
pre{white-space:pre-wrap;line-height:1.7}</style><h2>地面站位姿结果</h2>
<p>视频用地面站播放器或 VLC 打开 MediaMTX 的 /live RTSP 地址。</p>
<p>位置单位：毫米；旋转向量单位：弧度。PnP 无效时不使用其旋转和平移结果。</p>
<pre id="data">等待 Jetson UDP 数据…</pre><script>
async function refresh(){try{
const s=await(await fetch('/pose.json',{cache:'no-store'})).json();
document.getElementById('data').textContent=(s.live?'实时接收':'等待数据 / 数据已过期')+'\\n'+JSON.stringify(s,null,2);
}catch(e){document.getElementById('data').textContent='服务连接失败';}setTimeout(refresh,200);}refresh();
</script></html>""".encode("utf-8")


def make_http_server(host, port, store):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                kind, body = "text/html; charset=utf-8", PAGE
            elif path == "/pose.json":
                kind, body = "application/json", json.dumps(store.status(), allow_nan=False).encode()
            else:
                self.send_error(404)
                return
            try:
                self.connection.settimeout(2)
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--web-host", default="0.0.0.0")
    parser.add_argument("--web-port", type=int, default=8081)
    parser.add_argument("--sender-ip", help="Optional Jetson source IPv4 filter")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.web_port <= 65535:
        parser.error("Ports must be between 1 and 65535")
    store, csv_file, http = PoseStore(), None, None
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        receiver.bind((args.host, args.port))
        receiver.settimeout(0.5)
        http = make_http_server(args.web_host, args.web_port, store)
        threading.Thread(target=http.serve_forever, daemon=True).start()
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            csv_file = args.csv.open("w", newline="", encoding="utf-8")
            writer = csv.writer(csv_file)
            writer.writerow(["timestamp_ms", "pnp_valid", "x_predict_mm", "y_predict_mm", "z_predict_mm",
                             "x_filtered_mm", "y_filtered_mm", "z_filtered_mm", "rx_rad", "ry_rad", "rz_rad",
                             "x_pnp_mm", "y_pnp_mm", "z_pnp_mm"])
        print(f"[pose] Listening UDP {args.host}:{args.port}", flush=True)
        print(f"[web] http://127.0.0.1:{args.web_port}", flush=True)
        while True:
            try:
                data, address = receiver.recvfrom(4096)
            except socket.timeout:
                continue
            if args.sender_ip and address[0] != args.sender_ip:
                continue
            try:
                message = validate_message(data)
            except (ValueError, TypeError, OverflowError, UnicodeDecodeError):
                with store.lock:
                    store.rejected += 1
                continue
            if store.publish(message, address[0]) and csv_file:
                writer.writerow([message["timestamp_ms"], message["pnp_valid"], *message["predicted_xyz_mm"],
                                 *message["filtered_xyz_mm"], *(message["pnp_rvec_rad"] or [None] * 3),
                                 *(message["pnp_tvec_mm"] or [None] * 3)])
    except KeyboardInterrupt:
        pass
    finally:
        receiver.close()
        if http:
            http.shutdown()
            http.server_close()
        if csv_file:
            csv_file.close()


if __name__ == "__main__":
    main()
