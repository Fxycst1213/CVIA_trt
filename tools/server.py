"""Latest-frame image/pose viewer with optional per-frame camera calibration.

Jetson: python3 tools/server.py --host 127.0.0.1 --no-display
Jetson (another terminal): ./trt --preview --calibrated-pose --no-serial
PC: ssh -N -L 8080:127.0.0.1:8080 cvia@JETSON_IP
PC browser: http://127.0.0.1:8080
"""

import argparse
import csv
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import struct
import threading
import time
from typing import Optional

import cv2
import numpy as np

# Must match src/params/pose_params.hpp and the x_plane1.onnx publisher.
NUM_KEYPOINTS = 9
NUM_DATA = NUM_KEYPOINTS * 3 + 5
POSE_DAT_NUM = 8
MAX_JPEG_BYTES = 16 * 1024 * 1024
POINTS = struct.Struct(f"<{NUM_DATA}f")
POSE = struct.Struct("<6fQ")
POSE_EXTRA = struct.Struct("<I6f")
CAMERA = struct.Struct("<III16d")
SRC_W, SRC_H = 1920, 1080
# Match Pose's 1920x1080 calibration in src/yolov_pose/pose.cpp.
K = np.array([[1067.33054757922, 0, 949.935792304770],
              [0, 1067.37400981335, 525.523361276358], [0, 0, 1]], dtype=np.float64)
DIST = np.array([-0.0898188725781947, 0.0570779357792198, 0, 0, 0.0421749060858686], dtype=np.float64)


@dataclass(frozen=True)
class Frame:
    jpeg: bytes
    keypoints: tuple
    pose: tuple  # Predicted XYZ, then filtered XYZ, in millimeters; no Euler angles.
    timestamp: int
    arrived: float
    pnp_valid: Optional[bool] = None  # None for legacy IHV/IHD publishers.
    pnp_rvec: tuple = ()
    pnp_tvec: tuple = ()
    camera: Optional[dict] = None


def recv_all(sock, count, stop=None):
    """Read exactly one field, tolerating TCP fragmentation and timeouts."""
    data = bytearray(count)
    view = memoryview(data)
    offset = 0
    while offset < count:
        if stop is not None and stop.is_set():
            return None
        try:
            received = sock.recv_into(view[offset:])
        except socket.timeout:
            continue
        if received == 0:
            if offset:
                raise EOFError("Connection closed inside a protocol field")
            return None
        offset += received
    return bytes(data)


def receive_frame(sock, mode, stop=None):
    jpeg, points = b"", ()
    if mode in ("V", "X", "C"):
        length = recv_all(sock, 4, stop)
        if length is None:
            return None
        size = struct.unpack("!I", length)[0]
        if size > MAX_JPEG_BYTES:
            raise ValueError(f"JPEG too large: {size}")
        jpeg = recv_all(sock, size, stop)
        data = recv_all(sock, POINTS.size + POSE.size, stop)
        if jpeg is None or data is None:
            raise EOFError("Incomplete image frame")
        points = POINTS.unpack(data[:POINTS.size])
        data = data[POINTS.size:]
    elif mode == "D":
        data = recv_all(sock, POSE.size, stop)
        if data is None:
            return None
    else:
        raise ValueError("Unknown protocol mode")
    values = POSE.unpack(data)
    extra = None
    if mode in ("X", "C"):
        extension = recv_all(sock, POSE_EXTRA.size, stop)
        if extension is None:
            raise EOFError("Incomplete PnP extension")
        extra = POSE_EXTRA.unpack(extension)
        if extra[0] not in (0, 1) or not all(np.isfinite(value) for value in extra[1:]):
            raise ValueError("Invalid PnP extension")
    camera = None
    if mode == "C":
        description = recv_all(sock, CAMERA.size, stop)
        if description is None:
            raise EOFError("Incomplete camera calibration")
        calibration = CAMERA.unpack(description)
        width, height, right_eye = calibration[:3]
        parameters = calibration[3:]
        if (not 0 < width <= 8192 or not 0 < height <= 8192 or right_eye not in (0, 1)
                or not all(np.isfinite(value) for value in parameters)
                or parameters[0] <= 0 or parameters[1] <= 0):
            raise ValueError("Invalid camera calibration")
        camera = {"eye": "right" if right_eye else "left", "view": "unrectified",
                  "width": width, "height": height, "intrinsics": list(parameters[:4]),
                  "distortion": list(parameters[4:])}
    if not all(np.isfinite(value) for value in points + values[:6]):
        raise ValueError("Non-finite coordinates in frame")
    return Frame(jpeg, points, values[:6], values[6], time.monotonic(),
                 bool(extra[0]) if extra else None,
                 extra[1:4] if extra and extra[0] else (), extra[4:] if extra and extra[0] else (), camera)


class LatestFrames:
    """One raw slot and one rendered snapshot, irrespective of viewer speed."""

    def __init__(self):
        self.condition = threading.Condition()
        self.frame = None
        self.sequence = 0
        self.rendered_sequence = 0
        self.taken_sequence = 0
        self.rendered_frame = None
        self.jpeg = None
        self.received = 0
        self.rendered = 0
        self.replaced = 0
        self.connected = False
        self.rate_started = time.monotonic()
        self.rate_count = 0
        self.receive_fps = 0.0
        self.output_status = None

    def publish(self, frame):
        with self.condition:
            if self.sequence > self.taken_sequence:
                self.replaced += 1
            self.frame = frame
            self.sequence += 1
            self.received += 1
            self.rate_count += 1
            elapsed = time.monotonic() - self.rate_started
            if elapsed >= 1:
                self.receive_fps = self.rate_count / elapsed
                self.rate_count = 0
                self.rate_started = time.monotonic()
            self.condition.notify_all()

    def latest(self, previous, timeout=0.1):
        with self.condition:
            self.condition.wait_for(lambda: self.sequence != previous, timeout)
            self.taken_sequence = self.sequence
            return self.sequence, self.frame

    def rendered_image(self, sequence, frame, jpeg):
        with self.condition:
            self.rendered_sequence = sequence
            self.rendered_frame = frame
            self.jpeg = jpeg
            self.rendered += 1

    def status(self):
        with self.condition:
            frame = self.rendered_frame
            age = (time.monotonic() - frame.arrived) * 1000 if frame else None
            return {"connected": self.connected, "received": self.received,
                    "rendered": self.rendered, "skipped": self.replaced,
                    "receive_fps": round(self.receive_fps, 1) if self.connected else 0,
                    "age_ms": round(age) if age is not None else None,
                    "timestamp": frame.timestamp if frame else None,
                    "detected": bool(frame.keypoints[-1] > 0) if frame and frame.keypoints else None,
                    "predicted_xyz_mm": frame.pose[:3] if frame else None,
                    "filtered_xyz_mm": frame.pose[3:] if frame else None,
                    "pnp_valid": frame.pnp_valid if frame else None,
                    "pnp_rvec_rad": frame.pnp_rvec if frame and frame.pnp_valid else None,
                    "pnp_tvec_mm": frame.pnp_tvec if frame and frame.pnp_valid else None,
                    "camera": frame.camera if frame else None,
                    "rtsp": self.output_status() if self.output_status else None}


def receive_connections(listener, frames, stop, on_frame=None):
    listener.settimeout(0.25)
    while not stop.is_set():
        try:
            connection, address = listener.accept()
        except socket.timeout:
            continue
        except OSError:
            if stop.is_set():
                break
            raise
        with connection:
            connection.settimeout(0.25)
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                handshake = recv_all(connection, 3, stop)
                if handshake not in (b"IHV", b"IHD", b"IHX", b"IHC"):
                    raise ValueError("Expected IHV, IHD, IHX or IHC handshake")
                with frames.condition:
                    frames.connected = True
                    frames.rate_started = time.monotonic()
                    frames.rate_count = 0
                print(f"[tcp] Connected: {address}", flush=True)
                while not stop.is_set():
                    frame = receive_frame(connection, chr(handshake[2]), stop)
                    if frame is None:
                        break
                    frames.publish(frame)
                    if on_frame is not None:
                        on_frame(frame)
            except (OSError, EOFError, ValueError) as error:
                if not stop.is_set():
                    print(f"[tcp] {error}; waiting for reconnect", flush=True)
            finally:
                with frames.condition:
                    frames.connected = False


def scale_camera_matrix(camera_matrix, src_w, src_h, dst_w, dst_h):
    scaled = camera_matrix.copy()
    scaled[0, :] *= dst_w / float(src_w)
    scaled[1, :] *= dst_h / float(src_h)
    return scaled


def get_pixel_location(x, y, z, camera_matrix, dist_coeffs):
    if z <= 0:
        return -1000, -1000
    pixels, _ = cv2.projectPoints(np.array([[x, y, z]], dtype=np.float64),
                                  np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
    return tuple(pixels[0, 0].astype(int))


def projection_calibration(calibration, width, height):
    if calibration is None:
        return scale_camera_matrix(K, SRC_W, SRC_H, width, height), DIST
    fx, fy, cx, cy = calibration["intrinsics"]
    camera = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return (scale_camera_matrix(camera, calibration["width"], calibration["height"], width, height),
            np.asarray(calibration["distortion"], dtype=np.float64))


def draw_result_image(image, points, pose, pnp_valid=None, pnp_rvec=(), pnp_tvec=(), calibration=None):
    h, w = image.shape[:2]
    if points and points[-1] > 0:
        cv2.rectangle(image, (int(points[-5]), int(points[-4])),
                      (int(points[-3]), int(points[-2])), (255, 255, 0), 2)
        for i in range(NUM_KEYPOINTS):
            x, y, confidence = points[i * 3:i * 3 + 3]
            if confidence >= 0.5 and 0 <= x < w and 0 <= y < h:
                point = (int(x), int(y))
                cv2.circle(image, point, 4, (0, 255, 0), -1)
                cv2.putText(image, str(i + 1), (point[0] + 5, point[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        camera, distortion = projection_calibration(calibration, w, h)
        position = get_pixel_location(*(pnp_tvec if pnp_valid else pose[3:]), camera, distortion)
        if 0 <= position[0] < w and 0 <= position[1] < h:
            cv2.circle(image, position, 6, (0, 0, 255), -1)
        if pnp_valid:
            cv2.drawFrameAxes(image, camera, distortion, np.asarray(pnp_rvec, dtype=np.float64),
                              np.asarray(pnp_tvec, dtype=np.float64), 200, 2)
    lines = ["Filtered XYZ mm: " + "  ".join(f"{v:.1f}" for v in pose[3:]),
             "Predicted XYZ mm: " + "  ".join(f"{v:.1f}" for v in pose[:3])]
    if points and points[-1] <= 0:
        lines.append("No target detected")
    if pnp_valid is not None:
        lines.append("PnP OK" if pnp_valid else "PnP invalid: no fresh 6D pose")
    if calibration:
        lines.append("Camera: " + calibration["eye"].upper())
    for i, text in enumerate(lines):
        origin = (12, 27 + i * 26)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return image


PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>位姿解算实时预览</title><style>
body{margin:24px;background:#15191f;color:#edf2f7;font:16px system-ui}
img{max-width:100%;display:block;background:#080a0d;min-height:120px}
pre{white-space:pre-wrap;line-height:1.6}p{color:#adb8c6}
</style><h2>位姿解算实时预览</h2>
<p>绿点及编号：检测关键点；红点：输出位置投影。位置单位为毫米。</p>
<img id="image" alt="等待图像连接…"><pre id="status">等待数据…</pre>
<script>
const image=document.getElementById('image'), status=document.getElementById('status');
async function refresh(){
  let url;
  try {
    const response=await fetch('/snapshot.jpg',{cache:'no-store'});
    if(response.ok){
      url=URL.createObjectURL(await response.blob());
      image.src=url;
      await image.decode();
    }
  } catch(error) {} finally {
    if(url) URL.revokeObjectURL(url);
    setTimeout(refresh,65);
  }
}
async function updateStatus(){
  try {
    const s=await (await fetch('/status.json',{cache:'no-store'})).json();
    const live=s.connected && s.age_ms!==null && s.age_ms<1000;
    status.textContent=(live?'实时接收':'等待数据或画面已停止更新')+
      (s.detected===false?'；未检测到目标':'')+
      '\\n接收帧率: '+s.receive_fps+' FPS  已接收: '+s.received+'  已绘制: '+s.rendered+
      '\\n画面距本机收到该帧: '+(s.age_ms===null?'—':s.age_ms+' ms')+
      '\\n源时间戳: '+s.timestamp+
      '\\n滤波位置 XYZ (mm): '+JSON.stringify(s.filtered_xyz_mm)+
      '\\n预测位置 XYZ (mm): '+JSON.stringify(s.predicted_xyz_mm);
  } catch(error) { status.textContent='预览服务未连接'; }
  setTimeout(updateStatus,1000);
}
refresh(); updateStatus();
</script></html>""".encode("utf-8")


def make_http_server(host, port, frames):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            code, kind = 200, "text/html; charset=utf-8"
            if path == "/":
                body = PAGE
            elif path == "/status.json":
                kind, body = "application/json", json.dumps(frames.status()).encode()
            elif path == "/snapshot.jpg":
                with frames.condition:
                    body = frames.jpeg
                kind = "image/jpeg"
                if body is None:
                    code, kind, body = 503, "text/plain", b"Waiting for image"
            else:
                code, kind, body = 404, "text/plain", b"Not found"
            try:
                self.connection.settimeout(2)
                self.send_response(code)
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0", help="TCP listen address")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8080, help="0 disables browser preview")
    parser.add_argument("--no-display", action="store_true", help="Do not open an OpenCV window")
    parser.add_argument("--render-fps", type=float, default=20)
    parser.add_argument("--video-width", type=int, default=0, help="RTSP maximum width; 0 preserves source resolution")
    parser.add_argument("--web-width", type=int, default=1280, help="Browser snapshot maximum width; 0 preserves source resolution")
    parser.add_argument("--csv", type=Path, help="Optional CSV of rendered frames")
    parser.add_argument("--rtsp-url", help="Publish annotated H.264 to MediaMTX; Orin mode uses 127.0.0.1")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--bitrate", default="2M")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--pose-target", help="Ground station IPv4 for UDP JSON telemetry")
    parser.add_argument("--pose-port", type=int, default=9000)
    args = parser.parse_args()
    if not 0 < args.render_fps <= 120 or not 1 <= args.port <= 65535 or not 0 <= args.web_port <= 65535:
        parser.error("Invalid port or render FPS")
    if not 0 <= args.video_width <= 8192 or not 0 <= args.web_width <= 8192:
        parser.error("Invalid output width")
    display = not args.no_display and (os.name == "nt" or bool(os.environ.get("DISPLAY")))
    frames, stop = LatestFrames(), threading.Event()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    http = None
    csv_file = None
    receiver = None
    publisher = None
    telemetry = None
    try:
        if args.rtsp_url or args.pose_target:
            from groundstation_stream import RtspPublisher, PoseSender
            if args.rtsp_url:
                publisher = RtspPublisher(args.rtsp_url, args.video_fps, args.bitrate, args.ffmpeg)
                frames.output_status = publisher.status
                publisher.start()
            if args.pose_target:
                telemetry = PoseSender(args.pose_target, args.pose_port)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.host, args.port))
        listener.listen(4)
        if args.web_port:
            http = make_http_server(args.web_host, args.web_port, frames)
            threading.Thread(target=http.serve_forever, daemon=True).start()
            print(f"[web] http://{args.web_host}:{args.web_port}", flush=True)
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            csv_file = args.csv.open("w", newline="", encoding="utf-8")
            writer = csv.writer(csv_file)
            writer.writerow(["timestamp", "x_predict", "y_predict", "z_predict", "x", "y", "z"])
        receiver = threading.Thread(target=receive_connections,
                                    args=(listener, frames, stop, telemetry.send if telemetry else None), daemon=True)
        receiver.start()
        print(f"[tcp] Listening on {args.host}:{args.port}", flush=True)
        if display:
            cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Window", 1244, 700)
        sequence = 0
        last_camera = None
        next_render = time.monotonic()
        while not stop.is_set():
            # Pace before taking the slot, so rendering always consumes the newest frame.
            stop.wait(max(0, next_render - time.monotonic()))
            current, frame = frames.latest(sequence)
            if current != sequence and frame is not None:
                sequence = current
                next_render = time.monotonic() + 1 / args.render_fps
                jpeg = None
                if frame.jpeg:
                    image = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if image is not None:
                        if frame.camera and frame.camera != last_camera:
                            last_camera = frame.camera
                            print("[camera] " + json.dumps(last_camera), flush=True)
                        image = draw_result_image(image, frame.keypoints, frame.pose,
                                                  frame.pnp_valid, frame.pnp_rvec, frame.pnp_tvec, frame.camera)
                        if publisher:
                            # The RTSP worker owns its own latest-frame slot; it never blocks reception/rendering.
                            video = image
                            if args.video_width and image.shape[1] > args.video_width:
                                video = cv2.resize(image, (args.video_width,
                                    max(1, round(image.shape[0] * args.video_width / image.shape[1]))))
                            publisher.submit(video, frame.timestamp)
                        # RTSP-only mode skips browser resizing and JPEG encoding entirely.
                        if display or http:
                            preview = image
                            if args.web_width and image.shape[1] > args.web_width:
                                preview = cv2.resize(image, (args.web_width,
                                    max(1, round(image.shape[0] * args.web_width / image.shape[1]))))
                            if display:
                                cv2.imshow("Window", preview)
                            if http:
                                ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 75])
                                if ok:
                                    jpeg = encoded.tobytes()
                frames.rendered_image(current, frame, jpeg)
                if csv_file:
                    writer.writerow([frame.timestamp, *frame.pose])
            if display and cv2.waitKey(1) & 0xff == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        listener.close()
        if receiver:
            receiver.join(timeout=1)
        if publisher:
            publisher.close()
        if telemetry:
            telemetry.close()
        if http:
            http.shutdown()
            http.server_close()
        if csv_file:
            csv_file.close()
        if display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
