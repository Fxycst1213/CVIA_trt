"""UDP pose + H.264 RTSP integration. Optional real server: CVIA_MEDIAMTX_BIN=/path/mediamtx."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.error import URLError
from urllib.request import urlopen

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import server as viewer
from groundstation_stream import RtspPublisher, pose_message

SPEC = importlib.util.spec_from_file_location("receive_pose", ROOT / "docs/51_DiMianZhan/receive_pose.py")
receiver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(receiver)


def unused_port(kind=socket.SOCK_STREAM):
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for(predicate, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            result = predicate()
            if result:
                return result
        except (OSError, URLError):
            pass
        time.sleep(0.05)
    raise AssertionError("Expected state was not reached")


def get_json(url):
    with urlopen(url, timeout=1) as response:
        return json.load(response)


class PoseTests(unittest.TestCase):
    def test_offline_installer_rejects_corrupted_archive(self):
        with tempfile.TemporaryDirectory(prefix="cvia-mediamtx-install-") as directory:
            archive = Path(directory) / "bad.tar.gz"
            archive.write_bytes(b"not an official archive")
            destination = Path(directory) / "bin"
            result = subprocess.run([sys.executable, str(ROOT / "tools/install_mediamtx.py"),
                                     "--archive", str(archive), "--destination", str(destination)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SHA256 mismatch", result.stderr)
            self.assertFalse((destination / "mediamtx").exists())

    def test_pose_units_validity_and_out_of_order_packets(self):
        frame = viewer.Frame(b"", (0,) * 55 + (.9,), (1, 2, 3, 4, 5, 6), 100,
                             time.monotonic(), True, (.1, .2, .3), (400, 500, 600))
        message = pose_message(frame)
        self.assertEqual(receiver.validate_message(json.dumps(message))["pnp_tvec_mm"], [400, 500, 600])
        self.assertEqual(message["filtered_xyz_mm"], [4, 5, 6])
        store = receiver.PoseStore()
        self.assertTrue(store.publish(message, "127.0.0.1"))
        self.assertFalse(store.publish(dict(message, timestamp_ms=99), "127.0.0.1"))
        self.assertEqual(store.status()["pose"]["timestamp_ms"], 100)
        with self.assertRaises(ValueError):
            receiver.validate_message(json.dumps(dict(message, pnp_valid=False)))
        with self.assertRaises(ValueError):
            receiver.validate_message(json.dumps(dict(message, filtered_xyz_mm=[float("nan"), 1, 2])))

    def test_nonreading_encoder_does_not_block_submit_or_shutdown(self):
        with tempfile.TemporaryDirectory(prefix="cvia-stalled-encoder-") as directory:
            executable = Path(directory) / "ffmpeg"
            executable.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(30)\n")
            executable.chmod(0o755)
            publisher = RtspPublisher("rtsp://127.0.0.1:1/live", ffmpeg=str(executable), write_timeout=.15)
            publisher.start()
            try:
                image = np.zeros((480, 640, 3), dtype=np.uint8)
                start = time.monotonic()
                for i in range(100):
                    publisher.submit(image, i)
                self.assertLess(time.monotonic() - start, .5)
                wait_for(lambda: publisher.status()["restarts"] >= 1)
                self.assertFalse(publisher.status()["connected"])
            finally:
                start = time.monotonic()
                publisher.close()
                self.assertLess(time.monotonic() - start, 2)

    def test_udp_pose_reaches_ground_station_when_rtsp_is_unavailable(self):
        tcp_port, web_port, ground_web = unused_port(), unused_port(), unused_port()
        udp_port = unused_port(socket.SOCK_DGRAM)
        processes = []
        try:
            commands = [
                [sys.executable, str(ROOT / "docs/51_DiMianZhan/receive_pose.py"),
                 "--host", "127.0.0.1", "--port", str(udp_port), "--web-host", "127.0.0.1", "--web-port", str(ground_web)],
                [sys.executable, str(ROOT / "tools/server.py"), "--host", "127.0.0.1", "--port", str(tcp_port),
                 "--web-port", str(web_port), "--no-display", "--render-fps", "1", "--video-fps", "10",
                 "--rtsp-url", f"rtsp://127.0.0.1:{unused_port()}/live", "--pose-target", "127.0.0.1", "--pose-port", str(udp_port)],
            ]
            for command in commands:
                processes.append(subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
            local_url = f"http://127.0.0.1:{web_port}/status.json"
            pose_url = f"http://127.0.0.1:{ground_web}/pose.json"
            wait_for(lambda: get_json(local_url))
            wait_for(lambda: get_json(pose_url))
            _, image = cv2.imencode(".jpg", np.zeros((90, 160, 3), dtype=np.uint8))
            prefix = len(image).to_bytes(4, "big") + image.tobytes() + viewer.POINTS.pack(*([0] * viewer.NUM_DATA))
            with socket.create_connection(("127.0.0.1", tcp_port), timeout=2) as connection:
                connection.sendall(b"IHX")
                connection.sendall(b"".join(prefix + viewer.POSE.pack(1, 2, 3, 4, 5, 6, i)
                                            + viewer.POSE_EXTRA.pack(1, .1, .2, .3, 400, 500, 600)
                                            for i in range(1, 81)))
                state = wait_for(lambda: (value if (value := get_json(pose_url))["received"] == 80 else None))
                self.assertEqual(state["pose"]["timestamp_ms"], 80)
                self.assertEqual(state["pose"]["pnp_tvec_mm"], [400, 500, 600])
                self.assertTrue(state["pose"]["pnp_valid"])
                self.assertLess(get_json(local_url)["rendered"], 10)
                connection.sendall(prefix + viewer.POSE.pack(1, 2, 3, 4, 5, 6, 81)
                                   + viewer.POSE_EXTRA.pack(0, 0, 0, 0, 0, 0, 0))
                state = wait_for(lambda: (value if (value := get_json(pose_url))["received"] == 81 else None))
                self.assertFalse(state["pose"]["pnp_valid"])
                self.assertIsNone(state["pose"]["pnp_rvec_rad"])
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                try:
                    stdout, stderr = process.communicate(timeout=6)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                    self.fail("Process did not stop: " + stderr)
                self.assertEqual(process.returncode, 0, stdout + stderr)


@unittest.skipUnless(os.environ.get("CVIA_MEDIAMTX_BIN"), "Set CVIA_MEDIAMTX_BIN for real RTSP decode tests")
class RtspIntegrationTests(unittest.TestCase):
    def test_rtsp_only_skips_web_work_and_keeps_pose_output(self):
        executable = str(Path(os.environ["CVIA_MEDIAMTX_BIN"]).resolve())
        rtsp_port, tcp_port = unused_port(), unused_port()
        processes = []
        with tempfile.TemporaryDirectory(prefix="cvia-rtsp-only-") as directory, socket.socket(
                socket.AF_INET, socket.SOCK_DGRAM) as pose_socket:
            pose_socket.bind(("127.0.0.1", 0))
            pose_socket.settimeout(3)
            config = Path(directory) / "mediamtx.yml"
            config.write_text((ROOT / "tools/mediamtx_orin.yml").read_text().replace(
                "rtspAddress: :8554", f"rtspAddress: 127.0.0.1:{rtsp_port}"))
            # Fail if headless mode starts HTTP or performs unused browser resize/JPEG work.
            script = (f"import sys; sys.path.insert(0, {str(ROOT / 'tools')!r})\n"
                      "import server\n"
                      "def forbidden(*args, **kwargs):\n"
                      "    raise AssertionError('Browser processing called in RTSP-only mode')\n"
                      "server.make_http_server = forbidden\n"
                      "server.cv2.imencode = forbidden\n"
                      "server.cv2.resize = forbidden\n"
                      "server.main()\n")
            try:
                processes.append(subprocess.Popen([executable, str(config)], cwd=directory,
                                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
                processes.append(subprocess.Popen([
                    sys.executable, "-c", script, "--host", "127.0.0.1", "--port", str(tcp_port),
                    "--web-port", "0", "--no-display", "--render-fps", "10", "--video-fps", "10",
                    "--video-width", "1920", "--web-width", "1280", "--bitrate", "1M",
                    "--pose-target", "127.0.0.1", "--pose-port", str(pose_socket.getsockname()[1]),
                    "--rtsp-url", f"rtsp://127.0.0.1:{rtsp_port}/live"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))

                def listening():
                    with socket.create_connection(("127.0.0.1", tcp_port), timeout=.1):
                        return True

                def stream_ready():
                    with socket.create_connection(("127.0.0.1", rtsp_port), timeout=.2) as connection:
                        connection.settimeout(.2)
                        connection.sendall(f'DESCRIBE rtsp://127.0.0.1:{rtsp_port}/live RTSP/1.0\r\nCSeq: 1\r\n\r\n'.encode())
                        return connection.recv(1024).startswith(b"RTSP/1.0 200 ")

                wait_for(listening)
                _, jpeg = cv2.imencode(".jpg", np.full((1080, 1920, 3), 35, dtype=np.uint8))
                packet = (len(jpeg).to_bytes(4, "big") + jpeg.tobytes()
                          + viewer.POINTS.pack(*([0] * viewer.NUM_DATA))
                          + viewer.POSE.pack(1, 2, 3, 4, 5, 6, 12345)
                          + viewer.POSE_EXTRA.pack(0, 0, 0, 0, 0, 0, 0)
                          + viewer.CAMERA.pack(1920, 1080, 1, 1000, 1100, 960, 540, *([0] * 12)))
                with socket.create_connection(("127.0.0.1", tcp_port), timeout=2) as connection:
                    connection.sendall(b"IHC" + packet)
                    pose = json.loads(pose_socket.recvfrom(4096)[0])
                    self.assertEqual(pose["timestamp_ms"], 12345)
                    self.assertEqual(pose["filtered_xyz_mm"], [4, 5, 6])
                    self.assertEqual(pose["camera"]["eye"], "right")
                    self.assertEqual(pose["camera"]["intrinsics"], [1000, 1100, 960, 540])
                    wait_for(stream_ready)
                    probe = subprocess.run([
                        "ffprobe", "-v", "error", "-rtsp_transport", "tcp", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of", "json",
                        f"rtsp://127.0.0.1:{rtsp_port}/live"], capture_output=True, text=True, timeout=15)
                    self.assertEqual(probe.returncode, 0, probe.stderr)
                    video = json.loads(probe.stdout)["streams"][0]
                    self.assertEqual((video["width"], video["height"]), (1920, 1080))
            finally:
                for process in reversed(processes):
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                    try:
                        stdout, stderr = process.communicate(timeout=6)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        stdout, stderr = process.communicate()
                        self.fail("RTSP-only test did not stop: " + stderr)
                    self.assertEqual(process.returncode, 0, stdout + stderr)

    def test_rtsp_stays_1080p_while_browser_preview_is_720p(self):
        executable = str(Path(os.environ["CVIA_MEDIAMTX_BIN"]).resolve())
        rtsp_port, tcp_port, web_port = unused_port(), unused_port(), unused_port()
        processes = []
        with tempfile.TemporaryDirectory(prefix="cvia-1080p-") as directory:
            config = Path(directory) / "mediamtx.yml"
            config.write_text((ROOT / "tools/mediamtx_orin.yml").read_text().replace(
                "rtspAddress: :8554", f"rtspAddress: 127.0.0.1:{rtsp_port}"))
            try:
                processes.append(subprocess.Popen([executable, str(config)], cwd=directory,
                                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
                processes.append(subprocess.Popen([
                    sys.executable, str(ROOT / "tools/server.py"), "--host", "127.0.0.1", "--port", str(tcp_port),
                    "--web-port", str(web_port), "--no-display", "--render-fps", "10", "--video-fps", "10",
                    "--video-width", "1920", "--web-width", "1280", "--bitrate", "1M",
                    "--rtsp-url", f"rtsp://127.0.0.1:{rtsp_port}/live"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
                status_url = f"http://127.0.0.1:{web_port}/status.json"
                wait_for(lambda: get_json(status_url))
                _, jpeg = cv2.imencode(".jpg", np.full((1080, 1920, 3), 35, dtype=np.uint8))
                packet = (len(jpeg).to_bytes(4, "big") + jpeg.tobytes()
                          + viewer.POINTS.pack(*([0] * viewer.NUM_DATA))
                          + viewer.POSE.pack(1, 2, 3, 4, 5, 6, 12345))
                with socket.create_connection(("127.0.0.1", tcp_port), timeout=2) as connection:
                    connection.sendall(b"IHV" + packet)
                    wait_for(lambda: get_json(status_url)["rtsp"]["frames_written"] > 5)
                    with urlopen(f"http://127.0.0.1:{web_port}/snapshot.jpg", timeout=2) as response:
                        preview = cv2.imdecode(np.frombuffer(response.read(), np.uint8), 1)
                    self.assertEqual(preview.shape[:2], (720, 1280))
                    probe = subprocess.run([
                        "ffprobe", "-v", "error", "-rtsp_transport", "tcp", "-select_streams", "v:0",
                        "-show_entries", "stream=codec_name,width,height", "-of", "json",
                        f"rtsp://127.0.0.1:{rtsp_port}/live"], capture_output=True, text=True, timeout=15)
                    self.assertEqual(probe.returncode, 0, probe.stderr)
                    video = json.loads(probe.stdout)["streams"][0]
                    self.assertEqual((video["codec_name"], video["width"], video["height"]), ("h264", 1920, 1080))
            finally:
                for process in reversed(processes):
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                    try:
                        stdout, stderr = process.communicate(timeout=6)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        stdout, stderr = process.communicate()
                        self.fail("1080p test did not stop: " + stderr)
                    self.assertEqual(process.returncode, 0, stdout + stderr)

    def test_h264_decodes_and_recovers_after_mediamtx_restart(self):
        executable = str(Path(os.environ["CVIA_MEDIAMTX_BIN"]).resolve())
        port = unused_port()
        url = f"rtsp://127.0.0.1:{port}/live"
        with tempfile.TemporaryDirectory(prefix="cvia-rtsp-") as directory:
            config = Path(directory) / "mediamtx.yml"
            config.write_text((ROOT / "tools/mediamtx_orin.yml").read_text().replace(
                "rtspAddress: :8554", f"rtspAddress: 127.0.0.1:{port}"))
            mediamtx = None
            publisher = RtspPublisher(url, fps=10, bitrate="300k")

            def start_server():
                process = subprocess.Popen([executable, str(config)], cwd=directory,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                def listening():
                    if process.poll() is not None:
                        raise AssertionError(process.communicate()[0].decode())
                    with socket.create_connection(("127.0.0.1", port), timeout=.1):
                        return True
                wait_for(listening)
                return process

            def decode():
                # Capture real decoded pixels; merely accepting a TCP connection is insufficient.
                decoded = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp",
                                          "-i", url, "-frames:v", "3", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"],
                                         capture_output=True, timeout=15)
                self.assertEqual(decoded.returncode, 0, decoded.stderr.decode())
                self.assertEqual(len(decoded.stdout), 320 * 180 * 3 * 3)
                self.assertGreater(sum(decoded.stdout), 0)

            try:
                mediamtx = start_server()
                publisher.start()
                publisher.submit(np.full((180, 320, 3), 35, dtype=np.uint8), 123456789)
                wait_for(lambda: publisher.status()["frames_written"] > 5)
                decode()
                mediamtx.terminate()
                mediamtx.communicate(timeout=3)
                wait_for(lambda: publisher.status()["restarts"] >= 1)
                written = publisher.status()["frames_written"]
                mediamtx = start_server()
                wait_for(lambda: publisher.status()["frames_written"] >= written + 5)
                decode()
            finally:
                publisher.close()
                if mediamtx and mediamtx.poll() is None:
                    mediamtx.terminate()
                    mediamtx.communicate(timeout=3)


if __name__ == "__main__":
    unittest.main()
