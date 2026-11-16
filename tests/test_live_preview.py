"""Production IHV/IHD transport and headless HTTP preview regression tests.

python3 -m unittest discover -s tests -p test_live_preview.py -v
"""
import importlib.util
import json
from pathlib import Path
import re
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("live_preview", ROOT / "tools/server.py")
viewer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = viewer
SPEC.loader.exec_module(viewer)


def packet(timestamp=1234567890000, jpeg=None):
    if jpeg is None:
        _, data = cv2.imencode(".jpg", np.zeros((90, 160, 3), dtype=np.uint8))
        jpeg = data.tobytes()
    points = [0.0] * viewer.NUM_DATA
    return (struct.pack("!I", len(jpeg)) + jpeg + viewer.POINTS.pack(*points)
            + viewer.POSE.pack(1, 2, 3, 4, 5, 6, timestamp))


class FragmentedSocket:
    def __init__(self, data):
        self.data = data

    def recv_into(self, view):
        count = min(len(view), len(self.data), 3)
        view[:count] = self.data[:count]
        self.data = self.data[count:]
        return count


class ProtocolTests(unittest.TestCase):
    def test_legacy_preview_fallback_matches_solver_defaults(self):
        source = (ROOT / "src/yolov_pose/pose.cpp").read_text()
        camera = re.search(r"_K = \(cv::Mat_<double>\(3, 3\) <<(.*?)\);", source, re.S)
        distortion = re.search(r"_diff = \(cv::Mat_<float>\(1, 5\) <<(.*?)\);", source, re.S)
        expected = np.array([float(value) for value in camera.group(1).split(",")]).reshape(3, 3)
        np.testing.assert_allclose(viewer.K, expected, rtol=0, atol=1e-10)
        expected = np.array([float(value) for value in distortion.group(1).split(",")])
        np.testing.assert_allclose(viewer.DIST, expected, rtol=0, atol=1e-10)

    def test_fragmented_frames_do_not_mix_image_pose_and_timestamp(self):
        stream = FragmentedSocket(packet(111) + packet(222))
        self.assertEqual(viewer.receive_frame(stream, "V").timestamp, 111)
        frame = viewer.receive_frame(stream, "V")
        self.assertEqual(frame.timestamp, 222)
        self.assertEqual(frame.pose, (1, 2, 3, 4, 5, 6))
        self.assertIsNone(viewer.receive_frame(stream, "V"))

    def test_truncated_or_oversized_frames_are_rejected(self):
        for size in (2, 4, len(packet()) - 1):
            with self.subTest(size=size), self.assertRaises(EOFError):
                viewer.receive_frame(FragmentedSocket(packet()[:size]), "V")
        with self.assertRaises(ValueError):
            viewer.receive_frame(FragmentedSocket(struct.pack("!I", viewer.MAX_JPEG_BYTES + 1)), "V")

    def test_pure_data_mode_keeps_all_six_values_and_timestamp(self):
        frame = viewer.receive_frame(FragmentedSocket(viewer.POSE.pack(1, 2, 3, 4, 5, 6, 999)), "D")
        self.assertEqual((frame.pose, frame.timestamp, frame.jpeg), ((1, 2, 3, 4, 5, 6), 999, b""))

    def test_extended_pose_is_framed_and_invalid_pose_has_no_stale_vectors(self):
        stream = FragmentedSocket(packet(111) + viewer.POSE_EXTRA.pack(1, .1, .2, .3, 400, 500, 600)
                                  + packet(222) + viewer.POSE_EXTRA.pack(0, 0, 0, 0, 0, 0, 0))
        frame = viewer.receive_frame(stream, "X")
        self.assertTrue(frame.pnp_valid)
        self.assertEqual(frame.timestamp, 111)
        np.testing.assert_allclose(frame.pnp_rvec, (.1, .2, .3), atol=1e-6)
        self.assertEqual(frame.pnp_tvec, (400, 500, 600))
        frame = viewer.receive_frame(stream, "X")
        self.assertFalse(frame.pnp_valid)
        self.assertEqual((frame.timestamp, frame.pnp_rvec, frame.pnp_tvec), (222, (), ()))
        with self.assertRaises(ValueError):
            viewer.receive_frame(FragmentedSocket(packet() + viewer.POSE_EXTRA.pack(3, 0, 0, 0, 0, 0, 0)), "X")

    def test_calibration_follows_frame_and_scales_for_projection(self):
        values = [160, 90, 1, 100, 110, 80, 45] + [0] * 12
        prefix = packet(111) + viewer.POSE_EXTRA.pack(0, 0, 0, 0, 0, 0, 0)
        frame = viewer.receive_frame(FragmentedSocket(prefix + viewer.CAMERA.pack(*values)), "C")
        self.assertEqual(frame.camera["eye"], "right")
        self.assertEqual(frame.camera["intrinsics"], [100, 110, 80, 45])
        camera, distortion = viewer.projection_calibration(frame.camera, 80, 45)
        np.testing.assert_allclose(camera, [[50, 0, 40], [0, 55, 22.5], [0, 0, 1]])
        np.testing.assert_array_equal(distortion, np.zeros(12))
        for index, invalid in ((0, 0), (2, 2), (3, 0), (7, float("nan"))):
            bad = values.copy()
            bad[index] = invalid
            with self.subTest(index=index), self.assertRaises(ValueError):
                viewer.receive_frame(FragmentedSocket(prefix + viewer.CAMERA.pack(*bad)), "C")
        with self.assertRaises(EOFError):
            viewer.receive_frame(FragmentedSocket(prefix + viewer.CAMERA.pack(*values)[:-1]), "C")


class ClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="cvia-preview-")
        cls.executable = Path(cls.directory.name) / "sender"
        subprocess.run(["g++", "-std=c++14", "-O2", "-pthread", "-I" + str(ROOT / "src"),
                        "-I/usr/include/opencv4", str(ROOT / "tests/legacy_tcp_sender.cpp"),
                        str(ROOT / "src/communication/client.cpp"), "-lopencv_core",
                        "-lopencv_imgproc", "-lopencv_imgcodecs", "-o", str(cls.executable)],
                       check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen()
        self.listener.settimeout(5)
        self.process = None

    def tearDown(self):
        self.listener.close()
        if self.process and self.process.poll() is None:
            self.process.kill()
            self.process.communicate()

    def start_sender(self, width=0, mode=0, count=1, delay=0, pressure=False):
        self.process = subprocess.Popen([str(self.executable), str(self.listener.getsockname()[1]),
                                         str(width), str(mode), str(count), str(delay), str(int(pressure))],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def accept(self, expected=b"IHV"):
        connection, _ = self.listener.accept()
        connection.settimeout(3)
        self.assertEqual(viewer.recv_all(connection, 3), expected)
        return connection

    def finish(self):
        stdout, stderr = self.process.communicate(timeout=8)
        self.assertEqual(self.process.returncode, 0, stderr)
        return json.loads(stdout)

    def test_resized_jpeg_keypoints_and_pose_stay_aligned(self):
        self.start_sender(width=960)
        with self.accept() as connection:
            frame = viewer.receive_frame(connection, "V")
        self.assertEqual(cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), 1).shape[:2], (540, 960))
        self.assertEqual(len(frame.keypoints), 9 * 3 + 5)
        np.testing.assert_allclose(np.asarray(frame.keypoints[:-5]).reshape(9, 3),
                                   np.tile([480, 270, 0.95], (9, 1)), atol=1e-6)
        np.testing.assert_allclose(frame.keypoints[:3], (480, 270, 0.95), atol=1e-6)
        np.testing.assert_allclose(frame.keypoints[-5:], (240, 135, 720, 405, 0.9), atol=1e-6)
        self.assertEqual(frame.pose, (1, 2, 3, 4, 5, 6))
        self.assertEqual(frame.timestamp, 1234567890000)
        self.assertEqual(self.finish()["sent"], 1)

    def test_original_resolution_and_data_mode_remain_supported(self):
        self.start_sender()
        with self.accept() as connection:
            frame = viewer.receive_frame(connection, "V")
        self.assertEqual(cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), 1).shape[:2], (1080, 1920))
        self.assertEqual(frame.keypoints[:2], (960, 540))
        self.finish()
        self.start_sender(mode=1, count=2)
        with self.accept(b"IHD") as connection:
            for i in range(2):
                frame = viewer.receive_frame(connection, "D")
                self.assertEqual(frame.timestamp, 1234567890000 + i)
                self.assertEqual(frame.pose, (1, 2, 3, 4, 5, 6))
        self.assertEqual(self.finish()["sent"], 2)

    def test_peer_reset_reconnects_with_fresh_handshake(self):
        self.start_sender(width=640, count=40, delay=50)
        connection = self.accept()
        first = viewer.receive_frame(connection, "V")
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        connection.close()
        with self.accept() as connection:
            latest = viewer.receive_frame(connection, "V")
        self.assertGreater(latest.timestamp, first.timestamp)
        self.assertGreater(self.finish()["sent"], 1)

    def test_production_client_sends_full_pose_and_clears_invalid_frame(self):
        self.start_sender(width=640, mode=2, count=2)
        with self.accept(b"IHX") as connection:
            first = viewer.receive_frame(connection, "X")
            second = viewer.receive_frame(connection, "X")
        self.assertTrue(first.pnp_valid)
        np.testing.assert_allclose(first.pnp_rvec, (.1, .2, .3), atol=1e-6)
        self.assertEqual(first.pnp_tvec, (400, 500, 600))
        self.assertFalse(second.pnp_valid)
        self.assertEqual(second.pnp_rvec, ())
        self.assertEqual(second.timestamp, first.timestamp + 1)
        self.assertEqual(second.pose, (1, 2, 3, 4, 5, 6))
        self.assertEqual(self.finish()["sent"], 2)

    def test_production_client_scales_right_calibration_with_jpeg(self):
        self.start_sender(width=960, mode=3)
        with self.accept(b"IHC") as connection:
            frame = viewer.receive_frame(connection, "C")
        self.assertEqual(cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), 1).shape[:2], (540, 960))
        self.assertEqual((frame.camera["width"], frame.camera["height"], frame.camera["eye"]), (960, 540, "right"))
        np.testing.assert_allclose(frame.camera["intrinsics"], [500, 550, 480, 270])
        np.testing.assert_allclose(frame.camera["distortion"], [-.05, .01, .001, -.002, .003, .004,
                                                              .005, .006, .007, .008, .009, .01])
        self.assertEqual(frame.pose, (1, 2, 3, 4, 5, 6))
        self.assertEqual(frame.timestamp, 1234567890000)
        self.assertEqual(self.finish()["sent"], 1)

    def test_nonreading_peer_cannot_block_sender_indefinitely(self):
        self.start_sender(count=2, pressure=True)
        with self.accept():
            result = self.finish()
        self.assertEqual(result["sent"], 0)
        # Includes JPEG encoding as well as the 250 ms whole-packet network deadline.
        self.assertLess(result["maximum_ms"], 1200)


class BrowserTests(unittest.TestCase):
    def test_headless_preview_keeps_latest_frame_and_accepts_reconnect(self):
        def unused_port():
            with socket.socket() as candidate:
                candidate.bind(("127.0.0.1", 0))
                return candidate.getsockname()[1]

        tcp_port, web_port = unused_port(), unused_port()
        process = subprocess.Popen([sys.executable, str(ROOT / "tools/server.py"),
                                    "--host", "127.0.0.1", "--port", str(tcp_port),
                                    "--web-port", str(web_port), "--no-display", "--render-fps", "2"],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        base = f"http://127.0.0.1:{web_port}"

        def wait_status(predicate):
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                try:
                    with urlopen(base + "/status.json", timeout=1) as response:
                        status = json.load(response)
                    if predicate(status):
                        return status
                except (OSError, URLError):
                    pass
                time.sleep(0.05)
            self.fail("Headless preview did not reach expected state")

        try:
            wait_status(lambda s: s["received"] == 0)
            with self.assertRaises(HTTPError) as error:
                urlopen(base + "/snapshot.jpg", timeout=1)
            self.assertEqual(error.exception.code, 503)
            with urlopen(base, timeout=1) as response:
                self.assertIn("位姿解算".encode(), response.read())
            with socket.create_connection(("127.0.0.1", tcp_port), timeout=2) as connection:
                connection.sendall(b"IHV")
                connection.sendall(b"".join(packet(i) for i in range(1, 101)))
                status = wait_status(lambda s: s["received"] == 100 and s["timestamp"] == 100)
                self.assertLess(status["rendered"], 10)
                self.assertEqual(status["filtered_xyz_mm"], [4, 5, 6])
                with urlopen(base + "/snapshot.jpg", timeout=1) as response:
                    self.assertEqual(response.headers["Cache-Control"], "no-store")
                    image = cv2.imdecode(np.frombuffer(response.read(), np.uint8), 1)
                self.assertEqual(image.shape[:2], (90, 160))
                self.assertGreater(int(image.sum()), 0)  # Labels are present on the black source image.
            wait_status(lambda s: not s["connected"])
            # A partial old frame cannot contaminate the next connection.
            with socket.create_connection(("127.0.0.1", tcp_port), timeout=2) as connection:
                connection.sendall(b"IHV" + packet(999)[:20])
            with socket.create_connection(("127.0.0.1", tcp_port), timeout=2) as connection:
                connection.sendall(b"IHD" + viewer.POSE.pack(7, 8, 9, 10, 11, 12, 200))
                status = wait_status(lambda s: s["timestamp"] == 200)
                self.assertEqual(status["filtered_xyz_mm"], [10, 11, 12])
        finally:
            process.send_signal(signal.SIGINT)
            try:
                stdout, stderr = process.communicate(timeout=4)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                self.fail("Viewer did not stop promptly: " + stderr)
            self.assertEqual(process.returncode, 0, stdout + stderr)


if __name__ == "__main__":
    unittest.main()
