"""Protocol and C++/Python integration tests; no camera, GPU, or remote PC needed.

Run from the project root:
    python3 -m unittest discover -s tests -p test_tcp_stream.py -v
"""

import importlib.util
import json
from pathlib import Path
import select
import shlex
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("receive_stream", ROOT / "tools/receive_stream.py")
receiver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(receiver)


def packet_bytes(metadata=None, jpeg=b"\xff\xd8test\xff\xd9"):
    if metadata is None:
        metadata = {"frame_id": 1, "width": 160, "height": 120, "detections": []}
    data = json.dumps(metadata).encode()
    return receiver.HEADER.pack(b"CVIA", 1, len(data), len(jpeg)) + data + jpeg


class FragmentedSocket:
    def __init__(self, data, chunk=3):
        self.data = data
        self.chunk = chunk

    def recv_into(self, view):
        count = min(len(view), self.chunk, len(self.data))
        view[:count] = self.data[:count]
        self.data = self.data[count:]
        return count


class ProtocolTests(unittest.TestCase):
    def test_fragmented_and_coalesced_frames(self):
        stream = FragmentedSocket(packet_bytes() + packet_bytes(), chunk=3)
        for _ in range(2):
            metadata, jpeg = receiver.receive_frame(stream)
            self.assertEqual(metadata["frame_id"], 1)
            self.assertEqual(jpeg, b"\xff\xd8test\xff\xd9")
        self.assertIsNone(receiver.receive_frame(stream))

    def test_partial_eof_is_not_a_complete_frame(self):
        for size in (3, 16, 20, len(packet_bytes()) - 1):
            with self.subTest(size=size), self.assertRaises(EOFError):
                receiver.receive_frame(FragmentedSocket(packet_bytes()[:size]))

    def test_rejects_bad_headers_before_allocating_payloads(self):
        headers = [
            (b"WRNG", 1, 10, 10), (b"CVIA", 2, 10, 10),
            (b"CVIA", 1, 0, 10), (b"CVIA", 1, 10, 0),
            (b"CVIA", 1, receiver.MAX_JSON_BYTES + 1, 10),
            (b"CVIA", 1, 10, receiver.MAX_JPEG_BYTES + 1),
        ]
        for header in headers:
            with self.subTest(header=header), self.assertRaises(ValueError):
                receiver.receive_frame(FragmentedSocket(receiver.HEADER.pack(*header)))

    def test_rejects_bad_metadata_and_jpeg(self):
        for metadata in ([], {"detections": []}, {"width": -1, "height": 1, "detections": []}):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                receiver.receive_frame(FragmentedSocket(packet_bytes(metadata)))
        with self.assertRaises(ValueError):
            receiver.receive_frame(FragmentedSocket(packet_bytes(jpeg=b"not a JPEG")))


class NetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="cvia-tcp-tests-")
        cls.executable = Path(cls.temporary.name) / "sender"
        flags = shlex.split(subprocess.check_output(
            ["pkg-config", "--cflags", "--libs", "opencv4"], text=True))
        # Some Jetson packages keep an obsolete /usr/local include prefix.
        if not any(flag.startswith("-I") and (Path(flag[2:]) / "opencv2/core.hpp").exists()
                   for flag in flags):
            system_headers = Path("/usr/include/opencv4")
            if (system_headers / "opencv2/core.hpp").exists():
                flags.insert(0, "-I" + str(system_headers))
        command = ["g++", "-std=c++14", "-O1", "-pthread", "-Wall", "-Wextra",
                   "-I" + str(ROOT / "src/communication"),
                   str(ROOT / "tests/stream_sender_smoke.cpp"),
                   str(ROOT / "src/communication/frame_streamer.cpp"),
                   str(ROOT / "src/communication/client.cpp"),
                   "-o", str(cls.executable)] + flags
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError("Could not compile TCP test sender:\n" + result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def listener(self, port=0):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addCleanup(server.close)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", port))
        server.listen(2)
        server.settimeout(5)
        return server

    def sender(self, port, duration=1500, pressure=False):
        command = [str(self.executable), str(port), str(duration)]
        if pressure:
            command.append("pressure")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        def cleanup():
            if process.poll() is None:
                process.kill()
            process.communicate()

        self.addCleanup(cleanup)
        return process

    def receive(self, connection):
        connection.settimeout(2)
        return receiver.receive_frame(connection)

    def assert_sender_stopped(self, process):
        output, error = process.communicate(timeout=6)
        self.assertEqual(process.returncode, 0, output + error)
        self.assertIn("STOP_MS=", output)

    def test_cpp_sender_matches_image_and_coordinates(self):
        with self.listener() as server:
            process = self.sender(server.getsockname()[1])
            connection, _ = server.accept()
            with connection:
                metadata, jpeg = self.receive(connection)
            self.assertEqual(metadata["timestamp_ms"], 1234567890000 + metadata["frame_id"])
            self.assertEqual((metadata["width"], metadata["height"]), (160, 120))
            self.assertEqual(metadata["coordinate_system"], "original_image_pixels")
            self.assertEqual(metadata["num_keypoints"], 17)
            detection = metadata["detections"][0]
            self.assertEqual(detection["label"], 2)
            self.assertEqual(detection["bbox"], [10, 10, 110, 90])
            self.assertEqual(len(detection["keypoints"]), 17)
            self.assertAlmostEqual(detection["keypoints"][0][2], 0.95, places=5)
            # Check the annotated JPEG when Python OpenCV is installed.
            if importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"):
                import cv2
                import numpy as np
                image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(image.shape[:2], (120, 160))
                self.assertGreater(int(image[30, 40, 2]) - int(image[30, 40, 0]), 100)
            self.assert_sender_stopped(process)

    def test_reconnects_after_receiver_disconnects(self):
        with self.listener() as server:
            process = self.sender(server.getsockname()[1], duration=2500)
            connection, _ = server.accept()
            with connection:
                first, _ = self.receive(connection)
                connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            connection, _ = server.accept()
            with connection:
                second, _ = self.receive(connection)
            self.assertGreater(second["frame_id"], first["frame_id"])
            self.assert_sender_stopped(process)

    def test_late_receiver_gets_latest_pending_frame(self):
        reservation = self.listener()
        port = reservation.getsockname()[1]
        reservation.close()
        process = self.sender(port, duration=2000)
        time.sleep(0.25)
        with self.listener(port) as server:
            connection, _ = server.accept()
            with connection:
                metadata, _ = self.receive(connection)
            self.assertGreater(metadata["frame_id"], 20)
            self.assert_sender_stopped(process)

    def test_slow_receiver_cannot_hold_up_shutdown(self):
        with self.listener() as server:
            started = time.monotonic()
            process = self.sender(server.getsockname()[1], duration=800, pressure=True)
            connection, _ = server.accept()
            with connection:
                connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
                self.assert_sender_stopped(process)
            self.assertLess(time.monotonic() - started, 4)

    def test_headless_receiver_saves_both_payloads(self):
        reservation = self.listener()
        port = reservation.getsockname()[1]
        reservation.close()
        root = Path(self.temporary.name)
        coordinates, image = root / "detections.jsonl", root / "latest.jpg"
        command = [sys.executable, str(ROOT / "tools/receive_stream.py"), "--host", "127.0.0.1",
                   "--port", str(port), "--headless", "--max-frames", "3",
                   "--jsonl", str(coordinates), "--save-latest", str(image)]
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as pc:
            try:
                self.assertTrue(select.select([pc.stdout], [], [], 5)[0], "Receiver did not start")
                self.assertIn("Listening", pc.stdout.readline())
                process = self.sender(port)
                output, error = pc.communicate(timeout=5)
                self.assertEqual(pc.returncode, 0, output + error)
                self.assertIn("Received 3 frames", output)
                records = [json.loads(line) for line in coordinates.read_text().splitlines()]
                self.assertEqual(len(records), 3)
                self.assertTrue(all(len(record["detections"][0]["keypoints"]) == 17 for record in records))
                self.assertTrue(image.read_bytes().startswith(b"\xff\xd8"))
                self.assert_sender_stopped(process)
            finally:
                if pc.poll() is None:
                    pc.kill()
                    pc.communicate()


if __name__ == "__main__":
    unittest.main()
