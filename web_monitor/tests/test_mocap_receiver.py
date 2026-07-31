from __future__ import annotations

import math
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch


WEB_MONITOR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEB_MONITOR))

from mocap_receiver import (  # noqa: E402
    MocapPose,
    MocapReceiver,
    parse_pose_line,
    pose_validity,
    quaternion_to_euler_xyz_degrees,
)


def make_pose(**changes) -> MocapPose:
    values = {
        "selector": "name:Tracker 3",
        "tracker_id": 3,
        "tracker_name": "Tracker 3",
        "mocap_frame": 42,
        "mocap_timestamp_ms": 123456,
        "receive_unix_ns": time.time_ns(),
        "receive_monotonic_ns": time.monotonic_ns(),
        "x": 1.25,
        "y": 2.5,
        "z": 3.75,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
        "mean_error": 0.02,
        "tracking_params": 0,
    }
    values.update(changes)
    return MocapPose(**values)


class MocapReceiverTests(unittest.TestCase):
    def receiver_config(self) -> dict:
        return {
            "enabled": True,
            "server": "10.1.1.198",
            "tracker": "name:Tracker 3",
            "stale_ms": 500,
            "retry_seconds": 5,
        }

    def test_external_bridge_environment_override(self) -> None:
        with patch.dict(
            "os.environ",
            {"CVIA_MOCAP_BRIDGE": "/opt/nokov/bin/MocapBridge"},
        ):
            receiver = MocapReceiver(self.receiver_config(), Path("/tmp"))
        self.assertEqual(
            receiver.bridge_path,
            Path("/opt/nokov/bin/MocapBridge"),
        )

    def test_parse_bridge_pose_and_percent_encoded_names(self) -> None:
        pose = parse_pose_line(
            "POSE\tname%3ATracker%203\t3\tTracker%203\t42\t123456"
            "\t1700000000000000000\t9000000000\t1.25\t2.5\t3.75"
            "\t0\t0\t0\t1\t0.02\t0"
        )
        self.assertEqual(pose.selector, "name:Tracker 3")
        self.assertEqual(pose.tracker_name, "Tracker 3")
        self.assertEqual(pose.mocap_frame, 42)

    def test_params_zero_does_not_discard_numerically_valid_pose(self) -> None:
        valid, issue = pose_validity(make_pose(tracking_params=0))
        self.assertTrue(valid)
        self.assertEqual(issue, "")

    def test_sdk_sentinel_is_rejected(self) -> None:
        valid, issue = pose_validity(make_pose(x=9_999_999))
        self.assertFalse(valid)
        self.assertIn("9999999", issue)

    def test_quaternion_is_normalized_before_euler_conversion(self) -> None:
        half_turn = math.sqrt(0.5) * 5
        rx, ry, rz = quaternion_to_euler_xyz_degrees(
            0, 0, half_turn, half_turn
        )
        self.assertAlmostEqual(rx, 0, places=6)
        self.assertAlmostEqual(ry, 0, places=6)
        self.assertAlmostEqual(rz, 90, places=6)

    def test_live_snapshot_contains_xyz_and_euler_angles(self) -> None:
        receiver = MocapReceiver(
            self.receiver_config(),
            Path("/tmp"),
        )
        pose = make_pose(qz=math.sqrt(0.5), qw=math.sqrt(0.5))
        with receiver._lock:
            receiver._connection = "ready"
            receiver._sdk_version = "2.5.47.54"
            receiver._latest_pose = pose
            receiver._latest_valid = True
        snapshot = receiver.snapshot()
        self.assertEqual(snapshot["status"], "live")
        self.assertEqual(snapshot["pose"]["position"]["x"], 1.25)
        self.assertAlmostEqual(snapshot["pose"]["euler_deg"]["rz"], 90)


if __name__ == "__main__":
    unittest.main()
