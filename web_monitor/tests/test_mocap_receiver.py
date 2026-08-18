from __future__ import annotations

import math
import tempfile
import sys
import time
import unittest
import io
from pathlib import Path
from unittest.mock import patch


WEB_MONITOR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEB_MONITOR))

from mocap_receiver import (  # noqa: E402
    MocapClockMapper,
    MocapPose,
    MocapReceiver,
    MocapTimeline,
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

    def test_timeline_interpolates_position_and_quaternion(self) -> None:
        start = 1_700_000_000_000_000_000
        first = make_pose(
            mocap_frame=10,
            receive_unix_ns=start,
            x=0,
            qz=0,
            qw=1,
        )
        second = make_pose(
            mocap_frame=11,
            receive_unix_ns=start + 20_000_000,
            x=20,
            qz=math.sqrt(0.5),
            qw=math.sqrt(0.5),
        )
        result = MocapTimeline([first, second]).match(
            start + 10_000_000,
            max_error_ms=12,
            interpolate=True,
        )
        self.assertEqual(result["status"], "interpolated")
        self.assertAlmostEqual(result["pose"]["position"]["x"], 10)
        self.assertAlmostEqual(result["pose"]["euler_deg"]["rz"], 45, places=5)
        self.assertEqual(result["mocap_frames"], [10, 11])

    def test_timeline_applies_offset_and_rejects_large_error(self) -> None:
        base = 1_700_000_000_000_000_000
        pose = make_pose(receive_unix_ns=base)
        matched = MocapTimeline([pose]).match(
            base + 25_000_000,
            offset_ms=25,
            max_error_ms=2,
            interpolate=False,
        )
        self.assertEqual(matched["status"], "matched")
        rejected = MocapTimeline([pose]).match(
            base + 25_000_000,
            offset_ms=0,
            max_error_ms=12,
            interpolate=False,
        )
        self.assertEqual(rejected["status"], "unmatched")

    def test_independent_sdk_clock_rate_is_mapped_to_orin_time(self) -> None:
        mapper = MocapClockMapper(
            window_seconds=60,
            minimum_span_seconds=1,
            update_every=30,
        )
        base_unix_ns = 1_700_000_000_000_000_000
        base_monotonic_ns = 10_000_000_000
        fixed_delay_ns = 40_000_000
        expected_last_ns = 0
        last_pose = None
        for index in range(701):
            sdk_ms = index * 10
            orin_elapsed_ns = int(round(sdk_ms * 1_000_896.0))
            queue_jitter_ns = (index % 5) * 100_000
            receive_monotonic_ns = (
                base_monotonic_ns
                + fixed_delay_ns
                + orin_elapsed_ns
                + queue_jitter_ns
            )
            receive_unix_ns = (
                base_unix_ns
                + fixed_delay_ns
                + orin_elapsed_ns
                + queue_jitter_ns
            )
            last_pose = make_pose(
                mocap_frame=index,
                mocap_timestamp_ms=sdk_ms,
                receive_monotonic_ns=receive_monotonic_ns,
                receive_unix_ns=receive_unix_ns,
            )
            mapper.observe(last_pose)
            expected_last_ns = base_unix_ns + fixed_delay_ns + orin_elapsed_ns

        self.assertIsNotNone(last_pose)
        self.assertTrue(mapper.model.ready)
        self.assertAlmostEqual(mapper.model.rate_ppm, 896, delta=10)
        aligned = mapper.map_pose(last_pose)
        self.assertLess(abs(aligned.timeline_unix_ns - expected_last_ns), 1_000_000)

    def test_timeline_prefers_aligned_timestamp_over_receive_timestamp(self) -> None:
        base = 1_700_000_000_000_000_000
        pose = make_pose(
            receive_unix_ns=base + 40_000_000,
            aligned_unix_ns=base,
        )
        result = MocapTimeline([pose]).match(
            base,
            max_error_ms=2,
            interpolate=False,
        )
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["mocap_receive_unix_ns"], [base + 40_000_000])
        self.assertEqual(result["mocap_aligned_unix_ns"], [base])

    def test_clock_frames_warm_model_without_visible_rigid_body(self) -> None:
        receiver = MocapReceiver(self.receiver_config(), Path("/tmp"))
        base_unix_ns = 1_700_000_000_000_000_000
        base_monotonic_ns = 10_000_000_000
        lines = ["READY\t2.5.47.54"]
        for frame in range(721):
            sdk_ms = frame * 10
            elapsed_ns = int(round(sdk_ms * 1_001_000.0))
            monotonic_ns = base_monotonic_ns + elapsed_ns
            unix_ns = base_unix_ns + elapsed_ns
            lines.append(
                f"CLOCK\t{frame}\t{sdk_ms}\t{unix_ns}\t{monotonic_ns}"
            )

        receiver._read_stdout(io.StringIO("\n".join(lines) + "\n"))
        snapshot = receiver.snapshot()

        self.assertEqual(snapshot["status"], "waiting")
        self.assertIsNone(snapshot["pose"])
        self.assertTrue(snapshot["clock_model"]["ready"])
        self.assertEqual(snapshot["clock_model"]["mode"], "sdk_affine")
        self.assertAlmostEqual(snapshot["clock_model"]["rate_ppm"], 1000, delta=5)

    def test_session_recording_is_buffered_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mocap.csv"
            receiver = MocapReceiver(
                self.receiver_config(),
                Path("/tmp"),
                session_csv_path=path,
            )
            receiver._record_pose(make_pose(), True)
            receiver.finish_recording()
            receiver._record_pose(make_pose(mocap_frame=43), True)
            rows = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 2)
        self.assertIn("receive_unix_ns", rows[0])
        self.assertIn("Tracker 3", rows[1])


if __name__ == "__main__":
    unittest.main()
