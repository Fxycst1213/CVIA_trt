from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


WEB_MONITOR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEB_MONITOR))

from mocap_receiver import MocapPose  # noqa: E402
from offline_sync import (  # noqa: E402
    DetectionPose,
    estimate_time_offset,
    generate_offline_report,
    generate_offline_reports,
    load_detection_poses,
    load_mocap_poses,
)


def quaternion_from_euler(rx: float, ry: float, rz: float) -> tuple[float, float, float, float]:
    roll, pitch, yaw = (math.radians(value) / 2 for value in (rx, ry, rz))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def synthetic_motion(offset_ms: int) -> tuple[list[DetectionPose], list[MocapPose]]:
    base_ms = 1_700_000_000_000
    detections = []
    mocap = []
    for frame in range(801):
        elapsed_ms = frame * 10
        x = 80 * math.sin(elapsed_ms / 430) + 25 * math.sin(elapsed_ms / 97) + 0.0008 * elapsed_ms ** 2
        y = 50 * math.cos(elapsed_ms / 311) + 15 * math.sin(elapsed_ms / 71)
        z = 20 * math.sin(elapsed_ms / 179) + 0.004 * elapsed_ms
        rx = 7 * math.sin(elapsed_ms / 283)
        ry = 10 * math.sin(elapsed_ms / 347)
        rz = 35 * math.sin(elapsed_ms / 521) + 8 * math.sin(elapsed_ms / 83)
        detections.append(DetectionPose(base_ms + elapsed_ms, x, y, z, rx, ry, rz))
        qx, qy, qz, qw = quaternion_from_euler(rx, ry, rz)
        mocap.append(MocapPose(
            selector="test",
            tracker_id=1,
            tracker_name="test",
            mocap_frame=frame,
            mocap_timestamp_ms=elapsed_ms,
            receive_unix_ns=(base_ms + elapsed_ms - offset_ms) * 1_000_000,
            receive_monotonic_ns=elapsed_ms * 1_000_000,
            x=x * 2.5 + 1000,
            y=y * 2.5 - 200,
            z=z * 2.5 + 50,
            qx=qx,
            qy=qy,
            qz=qz,
            qw=qw,
            mean_error=0.01,
            tracking_params=0,
        ))
    return detections, mocap


class OfflineSyncTests(unittest.TestCase):
    def test_short_pose_interval_uses_persisted_prewarmed_clock_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mocap.csv"
            base_ns = 1_700_000_000_000_000_000
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow((
                    "receive_unix_ns", "receive_monotonic_ns",
                    "mocap_timestamp_ms", "aligned_unix_ns",
                    "clock_rate_ppm", "clock_model_ready", "frame",
                    "tracker_id", "tracker_name", "valid", "x", "y", "z",
                    "qx", "qy", "qz", "qw", "mean_error", "tracking_params",
                ))
                for frame in range(30):
                    sdk_ms = frame * 10
                    aligned_ns = base_ns + int(round(sdk_ms * 1_001_000.0))
                    writer.writerow((
                        base_ns + sdk_ms * 1_000_000 + 40_000_000,
                        10_000_000_000 + sdk_ms * 1_000_000,
                        sdk_ms, aligned_ns, 1000.0, 1, frame, 3, "Tracker3", 1,
                        frame, 0, 0, 0, 0, 0, 1, 0.01, 0,
                    ))

            poses = load_mocap_poses(path)

        self.assertEqual(len(poses), 30)
        self.assertEqual(poses[0].timeline_unix_ns, base_ns)
        self.assertEqual(
            poses[-1].timeline_unix_ns - poses[0].timeline_unix_ns,
            int(round(290 * 1_001_000.0)),
        )

    def test_prefers_nanosecond_capture_timestamp_and_reads_pipeline_latency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pnp.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow((
                    "timestamp", "capture_timestamp_ns", "publish_timestamp_ns",
                    "x", "y", "z", "rx", "ry", "rz",
                ))
                writer.writerow((
                    1700000000000,
                    1700000000000123456,
                    1700000000031123456,
                    1, 2, 3, 4, 5, 6,
                ))

            poses = load_detection_poses(path)

        self.assertEqual(len(poses), 1)
        self.assertEqual(poses[0].effective_timestamp_ns, 1700000000000123456)
        self.assertEqual(poses[0].publish_timestamp_ns, 1700000000031123456)

    def test_generates_synchronized_csv_and_six_axis_svg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detection_path = root / "pnp.csv"
            mocap_path = root / "mocap.csv"
            with detection_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("timestamp", "x", "y", "z", "rx", "ry", "rz"))
                writer.writerow((1000, 1, 2, 3, 4, 5, 6))
                writer.writerow((1010, 2, 3, 4, 5, 6, 7))
            with mocap_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow((
                    "receive_unix_ns", "receive_monotonic_ns", "mocap_timestamp_ms",
                    "frame", "tracker_id", "tracker_name", "valid", "x", "y", "z",
                    "qx", "qy", "qz", "qw", "mean_error", "tracking_params",
                ))
                writer.writerow((995_000_000, 1, 995, 1, 3, "Tracker 3", 1, 0, 1, 2, 0, 0, 0, 1, 0.01, 0))
                writer.writerow((1005_000_000, 2, 1005, 2, 3, "Tracker 3", 1, 2, 3, 4, 0, 0, 0, 1, 0.01, 0))
                writer.writerow((1015_000_000, 3, 1015, 3, 3, "Tracker 3", 1, 4, 5, 6, 0, 0, 0, 1, 0.01, 0))

            summary, synchronized_csv, reports = generate_offline_reports(
                detection_path,
                mocap_path,
                offset_ms=0,
                max_error_ms=8,
                interpolate=True,
                offset_estimation={
                    "offset_ms": 0,
                    "confidence": "high",
                    "correlation": 0.91,
                    "reliable": True,
                },
            )

        self.assertEqual(summary["matched_samples"], 2)
        self.assertEqual(summary["interpolated_matches"], 2)
        self.assertEqual(summary["coverage_percent"], 100)
        self.assertIn("interpolated", synchronized_csv)
        self.assertEqual(set(reports), {"camera", "mocap", "composite"})
        self.assertEqual(summary["coordinate_frame"], "mocap")
        self.assertEqual(summary["camera_pose_frame"], "mocap_via_T_M_C")
        self.assertEqual(summary["mocap_pose_frame"], "mocap_native")
        self.assertIn("CAMERA-DERIVED POSE / MOCAP FRAME", reports["camera"])
        self.assertIn("MOCAP-NATIVE POSE / MOCAP FRAME", reports["mocap"])
        self.assertIn("TEMPORAL COMPOSITE / MOCAP FRAME", reports["composite"])
        synchronized_records = list(csv.DictReader(synchronized_csv.splitlines()))
        self.assertTrue(synchronized_records)
        self.assertTrue(all(row["coordinate_frame"] == "mocap" for row in synchronized_records))
        self.assertNotIn('<polyline class="mocap"', reports["camera"])
        self.assertNotIn('<polyline class="vision"', reports["mocap"])
        self.assertIn('<polyline class="vision"', reports["composite"])
        self.assertIn('<polyline class="mocap"', reports["composite"])
        for svg in reports.values():
            self.assertIn("auto offset +0.0 ms", svg)
            self.assertIn("MOCAP FRAME", svg)
            self.assertIn("mocap frame / mm", svg)
            self.assertIn("mocap frame / deg", svg)
            ET.fromstring(svg)
        for axis in (">X<", ">Y<", ">Z<", ">Rx<", ">Ry<", ">Rz<"):
            for svg in reports.values():
                self.assertIn(axis, svg)

    def test_plot_starts_with_first_visual_pose_before_mocap_matches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detection_path = root / "pnp.csv"
            mocap_path = root / "mocap.csv"
            with detection_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("timestamp", "x", "y", "z", "rx", "ry", "rz"))
                # Deliberately vary every axis: the report start rule must not
                # encode a free-fall direction or assume X/Y are stationary.
                writer.writerow((1000, 10, 3, 100, 1, 2, 3))
                writer.writerow((1010, 20, 8, 80, 2, 4, 6))
                writer.writerow((1020, 40, 13, 45, 4, 8, 12))
            with mocap_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow((
                    "receive_unix_ns", "receive_monotonic_ns", "mocap_timestamp_ms",
                    "frame", "tracker_id", "tracker_name", "valid", "x", "y", "z",
                    "qx", "qy", "qz", "qw", "mean_error", "tracking_params",
                ))
                writer.writerow((1018_000_000, 1, 1018, 1, 3, "Tracker 3", 1, 1, 2, 3, 0, 0, 0, 1, 0.01, 0))
                writer.writerow((1022_000_000, 2, 1022, 2, 3, "Tracker 3", 1, 5, 6, 7, 0, 0, 0, 1, 0.01, 0))

            summary, synchronized_csv, reports = generate_offline_reports(
                detection_path,
                mocap_path,
                offset_ms=0,
                max_error_ms=3,
                interpolate=True,
            )

        records = list(csv.DictReader(synchronized_csv.splitlines()))
        self.assertEqual(summary["detection_samples"], 3)
        self.assertEqual(summary["matched_samples"], 1)
        self.assertEqual(summary["leading_unmatched_samples"], 2)
        self.assertEqual(summary["plot_start_reason"], "first_valid_visual_pose")
        self.assertEqual(summary["start_capture_timestamp_ns"], 1_000_000_000)
        self.assertEqual(summary["first_matched_capture_timestamp_ns"], 1_020_000_000)
        # Preserve the existing synchronized CSV contract: it contains only
        # successful pairs even though the SVG retains leading visual-only rows.
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["sync_method"], "interpolated")
        self.assertIn('class="vision" points="96.00,', reports["camera"])
        self.assertIn('class="mocap" points="560.00,', reports["mocap"])
        self.assertIn("不施加运动方向约束", reports["camera"])
        self.assertIn('class="vision" points="96.00,', reports["composite"])
        self.assertIn('class="mocap" points="560.00,', reports["composite"])
        for svg in reports.values():
            ET.fromstring(svg)

    def test_legacy_report_api_returns_composite_canvas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detection_path = root / "pnp.csv"
            mocap_path = root / "mocap.csv"
            detection_path.write_text(
                "timestamp,x,y,z,rx,ry,rz\n1000,1,2,3,4,5,6\n",
                encoding="utf-8",
            )
            mocap_path.write_text(
                "receive_unix_ns,receive_monotonic_ns,mocap_timestamp_ms,frame,tracker_id,tracker_name,valid,x,y,z,qx,qy,qz,qw,mean_error,tracking_params\n"
                "1000000000,1,1000,1,3,Tracker3,1,1,2,3,0,0,0,1,0.01,0\n",
                encoding="utf-8",
            )

            _summary, _csv_payload, svg = generate_offline_report(
                detection_path,
                mocap_path,
                offset_ms=0,
                max_error_ms=3,
                interpolate=True,
            )

        self.assertIn("TEMPORAL COMPOSITE / MOCAP FRAME", svg)
        self.assertIn('<polyline class="vision"', svg)
        self.assertIn('<polyline class="mocap"', svg)

    def test_rejects_explicit_non_mocap_visual_pose_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera_frame_pnp.csv"
            path.write_text(
                "timestamp,x,y,z,rx,ry,rz,coordinate_frame\n"
                "1000,1,2,3,4,5,6,camera\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "不是动捕坐标系"):
                load_detection_poses(path)

    def test_estimates_known_offset_from_motion_cross_correlation(self) -> None:
        detections, mocap = synthetic_motion(offset_ms=37)

        result = estimate_time_offset(detections, mocap, max_offset_ms=250)

        self.assertTrue(result["reliable"])
        self.assertEqual(result["confidence"], "high")
        self.assertAlmostEqual(result["offset_ms"], 37, delta=1)
        self.assertGreater(result["correlation"], 0.9)
        self.assertFalse(result["at_search_boundary"])

    def test_boundary_peak_is_not_automatically_trusted(self) -> None:
        detections, mocap = synthetic_motion(offset_ms=200)

        result = estimate_time_offset(detections, mocap, max_offset_ms=200)

        self.assertAlmostEqual(result["offset_ms"], 200, delta=1)
        self.assertTrue(result["at_search_boundary"])
        self.assertFalse(result["reliable"])
        self.assertEqual(result["confidence"], "low")

    def test_stationary_series_is_rejected(self) -> None:
        base_ms = 1_700_000_000_000
        detections = [
            DetectionPose(base_ms + index * 10, 1, 2, 3, 0, 0, 0)
            for index in range(100)
        ]
        mocap = [
            MocapPose(
                selector="test", tracker_id=1, tracker_name="test",
                mocap_frame=index, mocap_timestamp_ms=index * 10,
                receive_unix_ns=(base_ms + index * 10) * 1_000_000,
                receive_monotonic_ns=index * 10_000_000,
                x=10, y=20, z=30, qx=0, qy=0, qz=0, qw=1,
                mean_error=0.01, tracking_params=0,
            )
            for index in range(100)
        ]

        with self.assertRaisesRegex(ValueError, "运动变化不足"):
            estimate_time_offset(detections, mocap, max_offset_ms=200)


if __name__ == "__main__":
    unittest.main()
