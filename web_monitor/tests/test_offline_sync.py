from __future__ import annotations

import csv
import math
import struct
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
    continuous_euler_rows,
    estimate_time_offset,
    generate_offline_report,
    generate_offline_reports,
    interpolate_detection_pose_outliers,
    load_detection_poses,
    load_mocap_poses,
    resample_detection_poses,
    stabilize_detection_poses,
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
    def test_resamples_sixty_hz_camera_interval_to_three_hundred_hz(self) -> None:
        source = [
            DetectionPose(
                timestamp_ms=1000,
                capture_timestamp_ns=1_000_000_000,
                x=0, y=10, z=500, rx=0, ry=0, rz=0,
            ),
            DetectionPose(
                timestamp_ms=1017,
                capture_timestamp_ns=1_016_666_667,
                x=10, y=20, z=510, rx=0, ry=90, rz=0,
            ),
        ]

        resampled = resample_detection_poses(source, target_hz=300)

        self.assertEqual(len(resampled), 6)
        self.assertEqual(resampled[0], source[0])
        self.assertEqual(resampled[-1], source[-1])
        self.assertEqual(
            [pose.capture_timestamp_ns for pose in resampled],
            [
                1_000_000_000,
                1_003_333_333,
                1_006_666_667,
                1_010_000_000,
                1_013_333_333,
                1_016_666_667,
            ],
        )
        self.assertAlmostEqual(resampled[3].x, 6.0, places=5)
        self.assertAlmostEqual(resampled[3].y, 16.0, places=5)
        self.assertAlmostEqual(resampled[3].ry, 54.0, places=4)
        self.assertTrue(all(pose.interpolated for pose in resampled[1:-1]))

        one_second_at_sixty_hz = [
            DetectionPose(
                timestamp_ms=round(index * 1000 / 60),
                capture_timestamp_ns=2_000_000_000 + round(index * 1_000_000_000 / 60),
                x=float(index), y=0, z=500, rx=0, ry=0, rz=float(index),
            )
            for index in range(61)
        ]
        one_second_at_three_hundred_hz = resample_detection_poses(
            one_second_at_sixty_hz,
            target_hz=300,
        )
        self.assertEqual(len(one_second_at_three_hundred_hz), 301)
        intervals = {
            current.capture_timestamp_ns - previous.capture_timestamp_ns
            for previous, current in zip(
                one_second_at_three_hundred_hz,
                one_second_at_three_hundred_hz[1:],
            )
        }
        self.assertEqual(intervals, {3_333_333, 3_333_334})

    def test_rejected_pose_runs_are_interpolated_between_trusted_neighbours(self) -> None:
        def pose(timestamp_ms: int, x: float, ry: float) -> DetectionPose:
            return DetectionPose(
                timestamp_ms=timestamp_ms,
                capture_timestamp_ns=timestamp_ms * 1_000_000,
                x=x, y=x * 2, z=500 + x, rx=0, ry=ry, rz=0,
            )

        repaired, stats = interpolate_detection_pose_outliers([
            pose(1000, 0, 0),
            pose(2000, 500, 100),
            pose(3000, 10, 10),
            pose(4000, 600, 120),
        ])

        self.assertEqual([item.timestamp_ms for item in repaired], [1000, 2000, 3000])
        interpolated = repaired[1]
        self.assertTrue(interpolated.interpolated)
        self.assertAlmostEqual(interpolated.x, 5.0)
        self.assertAlmostEqual(interpolated.y, 10.0)
        self.assertAlmostEqual(interpolated.z, 505.0)
        self.assertAlmostEqual(interpolated.ry, 5.0, places=5)
        self.assertEqual(stats, {"interpolated": 1, "dropped": 1, "flagged": 2})

    def test_sparse_twelve_frame_camera_report_interpolates_across_capture_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            detection_path = Path(directory) / "pnp.csv"
            with detection_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("timestamp", "x", "y", "z", "rx", "ry", "rz"))
                # Twelve deliberately sparse captures over 55 seconds. The
                # camera report must connect them as piecewise-linear visual
                # interpolation instead of rendering isolated segments.
                for index in range(12):
                    writer.writerow((
                        1_000 + index * 5_000,
                        index * 2,
                        index * 3,
                        500 + index,
                        index * 0.5,
                        index * 0.75,
                        index,
                    ))

            summary, _csv_payload, reports = generate_offline_reports(
                detection_path,
                None,
                offset_ms=0,
                max_error_ms=8,
                interpolate=True,
            )

        self.assertEqual(summary["raw_detection_samples"], 12)
        self.assertEqual(summary["detection_samples"], 12)
        # One continuous visual polyline for each of the six pose axes.
        self.assertEqual(reports["camera"].count('<polyline class="vision"'), 6)
        self.assertNotIn('<polyline class="mocap"', reports["camera"])

    def test_equivalent_euler_branches_are_made_continuous_for_plotting(self) -> None:
        rows = [
            {
                "visual_capture_timestamp_ns": 1_000_000_000,
                "visual_rx": -8, "visual_ry": 88, "visual_rz": -9,
            },
            {
                "visual_capture_timestamp_ns": 1_010_000_000,
                "visual_rx": 170, "visual_ry": 89, "visual_rz": 170,
            },
            {
                "visual_capture_timestamp_ns": 1_020_000_000,
                "visual_rx": -179, "visual_ry": 88, "visual_rz": 179,
            },
        ]

        continuous = continuous_euler_rows(rows)

        self.assertAlmostEqual(continuous[1]["visual_rx"], -10)
        self.assertAlmostEqual(continuous[1]["visual_ry"], 91)
        self.assertAlmostEqual(continuous[1]["visual_rz"], -10)
        self.assertLess(
            abs(continuous[2]["visual_rx"] - continuous[1]["visual_rx"]),
            180,
        )

    def test_pose_stabilizer_rejects_spikes_and_reacquires_persistent_pose(self) -> None:
        def pose(timestamp_ms: int, x: float, ry: float) -> DetectionPose:
            return DetectionPose(
                timestamp_ms=timestamp_ms,
                capture_timestamp_ns=timestamp_ms * 1_000_000,
                x=x, y=0, z=500, rx=0, ry=ry, rz=0,
            )

        isolated = [
            pose(1000, 0, 0),
            pose(2000, 500, 100),
            pose(3000, 5, 1),
        ]
        stable, rejected = stabilize_detection_poses(isolated)
        self.assertEqual([item.timestamp_ms for item in stable], [1000, 3000])
        self.assertEqual(rejected, 1)

        relocated = [
            pose(1000, 0, 0),
            pose(2000, 500, 100),
            pose(3000, 510, 105),
            pose(4000, 520, 110),
        ]
        stable, rejected = stabilize_detection_poses(relocated)
        self.assertEqual([item.timestamp_ms for item in stable], [1000, 4000])
        self.assertEqual(rejected, 2)

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

    def test_generates_three_real_rgb_png_reports(self) -> None:
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
            summary, _csv_payload, reports = generate_offline_reports(
                detection_path,
                mocap_path,
                offset_ms=0,
                max_error_ms=8,
                interpolate=True,
                report_format="png",
            )

        self.assertEqual(summary["matched_samples"], 2)
        self.assertEqual(summary["camera_source_samples"], 2)
        self.assertEqual(summary["camera_plot_hz"], 300.0)
        self.assertEqual(summary["camera_plot_samples"], 4)
        self.assertEqual(summary["camera_plot_interpolated_samples"], 2)
        self.assertEqual(set(reports), {"camera", "mocap", "composite"})
        for png in reports.values():
            self.assertIsInstance(png, bytes)
            self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
            width, height, bit_depth, color_type = struct.unpack(
                "!IIBB", png[16:26]
            )
            self.assertEqual((width, height), (1200, 990))
            self.assertEqual(bit_depth, 8)
            self.assertEqual(color_type, 2)

    def test_generates_camera_png_when_mocap_file_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            detection_path = Path(directory) / "pnp.csv"
            with detection_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("timestamp", "x", "y", "z", "rx", "ry", "rz"))
                writer.writerow((1000, 1, 2, 3, 4, 5, 6))
                writer.writerow((1010, 2, 3, 4, 5, 6, 7))

            summary, synchronized_csv, reports = generate_offline_reports(
                detection_path,
                None,
                offset_ms=0,
                max_error_ms=8,
                interpolate=True,
                report_format="png",
            )

        self.assertEqual(summary["detection_samples"], 2)
        self.assertEqual(summary["mocap_samples"], 0)
        self.assertEqual(summary["matched_samples"], 0)
        self.assertEqual(summary["coverage_percent"], 0)
        self.assertTrue(synchronized_csv.startswith("coordinate_frame,"))
        camera_png = reports["camera"]
        self.assertTrue(camera_png.startswith(b"\x89PNG\r\n\x1a\n"))
        width, height, bit_depth, color_type = struct.unpack("!IIBB", camera_png[16:26])
        self.assertEqual((width, height, bit_depth, color_type), (1200, 990, 8, 2))

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
        self.assertIn('<polyline class="mocap"', reports["mocap"])
        self.assertNotIn('class="mocap" points="96.00,', reports["mocap"])
        self.assertIn("不施加运动方向约束", reports["camera"])
        self.assertIn('class="vision" points="96.00,', reports["composite"])
        self.assertIn('<polyline class="mocap"', reports["composite"])
        self.assertNotIn('class="mocap" points="96.00,', reports["composite"])
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
