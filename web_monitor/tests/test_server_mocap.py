from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


WEB_MONITOR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEB_MONITOR))

from server import (  # noqa: E402
    build_offline_report,
    comparison_snapshot,
    estimate_offline_offset,
    latest_detection_pose,
    tracker_selector,
    validate,
)


class MocapTargetSelectorTests(unittest.TestCase):
    def test_reprojection_exposes_capture_time_and_pipeline_latency(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            (runtime / "reprojection.json").write_text(json.dumps({
                "timestamp": 1700000000000,
                "capture_timestamp_ns": 1700000000000123456,
                "publish_timestamp_ns": 1700000000030123456,
                "timestamp_source": "v4l2_driver",
                "timestamp_point": "end_of_frame_or_unknown",
                "coordinate_frame": "mocap",
                "valid": True,
                "pose": [1, 2, 3, 4, 5, 6],
            }), encoding="utf-8")
            with patch("server.RUNTIME_PATH", runtime):
                result = latest_detection_pose()

        self.assertEqual(result["capture_timestamp_ns"], 1700000000000123456)
        self.assertEqual(result["pipeline_latency_ms"], 30.0)
        self.assertEqual(result["timestamp_source"], "v4l2_driver")
        self.assertEqual(result["coordinate_frame"], "mocap")

    def test_name_selector_preserves_spaces(self) -> None:
        self.assertEqual(
            tracker_selector("name", "  Tracker 4  "),
            "name:Tracker 4",
        )

    def test_id_selector_accepts_integer_or_canonical_text(self) -> None:
        self.assertEqual(tracker_selector("id", 0), "id:0")
        self.assertEqual(tracker_selector("id", "17"), "id:17")

    def test_invalid_target_values_are_rejected(self) -> None:
        for mode, value in (
            ("name", ""),
            ("name", "Tracker\n4"),
            ("id", -1),
            ("id", "1.5"),
            ("id", True),
            ("other", "Tracker4"),
        ):
            with self.subTest(mode=mode, value=value):
                with self.assertRaises(ValueError):
                    tracker_selector(mode, value)

    def test_sync_defaults_are_added_to_legacy_config(self) -> None:
        import json

        config = json.loads((WEB_MONITOR / "config.json").read_text(encoding="utf-8"))
        config.pop("sync")
        validated = validate(config)
        self.assertTrue(validated["sync"]["enabled"])
        self.assertEqual(validated["sync"]["max_error_ms"], 12.0)
        self.assertEqual(validated["sync"]["auto_offset_search_ms"], 1000)

    def test_comparison_uses_visual_timestamp_for_realtime_match(self) -> None:
        class Receiver:
            def snapshot(self):
                return {"status": "live", "pose": {"receive_unix_ns": 1}}

            def match_at_unix_ns(self, target, **options):
                return {
                    "status": "matched",
                    "pose": {"frame": 7},
                    "target": target,
                    "options": options,
                }

        detection = {
            "available": True,
            "source": "reprojection",
            "timestamp": 1234,
            "updated_unix_ns": 2,
            "pose": {"x": 1},
        }
        with patch("server.MOCAP_RECEIVER", Receiver()), patch(
            "server.latest_detection_pose", return_value=detection
        ), patch(
            "server.load_config",
            return_value={"sync": {"enabled": True, "offset_ms": 3, "max_error_ms": 9, "interpolate": False}},
        ):
            result = comparison_snapshot()
        self.assertEqual(result["synchronization"]["target"], 1_234_000_000)
        self.assertEqual(result["synchronization"]["options"]["offset_ms"], 3)

    def test_reliable_offline_estimate_is_saved_as_sync_offset(self) -> None:
        import json

        config = json.loads((WEB_MONITOR / "config.json").read_text(encoding="utf-8"))
        estimate = {
            "offset_ms": 37.0,
            "correlation": 0.91,
            "peak_margin": 0.12,
            "confidence": "high",
            "reliable": True,
            "message": "互相关峰清晰",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detection_path = root / "pnp.csv"
            mocap_path = root / "mocap.csv"
            detection_path.touch()
            mocap_path.touch()
            with patch("server.inference_process", return_value=(False, None)), patch(
                "server.SESSION_CSV_PATH", detection_path
            ), patch("server.MOCAP_SESSION_CSV_PATH", mocap_path), patch(
                "server.SYNCED_SESSION_CSV_PATH", root / "synced.csv"
            ), patch("server.OFFLINE_CAMERA_REPORT_PATH", root / "camera.svg"), patch(
                "server.OFFLINE_MOCAP_REPORT_PATH", root / "mocap-report.svg"
            ), patch("server.OFFLINE_REPORT_PATH", root / "report.svg"), patch(
                "server.OFFLINE_SUMMARY_PATH", root / "summary.json"
            ), patch("server.OFFSET_ESTIMATE_PATH", root / "estimate.json"), patch(
                "server.MOCAP_RECEIVER", None
            ), patch("server.load_config", return_value=config), patch(
                "server.load_detection_poses", return_value=[]
            ), patch("server.load_mocap_poses", return_value=[]), patch(
                "server.estimate_time_offset", return_value=dict(estimate)
            ), patch("server.atomic_save") as atomic_save:
                result = estimate_offline_offset()

            saved_estimate = json.loads(
                (root / "estimate.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result["config_offset_ms"], 37.0)
        self.assertTrue(result["offset_estimation"]["applied"])
        self.assertEqual(saved_estimate["offset_ms"], 37.0)
        self.assertEqual(atomic_save.call_args.args[0]["sync"]["offset_ms"], 37.0)

    def test_offline_report_build_writes_and_exposes_three_canvases(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detection_path = root / "pnp.csv"
            mocap_path = root / "mocap.csv"
            detection_path.touch()
            mocap_path.touch()
            summary_path = root / "summary.json"
            synced_path = root / "synced.csv"
            camera_path = root / "camera.svg"
            mocap_report_path = root / "mocap.svg"
            composite_path = root / "composite.svg"
            generated = (
                {"matched_samples": 1, "detection_samples": 1, "coverage_percent": 100},
                "header\nrow\n",
                {
                    "camera": "<svg><title>camera</title></svg>",
                    "mocap": "<svg><title>mocap</title></svg>",
                    "composite": "<svg><title>composite</title></svg>",
                },
            )
            with patch("server.inference_process", return_value=(False, None)), patch(
                "server.SESSION_CSV_PATH", detection_path
            ), patch("server.MOCAP_SESSION_CSV_PATH", mocap_path), patch(
                "server.SYNCED_SESSION_CSV_PATH", synced_path
            ), patch("server.OFFLINE_CAMERA_REPORT_PATH", camera_path), patch(
                "server.OFFLINE_MOCAP_REPORT_PATH", mocap_report_path
            ), patch("server.OFFLINE_REPORT_PATH", composite_path), patch(
                "server.OFFLINE_SUMMARY_PATH", summary_path
            ), patch("server.OFFSET_ESTIMATE_PATH", root / "estimate.json"), patch(
                "server.MOCAP_RECEIVER", None
            ), patch(
                "server.load_config",
                return_value={"sync": {"offset_ms": 0, "max_error_ms": 10, "interpolate": True}},
            ), patch("server.generate_offline_reports", return_value=generated):
                result = build_offline_report()

            saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            camera_svg = camera_path.read_text(encoding="utf-8")
            mocap_svg = mocap_report_path.read_text(encoding="utf-8")
            composite_svg = composite_path.read_text(encoding="utf-8")

        self.assertTrue(result["available"])
        self.assertEqual(
            set(result["report_urls"]), {"camera", "mocap", "composite"}
        )
        self.assertEqual(camera_svg, generated[2]["camera"])
        self.assertEqual(mocap_svg, generated[2]["mocap"])
        self.assertEqual(composite_svg, generated[2]["composite"])
        self.assertIn("generated_unix_ns", saved_summary)


if __name__ == "__main__":
    unittest.main()
