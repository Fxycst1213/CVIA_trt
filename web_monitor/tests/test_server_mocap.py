from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


WEB_MONITOR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEB_MONITOR))

from server import (  # noqa: E402
    build_offline_report,
    changed_config_paths,
    comparison_snapshot,
    config_apply_metadata,
    estimate_offline_offset,
    latest_detection_pose,
    runtime_config_snapshot,
    tracker_selector,
    validate,
)
import server as server_module  # noqa: E402


class MocapTargetSelectorTests(unittest.TestCase):
    def test_start_inference_uses_latest_saved_config_and_writes_pid(self) -> None:
        class FakeProcess:
            pid = 43210
            returncode = None

            def poll(self):
                return None

            def terminate(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            runtime.mkdir()
            config_path = root / "config.json"
            config_path.write_text(
                (WEB_MONITOR / "config.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            binary_path = root / "trt"
            binary_path.write_bytes(b"test executable")
            binary_path.chmod(0o755)
            fake_process = FakeProcess()
            path_patches = {
                "RUNTIME_PATH": runtime,
                "CONFIG_PATH": config_path,
                "TRT_BINARY_PATH": binary_path,
                "PID_PATH": runtime / "trt.pid",
                "SESSION_CSV_PATH": runtime / "pnp_session.csv",
                "PREVIEW_PAIR_PATH": runtime / "preview_pair.bin",
                "RUNTIME_CONFIG_STATUS_PATH": runtime / "runtime_config_status.json",
                "CONFIG_APPLY_META_PATH": runtime / "config_apply_meta.json",
                "MOCAP_SESSION_CSV_PATH": runtime / "mocap_session.csv",
                "SYNCED_SESSION_CSV_PATH": runtime / "sync.csv",
                "OFFLINE_CAMERA_REPORT_PATH": runtime / "camera.png",
                "OFFLINE_MOCAP_REPORT_PATH": runtime / "mocap.png",
                "OFFLINE_REPORT_PATH": runtime / "report.png",
                "LEGACY_OFFLINE_REPORT_PATHS": (
                    runtime / "report.svg",
                    runtime / "camera.svg",
                    runtime / "mocap.svg",
                ),
                "OFFLINE_SUMMARY_PATH": runtime / "summary.json",
                "OFFSET_ESTIMATE_PATH": runtime / "offset.json",
            }
            with ExitStack() as stack:
                for name, value in path_patches.items():
                    stack.enter_context(patch(f"server.{name}", value))
                popen = stack.enter_context(
                    patch("server.subprocess.Popen", return_value=fake_process)
                )
                restart_mocap = stack.enter_context(
                    patch("server.restart_mocap_for_session")
                )
                ensure_assets = stack.enter_context(patch(
                    "server.ensure_model_assets",
                    return_value={
                        "available": True,
                        "onnx_exists": False,
                        "engine_exists": True,
                        "onnx_path": "/test/model.onnx",
                        "engine_path": "/test/model-fp16.engine",
                        "message": "已找到 FP16 TensorRT engine",
                    },
                ))
                stack.enter_context(
                    patch("server.session_status", return_value={"available": False})
                )
                stack.enter_context(patch("server.INFERENCE_PROCESS", None))
                stack.enter_context(patch("server.INFERENCE_LAST_EXIT_CODE", None))
                stack.enter_context(patch("server.INFERENCE_STARTED_NS", None))
                result = server_module.start_inference()

            self.assertEqual(result["pid"], 43210)
            self.assertEqual((runtime / "trt.pid").read_text(encoding="utf-8"), "43210\n")
            popen.assert_called_once_with(
                [str(binary_path), str(config_path)],
                cwd=str(server_module.PROJECT_ROOT),
            )
            restart_mocap.assert_called_once()
            ensure_assets.assert_called_once()
            self.assertTrue(result["model"]["engine_exists"])
            self.assertTrue((runtime / "config_apply_meta.json").exists())

    def test_start_inference_requires_confirmation_before_overwriting_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary_path = root / "trt"
            binary_path.write_bytes(b"test executable")
            binary_path.chmod(0o755)
            config_path = root / "config.json"
            config_path.write_text("{}\n", encoding="utf-8")
            with patch("server.TRT_BINARY_PATH", binary_path), patch(
                "server.CONFIG_PATH", config_path
            ), patch("server.PID_PATH", root / "trt.pid"), patch(
                "server.INFERENCE_PROCESS", None
            ), patch("server.session_status", return_value={
                "available": True,
                "current_run": True,
                "archived": False,
            }), patch("server.subprocess.Popen") as popen:
                with self.assertRaises(FileExistsError):
                    server_module.start_inference()

            popen.assert_not_called()

    def test_config_change_paths_keep_calibration_matrices_atomic(self) -> None:
        before = {
            "calibration": {"camera_matrix": [1, 0, 0, 0, 1, 0, 0, 0, 1]},
            "detect_camera": {"fps": 60},
        }
        after = {
            "calibration": {"camera_matrix": [2, 0, 0, 0, 2, 0, 0, 0, 1]},
            "detect_camera": {"fps": 30},
        }
        self.assertEqual(changed_config_paths(before, after), [
            "calibration.camera_matrix",
            "detect_camera.fps",
        ])

    def test_config_apply_metadata_separates_hot_and_restart_changes(self) -> None:
        before = {
            "calibration": {"extrinsic": [1, 0, 0, 0]},
            "network": {"udp_port": 1234},
        }
        after = {
            "calibration": {"extrinsic": [1, 0, 0, 5]},
            "network": {"udp_port": 5678},
        }
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text("{}\n", encoding="utf-8")
            expected_mtime = str(config_path.stat().st_mtime_ns)
            with patch("server.CONFIG_PATH", config_path):
                metadata = config_apply_metadata(before, after)

        self.assertEqual(metadata["hot_apply_paths"], ["calibration.extrinsic"])
        self.assertEqual(metadata["restart_required_paths"], ["network.udp_port"])
        self.assertEqual(metadata["config_mtime_ns"], expected_mtime)

    def test_model_keypoint_coordinates_are_hot_applied_when_count_is_unchanged(self) -> None:
        config = json.loads((WEB_MONITOR / "config.json").read_text(encoding="utf-8"))
        validated = validate(config)
        points = validated["calibration"]["model_keypoints_3d"]
        self.assertEqual(len(points), 10)
        self.assertTrue(all(len(point) == 3 for point in points))

        before = json.loads(json.dumps(validated))
        after = json.loads(json.dumps(validated))
        after["calibration"]["model_keypoints_3d"][3][1] += 0.25
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text("{}\n", encoding="utf-8")
            with patch("server.CONFIG_PATH", config_path):
                metadata = config_apply_metadata(before, after)
        self.assertEqual(
            metadata["hot_apply_paths"],
            ["calibration.model_keypoints_3d"],
        )
        self.assertEqual(metadata["restart_required_paths"], [])

    def test_legacy_config_gets_default_model_keypoints(self) -> None:
        config = json.loads((WEB_MONITOR / "config.json").read_text(encoding="utf-8"))
        config["calibration"].pop("model_keypoints_3d")
        validated = validate(config)
        self.assertEqual(
            validated["calibration"]["model_keypoints_3d"],
            [list(point) for point in server_module.DEFAULT_MODEL_KEYPOINTS_3D],
        )

    def test_legacy_config_gets_default_onnx_path(self) -> None:
        config = json.loads((WEB_MONITOR / "config.json").read_text(encoding="utf-8"))
        config.pop("model")
        validated = validate(config)
        self.assertEqual(
            validated["model"]["onnx_path"],
            server_module.DEFAULT_ONNX_PATH,
        )

    def test_invalid_onnx_paths_are_rejected(self) -> None:
        base = json.loads((WEB_MONITOR / "config.json").read_text(encoding="utf-8"))
        for value in ("", "models/engine/model.engine", 42):
            with self.subTest(value=value):
                config = json.loads(json.dumps(base))
                config["model"]["onnx_path"] = value
                with self.assertRaises(ValueError):
                    validate(config)

    def test_model_asset_paths_follow_sibling_engine_convention(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"model": {"onnx_path": "models/onnx/custom.onnx"}}
            with patch("server.PROJECT_ROOT", root):
                onnx, engine = server_module.model_asset_paths(config)
        self.assertEqual(onnx, root / "models/onnx/custom.onnx")
        self.assertEqual(engine, root / "models/engine/custom-fp16.engine")

    def test_bare_onnx_name_uses_root_engine_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"model": {"onnx_path": "custom.onnx"}}
            with patch("server.PROJECT_ROOT", root):
                onnx, engine = server_module.model_asset_paths(config)
        self.assertEqual(onnx, root / "custom.onnx")
        self.assertEqual(engine, root / "engine/custom-fp16.engine")

    def test_model_assets_accept_engine_without_shipping_onnx(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "models/engine/custom-fp16.engine"
            engine.parent.mkdir(parents=True)
            engine.write_bytes(b"engine")
            config = {"model": {"onnx_path": "models/onnx/custom.onnx"}}
            with patch("server.PROJECT_ROOT", root):
                status = server_module.ensure_model_assets(config)
        self.assertTrue(status["available"])
        self.assertTrue(status["engine_exists"])
        self.assertFalse(status["onnx_exists"])

    def test_model_assets_create_engine_directory_for_onnx_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            onnx = root / "models/onnx/custom.onnx"
            onnx.parent.mkdir(parents=True)
            onnx.write_bytes(b"onnx")
            config = {"model": {"onnx_path": "models/onnx/custom.onnx"}}
            with patch("server.PROJECT_ROOT", root):
                status = server_module.ensure_model_assets(config)
            self.assertTrue((root / "models/engine").is_dir())
        self.assertTrue(status["onnx_exists"])
        self.assertFalse(status["engine_exists"])

    def test_missing_onnx_and_engine_prevent_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = {"model": {"onnx_path": "models/onnx/missing.onnx"}}
            with patch("server.PROJECT_ROOT", Path(directory)):
                with self.assertRaises(FileNotFoundError):
                    server_module.ensure_model_assets(config)

    def test_model_path_change_requires_next_inference_start(self) -> None:
        before = {"model": {"onnx_path": "models/onnx/a.onnx"}}
        after = {"model": {"onnx_path": "models/onnx/b.onnx"}}
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text("{}\n", encoding="utf-8")
            with patch("server.CONFIG_PATH", config_path):
                metadata = config_apply_metadata(before, after)
        self.assertEqual(metadata["hot_apply_paths"], [])
        self.assertEqual(metadata["restart_required_paths"], ["model.onnx_path"])

    def test_variable_model_keypoint_counts_are_accepted(self) -> None:
        base = json.loads((WEB_MONITOR / "config.json").read_text(encoding="utf-8"))
        for count in (4, 7, 10, 16):
            with self.subTest(count=count):
                config = json.loads(json.dumps(base))
                config["calibration"]["model_keypoints_3d"] = [
                    [float(index), float(index + 1), float(index + 2)]
                    for index in range(count)
                ]
                self.assertEqual(
                    len(validate(config)["calibration"]["model_keypoints_3d"]),
                    count,
                )

    def test_model_keypoint_count_change_requires_restart(self) -> None:
        before = json.loads((WEB_MONITOR / "config.json").read_text(encoding="utf-8"))
        after = json.loads(json.dumps(before))
        after["calibration"]["model_keypoints_3d"].append([0.0, 0.0, 0.0])
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text("{}\n", encoding="utf-8")
            with patch("server.CONFIG_PATH", config_path):
                metadata = config_apply_metadata(before, after)
        self.assertNotIn(
            "calibration.model_keypoints_3d", metadata["hot_apply_paths"]
        )
        self.assertIn(
            "calibration.model_keypoints_3d", metadata["restart_required_paths"]
        )

    def test_invalid_model_keypoint_count_shape_and_values_are_rejected(self) -> None:
        base = json.loads((WEB_MONITOR / "config.json").read_text(encoding="utf-8"))
        invalid_values = (
            [[1, 2, 3]] * 3,
            [[1, 2, 3]] * 257,
            [[1, 2]] * 10,
            [[1, 2, 3]] * 9 + [[1, 2, float("nan")]],
        )
        for points in invalid_values:
            with self.subTest(points=points[-1]):
                config = json.loads(json.dumps(base))
                config["calibration"]["model_keypoints_3d"] = points
                with self.assertRaises(ValueError):
                    validate(config)

    def test_runtime_config_snapshot_confirms_matching_cpp_timestamp(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            runtime_status_path = root / "runtime_config_status.json"
            apply_meta_path = root / "config_apply_meta.json"
            config_path.write_text("{}\n", encoding="utf-8")
            mtime = str(config_path.stat().st_mtime_ns)
            runtime_status_path.write_text(json.dumps({
                "result": "ok",
                "observed_config_mtime_ns": mtime,
                "calibration_revision": 2,
            }), encoding="utf-8")
            apply_meta_path.write_text(json.dumps({
                "config_mtime_ns": mtime,
                "hot_apply_paths": ["calibration.extrinsic"],
                "restart_required_paths": [],
            }), encoding="utf-8")
            with patch("server.CONFIG_PATH", config_path), patch(
                "server.RUNTIME_CONFIG_STATUS_PATH", runtime_status_path
            ), patch("server.CONFIG_APPLY_META_PATH", apply_meta_path):
                snapshot = runtime_config_snapshot()

        self.assertTrue(snapshot["runtime"]["current"])
        self.assertEqual(snapshot["runtime"]["calibration_revision"], 2)
        self.assertEqual(
            snapshot["last_save"]["hot_apply_paths"],
            ["calibration.extrinsic"],
        )

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
            ), patch("server.OFFLINE_CAMERA_REPORT_PATH", root / "camera.png"), patch(
                "server.OFFLINE_MOCAP_REPORT_PATH", root / "mocap-report.png"
            ), patch("server.OFFLINE_REPORT_PATH", root / "report.png"), patch(
                "server.LEGACY_OFFLINE_REPORT_PATHS", (
                    root / "report.svg", root / "camera.svg", root / "mocap.svg"
                )
            ), patch(
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
            camera_path = root / "camera.png"
            mocap_report_path = root / "mocap.png"
            composite_path = root / "composite.png"
            generated = (
                {
                    "matched_samples": 1,
                    "detection_samples": 1,
                    "mocap_samples": 2,
                    "coverage_percent": 100,
                },
                "header\nrow\n",
                {
                    "camera": b"\x89PNG\r\n\x1a\ncamera",
                    "mocap": b"\x89PNG\r\n\x1a\nmocap",
                    "composite": b"\x89PNG\r\n\x1a\ncomposite",
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
            camera_png = camera_path.read_bytes()
            mocap_png = mocap_report_path.read_bytes()
            composite_png = composite_path.read_bytes()

        self.assertTrue(result["available"])
        self.assertEqual(
            set(result["report_urls"]), {"camera", "mocap", "composite"}
        )
        self.assertEqual(camera_png, generated[2]["camera"])
        self.assertEqual(mocap_png, generated[2]["mocap"])
        self.assertEqual(composite_png, generated[2]["composite"])
        self.assertIn("generated_unix_ns", saved_summary)

    def test_offline_report_without_mocap_keeps_camera_png_only(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detection_path = root / "pnp.csv"
            detection_path.touch()
            mocap_path = root / "missing-mocap.csv"
            summary_path = root / "summary.json"
            synced_path = root / "synced.csv"
            camera_path = root / "camera.png"
            mocap_report_path = root / "mocap.png"
            composite_path = root / "composite.png"
            mocap_report_path.write_bytes(b"stale mocap")
            composite_path.write_bytes(b"stale composite")
            generated = (
                {
                    "matched_samples": 0,
                    "detection_samples": 2,
                    "mocap_samples": 0,
                    "coverage_percent": 0,
                },
                "header\n",
                {
                    "camera": b"\x89PNG\r\n\x1a\ncamera-only",
                    "mocap": b"unused mocap",
                    "composite": b"unused composite",
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
            ), patch(
                "server.generate_offline_reports", return_value=generated
            ) as generate_reports:
                result = build_offline_report()

            saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            camera_png = camera_path.read_bytes()
            mocap_report_exists = mocap_report_path.exists()
            composite_exists = composite_path.exists()

        self.assertTrue(result["available"])
        self.assertTrue(result["camera_only"])
        self.assertFalse(result["mocap_available"])
        self.assertEqual(set(result["report_urls"]), {"camera"})
        self.assertEqual(result["report_url"], result["report_urls"]["camera"])
        self.assertEqual(camera_png, generated[2]["camera"])
        self.assertFalse(mocap_report_exists)
        self.assertFalse(composite_exists)
        self.assertEqual(saved_summary["report_mode"], "camera_only")
        self.assertIn("未记录到动捕数据", saved_summary["report_message"])
        self.assertIsNone(generate_reports.call_args.args[1])


if __name__ == "__main__":
    unittest.main()
