"""2D preview must render sender coordinates without accessing 3D pose."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("preview_2d", ROOT / "tools/server.py")
viewer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = viewer
SPEC.loader.exec_module(viewer)


class KeypointPreviewTests(unittest.TestCase):
    def test_points_render_without_accessing_pose_or_calibration(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        points = []
        for i in range(viewer.NUM_KEYPOINTS):
            points.extend((60 + 45 * i, 200, .49 if i == 0 else .85))
        points.extend((20, 100, 620, 300, .9))
        with patch.object(viewer, "projection_calibration", side_effect=AssertionError("Used 3D calibration")), \
             patch.object(viewer.cv2, "drawFrameAxes", side_effect=AssertionError("Drew 3D axes")), \
             patch.object(viewer.cv2, "putText", wraps=cv2.putText) as labels:
            result = viewer.draw_result_image(image, points, None, pnp_valid=True,
                                               calibration=None, keypoints_only=True)
        np.testing.assert_array_equal(result[200, 60], [0, 0, 0])
        for i in range(1, viewer.NUM_KEYPOINTS):
            np.testing.assert_array_equal(result[200, 60 + 45 * i], [0, 255, 0])
        text = [call.args[1] for call in labels.call_args_list]
        self.assertIn(f"{viewer.NUM_KEYPOINTS - 1}: 0.85", text)
        self.assertTrue(any("2D keypoints" in label for label in text))
        self.assertFalse(any("XYZ" in label or "PnP" in label for label in text))

    def test_subpixel_coordinates_are_drawn_as_received(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        points = [0] * viewer.NUM_DATA
        points[:3] = [150.375, 210.625, .95]
        points[-5:] = [100, 100, 600, 300, .9]
        with patch.object(viewer.cv2, "putText", wraps=cv2.putText) as labels, \
             patch.object(viewer.cv2, "circle", wraps=cv2.circle) as circles:
            viewer.draw_result_image(image, points, None, keypoints_only=True)
        self.assertFalse(any("raw" in call.args[1] or "refinement" in call.args[1] for call in labels.call_args_list))
        self.assertEqual(circles.call_args.args[1], (round(150.375 * 16), round(210.625 * 16)))
        self.assertEqual(circles.call_args.kwargs["shift"], 4)

    def test_no_detection_renders_without_pose(self):
        image = np.zeros((240, 640, 3), dtype=np.uint8)
        with patch.object(viewer.cv2, "putText", wraps=cv2.putText) as labels:
            viewer.draw_result_image(image, [0] * viewer.NUM_DATA, None, keypoints_only=True)
        self.assertIn("No target detected", [call.args[1] for call in labels.call_args_list])


if __name__ == "__main__":
    unittest.main()
