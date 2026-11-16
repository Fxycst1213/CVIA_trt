"""Check the actual production 3D initializer and its PnP coordinate convention."""

import csv
import hashlib
import json
from pathlib import Path
import re
import unittest

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class Plane9GeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (ROOT / 'src/yolov_pose/pose.cpp').read_text()
        match = re.search(r'_p3d = \(cv::Mat_<double>\(NUM_KEYPOINTS, 3\) <<(.*?)\);',
                          source, re.S)
        if match is None:
            raise AssertionError('Production 3D point initializer not found')
        numbers = re.sub(r'//[^\n]*', '', match.group(1))
        cls.points = np.array([float(value) for value in numbers.split(',')]).reshape(-1, 3)
        cls.calibrated = json.loads((ROOT / 'data/plane9/points_body_mm.json').read_text())
        cls.rotation = np.array(cls.calibrated['rotation_body_from_obj'], dtype=float)
        cls.scale = 3445 / cls.calibrated['model_wingspan_obj_units']
        cls.origin = np.array(cls.calibrated['origin_obj'])
        cls.obj = np.array([record['xyz_obj'] for record in cls.calibrated['keypoints']])
        intrinsics = re.search(r'_K = \(cv::Mat_<double>\(3, 3\) <<(.*?)\);', source, re.S)
        distortion = re.search(r'_diff = \(cv::Mat_<float>\(1, 5\) <<(.*?)\);', source, re.S)
        cls.camera = np.array([float(value) for value in intrinsics.group(1).split(',')]).reshape(3, 3)
        cls.distortion = np.array([float(value) for value in distortion.group(1).split(',')])

    def test_point_order_scale_and_export_match_production(self):
        self.assertEqual(self.points.shape, (9, 3))
        np.testing.assert_allclose(self.points, (self.obj - self.origin) @ self.rotation.T * self.scale,
                                   rtol=0, atol=0.000001)
        with (ROOT / 'data/plane9/points_body_mm.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row['onnx_index']) for row in rows], list(range(9)))
        exported = np.array([[float(row[key]) for key in ('x_mm', 'y_mm', 'z_mm')] for row in rows])
        np.testing.assert_array_equal(self.points, exported)
        span = self.calibrated['obj_bounds_max'][0] - self.calibrated['obj_bounds_min'][0]
        self.assertAlmostEqual(span * self.scale, 3445, places=9)

    def test_active_model_matches_verified_keypoint_mapping(self):
        main = (ROOT / 'src/main.cpp').read_text()
        path = re.search(r'string onnxPath = "([^"]+)";', main).group(1)
        self.assertEqual(path, self.calibrated['source_onnx'])
        self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
                         self.calibrated['source_onnx_sha256'])
        shape = json.loads(self.calibrated['onnx_metadata']['kpt_shape'])
        params = (ROOT / 'src/params/pose_params.hpp').read_text()
        count = int(re.search(r'NUM_KEYPOINTS = (\d+)', params).group(1))
        self.assertEqual(count, shape[0])
        self.assertEqual(self.points.shape[0], count)
        self.assertEqual([r['onnx_index'] for r in self.calibrated['keypoints']], list(range(count)))
        self.assertIsNone(self.calibrated['excluded_marker']['onnx_index'])
        self.assertTrue((np.linalg.norm(self.obj - self.origin, axis=1) > 0.01).all())

    def test_body_frame_is_right_handed_and_points_toward_nose(self):
        np.testing.assert_array_equal(self.rotation @ self.rotation.T, np.eye(3))
        self.assertAlmostEqual(np.linalg.det(self.rotation), 1.0)
        self.assertGreater(self.points[8, 0], self.points[:3, 0].max())
        self.assertGreater(self.points[3, 1], 0)
        self.assertLess(self.points[7, 1], 0)
        self.assertTrue((self.points[:3, 2] < 0).all())
        self.assertTrue((self.points[3:, 2] > 0).all())
        self.assertEqual(np.linalg.matrix_rank(self.points - self.points.mean(axis=0)), 3)

    def test_frame_conversion_preserves_camera_projections(self):
        body_rotation, _ = cv2.Rodrigues(np.array([0.10, -0.12, 1.57]))
        body_vector, _ = cv2.Rodrigues(body_rotation)
        obj_rotation = body_rotation @ self.rotation
        obj_vector, _ = cv2.Rodrigues(obj_rotation)
        translation = np.array([150.0, -80.0, 6000.0])
        obj_translation = translation - obj_rotation @ (self.origin * self.scale)
        body_pixels, _ = cv2.projectPoints(self.points, body_vector, translation, self.camera, self.distortion)
        obj_pixels, _ = cv2.projectPoints(self.obj * self.scale, obj_vector, obj_translation,
                                        self.camera, self.distortion)
        np.testing.assert_allclose(body_pixels, obj_pixels, rtol=0, atol=0.000001)

    def test_pnp_recovers_millimeter_translation_with_outliers(self):
        rotation = np.array([0.10, -0.12, 1.57])
        translation = np.array([150.0, -80.0, 6000.0])
        pixels, _ = cv2.projectPoints(self.points, rotation, translation, self.camera, self.distortion)
        pixels[[3, 7], 0] += [180, -140]
        cv2.setRNGSeed(123)
        success, recovered_rotation, recovered_translation, inliers = cv2.solvePnPRansac(
            self.points, pixels, self.camera, self.distortion, iterationsCount=100,
            reprojectionError=2.0, confidence=0.99, flags=cv2.SOLVEPNP_SQPNP)
        self.assertTrue(success)
        self.assertEqual(len(inliers), 7)
        self.assertFalse(set(inliers.ravel()) & {3, 7})
        np.testing.assert_allclose(recovered_translation.ravel(), translation, rtol=0, atol=0.05)
        expected_matrix, _ = cv2.Rodrigues(rotation)
        actual_matrix, _ = cv2.Rodrigues(recovered_rotation)
        np.testing.assert_allclose(actual_matrix, expected_matrix, rtol=0, atol=0.00001)

    def test_pnp_recovers_pose_from_visible_subset(self):
        indices = [0, 2, 3, 5, 6, 8]
        points = self.points[indices].copy()
        rotation = np.array([0.10, -0.12, 1.57])
        translation = np.array([150.0, -80.0, 6000.0])
        pixels, _ = cv2.projectPoints(points, rotation, translation, self.camera, self.distortion)
        success, _, recovered_translation, inliers = cv2.solvePnPRansac(
            points, pixels, self.camera, self.distortion, iterationsCount=100,
            reprojectionError=2.0, confidence=0.99, flags=cv2.SOLVEPNP_SQPNP)
        self.assertTrue(success)
        self.assertEqual(len(inliers), len(indices))
        np.testing.assert_allclose(recovered_translation.ravel(), translation, rtol=0, atol=0.01)


if __name__ == '__main__':
    unittest.main()
