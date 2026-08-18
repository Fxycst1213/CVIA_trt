from __future__ import annotations

import sys
import unittest
from pathlib import Path


MOCAP_DIR = Path(__file__).resolve().parents[1] / "mocap"
sys.path.insert(0, str(MOCAP_DIR))

from measure_clock_offset import (  # noqa: E402
    ClockSample,
    analyse,
    parse_clock_line,
    parse_pose_line,
)


def make_samples(
    *,
    sdk_start_ms: int,
    orin_start_ms: float,
    sdk_period_ms: int = 10,
    orin_period_ms: float = 10.0,
    count: int = 101,
) -> list[ClockSample]:
    samples = []
    monotonic_start_ns = 8_000_000_000
    for index in range(count):
        receive_ns = int(round((orin_start_ms + index * orin_period_ms) * 1_000_000))
        samples.append(ClockSample(
            sample_index=index + 1,
            mocap_frame=100 + index,
            mocap_timestamp_ms=sdk_start_ms + index * sdk_period_ms,
            orin_receive_unix_ns=receive_ns,
            orin_receive_monotonic_ns=monotonic_start_ns + int(round(index * orin_period_ms * 1_000_000)),
            python_read_unix_ns=receive_ns + 200_000,
            tracker_id=1,
            tracker_name="Tracker 0",
        ))
    return samples


class MeasureClockOffsetTests(unittest.TestCase):
    def test_parse_clock_does_not_require_visible_rigid_body(self) -> None:
        sample = parse_clock_line(
            "CLOCK\t42\t1800000000000\t1800000000012500000\t8000000000",
            sample_index=7,
            read_unix_ns=1_800_000_000_012_700_000,
        )
        self.assertEqual(sample.mocap_frame, 42)
        self.assertEqual(sample.tracker_name, "FrameGroup CLOCK")
        self.assertAlmostEqual(sample.orin_minus_mocap_ms, 12.5)

    def test_parse_pose_uses_bridge_callback_timestamp(self) -> None:
        sample = parse_pose_line(
            "POSE\tname%3ATracker%200\t1\tTracker%200\t42\t1800000000000"
            "\t1800000000012500000\t8000000000\t1\t2\t3\t0\t0\t0\t1\t0.01\t1",
            sample_index=7,
            read_unix_ns=1_800_000_000_012_700_000,
        )
        self.assertEqual(sample.tracker_name, "Tracker 0")
        self.assertEqual(sample.sample_index, 7)
        self.assertAlmostEqual(sample.orin_minus_mocap_ms, 12.5)
        self.assertAlmostEqual(sample.python_pipe_delay_ms, 0.2)

    def test_common_unix_epoch_reports_apparent_offset(self) -> None:
        samples = make_samples(
            sdk_start_ms=1_800_000_000_000,
            orin_start_ms=1_800_000_000_012.5,
        )
        result = analyse(samples)
        self.assertEqual(result.time_domain, "unix_like")
        self.assertAlmostEqual(result.raw_offset.p50, 12.5, places=6)
        self.assertAlmostEqual(result.drift_ppm, 0.0, places=6)
        self.assertAlmostEqual(result.orin_rate_hz, 100.0, places=6)

    def test_relative_sdk_clock_reports_rate_drift_not_absolute_offset(self) -> None:
        samples = make_samples(
            sdk_start_ms=50_000,
            orin_start_ms=1_800_000_000_000.0,
            sdk_period_ms=10,
            orin_period_ms=10.001,
        )
        result = analyse(samples)
        self.assertEqual(result.time_domain, "relative")
        self.assertAlmostEqual(result.drift_ppm, 99.99, delta=0.05)
        self.assertAlmostEqual(result.affine_rate_ppm, 100.0, delta=0.01)
        self.assertLess(result.affine_residual.peak_to_peak, 1e-6)
        self.assertAlmostEqual(result.relative_residual.maximum, 0.1, places=5)

    def test_tai_utc_difference_is_flagged(self) -> None:
        samples = make_samples(
            sdk_start_ms=1_800_000_037_000,
            orin_start_ms=1_800_000_000_000.0,
        )
        result = analyse(samples)
        self.assertEqual(result.time_domain, "possible_tai")
        self.assertAlmostEqual(result.raw_offset.p50, -37_000.0, places=6)


if __name__ == "__main__":
    unittest.main()
