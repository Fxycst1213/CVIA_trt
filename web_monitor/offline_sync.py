"""Offline visual/mocap pairing and dependency-free SVG report generation."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from xml.sax.saxutils import escape

from mocap_receiver import (
    MocapPose,
    MocapTimeline,
    align_mocap_poses,
    pose_validity,
)


SYNCED_COLUMNS = (
    "coordinate_frame",
    "visual_timestamp_ms",
    "visual_capture_timestamp_ns",
    "visual_publish_timestamp_ns",
    "visual_pipeline_latency_ms",
    "visual_x",
    "visual_y",
    "visual_z",
    "visual_rx",
    "visual_ry",
    "visual_rz",
    "mocap_x",
    "mocap_y",
    "mocap_z",
    "mocap_rx",
    "mocap_ry",
    "mocap_rz",
    "mocap_frames",
    "sync_method",
    "sync_error_ms",
    "bracket_span_ms",
    "interpolation_alpha",
)

MOCAP_COORDINATE_FRAME = "mocap"


@dataclass(frozen=True)
class DetectionPose:
    timestamp_ms: int
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float
    capture_timestamp_ns: int = 0
    publish_timestamp_ns: int = 0
    coordinate_frame: str = MOCAP_COORDINATE_FRAME

    @property
    def effective_timestamp_ns(self) -> int:
        return self.capture_timestamp_ns or self.timestamp_ms * 1_000_000


@dataclass(frozen=True)
class MotionSeries:
    """Smoothed scalar motion signatures on one device time line."""

    times_ns: tuple[int, ...]
    translation_speed: tuple[float, ...]
    rotation_speed: tuple[float, ...]


def load_detection_poses(path: Path) -> list[DetectionPose]:
    poses: list[DetectionPose] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                # Main-program PNP output is T_M_O: solvePnP's T_C_O has
                # already been left-multiplied by calibration.extrinsic
                # (T_M_C). Legacy CSVs predate this metadata column but were
                # produced by the same transformed m_result path.
                coordinate_frame = (
                    row.get("coordinate_frame") or MOCAP_COORDINATE_FRAME
                ).strip().lower()
                if coordinate_frame != MOCAP_COORDINATE_FRAME:
                    raise ValueError(
                        "视觉 PnP CSV 不是动捕坐标系，拒绝生成混合坐标系报告："
                        f"coordinate_frame={coordinate_frame!r}"
                    )
                pose = DetectionPose(
                    timestamp_ms=int(row["timestamp"]),
                    x=float(row["x"]),
                    y=float(row["y"]),
                    z=float(row["z"]),
                    rx=float(row["rx"]),
                    ry=float(row["ry"]),
                    rz=float(row["rz"]),
                    capture_timestamp_ns=int(
                        row.get("capture_timestamp_ns")
                        or int(row["timestamp"]) * 1_000_000
                    ),
                    publish_timestamp_ns=int(row.get("publish_timestamp_ns") or 0),
                    coordinate_frame=coordinate_frame,
                )
            except ValueError as error:
                if "不是动捕坐标系" in str(error):
                    raise
                continue
            except (KeyError, TypeError):
                continue
            if all(math.isfinite(value) for value in (
                pose.x, pose.y, pose.z, pose.rx, pose.ry, pose.rz
            )):
                poses.append(pose)
    return sorted(poses, key=lambda pose: pose.effective_timestamp_ns)


def load_mocap_poses(path: Path) -> list[MocapPose]:
    poses: list[MocapPose] = []
    persisted_models: list[tuple[MocapPose, float]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                if int(row["valid"]) != 1:
                    continue
                pose = MocapPose(
                    selector="offline",
                    tracker_id=int(row["tracker_id"]),
                    tracker_name=row["tracker_name"],
                    mocap_frame=int(row["frame"]),
                    mocap_timestamp_ms=int(row["mocap_timestamp_ms"]),
                    receive_unix_ns=int(row["receive_unix_ns"]),
                    receive_monotonic_ns=int(row["receive_monotonic_ns"]),
                    x=float(row["x"]),
                    y=float(row["y"]),
                    z=float(row["z"]),
                    qx=float(row["qx"]),
                    qy=float(row["qy"]),
                    qz=float(row["qz"]),
                    qw=float(row["qw"]),
                    mean_error=float(row["mean_error"]),
                    tracking_params=int(row["tracking_params"]),
                    aligned_unix_ns=int(row.get("aligned_unix_ns") or 0),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if pose_validity(pose)[0]:
                poses.append(pose)
                try:
                    model_ready = int(row.get("clock_model_ready") or 0) == 1
                    rate_ppm = float(row.get("clock_rate_ppm") or 0.0)
                except (TypeError, ValueError):
                    model_ready = False
                    rate_ppm = 0.0
                if (
                    model_ready
                    and pose.aligned_unix_ns > 0
                    and math.isfinite(rate_ppm)
                ):
                    persisted_models.append((pose, rate_ppm))
    if persisted_models:
        # The last ready row was generated by the most mature online model.
        # Rebuild one consistent affine time line around that persisted anchor,
        # including short POSE intervals whose clock was pre-warmed by CLOCK
        # records while the rigid body was not visible.
        anchor, rate_ppm = persisted_models[-1]
        ns_per_sdk_ms = 1_000_000.0 * (1.0 + rate_ppm / 1_000_000.0)
        aligned = [
            replace(
                pose,
                aligned_unix_ns=anchor.aligned_unix_ns + int(round(
                    (pose.mocap_timestamp_ms - anchor.mocap_timestamp_ms)
                    * ns_per_sdk_ms
                )),
            )
            for pose in poses
        ]
        return sorted(aligned, key=lambda pose: pose.timeline_unix_ns)
    # Refit over the complete session so early warm-up rows and legacy CSVs use
    # one consistent SDK-clock rate.  Short legacy fixtures safely fall back to
    # receive time when there are not enough samples for a reliable model.
    return align_mocap_poses(poses)[0]


def _normalize_quaternion(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1e-12:
        raise ValueError("四元数长度为零")
    return tuple(value / norm for value in quaternion)


def _euler_xyz_degrees_to_quaternion(
    rx: float, ry: float, rz: float
) -> tuple[float, float, float, float]:
    """Convert the dashboard Roll-X/Pitch-Y/Yaw-Z convention to a quaternion."""
    roll, pitch, yaw = (math.radians(value) / 2 for value in (rx, ry, rz))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return _normalize_quaternion((
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ))


def _quaternion_angular_distance(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first = _normalize_quaternion(first)
    second = _normalize_quaternion(second)
    dot = abs(sum(a * b for a, b in zip(first, second)))
    return 2 * math.acos(min(1.0, max(-1.0, dot)))


def _moving_average(values: list[float], radius: int = 2) -> tuple[float, ...]:
    if not values:
        return ()
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        smoothed.append((prefix[end] - prefix[start]) / (end - start))
    return tuple(smoothed)


def _motion_series_from_detection(poses: list[DetectionPose]) -> MotionSeries:
    times: list[int] = []
    translation: list[float] = []
    rotation: list[float] = []
    previous_quaternion = None
    for previous, current in zip(poses, poses[1:]):
        start_ns = previous.effective_timestamp_ns
        end_ns = current.effective_timestamp_ns
        dt_seconds = (end_ns - start_ns) / 1_000_000_000
        if not 0 < dt_seconds <= 0.25:
            previous_quaternion = None
            continue
        displacement = math.sqrt(
            (current.x - previous.x) ** 2
            + (current.y - previous.y) ** 2
            + (current.z - previous.z) ** 2
        )
        first_quaternion = previous_quaternion or _euler_xyz_degrees_to_quaternion(
            previous.rx, previous.ry, previous.rz
        )
        second_quaternion = _euler_xyz_degrees_to_quaternion(
            current.rx, current.ry, current.rz
        )
        times.append((start_ns + end_ns) // 2)
        translation.append(displacement / dt_seconds)
        rotation.append(
            math.degrees(_quaternion_angular_distance(first_quaternion, second_quaternion))
            / dt_seconds
        )
        previous_quaternion = second_quaternion
    return MotionSeries(tuple(times), _moving_average(translation), _moving_average(rotation))


def _motion_series_from_mocap(poses: list[MocapPose]) -> MotionSeries:
    ordered = sorted(poses, key=lambda pose: pose.timeline_unix_ns)
    times: list[int] = []
    translation: list[float] = []
    rotation: list[float] = []
    for previous, current in zip(ordered, ordered[1:]):
        dt_seconds = (current.timeline_unix_ns - previous.timeline_unix_ns) / 1_000_000_000
        if not 0 < dt_seconds <= 0.25:
            continue
        displacement = math.sqrt(
            (current.x - previous.x) ** 2
            + (current.y - previous.y) ** 2
            + (current.z - previous.z) ** 2
        )
        times.append((previous.timeline_unix_ns + current.timeline_unix_ns) // 2)
        translation.append(displacement / dt_seconds)
        rotation.append(
            math.degrees(_quaternion_angular_distance(
                (previous.qx, previous.qy, previous.qz, previous.qw),
                (current.qx, current.qy, current.qz, current.qw),
            )) / dt_seconds
        )
    return MotionSeries(tuple(times), _moving_average(translation), _moving_average(rotation))


def _pearson_from_sums(
    count: int,
    sum_x: float,
    sum_y: float,
    sum_xx: float,
    sum_yy: float,
    sum_xy: float,
) -> float | None:
    if count < 60:
        return None
    variance_x = count * sum_xx - sum_x * sum_x
    variance_y = count * sum_yy - sum_y * sum_y
    denominator = math.sqrt(max(0.0, variance_x) * max(0.0, variance_y))
    if denominator <= 1e-12:
        return None
    return min(1.0, max(-1.0, (count * sum_xy - sum_x * sum_y) / denominator))


def _correlation_at_offset(
    vision: MotionSeries,
    mocap: MotionSeries,
    offset_ms: float,
    *,
    max_samples: int = 12_000,
) -> dict | None:
    if len(vision.times_ns) < 60 or len(mocap.times_ns) < 60:
        return None
    offset_ns = int(round(offset_ms * 1_000_000))
    stride = max(1, math.ceil(len(vision.times_ns) / max_samples))
    sums = {
        "translation": [0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "rotation": [0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    mocap_index = 0
    for vision_index in range(0, len(vision.times_ns), stride):
        raw_target_ns = vision.times_ns[vision_index] - offset_ns
        if raw_target_ns < mocap.times_ns[0] or raw_target_ns > mocap.times_ns[-1]:
            continue
        while (
            mocap_index + 1 < len(mocap.times_ns)
            and mocap.times_ns[mocap_index + 1] < raw_target_ns
        ):
            mocap_index += 1
        if mocap_index + 1 >= len(mocap.times_ns):
            break
        start_time = mocap.times_ns[mocap_index]
        end_time = mocap.times_ns[mocap_index + 1]
        if end_time <= start_time or end_time - start_time > 250_000_000:
            continue
        alpha = (raw_target_ns - start_time) / (end_time - start_time)
        for channel, vision_values, mocap_values in (
            ("translation", vision.translation_speed, mocap.translation_speed),
            ("rotation", vision.rotation_speed, mocap.rotation_speed),
        ):
            vision_value = vision_values[vision_index]
            mocap_value = mocap_values[mocap_index] + alpha * (
                mocap_values[mocap_index + 1] - mocap_values[mocap_index]
            )
            accumulator = sums[channel]
            accumulator[0] += 1
            accumulator[1] += vision_value
            accumulator[2] += mocap_value
            accumulator[3] += vision_value * vision_value
            accumulator[4] += mocap_value * mocap_value
            accumulator[5] += vision_value * mocap_value

    translation_correlation = _pearson_from_sums(*sums["translation"])
    rotation_correlation = _pearson_from_sums(*sums["rotation"])
    weighted = []
    if translation_correlation is not None:
        weighted.append((translation_correlation, 0.7))
    if rotation_correlation is not None:
        weighted.append((rotation_correlation, 0.3))
    if not weighted:
        return None
    score = sum(value * weight for value, weight in weighted) / sum(
        weight for _, weight in weighted
    )
    return {
        "offset_ms": float(offset_ms),
        "score": score,
        "translation_correlation": translation_correlation,
        "rotation_correlation": rotation_correlation,
        "samples": max(sums["translation"][0], sums["rotation"][0]),
    }


def estimate_time_offset(
    detections: list[DetectionPose],
    mocap_poses: list[MocapPose],
    *,
    max_offset_ms: float = 1000.0,
) -> dict:
    """Estimate the correction added to mocap receive time using motion NCC.

    Translation-speed and angular-speed magnitudes are invariant to coordinate
    origin and, after Pearson normalization, insensitive to constant unit scale.
    This permits temporal calibration before the two spatial frames are aligned.
    """
    if not 50 <= max_offset_ms <= 10_000:
        raise ValueError("自动估计搜索范围必须在 50～10000 ms")
    vision = _motion_series_from_detection(detections)
    mocap = _motion_series_from_mocap(mocap_poses)
    if len(vision.times_ns) < 60 or len(mocap.times_ns) < 60:
        raise ValueError("至少需要约 60 个连续有效样本才能估计时间偏移")

    coarse_step_ms = 5.0
    coarse_limit = int(math.floor(max_offset_ms / coarse_step_ms))
    candidates: dict[float, dict] = {}
    for index in range(-coarse_limit, coarse_limit + 1):
        offset = index * coarse_step_ms
        result = _correlation_at_offset(vision, mocap, offset)
        if result is not None:
            candidates[offset] = result
    if not candidates:
        raise ValueError("运动变化不足，无法计算有效互相关")

    coarse_best = max(candidates.values(), key=lambda item: item["score"])
    fine_start = max(-max_offset_ms, coarse_best["offset_ms"] - coarse_step_ms)
    fine_end = min(max_offset_ms, coarse_best["offset_ms"] + coarse_step_ms)
    fine_index = math.ceil(fine_start)
    while fine_index <= math.floor(fine_end):
        offset = float(fine_index)
        if offset not in candidates:
            result = _correlation_at_offset(vision, mocap, offset)
            if result is not None:
                candidates[offset] = result
        fine_index += 1

    best = max(candidates.values(), key=lambda item: item["score"])
    distinct_distance_ms = max(50.0, coarse_step_ms * 4)
    distinct = [
        result for result in candidates.values()
        if abs(result["offset_ms"] - best["offset_ms"]) >= distinct_distance_ms
    ]
    second_score = max((result["score"] for result in distinct), default=-1.0)
    peak_margin = best["score"] - second_score
    at_boundary = abs(best["offset_ms"]) >= max_offset_ms - coarse_step_ms
    if best["score"] >= 0.70 and peak_margin >= 0.08 and not at_boundary:
        confidence = "high"
    elif best["score"] >= 0.45 and peak_margin >= 0.03 and not at_boundary:
        confidence = "medium"
    else:
        confidence = "low"
    reliable = confidence in {"high", "medium"}
    if at_boundary:
        message = "最佳相关峰位于搜索边界；请扩大搜索范围后重试"
    elif not reliable:
        message = "相关峰不够清晰；请增加非周期的平移或转动后重试"
    else:
        message = "互相关峰清晰，已得到可应用的时间补偿"
    return {
        "offset_ms": round(best["offset_ms"], 3),
        "correlation": round(best["score"], 6),
        "translation_correlation": (
            round(best["translation_correlation"], 6)
            if best["translation_correlation"] is not None else None
        ),
        "rotation_correlation": (
            round(best["rotation_correlation"], 6)
            if best["rotation_correlation"] is not None else None
        ),
        "peak_margin": round(peak_margin, 6),
        "confidence": confidence,
        "reliable": reliable,
        "at_search_boundary": at_boundary,
        "samples": best["samples"],
        "vision_motion_samples": len(vision.times_ns),
        "mocap_motion_samples": len(mocap.times_ns),
        "search_range_ms": max_offset_ms,
        "coarse_step_ms": coarse_step_ms,
        "fine_step_ms": 1.0,
        "method": "normalized_cross_correlation_motion_magnitude",
        "message": message,
    }


def synchronize(
    detections: list[DetectionPose],
    mocap_poses: list[MocapPose],
    *,
    offset_ms: float,
    max_error_ms: float,
    interpolate: bool,
    retain_unmatched: bool = False,
) -> tuple[list[dict], dict]:
    timeline = MocapTimeline(mocap_poses)
    rows: list[dict] = []
    errors: list[float] = []
    method_counts = {"matched": 0, "interpolated": 0}
    first_matched_detection_index: int | None = None
    for detection_index, detection in enumerate(detections):
        row = {
            "coordinate_frame": MOCAP_COORDINATE_FRAME,
            "visual_timestamp_ms": detection.timestamp_ms,
            "visual_capture_timestamp_ns": detection.effective_timestamp_ns,
            "visual_publish_timestamp_ns": detection.publish_timestamp_ns,
            "visual_pipeline_latency_ms": (
                round(
                    (detection.publish_timestamp_ns - detection.effective_timestamp_ns)
                    / 1_000_000,
                    6,
                )
                if detection.publish_timestamp_ns >= detection.effective_timestamp_ns
                and detection.publish_timestamp_ns > 0 else ""
            ),
            "visual_x": detection.x,
            "visual_y": detection.y,
            "visual_z": detection.z,
            "visual_rx": detection.rx,
            "visual_ry": detection.ry,
            "visual_rz": detection.rz,
            "mocap_x": "",
            "mocap_y": "",
            "mocap_z": "",
            "mocap_rx": "",
            "mocap_ry": "",
            "mocap_rz": "",
            "mocap_frames": "",
            "sync_method": "waiting",
            "sync_error_ms": "",
            "bracket_span_ms": "",
            "interpolation_alpha": "",
        }
        match = timeline.match(
            detection.effective_timestamp_ns,
            offset_ms=offset_ms,
            max_error_ms=max_error_ms,
            interpolate=interpolate,
        )
        if match["status"] not in method_counts or not match.get("pose"):
            row["sync_method"] = match["status"]
            row["sync_error_ms"] = match.get("sync_error_ms", "")
            if retain_unmatched:
                rows.append(row)
            continue
        if first_matched_detection_index is None:
            first_matched_detection_index = detection_index
        method_counts[match["status"]] += 1
        error = float(match["sync_error_ms"])
        errors.append(abs(error))
        mocap = match["pose"]
        row.update({
            "mocap_x": mocap["position"]["x"],
            "mocap_y": mocap["position"]["y"],
            "mocap_z": mocap["position"]["z"],
            "mocap_rx": mocap["euler_deg"]["rx"],
            "mocap_ry": mocap["euler_deg"]["ry"],
            "mocap_rz": mocap["euler_deg"]["rz"],
            "mocap_frames": ":".join(str(value) for value in match["mocap_frames"]),
            "sync_method": match["status"],
            "sync_error_ms": error,
            "bracket_span_ms": match.get("bracket_span_ms"),
            "interpolation_alpha": match.get("alpha"),
        })
        rows.append(row)

    sorted_errors = sorted(errors)
    p95_index = max(0, math.ceil(len(sorted_errors) * 0.95) - 1)
    pipeline_latencies = sorted(
        (pose.publish_timestamp_ns - pose.effective_timestamp_ns) / 1_000_000
        for pose in detections
        if pose.publish_timestamp_ns >= pose.effective_timestamp_ns
        and pose.publish_timestamp_ns > 0
    )
    latency_p95_index = max(0, math.ceil(len(pipeline_latencies) * 0.95) - 1)
    matched_samples = sum(method_counts.values())
    summary = {
        "coordinate_frame": MOCAP_COORDINATE_FRAME,
        "camera_pose_frame": "mocap_via_T_M_C",
        "mocap_pose_frame": "mocap_native",
        "detection_samples": len(detections),
        "mocap_samples": len(mocap_poses),
        "matched_samples": matched_samples,
        "unmatched_samples": len(detections) - matched_samples,
        "coverage_percent": round(matched_samples * 100 / len(detections), 3) if detections else 0.0,
        "nearest_matches": method_counts["matched"],
        "interpolated_matches": method_counts["interpolated"],
        "mean_abs_error_ms": round(sum(errors) / len(errors), 6) if errors else None,
        "p95_abs_error_ms": round(sorted_errors[p95_index], 6) if errors else None,
        "max_abs_error_ms": round(sorted_errors[-1], 6) if errors else None,
        "offset_ms": offset_ms,
        "max_error_ms": max_error_ms,
        "interpolate": interpolate,
        "mean_pipeline_latency_ms": (
            round(sum(pipeline_latencies) / len(pipeline_latencies), 6)
            if pipeline_latencies else None
        ),
        "p95_pipeline_latency_ms": (
            round(pipeline_latencies[latency_p95_index], 6)
            if pipeline_latencies else None
        ),
        "start_timestamp_ms": detections[0].timestamp_ms if detections else None,
        "end_timestamp_ms": detections[-1].timestamp_ms if detections else None,
        "start_capture_timestamp_ns": (
            detections[0].effective_timestamp_ns if detections else None
        ),
        "end_capture_timestamp_ns": (
            detections[-1].effective_timestamp_ns if detections else None
        ),
        "plot_start_reason": "first_valid_visual_pose",
        "first_matched_capture_timestamp_ns": (
            detections[first_matched_detection_index].effective_timestamp_ns
            if first_matched_detection_index is not None else None
        ),
        "leading_unmatched_samples": (
            first_matched_detection_index
            if first_matched_detection_index is not None else len(detections)
        ),
    }
    return rows, summary


def synced_csv(rows: list[dict]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=SYNCED_COLUMNS)
    writer.writeheader()
    writer.writerows(
        row for row in rows
        if row.get("sync_method") in {"matched", "interpolated"}
    )
    return output.getvalue()


def _sample_rows(rows: list[dict], limit: int = 900) -> list[dict]:
    if len(rows) <= limit:
        return rows
    stride = len(rows) / limit
    return [rows[min(len(rows) - 1, int(index * stride))] for index in range(limit)]


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _polyline_segments(
    rows: list[dict],
    key: str,
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    time_min_ns: int,
    time_max_ns: int,
    value_min: float,
    value_max: float,
) -> list[str]:
    """Build separate SVG polylines without bridging missing mocap intervals."""
    time_span = max(1, time_max_ns - time_min_ns)
    value_span = max(1e-12, value_max - value_min)
    segments: list[str] = []
    points: list[str] = []
    for row in rows:
        value = _finite_number(row.get(key))
        try:
            timestamp_ns = int(row.get("visual_capture_timestamp_ns"))
        except (TypeError, ValueError):
            timestamp_ns = None
        if value is None or timestamp_ns is None:
            if points:
                segments.append(" ".join(points))
                points = []
            continue
        x = x0 + (timestamp_ns - time_min_ns) / time_span * width
        y = y0 + height - (value - value_min) / value_span * height
        points.append(f"{x:.2f},{y:.2f}")
    if points:
        segments.append(" ".join(points))
    return segments


def report_svg(rows: list[dict], summary: dict, *, mode: str = "composite") -> str:
    """Render one source-specific report on the shared visual acquisition axis."""
    report_modes = {
        "camera": {
            "title": "CVIA / CAMERA-DERIVED POSE / MOCAP FRAME",
            "description": "相机首次有效位姿之后，经 T_M_C 转换到动捕坐标系的六轴 PnP 曲线",
            "note": "相机首次有效位姿定义 0 秒；视觉位姿已按 T_M_O = T_M_C × T_C_O 转到动捕坐标系；六轴均按测量绘制，不施加运动方向约束。",
            "show_vision": True,
            "show_mocap": False,
        },
        "mocap": {
            "title": "CVIA / MOCAP-NATIVE POSE / MOCAP FRAME",
            "description": "时间匹配后的 NOKOV 动捕坐标系原生六轴曲线",
            "note": "与相机画布共用 0 秒；只绘制成功时间配对的动捕坐标系位姿，失配区保持断线。",
            "show_vision": False,
            "show_mocap": True,
        },
        "composite": {
            "title": "CVIA / TEMPORAL COMPOSITE / MOCAP FRAME",
            "description": "动捕坐标系中的视觉 PnP 与时间匹配 NOKOV 六轴叠加曲线",
            "note": "两路位姿均在动捕坐标系中，仅作时间趋势叠加、不做数值融合；失配区不跨空档连接动捕。",
            "show_vision": True,
            "show_mocap": True,
        },
    }
    if mode not in report_modes:
        raise ValueError(f"未知离线报告模式：{mode}")
    report = report_modes[mode]
    width, height = 1200, 990
    sampled = _sample_rows(rows)
    time_min_ns = summary.get("start_capture_timestamp_ns") or 0
    time_max_ns = summary.get("end_capture_timestamp_ns") or time_min_ns + 1
    duration = max(0.0, (time_max_ns - time_min_ns) / 1_000_000_000)
    axes = (
        ("X", "visual_x", "mocap_x", "mocap frame / mm"),
        ("Rx", "visual_rx", "mocap_rx", "mocap frame / deg"),
        ("Y", "visual_y", "mocap_y", "mocap frame / mm"),
        ("Ry", "visual_ry", "mocap_ry", "mocap frame / deg"),
        ("Z", "visual_z", "mocap_z", "mocap frame / mm"),
        ("Rz", "visual_rz", "mocap_rz", "mocap frame / deg"),
    )
    estimation = summary.get("offset_estimation") or {}
    if estimation and estimation.get("applied", estimation.get("reliable", False)):
        estimate_note = (
            f'auto offset {float(estimation.get("offset_ms", 0)):+.1f} ms · '
            f'{str(estimation.get("confidence", "unknown")).upper()} · '
            f'corr {float(estimation.get("correlation", 0)):.3f}'
        )
    elif estimation:
        estimate_note = (
            f'rejected auto candidate {float(estimation.get("offset_ms", 0)):+.1f} ms · '
            f'applied offset {float(summary.get("offset_ms", 0)):+.1f} ms'
        )
    else:
        estimate_note = f'manual offset {float(summary.get("offset_ms", 0)):+.1f} ms'
    if mode == "camera":
        metrics = (
            f'visual samples {summary["detection_samples"]} · '
            f'duration {duration:.2f} s · t₀ first valid camera pose'
        )
        legend = (
            '<line x1="960" y1="48" x2="995" y2="48" class="vision"/>'
            '<text class="meta" x="1005" y="52">PNP · T_M_C</text>'
        )
    elif mode == "mocap":
        metrics = (
            f'matched {summary["matched_samples"]}/{summary["detection_samples"]} · '
            f'coverage {summary["coverage_percent"]:.1f}% · '
            f'P95 |Δt| {summary.get("p95_abs_error_ms") if summary.get("p95_abs_error_ms") is not None else "--"} ms · '
            f'duration {duration:.2f} s'
        )
        legend = (
            '<line x1="960" y1="48" x2="995" y2="48" class="mocap"/>'
            '<text class="meta" x="1005" y="52">NOKOV · NATIVE M</text>'
        )
    else:
        metrics = (
            f'matched {summary["matched_samples"]}/{summary["detection_samples"]} · '
            f'coverage {summary["coverage_percent"]:.1f}% · '
            f'P95 |Δt| {summary.get("p95_abs_error_ms") if summary.get("p95_abs_error_ms") is not None else "--"} ms · '
            f'duration {duration:.2f} s'
        )
        legend = (
            '<line x1="790" y1="48" x2="825" y2="48" class="vision"/>'
            '<text class="meta" x="835" y="52">PNP · T_M_C</text>'
            '<line x1="960" y1="48" x2="995" y2="48" class="mocap"/>'
            '<text class="meta" x="1005" y="52">NOKOV · NATIVE M</text>'
        )
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(str(report["title"]))}</title>',
        f'<desc id="desc">{escape(str(report["description"]))}</desc>',
        '<rect width="1200" height="990" fill="#071114"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}.title{fill:#dbe7e9;font-size:22px;font-weight:600}.meta{fill:#758b92;font-size:11px}.label{fill:#b8c8cc;font-size:13px;font-weight:600}.unit{fill:#60767d;font-size:9px}.tick{fill:#60767d;font-size:9px}.grid{stroke:#20343a;stroke-width:1}.frame{fill:#09161a;stroke:#2b4148}.vision{fill:none;stroke:#65d7e4;stroke-width:1.8}.mocap{fill:none;stroke:#9d8cf2;stroke-width:1.8}</style>',
        f'<text class="title" x="48" y="44">{escape(str(report["title"]))}</text>',
        f'<text class="meta" x="48" y="69">{escape(metrics)}</text>',
        legend,
        f'<text class="meta" x="48" y="91">{escape(estimate_note)}</text>',
        f'<text class="meta" x="48" y="108">{escape(str(report["note"]))}</text>',
    ]

    panel_width, panel_height = 540, 260
    for index, (label, visual_key, mocap_key, unit) in enumerate(axes):
        column, row_index = index % 2, index // 2
        panel_x = 40 + column * 575
        panel_y = 125 + row_index * 285
        plot_x, plot_y = panel_x + 56, panel_y + 42
        plot_width, plot_height = panel_width - 76, panel_height - 76
        visible_keys = []
        if report["show_vision"]:
            visible_keys.append(visual_key)
        if report["show_mocap"]:
            visible_keys.append(mocap_key)
        values = [
            number
            for row in sampled
            for key in visible_keys
            if (number := _finite_number(row.get(key))) is not None
        ]
        if values:
            value_min, value_max = min(values), max(values)
            padding = max((value_max - value_min) * 0.08, 1e-6)
            value_min -= padding
            value_max += padding
        else:
            value_min, value_max = -1.0, 1.0
        elements.extend([
            f'<rect class="frame" x="{panel_x}" y="{panel_y}" width="{panel_width}" height="{panel_height}"/>',
            f'<text class="label" x="{panel_x + 16}" y="{panel_y + 24}">{escape(label)}</text>',
            f'<text class="unit" x="{panel_x + 48}" y="{panel_y + 24}">{escape(unit)}</text>',
        ])
        for grid_index in range(5):
            grid_y = plot_y + plot_height * grid_index / 4
            grid_value = value_max - (value_max - value_min) * grid_index / 4
            elements.append(
                f'<line class="grid" x1="{plot_x}" y1="{grid_y:.2f}" x2="{plot_x + plot_width}" y2="{grid_y:.2f}"/>'
            )
            elements.append(
                f'<text class="tick" x="{plot_x - 7}" y="{grid_y + 3:.2f}" text-anchor="end">{grid_value:.3g}</text>'
            )
        for grid_index in range(5):
            grid_x = plot_x + plot_width * grid_index / 4
            grid_time = duration * grid_index / 4
            elements.append(
                f'<line class="grid" x1="{grid_x:.2f}" y1="{plot_y}" x2="{grid_x:.2f}" y2="{plot_y + plot_height}"/>'
            )
            elements.append(
                f'<text class="tick" x="{grid_x:.2f}" y="{plot_y + plot_height + 17}" text-anchor="middle">{grid_time:.2f}s</text>'
            )
        rendered_segments = 0
        if sampled and report["show_vision"]:
            vision_segments = _polyline_segments(
                sampled, visual_key, x0=plot_x, y0=plot_y,
                width=plot_width, height=plot_height,
                time_min_ns=time_min_ns, time_max_ns=time_max_ns,
                value_min=value_min, value_max=value_max,
            )
            elements.extend(
                f'<polyline class="vision" points="{points}"/>'
                for points in vision_segments
            )
            rendered_segments += len(vision_segments)
        if sampled and report["show_mocap"]:
            mocap_segments = _polyline_segments(
                sampled, mocap_key, x0=plot_x, y0=plot_y,
                width=plot_width, height=plot_height,
                time_min_ns=time_min_ns, time_max_ns=time_max_ns,
                value_min=value_min, value_max=value_max,
            )
            elements.extend(
                f'<polyline class="mocap" points="{points}"/>'
                for points in mocap_segments
            )
            rendered_segments += len(mocap_segments)
        if rendered_segments == 0:
            empty_message = (
                "NO MATCHED MOCAP SAMPLES"
                if report["show_mocap"] and not report["show_vision"]
                else "NO VISUAL POSE SAMPLES"
            )
            elements.append(
                f'<text class="meta" x="{plot_x + plot_width / 2}" y="{plot_y + plot_height / 2}" text-anchor="middle">{empty_message}</text>'
            )
    elements.append('</svg>')
    return "\n".join(elements)


def generate_offline_report(
    detection_path: Path,
    mocap_path: Path,
    *,
    offset_ms: float,
    max_error_ms: float,
    interpolate: bool,
    offset_estimation: dict | None = None,
) -> tuple[dict, str, str]:
    """Compatibility wrapper returning the historical composite SVG."""
    summary, csv_payload, reports = generate_offline_reports(
        detection_path,
        mocap_path,
        offset_ms=offset_ms,
        max_error_ms=max_error_ms,
        interpolate=interpolate,
        offset_estimation=offset_estimation,
    )
    return summary, csv_payload, reports["composite"]


def generate_offline_reports(
    detection_path: Path,
    mocap_path: Path,
    *,
    offset_ms: float,
    max_error_ms: float,
    interpolate: bool,
    offset_estimation: dict | None = None,
) -> tuple[dict, str, dict[str, str]]:
    """Generate camera, mocap and composite canvases from one synchronized pass."""
    detections = load_detection_poses(detection_path)
    mocap = load_mocap_poses(mocap_path)
    rows, summary = synchronize(
        detections,
        mocap,
        offset_ms=offset_ms,
        max_error_ms=max_error_ms,
        interpolate=interpolate,
        retain_unmatched=True,
    )
    if offset_estimation is not None:
        summary["offset_estimation"] = offset_estimation
    reports = {
        mode: report_svg(rows, summary, mode=mode)
        for mode in ("camera", "mocap", "composite")
    }
    return summary, synced_csv(rows), reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按 Orin 本机时间同步视觉 PnP 与 NOKOV 动捕并生成六轴 SVG"
    )
    parser.add_argument("--detection", type=Path, required=True, help="视觉 PnP CSV")
    parser.add_argument("--mocap", type=Path, required=True, help="动捕会话 CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="报告输出目录")
    parser.add_argument("--offset-ms", type=float, default=0.0, help="加到动捕接收时间的固定补偿")
    parser.add_argument("--max-error-ms", type=float, default=12.0, help="最近帧最大允许时间误差")
    parser.add_argument("--nearest-only", action="store_true", help="禁用位置/四元数插值")
    parser.add_argument(
        "--estimate-offset",
        action="store_true",
        help="先用运动强度归一化互相关自动估计时间补偿",
    )
    parser.add_argument(
        "--search-ms",
        type=float,
        default=1000.0,
        help="自动估计时搜索的正负时间范围；默认 1000 ms",
    )
    args = parser.parse_args()
    if args.max_error_ms <= 0:
        parser.error("--max-error-ms 必须大于 0")
    offset_ms = args.offset_ms
    estimation = None
    if args.estimate_offset:
        detections = load_detection_poses(args.detection)
        mocap = load_mocap_poses(args.mocap)
        estimation = estimate_time_offset(
            detections,
            mocap,
            max_offset_ms=args.search_ms,
        )
        if not estimation["reliable"]:
            raise SystemExit(
                f"自动估计未通过可靠性检查：{estimation['message']} "
                f"(correlation={estimation['correlation']:.3f}, "
                f"margin={estimation['peak_margin']:.3f})"
            )
        offset_ms = estimation["offset_ms"]
        rows, summary = synchronize(
            detections,
            mocap,
            offset_ms=offset_ms,
            max_error_ms=args.max_error_ms,
            interpolate=not args.nearest_only,
            retain_unmatched=True,
        )
        if estimation is not None:
            summary["offset_estimation"] = estimation
        csv_payload = synced_csv(rows)
        reports = {
            mode: report_svg(rows, summary, mode=mode)
            for mode in ("camera", "mocap", "composite")
        }
    else:
        summary, csv_payload, reports = generate_offline_reports(
            args.detection,
            args.mocap,
            offset_ms=offset_ms,
            max_error_ms=args.max_error_ms,
            interpolate=not args.nearest_only,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "synchronized_session.csv").write_text(csv_payload, encoding="utf-8")
    for filename, mode in (
        ("synchronized_camera_report.svg", "camera"),
        ("synchronized_mocap_report.svg", "mocap"),
        # Preserve the historical filename as the composite canvas.
        ("synchronized_report.svg", "composite"),
    ):
        (args.output_dir / filename).write_text(reports[mode], encoding="utf-8")
    (args.output_dir / "synchronized_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
