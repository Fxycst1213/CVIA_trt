"""Offline visual/mocap pairing and dependency-free SVG/PNG report generation."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import struct
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from xml.sax.saxutils import escape

from mocap_receiver import (
    MocapPose,
    MocapTimeline,
    align_mocap_poses,
    pose_validity,
    quaternion_to_euler_xyz_degrees,
)


SYNCED_COLUMNS = (
    "coordinate_frame",
    "visual_timestamp_ms",
    "visual_capture_timestamp_ns",
    "visual_publish_timestamp_ns",
    "visual_pipeline_latency_ms",
    "visual_interpolated",
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
CAMERA_REPORT_HZ = 300.0


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
    interpolated: bool = False

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


def stabilize_detection_poses(
    poses: list[DetectionPose],
) -> tuple[list[DetectionPose], int]:
    """Reject isolated SE(3) jumps while allowing persistent reacquisition.

    This mirrors the runtime gate so reports created from older CSV sessions do
    not connect one-frame SQPnP branch flips into physically impossible spikes.
    A genuinely relocated target is accepted after three mutually consistent
    candidates instead of being rejected forever.
    """
    ordered = sorted(poses, key=lambda pose: pose.effective_timestamp_ns)
    if not ordered:
        return [], 0

    accepted = ordered[0]
    stable = [accepted]
    pending: DetectionPose | None = None
    pending_count = 0
    rejected = 0

    def deltas(first: DetectionPose, second: DetectionPose) -> tuple[float, float]:
        position = math.sqrt(
            (second.x - first.x) ** 2
            + (second.y - first.y) ** 2
            + (second.z - first.z) ** 2
        )
        rotation = math.degrees(_quaternion_angular_distance(
            _euler_xyz_degrees_to_quaternion(first.rx, first.ry, first.rz),
            _euler_xyz_degrees_to_quaternion(second.rx, second.ry, second.rz),
        ))
        return position, rotation

    def elapsed_seconds(first: DetectionPose, second: DetectionPose) -> float:
        elapsed = (second.effective_timestamp_ns - first.effective_timestamp_ns) / 1e9
        return max(0.001, min(elapsed, 1.5))

    for pose in ordered[1:]:
        dt = elapsed_seconds(accepted, pose)
        position_delta, rotation_delta = deltas(accepted, pose)
        continuous = (
            position_delta <= 60.0 + 140.0 * dt
            and rotation_delta <= 12.0 + 30.0 * dt
        )
        if continuous:
            stable.append(pose)
            accepted = pose
            pending = None
            pending_count = 0
            continue

        agrees_with_pending = False
        if pending is not None:
            pending_dt = elapsed_seconds(pending, pose)
            pending_position, pending_rotation = deltas(pending, pose)
            agrees_with_pending = (
                pending_position <= 100.0 + 300.0 * pending_dt
                and pending_rotation <= 20.0 + 90.0 * pending_dt
            )
        pending_count = pending_count + 1 if agrees_with_pending else 1
        pending = pose
        if pending_count >= 3:
            stable.append(pose)
            accepted = pose
            pending = None
            pending_count = 0
        else:
            rejected += 1
    return stable, rejected


def _slerp_quaternions(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    first = _normalize_quaternion(first)
    second = _normalize_quaternion(second)
    dot = sum(a * b for a, b in zip(first, second))
    if dot < 0.0:
        second = tuple(-value for value in second)
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return _normalize_quaternion(tuple(
            a + alpha * (b - a) for a, b in zip(first, second)
        ))
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    first_weight = math.sin((1.0 - alpha) * theta) / sin_theta
    second_weight = math.sin(alpha * theta) / sin_theta
    return tuple(
        first_weight * a + second_weight * b
        for a, b in zip(first, second)
    )


def interpolate_detection_pose_outliers(
    poses: list[DetectionPose],
) -> tuple[list[DetectionPose], dict[str, int]]:
    """Repair bounded rejected runs from their trusted temporal neighbours."""
    ordered = sorted(poses, key=lambda pose: pose.effective_timestamp_ns)
    stable, _ = stabilize_detection_poses(ordered)
    accepted_ids = {id(pose) for pose in stable}
    repaired: list[DetectionPose] = []
    interpolated_count = 0
    dropped_count = 0
    index = 0
    while index < len(ordered):
        pose = ordered[index]
        if id(pose) in accepted_ids:
            repaired.append(pose)
            index += 1
            continue

        run_start = index
        while index < len(ordered) and id(ordered[index]) not in accepted_ids:
            index += 1
        run_end = index
        previous = ordered[run_start - 1] if run_start > 0 else None
        following = ordered[run_end] if run_end < len(ordered) else None
        if (
            previous is None
            or following is None
            or id(previous) not in accepted_ids
            or id(following) not in accepted_ids
            or following.effective_timestamp_ns <= previous.effective_timestamp_ns
        ):
            dropped_count += run_end - run_start
            continue

        first_quaternion = _euler_xyz_degrees_to_quaternion(
            previous.rx, previous.ry, previous.rz
        )
        second_quaternion = _euler_xyz_degrees_to_quaternion(
            following.rx, following.ry, following.rz
        )
        duration_ns = following.effective_timestamp_ns - previous.effective_timestamp_ns
        for rejected_pose in ordered[run_start:run_end]:
            alpha = (
                rejected_pose.effective_timestamp_ns - previous.effective_timestamp_ns
            ) / duration_ns
            alpha = min(1.0, max(0.0, alpha))
            quaternion = _slerp_quaternions(
                first_quaternion, second_quaternion, alpha
            )
            rx, ry, rz = quaternion_to_euler_xyz_degrees(*quaternion)
            repaired.append(replace(
                rejected_pose,
                x=previous.x + (following.x - previous.x) * alpha,
                y=previous.y + (following.y - previous.y) * alpha,
                z=previous.z + (following.z - previous.z) * alpha,
                rx=rx,
                ry=ry,
                rz=rz,
                interpolated=True,
            ))
            interpolated_count += 1
    repaired.sort(key=lambda pose: pose.effective_timestamp_ns)
    return repaired, {
        "interpolated": interpolated_count,
        "dropped": dropped_count,
        "flagged": len(ordered) - len(stable),
    }


def resample_detection_poses(
    poses: list[DetectionPose],
    *,
    target_hz: float = 300.0,
    max_samples: int = 1_000_000,
) -> list[DetectionPose]:
    """Resample camera poses on a uniform report-only SE(3) timeline.

    Translation is linear in capture time and rotation uses quaternion SLERP.
    Source samples that coincide with the uniform grid are retained unchanged.
    The result is used only for synchronization and plotting; the persisted
    detection CSV remains the original camera-rate measurement stream.
    """
    if not math.isfinite(target_hz) or not 1.0 <= target_hz <= 1_000.0:
        raise ValueError("绘图重采样频率必须在 1～1000 Hz")
    if max_samples < 2:
        raise ValueError("绘图重采样最大点数必须至少为 2")

    ordered: list[DetectionPose] = []
    for pose in sorted(poses, key=lambda item: item.effective_timestamp_ns):
        if ordered and pose.effective_timestamp_ns == ordered[-1].effective_timestamp_ns:
            ordered[-1] = pose
        else:
            ordered.append(pose)
    if len(ordered) < 2:
        return ordered

    start_ns = ordered[0].effective_timestamp_ns
    end_ns = ordered[-1].effective_timestamp_ns
    duration_ns = end_ns - start_ns
    interval_count = int(math.floor(duration_ns * target_hz / 1_000_000_000))
    sample_times = [
        start_ns + round(index * 1_000_000_000 / target_hz)
        for index in range(interval_count + 1)
    ]
    if sample_times[-1] < end_ns:
        sample_times.append(end_ns)
    else:
        sample_times[-1] = end_ns
    if len(sample_times) > max_samples:
        raise ValueError(
            f"300 Hz 绘图需要 {len(sample_times)} 个插值点，超过安全上限 {max_samples}；"
            "请缩短会话后重试"
        )

    result: list[DetectionPose] = []
    source_index = 0
    for timestamp_ns in sample_times:
        while (
            source_index + 1 < len(ordered)
            and ordered[source_index + 1].effective_timestamp_ns <= timestamp_ns
        ):
            source_index += 1
        previous = ordered[source_index]
        if previous.effective_timestamp_ns == timestamp_ns or source_index + 1 >= len(ordered):
            result.append(previous)
            continue
        following = ordered[source_index + 1]
        span_ns = following.effective_timestamp_ns - previous.effective_timestamp_ns
        alpha = (timestamp_ns - previous.effective_timestamp_ns) / span_ns
        quaternion = _slerp_quaternions(
            _euler_xyz_degrees_to_quaternion(previous.rx, previous.ry, previous.rz),
            _euler_xyz_degrees_to_quaternion(following.rx, following.ry, following.rz),
            alpha,
        )
        rx, ry, rz = quaternion_to_euler_xyz_degrees(*quaternion)
        result.append(DetectionPose(
            timestamp_ms=round(timestamp_ns / 1_000_000),
            capture_timestamp_ns=timestamp_ns,
            publish_timestamp_ns=0,
            coordinate_frame=previous.coordinate_frame,
            x=previous.x + (following.x - previous.x) * alpha,
            y=previous.y + (following.y - previous.y) * alpha,
            z=previous.z + (following.z - previous.z) * alpha,
            rx=rx,
            ry=ry,
            rz=rz,
            interpolated=True,
        ))
    return result


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
            "visual_interpolated": int(detection.interpolated),
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


def _nearest_equivalent_angle(angle: float, reference: float) -> float:
    return angle + 360.0 * round((reference - angle) / 360.0)


def _nearest_equivalent_euler(
    raw: tuple[float, float, float],
    previous: tuple[float, float, float],
) -> tuple[float, float, float]:
    candidates = (
        raw,
        (raw[0] + 180.0, 180.0 - raw[1], raw[2] + 180.0),
    )
    expanded = [
        tuple(_nearest_equivalent_angle(value, reference)
              for value, reference in zip(candidate, previous))
        for candidate in candidates
    ]
    return min(
        expanded,
        key=lambda candidate: sum(
            (value - reference) ** 2
            for value, reference in zip(candidate, previous)
        ),
    )


def continuous_euler_rows(rows: list[dict]) -> list[dict]:
    """Use the nearest equivalent XYZ Euler representation for plotting."""
    states: dict[str, tuple[float, float, float]] = {}
    result = []
    for source_row in rows:
        row = dict(source_row)
        for prefix in ("visual", "mocap"):
            values = tuple(
                _finite_number(row.get(f"{prefix}_{axis}"))
                for axis in ("rx", "ry", "rz")
            )
            if any(value is None for value in values):
                continue
            raw = tuple(float(value) for value in values)
            continuous = (
                _nearest_equivalent_euler(raw, states[prefix])
                if prefix in states
                else raw
            )
            for axis, value in zip(("rx", "ry", "rz"), continuous):
                row[f"{prefix}_{axis}"] = value
            states[prefix] = continuous
        result.append(row)
    return result


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
    break_on_time_gap: bool = True,
) -> list[str]:
    """Build SVG polylines, optionally preserving real source outages.

    Visual poses can be intentionally sparse (for example, a folder containing
    only a dozen test images). Connecting two measured visual poses is the
    report's piecewise-linear interpolation and must not be confused with a
    missing sample. Mocap values remain discontinuous across unmatched
    intervals so the report never invents tracking data.
    """
    time_span = max(1, time_max_ns - time_min_ns)
    value_span = max(1e-12, value_max - value_min)
    segments: list[str] = []
    points: list[str] = []
    valid_timestamps = []
    for row in rows:
        if _finite_number(row.get(key)) is None:
            continue
        try:
            valid_timestamps.append(int(row.get("visual_capture_timestamp_ns")))
        except (TypeError, ValueError):
            pass
    gaps = sorted(
        current - previous
        for previous, current in zip(valid_timestamps, valid_timestamps[1:])
        if current > previous
    )
    median_gap = gaps[len(gaps) // 2] if gaps else 0
    gap_limit_ns = max(100_000_000, median_gap * 3) if median_gap else None
    previous_timestamp_ns = None
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
        if (break_on_time_gap and previous_timestamp_ns is not None and
                gap_limit_ns is not None and
                timestamp_ns - previous_timestamp_ns > gap_limit_ns):
            if points:
                segments.append(" ".join(points))
                points = []
        x = x0 + (timestamp_ns - time_min_ns) / time_span * width
        y = y0 + height - (value - value_min) / value_span * height
        points.append(f"{x:.2f},{y:.2f}")
        previous_timestamp_ns = timestamp_ns
    if points:
        segments.append(" ".join(points))
    return segments


_FONT_5X7 = {
    " ": ("00000",) * 7,
    "A": ("01110","10001","10001","11111","10001","10001","10001"),
    "B": ("11110","10001","10001","11110","10001","10001","11110"),
    "C": ("01111","10000","10000","10000","10000","10000","01111"),
    "D": ("11110","10001","10001","10001","10001","10001","11110"),
    "E": ("11111","10000","10000","11110","10000","10000","11111"),
    "F": ("11111","10000","10000","11110","10000","10000","10000"),
    "G": ("01111","10000","10000","10111","10001","10001","01111"),
    "H": ("10001","10001","10001","11111","10001","10001","10001"),
    "I": ("11111","00100","00100","00100","00100","00100","11111"),
    "J": ("00111","00010","00010","00010","10010","10010","01100"),
    "K": ("10001","10010","10100","11000","10100","10010","10001"),
    "L": ("10000","10000","10000","10000","10000","10000","11111"),
    "M": ("10001","11011","10101","10101","10001","10001","10001"),
    "N": ("10001","11001","10101","10011","10001","10001","10001"),
    "O": ("01110","10001","10001","10001","10001","10001","01110"),
    "P": ("11110","10001","10001","11110","10000","10000","10000"),
    "Q": ("01110","10001","10001","10001","10101","10010","01101"),
    "R": ("11110","10001","10001","11110","10100","10010","10001"),
    "S": ("01111","10000","10000","01110","00001","00001","11110"),
    "T": ("11111","00100","00100","00100","00100","00100","00100"),
    "U": ("10001","10001","10001","10001","10001","10001","01110"),
    "V": ("10001","10001","10001","10001","10001","01010","00100"),
    "W": ("10001","10001","10001","10101","10101","10101","01010"),
    "X": ("10001","10001","01010","00100","01010","10001","10001"),
    "Y": ("10001","10001","01010","00100","00100","00100","00100"),
    "Z": ("11111","00001","00010","00100","01000","10000","11111"),
    "0": ("01110","10001","10011","10101","11001","10001","01110"),
    "1": ("00100","01100","00100","00100","00100","00100","01110"),
    "2": ("01110","10001","00001","00010","00100","01000","11111"),
    "3": ("11110","00001","00001","01110","00001","00001","11110"),
    "4": ("00010","00110","01010","10010","11111","00010","00010"),
    "5": ("11111","10000","10000","11110","00001","00001","11110"),
    "6": ("01110","10000","10000","11110","10001","10001","01110"),
    "7": ("11111","00001","00010","00100","01000","01000","01000"),
    "8": ("01110","10001","10001","01110","10001","10001","01110"),
    "9": ("01110","10001","10001","01111","00001","00001","01110"),
    ".": ("00000","00000","00000","00000","00000","00110","00110"),
    ",": ("00000","00000","00000","00000","00110","00100","01000"),
    ":": ("00000","00110","00110","00000","00110","00110","00000"),
    "+": ("00000","00100","00100","11111","00100","00100","00000"),
    "-": ("00000","00000","00000","11111","00000","00000","00000"),
    "/": ("00001","00010","00010","00100","01000","01000","10000"),
    "%": ("11001","11010","00100","01000","10110","00110","00000"),
    "(": ("00010","00100","01000","01000","01000","00100","00010"),
    ")": ("01000","00100","00010","00010","00010","00100","01000"),
    "=": ("00000","11111","00000","11111","00000","00000","00000"),
    "_": ("00000","00000","00000","00000","00000","00000","11111"),
    "?": ("01110","10001","00001","00010","00100","00000","00100"),
}


class _RasterCanvas:
    """Small RGB canvas used to keep customer PNG generation dependency-free."""

    def __init__(self, width: int, height: int, background: tuple[int, int, int]):
        self.width = width
        self.height = height
        self.pixels = bytearray(bytes(background) * width * height)

    def fill_rect(self, x: int, y: int, width: int, height: int,
                  color: tuple[int, int, int]) -> None:
        left, top = max(0, x), max(0, y)
        right, bottom = min(self.width, x + width), min(self.height, y + height)
        if left >= right or top >= bottom:
            return
        row = bytes(color) * (right - left)
        for py in range(top, bottom):
            start = (py * self.width + left) * 3
            self.pixels[start:start + len(row)] = row

    def line(self, x0: float, y0: float, x1: float, y1: float,
             color: tuple[int, int, int], thickness: int = 1) -> None:
        x0i, y0i, x1i, y1i = map(lambda value: int(round(value)), (x0, y0, x1, y1))
        dx, dy = abs(x1i - x0i), -abs(y1i - y0i)
        step_x = 1 if x0i < x1i else -1
        step_y = 1 if y0i < y1i else -1
        error = dx + dy
        radius = max(0, thickness // 2)
        while True:
            self.fill_rect(x0i - radius, y0i - radius,
                           max(1, thickness), max(1, thickness), color)
            if x0i == x1i and y0i == y1i:
                break
            doubled = 2 * error
            if doubled >= dy:
                error += dy
                x0i += step_x
            if doubled <= dx:
                error += dx
                y0i += step_y

    def polyline(self, points: list[tuple[float, float]],
                 color: tuple[int, int, int], thickness: int = 2) -> None:
        for start, end in zip(points, points[1:]):
            self.line(start[0], start[1], end[0], end[1], color, thickness)

    def text(self, x: int, y: int, value: str, color: tuple[int, int, int],
             scale: int = 1) -> None:
        cursor = x
        for character in value.upper():
            glyph = _FONT_5X7.get(character, _FONT_5X7["?"])
            for row_index, row in enumerate(glyph):
                for column_index, bit in enumerate(row):
                    if bit == "1":
                        self.fill_rect(cursor + column_index * scale,
                                       y + row_index * scale,
                                       scale, scale, color)
            cursor += 6 * scale

    @staticmethod
    def _chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind)
        checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
        return struct.pack("!I", len(payload)) + kind + payload + struct.pack("!I", checksum)

    def png(self) -> bytes:
        stride = self.width * 3
        raw = b"".join(
            b"\x00" + self.pixels[offset:offset + stride]
            for offset in range(0, len(self.pixels), stride)
        )
        header = struct.pack("!IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        return (
            b"\x89PNG\r\n\x1a\n"
            + self._chunk(b"IHDR", header)
            + self._chunk(b"IDAT", zlib.compress(raw, 6))
            + self._chunk(b"IEND", b"")
        )


def _numeric_segments(rows: list[dict], key: str, *, x0: float, y0: float,
                      width: float, height: float, time_min_ns: int,
                      time_max_ns: int, value_min: float,
                      value_max: float,
                      break_on_time_gap: bool = True
                      ) -> list[list[tuple[float, float]]]:
    time_span = max(1, time_max_ns - time_min_ns)
    value_span = max(1e-12, value_max - value_min)
    segments: list[list[tuple[float, float]]] = []
    points: list[tuple[float, float]] = []
    valid_timestamps = []
    for row in rows:
        if _finite_number(row.get(key)) is None:
            continue
        try:
            valid_timestamps.append(int(row.get("visual_capture_timestamp_ns")))
        except (TypeError, ValueError):
            pass
    gaps = sorted(
        current - previous
        for previous, current in zip(valid_timestamps, valid_timestamps[1:])
        if current > previous
    )
    median_gap = gaps[len(gaps) // 2] if gaps else 0
    gap_limit_ns = max(100_000_000, median_gap * 3) if median_gap else None
    previous_timestamp_ns = None
    for row in rows:
        value = _finite_number(row.get(key))
        try:
            timestamp_ns = int(row.get("visual_capture_timestamp_ns"))
        except (TypeError, ValueError):
            timestamp_ns = None
        if value is None or timestamp_ns is None:
            if points:
                segments.append(points)
                points = []
            continue
        if (break_on_time_gap and previous_timestamp_ns is not None and
                gap_limit_ns is not None and
                timestamp_ns - previous_timestamp_ns > gap_limit_ns):
            if points:
                segments.append(points)
                points = []
        points.append((
            x0 + (timestamp_ns - time_min_ns) / time_span * width,
            y0 + height - (value - value_min) / value_span * height,
        ))
        previous_timestamp_ns = timestamp_ns
    if points:
        segments.append(points)
    return segments


def report_png(rows: list[dict], summary: dict, *, mode: str = "composite") -> bytes:
    """Render a real 1200x990 RGB PNG without optional plotting packages."""
    modes = {
        "camera": ("CVIA / CAMERA-DERIVED POSE / MOCAP FRAME", True, False),
        "mocap": ("CVIA / MOCAP-NATIVE POSE / MOCAP FRAME", False, True),
        "composite": ("CVIA / TEMPORAL COMPOSITE / MOCAP FRAME", True, True),
    }
    if mode not in modes:
        raise ValueError(f"未知离线报告模式：{mode}")
    title, show_vision, show_mocap = modes[mode]
    canvas = _RasterCanvas(1200, 990, (7, 17, 20))
    colors = {
        "panel": (9, 22, 26), "frame": (43, 65, 72), "grid": (32, 52, 58),
        "title": (219, 231, 233), "label": (184, 200, 204),
        "muted": (96, 118, 125), "vision": (101, 215, 228),
        "mocap": (157, 140, 242),
    }
    sampled = continuous_euler_rows(_sample_rows(rows))
    time_min_ns = int(summary.get("start_capture_timestamp_ns") or 0)
    time_max_ns = int(summary.get("end_capture_timestamp_ns") or time_min_ns + 1)
    duration = max(0.0, (time_max_ns - time_min_ns) / 1_000_000_000)
    canvas.text(48, 35, title, colors["title"], 2)
    metrics = (
        f'SAMPLES {summary.get("detection_samples", 0)}  REPAIRED '
        f'{summary.get("interpolated_pose_outliers", 0)}  DROPPED '
        f'{summary.get("unrecoverable_pose_outliers", 0)}  MATCHED '
        f'{summary.get("matched_samples", 0)}  COVERAGE '
        f'{float(summary.get("coverage_percent", 0)):.1f}%  DURATION {duration:.2f} S'
    )
    canvas.text(48, 67, metrics, colors["muted"])
    canvas.text(48, 88,
                f'OFFSET {float(summary.get("offset_ms", 0)):+.1f} MS  '
                'TIME ZERO = FIRST VALID CAMERA POSE  '
                f'PLOT {float(summary.get("camera_plot_hz", 0)):.0f} HZ / '
                f'{summary.get("camera_plot_samples", 0)} PTS', colors["muted"])
    legend_x = 790
    if show_vision:
        canvas.line(legend_x, 49, legend_x + 34, 49, colors["vision"], 2)
        canvas.text(legend_x + 44, 45, "PNP / T_M_C", colors["muted"])
        legend_x += 190
    if show_mocap:
        canvas.line(legend_x, 49, legend_x + 34, 49, colors["mocap"], 2)
        canvas.text(legend_x + 44, 45, "NOKOV / NATIVE M", colors["muted"])

    axes = (
        ("X", "visual_x", "mocap_x", "MOCAP FRAME / MM"),
        ("RX", "visual_rx", "mocap_rx", "MOCAP FRAME / DEG"),
        ("Y", "visual_y", "mocap_y", "MOCAP FRAME / MM"),
        ("RY", "visual_ry", "mocap_ry", "MOCAP FRAME / DEG"),
        ("Z", "visual_z", "mocap_z", "MOCAP FRAME / MM"),
        ("RZ", "visual_rz", "mocap_rz", "MOCAP FRAME / DEG"),
    )
    panel_width, panel_height = 540, 260
    for index, (label, visual_key, mocap_key, unit) in enumerate(axes):
        column, row_index = index % 2, index // 2
        panel_x, panel_y = 40 + column * 575, 125 + row_index * 285
        plot_x, plot_y = panel_x + 56, panel_y + 42
        plot_width, plot_height = panel_width - 76, panel_height - 76
        canvas.fill_rect(panel_x, panel_y, panel_width, panel_height, colors["panel"])
        canvas.line(panel_x, panel_y, panel_x + panel_width, panel_y, colors["frame"])
        canvas.line(panel_x, panel_y + panel_height, panel_x + panel_width,
                    panel_y + panel_height, colors["frame"])
        canvas.line(panel_x, panel_y, panel_x, panel_y + panel_height, colors["frame"])
        canvas.line(panel_x + panel_width, panel_y, panel_x + panel_width,
                    panel_y + panel_height, colors["frame"])
        canvas.text(panel_x + 16, panel_y + 14, label, colors["label"], 2)
        canvas.text(panel_x + 62, panel_y + 18, unit, colors["muted"])
        visible_keys = ([visual_key] if show_vision else []) + ([mocap_key] if show_mocap else [])
        values = [number for row in sampled for key in visible_keys
                  if (number := _finite_number(row.get(key))) is not None]
        if values:
            value_min, value_max = min(values), max(values)
            padding = max((value_max - value_min) * 0.08, 1e-6)
            value_min -= padding
            value_max += padding
        else:
            value_min, value_max = -1.0, 1.0
        for grid_index in range(5):
            grid_y = plot_y + plot_height * grid_index / 4
            grid_x = plot_x + plot_width * grid_index / 4
            grid_value = value_max - (value_max - value_min) * grid_index / 4
            canvas.line(plot_x, grid_y, plot_x + plot_width, grid_y, colors["grid"])
            canvas.line(grid_x, plot_y, grid_x, plot_y + plot_height, colors["grid"])
            canvas.text(panel_x + 4, int(grid_y) - 3, f"{grid_value:.3G}", colors["muted"])
            canvas.text(int(grid_x) - 12, int(plot_y + plot_height + 10),
                        f"{duration * grid_index / 4:.2F}S", colors["muted"])
        rendered = 0
        for enabled, key, color in (
            (show_vision, visual_key, colors["vision"]),
            (show_mocap, mocap_key, colors["mocap"]),
        ):
            if not enabled:
                continue
            segments = _numeric_segments(
                sampled, key, x0=plot_x, y0=plot_y, width=plot_width,
                height=plot_height, time_min_ns=time_min_ns,
                time_max_ns=time_max_ns, value_min=value_min, value_max=value_max,
                break_on_time_gap=key.startswith("mocap_"),
            )
            for points in segments:
                canvas.polyline(points, color, 2)
            rendered += len(segments)
        if rendered == 0:
            canvas.text(plot_x + 120, plot_y + 88, "NO VALID SAMPLES", colors["muted"])
    return canvas.png()


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
    sampled = continuous_euler_rows(_sample_rows(rows))
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
            f'plot {float(summary.get("camera_plot_hz", 0)):.0f} Hz / '
            f'{summary.get("camera_plot_samples", 0)} points · '
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
            f'plot {float(summary.get("camera_plot_hz", 0)):.0f} Hz · '
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
                break_on_time_gap=False,
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
                break_on_time_gap=True,
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
    mocap_path: Path | None,
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
    mocap_path: Path | None,
    *,
    offset_ms: float,
    max_error_ms: float,
    interpolate: bool,
    offset_estimation: dict | None = None,
    report_format: str = "svg",
) -> tuple[dict, str, dict[str, str] | dict[str, bytes]]:
    """Generate source-specific canvases; mocap is optional for camera plots."""
    raw_detections = load_detection_poses(detection_path)
    detections, interpolation = interpolate_detection_pose_outliers(raw_detections)
    mocap = (
        load_mocap_poses(mocap_path)
        if mocap_path is not None and mocap_path.is_file()
        else []
    )
    rows, summary = synchronize(
        detections,
        mocap,
        offset_ms=offset_ms,
        max_error_ms=max_error_ms,
        interpolate=interpolate,
        retain_unmatched=True,
    )
    camera_plot_hz = CAMERA_REPORT_HZ
    plot_detections = resample_detection_poses(
        detections,
        target_hz=camera_plot_hz,
    )
    plot_rows, _plot_summary = synchronize(
        plot_detections,
        mocap,
        offset_ms=offset_ms,
        max_error_ms=max_error_ms,
        interpolate=interpolate,
        retain_unmatched=True,
    )
    summary["raw_detection_samples"] = len(raw_detections)
    summary["rejected_pose_outliers"] = interpolation["flagged"]
    summary["interpolated_pose_outliers"] = interpolation["interpolated"]
    summary["unrecoverable_pose_outliers"] = interpolation["dropped"]
    summary["camera_source_samples"] = len(detections)
    summary["camera_plot_hz"] = camera_plot_hz
    summary["camera_plot_samples"] = len(plot_detections)
    source_timestamps = {
        pose.effective_timestamp_ns for pose in detections
    }
    summary["camera_plot_interpolated_samples"] = sum(
        pose.effective_timestamp_ns not in source_timestamps
        for pose in plot_detections
    )
    if offset_estimation is not None:
        summary["offset_estimation"] = offset_estimation
    if report_format == "svg":
        reports = {
            mode: report_svg(plot_rows, summary, mode=mode)
            for mode in ("camera", "mocap", "composite")
        }
    elif report_format == "png":
        reports = {
            mode: report_png(plot_rows, summary, mode=mode)
            for mode in ("camera", "mocap", "composite")
        }
    else:
        raise ValueError("report_format 只能是 svg 或 png")
    return summary, synced_csv(rows), reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按 Orin 本机时间同步视觉 PnP 与 NOKOV 动捕并生成三张六轴 PNG"
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
        detections, _ = stabilize_detection_poses(
            load_detection_poses(args.detection)
        )
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
            mode: report_png(rows, summary, mode=mode)
            for mode in ("camera", "mocap", "composite")
        }
    else:
        summary, csv_payload, reports = generate_offline_reports(
            args.detection,
            args.mocap,
            offset_ms=offset_ms,
            max_error_ms=args.max_error_ms,
            interpolate=not args.nearest_only,
            report_format="png",
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "synchronized_session.csv").write_text(csv_payload, encoding="utf-8")
    for filename, mode in (
        ("synchronized_camera_report.png", "camera"),
        ("synchronized_mocap_report.png", "mocap"),
        # Preserve the historical filename as the composite canvas.
        ("synchronized_report.png", "composite"),
    ):
        (args.output_dir / filename).write_bytes(reports[mode])
    (args.output_dir / "synchronized_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
