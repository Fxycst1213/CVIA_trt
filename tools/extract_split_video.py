"""视频左右拆帧，以及连续图像中的飞机圆形靶标中心标注。

默认保留原拆帧用法。标注入口：
    python extract_split_video.py --mode annotate --images IMAGE_DIRECTORY
第一张图须已有完整的 Labelme 0-10 点和 target 框，作为身份参考。
标注结果为图片旁的同名 JSON；质量报告与预览默认位于旁边的目录。
完整使用说明见 aircraft_annotation_usage.txt。
"""

from pathlib import Path

# ================== 常用设置：后续主要修改这里 ==================
VIDEO_PATH = r"C:\Users\10731\Desktop\terst1118\video\WIN_20261118_17_04_13_Pro.mp4"
OUTPUT_DIR = None        # None：保存到视频旁的“视频名_frames”目录；也可填写 r"D:\输出目录"
FRAME_INTERVAL = 1      # 1 = 每帧保存；30 = 从第1帧起，每隔30帧保存一对
IMAGE_FORMAT = "jpg"   # "jpg"：体积较小；"png"：无损，体积较大
JPEG_QUALITY = 95      # JPG质量，范围1~100；仅对jpg生效
# ==============================================================

import argparse
import csv
import json
import time
import math
import hashlib
import os
from collections import Counter

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise SystemExit("缺少OpenCV，请先运行：python -m pip install opencv-python") from exc


def create_output_directory(base_dir):
    """重复运行时使用新目录，保留之前生成的图片。"""
    base_dir = Path(base_dir)
    candidate = base_dir
    suffix = 2
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = base_dir.with_name(f"{base_dir.name}_{suffix:02d}")
            suffix += 1


def save_image(path, image, extension, parameters):
    """先编码再写文件，兼容含中文字符的输出路径。"""
    success, encoded = cv2.imencode(extension, image, parameters)
    if not success:
        raise RuntimeError(f"图片编码失败：{path}")
    path.write_bytes(encoded.tobytes())


def extract_and_split(video_path, output_dir=None, frame_interval=1,
                      image_format="jpg", jpeg_quality=95):
    """返回实际输出目录；也可以在其他Python程序中导入并调用此函数。"""
    video_path = Path(video_path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"视频不存在：{video_path}")
    if not isinstance(frame_interval, int) or isinstance(frame_interval, bool) or frame_interval < 1:
        raise ValueError("FRAME_INTERVAL必须是大于等于1的整数")
    image_format = image_format.lower().lstrip(".")
    if image_format == "jpeg":
        image_format = "jpg"
    if image_format not in {"jpg", "png"}:
        raise ValueError("IMAGE_FORMAT只能设置为jpg或png")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("JPEG_QUALITY必须在1~100之间")

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频：{video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        success, frame = cap.read()
        if not success:
            raise RuntimeError("无法读取视频第一帧")
        height, width = frame.shape[:2]
        if width % 2:
            raise ValueError(f"视频宽度为{width}，无法等宽切分成左右两半")
        middle = width // 2

        base_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else (
            video_path.parent / f"{video_path.stem}_frames"
        )
        actual_output_dir = create_output_directory(base_dir)
        left_dir = actual_output_dir / "left"
        right_dir = actual_output_dir / "right"
        left_dir.mkdir()
        right_dir.mkdir()
        extension = f".{image_format}"
        parameters = ([cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if image_format == "jpg"
                      else [cv2.IMWRITE_PNG_COMPRESSION, 3])
        decoded_frames = 0
        saved_pairs = 0
        start_time = last_progress = time.monotonic()
        print(f"Input: {video_path}", flush=True)
        print(f"Video: {width}x{height}, {fps:.6f} fps, {reported_frames} frames", flush=True)
        print(f"Output: {actual_output_dir}", flush=True)

        # 帧在内存中提取后直接切分，无需另外保存完整的3840x1080图片。
        # 两个文件夹使用相同文件名，帧号从1开始，并保留原视频的帧号。
        with (actual_output_dir / "frames.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["source_frame_1based", "timestamp_ms", "left_file", "right_file"])
            while success:
                if frame.shape[:2] != (height, width):
                    raise RuntimeError(f"第{decoded_frames + 1}帧的分辨率发生变化")
                frame_index = decoded_frames
                decoded_frames += 1
                if frame_index % frame_interval == 0:
                    filename = f"frame_{decoded_frames:06d}{extension}"
                    left_path = left_dir / filename
                    right_path = right_dir / filename
                    try:
                        save_image(left_path, frame[:, :middle], extension, parameters)
                        save_image(right_path, frame[:, middle:], extension, parameters)
                    except Exception:
                        # 如果一侧写入失败，移除本对图片，避免产生无法配对的文件。
                        left_path.unlink(missing_ok=True)
                        right_path.unlink(missing_ok=True)
                        raise
                    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    writer.writerow([decoded_frames, f"{timestamp_ms:.6f}",
                                     f"left/{filename}", f"right/{filename}"])
                    saved_pairs += 1
                now = time.monotonic()
                if now - last_progress >= 5:
                    print(f"Processed {decoded_frames}/{reported_frames or '?'} frames; "
                          f"saved {saved_pairs} pairs; elapsed {now - start_time:.1f}s", flush=True)
                    stream.flush()
                    last_progress = now
                success, frame = cap.read()

        if reported_frames > 0 and decoded_frames != reported_frames:
            raise RuntimeError(
                f"仅解码{decoded_frames}帧，视频元数据报告{reported_frames}帧。"
                f"已生成的图片保留在{actual_output_dir}，请检查视频是否完整。"
            )
        summary = {
            "video_path": str(video_path),
            "output_dir": str(actual_output_dir),
            "source_size": [width, height],
            "left_image_size": [middle, height],
            "right_image_size": [width - middle, height],
            "fps": fps,
            "reported_frames": reported_frames,
            "decoded_frames": decoded_frames,
            "frame_interval": frame_interval,
            "saved_pairs": saved_pairs,
            "saved_images": saved_pairs * 2,
            "image_format": image_format,
            "jpeg_quality": jpeg_quality if image_format == "jpg" else None,
            "elapsed_seconds": round(time.monotonic() - start_time, 2),
            "frame_numbering": "1-based source frame number; first selected frame is 1",
            "timestamp_source": "OpenCV CAP_PROP_POS_MSEC after decoding each selected frame",
        }
        (actual_output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Done: {saved_pairs} pairs ({saved_pairs * 2} images), "
              f"each {middle}x{height}.", flush=True)
        print(f"Left:  {left_dir}", flush=True)
        print(f"Right: {right_dir}", flush=True)
        return actual_output_dir
    finally:
        cap.release()


ANNOTATION_ALGORITHM_VERSION = "ellipse-edge-tracker-4.2"


class AnnotationTrackingError(RuntimeError):
    """No reliable correspondence: quarantine this image for visual review."""


def read_annotation_image(path):
    """Read without changing the source image; support Unicode Windows paths."""
    data = Path(path).read_bytes()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image, hashlib.sha256(data).hexdigest()


def ellipse_samples(ellipse, count=192):
    center, axes, angle = ellipse
    radii = np.asarray(axes, dtype=float) / 2
    theta = np.linspace(0, 2*np.pi, count, endpoint=False)
    phi = np.deg2rad(angle)
    rotation = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    unit = np.column_stack([np.cos(theta), np.sin(theta)])
    boundary = (unit*radii) @ rotation.T + center
    normals = (unit/radii) @ rotation.T
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return boundary, normals


def ellipse_residuals(points, ellipse):
    center, axes, angle = ellipse
    a, b = np.asarray(axes)/2
    phi = np.deg2rad(angle)
    rotation = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    local = (points-center) @ rotation
    f = (local[:, 0]/a)**2 + (local[:, 1]/b)**2 - 1
    df = 2*np.sqrt((local[:, 0]/a**2)**2+(local[:, 1]/b**2)**2)
    return f/np.maximum(df, 1e-6)


def transform_xy(points, homography):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, homography).reshape(-1, 2)


def transform_ellipse(ellipse, homography):
    boundary, _ = ellipse_samples(ellipse, 32)
    mapped = transform_xy(boundary, homography).astype(np.float32)
    result = cv2.fitEllipse(mapped)
    if not np.isfinite(np.array([*result[0], *result[1], result[2]])).all():
        return ellipse
    return result


class EllipseMarkerDetector:
    """Locate centers from image boundaries, never from predicted positions alone.

    Motion supplies a search prior, not an annotation. Positive normal gradients
    locate the dark-to-light outer edge. Robust fitting rejects glare fragments;
    angular support, contrast, residual and leave-sector-out stability gate the
    result. Unobservable centers are omitted instead of filled with fake points.
    """

    def __init__(self, min_coverage=0.65):
        self.min_coverage = min_coverage

    def refine(self, gray, initial, iterations=3, allow_asymmetry=False):
        height, width = gray.shape
        center, axes, angle = initial
        major = max(axes)
        minor = min(axes)
        if not (8 <= minor <= 230 and 16 <= major <= 280):
            return None, "invalid_size"
        cx, cy = center
        if cx < -major or cx > width+major or cy < -major or cy > height+major:
            return None, "out_of_frame"
        half = int(major*0.8+20)
        x0, y0 = max(0, int(cx)-half), max(0, int(cy)-half)
        x2, y2 = min(width, int(cx)+half+1), min(height, int(cy)+half+1)
        if x2-x0 < 10 or y2-y0 < 10:
            return None, "out_of_frame"
        origin = np.array([x0, y0])
        roi = gray[y0:y2, x0:x2].astype(np.float32)
        smooth = cv2.GaussianBlur(roi, (5, 5), 0.8)
        ellipse = (tuple(np.asarray(center)-origin), axes, angle)
        search_radius = max(5.0, min(11.0, major*0.10))
        offsets = np.linspace(-search_radius, search_radius, 81, dtype=np.float32)
        points = None
        for _ in range(iterations):
            boundary, normals = ellipse_samples(ellipse)
            positions = boundary[:, None, :] + normals[:, None, :]*offsets[None, :, None]
            profiles = cv2.remap(smooth, positions[:, :, 0].astype(np.float32),
                                 positions[:, :, 1].astype(np.float32), cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
            gradients = np.gradient(profiles, offsets, axis=1)
            peak_index = np.argmax(gradients[:, 3:-3], axis=1)+3
            rows = np.arange(len(boundary))
            peaks = gradients[rows, peak_index]
            left = gradients[rows, peak_index-1]
            right = gradients[rows, peak_index+1]
            denominator = left-2*peaks+right
            fraction = np.divide(0.5*(left-right), denominator,
                                 out=np.zeros_like(peaks), where=np.abs(denominator)>1e-6)
            displacement = offsets[peak_index]+np.clip(fraction, -0.5, 0.5)*(offsets[1]-offsets[0])
            edge = boundary+normals*displacement[:, None]
            global_edge = edge+origin
            valid = ((peaks > 6) & (peak_index > 5) & (peak_index < len(offsets)-6)
                     & (global_edge[:, 0] >= 2) & (global_edge[:, 0] < width-2)
                     & (global_edge[:, 1] >= 2) & (global_edge[:, 1] < height-2))
            if valid.sum() < 192*self.min_coverage:
                return None, "insufficient_boundary"
            points = edge[valid]
            ellipse = cv2.fitEllipse(points.astype(np.float32))
            for _ in range(3):
                errors = ellipse_residuals(points, ellipse)
                median = np.median(errors)
                mad = np.median(np.abs(errors-median))
                inliers = np.abs(errors-median) < max(0.85, 3*1.4826*mad)
                if inliers.sum() < 192*0.55:
                    return None, "fragmented_boundary"
                ellipse = cv2.fitEllipse(points[inliers].astype(np.float32))
            if min(ellipse[1]) < 6 or max(ellipse[1]) > 300:
                return None, "degenerate_fit"
        fitted_center = np.asarray(ellipse[0])+origin
        if not (0 <= fitted_center[0] < width and 0 <= fitted_center[1] < height):
            return None, "center_out_of_frame"
        if np.linalg.norm(fitted_center-center) > max(12, major*0.32):
            return None, "excessive_fit_drift"
        axis_ratios = np.sort(ellipse[1])/np.sort(axes)
        if min(axis_ratios) < 0.65 or max(axis_ratios) > 1.55:
            return None, "shape_changed"
        rms = float(np.sqrt(np.mean(ellipse_residuals(points[inliers], ellipse)**2)))
        boundary, normals = ellipse_samples(ellipse)
        inside = boundary-normals*3
        outside = boundary+normals*3
        inner_values = cv2.remap(smooth, inside[:, 0].astype(np.float32), inside[:, 1].astype(np.float32), cv2.INTER_LINEAR).ravel()
        outer_values = cv2.remap(smooth, outside[:, 0].astype(np.float32), outside[:, 1].astype(np.float32), cv2.INTER_LINEAR).ravel()
        contrast = float(np.median((outer_values-inner_values)[valid]))
        if contrast < 24:
            return None, "low_contrast"
        if rms > max(1.8, max(ellipse[1])*0.035):
            return None, "non_elliptic_boundary"
        # Removing one angular quarter exposes an asymmetric glare-biased fit.
        original_indices = np.flatnonzero(valid)[inliers]
        spread = 0.0
        for sector in range(4):
            subset = points[inliers][original_indices//48 != sector]
            if len(subset) < 70:
                continue
            alternate = cv2.fitEllipse(subset.astype(np.float32))
            spread = max(spread, float(np.linalg.norm(np.asarray(alternate[0])-ellipse[0])))
        if spread > max(2.3, major*0.065) and not allow_asymmetry:
            return None, "unstable_center"
        if spread > max(6.0, major*0.16):
            return None, "severely_unstable_center"
        quality = {
            "coverage": round(float(valid.mean()), 4),
            "inlier_fraction": round(float(inliers.sum()/192), 4),
            "edge_rms_px": round(rms, 4),
            "contrast": round(contrast, 2),
            "center_spread_px": round(spread, 4),
        }
        quality["review"] = bool(spread > max(1.25, major*0.025) or valid.mean() < 0.82)
        global_ellipse = (tuple(map(float, fitted_center)), tuple(map(float, ellipse[1])), float(ellipse[2]))
        return {"ellipse": global_ellipse, "quality": quality}, None

    def contour_candidates(self, gray, seed, prior=None, wide=False):
        height, width = gray.shape
        major = max(prior[1]) if prior is not None else 75
        radius = max(55, int(major*(1.7 if wide else 1.05)))
        sx, sy = np.asarray(seed, dtype=float)
        x0, y0 = max(0, int(sx)-radius), max(0, int(sy)-radius)
        x2, y2 = min(width, int(sx)+radius+1), min(height, int(sy)+radius+1)
        if x2-x0 < 10 or y2-y0 < 10:
            return []
        roi = cv2.GaussianBlur(gray[y0:y2, x0:x2], (3, 3), 0.7)
        candidates = []
        for threshold in range(70, 226, 15):
            mask = (roi < threshold).astype(np.uint8)*255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                if len(contour) < 18 or cv2.contourArea(contour) < 100:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if x <= 0 or y <= 0 or x+w >= roi.shape[1] or y+h >= roi.shape[0]:
                    continue
                ellipse = cv2.fitEllipse(contour)
                if not (8 < min(ellipse[1]) and 18 < max(ellipse[1]) < 240):
                    continue
                center = np.asarray(ellipse[0])+[x0, y0]
                distance = float(np.linalg.norm(center-seed))
                if distance > max(25, major*(0.9 if wide else 0.5)):
                    continue
                residual = float(np.sqrt(np.mean(ellipse_residuals(contour[:, 0, :], ellipse)**2)))
                if residual > max(1.8, max(ellipse[1])*0.04):
                    continue
                size_penalty = 0
                if prior is not None:
                    ratios = np.sort(ellipse[1])/np.sort(prior[1])
                    if min(ratios) < 0.5 or max(ratios) > 1.7:
                        continue
                    size_penalty = float(np.abs(np.log(ratios)).sum())*8
                score = distance+residual*3+size_penalty
                candidates.append((score, (tuple(map(float, center)), ellipse[1], ellipse[2])))
        candidates.sort(key=lambda pair: pair[0])
        distinct = []
        for score, ellipse in candidates:
            if any(np.linalg.norm(np.asarray(ellipse[0])-old[0]) < 0.1
                   and np.linalg.norm(np.asarray(ellipse[1])-old[1]) < 0.3 for old in distinct):
                continue
            distinct.append(ellipse)
            if len(distinct) >= 7:
                break
        return distinct

    def detect(self, gray, seed, prior=None, wide=False):
        reason = "no_ellipse"
        direct = None
        if prior is not None:
            direct, reason = self.refine(gray, prior)
            if direct is not None:
                q=direct["quality"]
                if q["coverage"]>=0.985 and q["center_spread_px"]<=max(1.6,max(direct["ellipse"][1])*0.035):
                    return direct, None
        candidates=self.contour_candidates(gray, seed, prior, wide=wide)
        refined=[]
        for initial in candidates:
            observation, candidate_reason = self.refine(gray, initial, allow_asymmetry=True)
            if observation is not None:
                refined.append(observation)
            else:
                reason=candidate_reason
        # A curved-surface sticker need not be a perfect projected conic. In a
        # complete silhouette, independent threshold initializations converging
        # to the same edge-based center are stronger evidence than fitting only
        # three quarters of a very flat ellipse. Partial/clipped disks cannot use
        # this exception: they still need the ordinary stability gate.
        complete=[item for item in refined if item["quality"]["coverage"]>=0.90
                  and item["quality"]["inlier_fraction"]>=0.85
                  and item["quality"]["contrast"]>=30
                  and item["quality"]["edge_rms_px"]<=max(1.4,max(item["ellipse"][1])*0.03)]
        best_cluster=[]
        for candidate in complete:
            cluster=[item for item in complete if np.linalg.norm(np.asarray(item["ellipse"][0])-candidate["ellipse"][0])<=0.65]
            if len(cluster)>len(best_cluster):
                best_cluster=cluster
        if len(best_cluster)>=2:
            selected=min(best_cluster,key=lambda item:(-item["quality"]["coverage"],item["quality"]["edge_rms_px"]))
            centers=np.array([item["ellipse"][0] for item in best_cluster])
            selected["quality"]["threshold_consensus_count"]=len(best_cluster)
            selected["quality"]["threshold_consensus_spread_px"]=round(float(np.max(np.linalg.norm(centers-centers.mean(axis=0),axis=1))),4)
            return selected,None
        for observation in refined:
            major=max(observation["ellipse"][1])
            if observation["quality"]["center_spread_px"]<=max(2.3,major*0.065):
                return observation,None
        if direct is not None:
            return direct,None
        return None, reason


class AircraftMarkerTracker:
    """Track identities through a sequence; missing IDs are never renumbered."""

    def __init__(self, reference, detector=None, config=None):
        self.config = config or {}
        coverage=float(self.config.get("min_coverage",0.65))
        if not 0.55<=coverage<=1:
            raise ValueError("min_coverage must be between 0.55 and 1")
        self.detector = detector or EllipseMarkerDetector(min_coverage=coverage)
        self.ids = sorted(int(s["label"]) for s in reference["shapes"]
                          if s["shape_type"] == "point" and s["label"].isdigit())
        if self.ids != list(range(11)):
            raise ValueError("The reference must contain point labels 0 through 10.")
        self.reference_points = np.array([next(s["points"][0] for s in reference["shapes"]
                                              if s["label"] == str(i) and s["shape_type"] == "point") for i in self.ids])
        self.states = {}
        self.visible = set()
        self.previous_small = None
        self.anchor_states = None
        self.anchor_points = None
        self.center_spread_baseline = None

    def predict(self, gray):
        small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        if self.previous_small is None:
            self.previous_small = small
            return None
        old_points, owners = [], []
        for marker_id in sorted(self.visible):
            boundary, _ = ellipse_samples(self.states[marker_id], 12)
            center = np.array(self.states[marker_id][0])
            features = np.vstack([center, center+(boundary-center)*0.92])
            for xy in features:
                if 4 < xy[0] < gray.shape[1]-4 and 4 < xy[1] < gray.shape[0]-4:
                    old_points.append(xy)
                    owners.append(marker_id)
        predictions = dict(self.states)
        if len(old_points) >= 12:
            previous = (np.array(old_points, dtype=np.float32)*0.5).reshape(-1,1,2)
            current, status, _ = cv2.calcOpticalFlowPyrLK(self.previous_small, small, previous, None,
                winSize=(31,31), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,25,0.01))
            backward, status_back, _ = cv2.calcOpticalFlowPyrLK(small, self.previous_small, current, None,
                winSize=(31,31), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,25,0.01))
            valid = status.ravel().astype(bool) & status_back.ravel().astype(bool)
            valid &= np.linalg.norm(backward[:,0]-previous[:,0],axis=1)<0.8
            old = previous[:,0]*2
            new = current[:,0]*2
            H = None
            if valid.sum() >= 12:
                H, _ = cv2.findHomography(old[valid],new[valid],cv2.RANSAC,3.0)
            if H is None or not np.isfinite(H).all():
                H = np.eye(3)
                if valid.any():
                    H[:2,2] = np.median(new[valid]-old[valid],axis=0)
            for marker_id, state in self.states.items():
                prediction = transform_ellipse(state,H)
                chosen = valid & (np.array(owners)==marker_id)
                if chosen.sum() >= 5:
                    shift = np.median(new[chosen]-old[chosen],axis=0)
                    local_center = np.asarray(state[0])+shift
                    if np.linalg.norm(local_center-prediction[0])<max(15,max(state[1])*0.25):
                        prediction = (tuple(map(float,local_center)),prediction[1],prediction[2])
                predictions[marker_id] = prediction
        self.previous_small = small
        return predictions

    def update_missing_priors(self, observations, predictions):
        good = sorted(observations)
        if self.anchor_points is None or len(good)<5:
            return
        current = np.array([observations[i]["ellipse"][0] for i in good])
        original = self.anchor_points[good]
        H, inliers = cv2.findHomography(original,current,cv2.RANSAC,6.0)
        if H is None or not np.isfinite(H).all() or inliers.sum()<5:
            return
        for i in self.ids:
            if i not in observations:
                candidate = transform_ellipse(self.anchor_states[i],H)
                if 8 < min(candidate[1]) and max(candidate[1])<270:
                    predictions[i] = candidate
        # Extrapolate a missing wing tip from neighboring disks on the same wing.
        # A global body homography is less reliable when the nose is out of view.
        neighbors={0:(1,2),7:(6,5),6:(5,4)}
        for i,(a,b) in neighbors.items():
            if i in observations or a not in observations or b not in observations:
                continue
            source=self.anchor_points[b]-self.anchor_points[a]
            target=np.asarray(observations[b]["ellipse"][0])-observations[a]["ellipse"][0]
            ratio=complex(*target)/complex(*source)
            local=np.eye(3)
            local[:2,:2]=[[ratio.real,-ratio.imag],[ratio.imag,ratio.real]]
            local[:2,2]=np.asarray(observations[a]["ellipse"][0])-local[:2,:2]@self.anchor_points[a]
            predictions[i]=transform_ellipse(self.anchor_states[i],local)

    def aircraft_box(self, observations, gray):
        """Fit the current fuselage/wing silhouette inside a geometric search mask.

        A projected reference rectangle can explode near a homography horizon.
        This mask uses the individual observed disks and the fore/aft body axis;
        connected bright components adjacent to the disks give image evidence
        for the wing tips and fuselage outline. Predictions only bound the search.
        """
        height,width = gray.shape
        scale = float(np.median([max(item["ellipse"][1]) for item in observations.values()]))/60
        padding = self.config.get("bbox_padding",20)*scale
        samples=[]
        ring_samples=[]
        for i in self.ids:
            ellipse=self.states[i]
            boundary,normals=ellipse_samples(ellipse,32)
            samples.append(boundary+normals*max(18,max(ellipse[1])*0.4))
            if i in observations:
                ring_samples.append(boundary+normals*7)
        rear=np.array(self.states[3][0])
        front=np.array(self.states[10][0])
        nose=front+0.72*(front-rear)
        apex=rear-0.55*(front-rear)
        samples.append(np.vstack([nose,apex]))
        polygon=np.concatenate(samples)
        polygon[:,0]=np.clip(polygon[:,0],-2*width,3*width)
        polygon[:,1]=np.clip(polygon[:,1],-2*height,3*height)
        hint=np.zeros_like(gray)
        hull=cv2.convexHull(polygon.astype(np.int32))
        cv2.fillConvexPoly(hint,hull,255)
        rings=np.concatenate(ring_samples).round().astype(int)
        rings=rings[(rings[:,0]>=0)&(rings[:,0]<width)&(rings[:,1]>=0)&(rings[:,1]<height)]
        white_level=float(np.median(gray[rings[:,1],rings[:,0]]))
        threshold=float(np.clip(white_level*0.68,110,170))
        mask=((gray>=threshold)&(hint>0)).astype(np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
        votes=Counter(int(v) for v in labels[rings[:,1],rings[:,0]] if v>0)
        chosen=[label for label,vote in votes.items() if vote>=4 and stats[label,cv2.CC_STAT_AREA]>100]
        observed_boundaries=np.concatenate([ellipse_samples(item["ellipse"],32)[0] for item in observations.values()])
        evidence=[observed_boundaries]
        for label in chosen:
            x,y,w,h,area=stats[label]
            evidence.append(np.array([[x,y],[x+w,y+h]],dtype=float))
        if not chosen:
            evidence.append(np.array([nose,apex]))
        bounds=np.concatenate(evidence)
        x1,y1=bounds.min(axis=0)-padding
        x2,y2=bounds.max(axis=0)+padding
        return [round(float(np.clip(x1,0,width-1)),2), round(float(np.clip(y1,0,height-1)),2),
                round(float(np.clip(x2,1,width)),2), round(float(np.clip(y2,1,height)),2)]

    def process(self, gray, known=None):
        previous_small=self.previous_small
        try:
            return self._process(gray,known)
        except AnnotationTrackingError:
            # A corrupt/stitched image must not poison optical flow for the next.
            self.previous_small=previous_small
            raise

    def _process(self, gray, known=None):
        predictions = self.predict(gray)
        observations, failures = {}, {}
        if predictions is None:
            predictions = {}
            for marker_id,seed in enumerate(self.reference_points):
                observation,reason = self.detector.detect(gray,seed)
                if observation is None:
                    raise RuntimeError(f"Cannot initialize reference ID {marker_id}: {reason}")
                observations[marker_id] = observation
                predictions[marker_id] = observation["ellipse"]
        else:
            for marker_id,prior in predictions.items():
                observation,reason = self.detector.detect(gray,prior[0],prior)
                if observation is not None:
                    observations[marker_id] = observation
                else:
                    failures[marker_id] = reason
            self.update_missing_priors(observations,predictions)
            for marker_id in self.ids:
                if marker_id in observations:
                    continue
                prior = predictions[marker_id]
                x,y = prior[0]
                if not (0<=x<gray.shape[1] and 0<=y<gray.shape[0]):
                    failures[marker_id] = "out_of_frame"
                    continue
                observation,reason = self.detector.detect(gray,prior[0],prior,wide=True)
                if observation is not None:
                    observations[marker_id] = observation
                else:
                    failures[marker_id] = reason
        if known is not None:
            for shape in known["shapes"]:
                if shape["shape_type"]=="point" and shape["label"].isdigit():
                    i=int(shape["label"])
                    if i in observations:
                        ellipse=observations[i]["ellipse"]
                        observations[i]["ellipse"]=(tuple(shape["points"][0]),ellipse[1],ellipse[2])
        # A detector is not allowed to assign the same disk to two identities.
        for i in sorted(list(observations)):
            for j in sorted(list(observations)):
                if j<=i or i not in observations or j not in observations:
                    continue
                ei,ej=observations[i]["ellipse"],observations[j]["ellipse"]
                if np.linalg.norm(np.asarray(ei[0])-ej[0]) < 0.4*(min(ei[1])+min(ej[1])):
                    di=np.linalg.norm(np.asarray(ei[0])-predictions[i][0])
                    dj=np.linalg.norm(np.asarray(ej[0])-predictions[j][0])
                    loser=i if di>dj else j
                    del observations[loser]
                    failures[loser]="duplicate_candidate"
        if len(observations)<5:
            raise AnnotationTrackingError(f"Tracking lost: only {len(observations)} reliably observed markers")
        self.states = {i:observations[i]["ellipse"] if i in observations else predictions[i] for i in self.ids}
        self.visible = set(observations)
        if self.center_spread_baseline is None:
            self.center_spread_baseline={i:item["quality"]["center_spread_px"]/max(item["ellipse"][1])
                                         for i,item in observations.items()}
        for i,item in observations.items():
            quality=item["quality"]
            baseline=self.center_spread_baseline.get(i,0.025)
            tolerance=max(2.0,max(item["ellipse"][1])*max(0.04,baseline*1.6))
            quality["review"]=bool(quality["coverage"]<0.82 or quality["center_spread_px"]>tolerance)
        if len(observations)==11:
            self.anchor_states = dict(self.states)
            self.anchor_points = np.array([self.states[i][0] for i in self.ids])
        missing=[{"id":i,"reason":failures.get(i,"unobservable"),
                  "prediction":[round(float(v),3) for v in self.states[i][0]],
                  "predicted_ellipse":self.states[i]} for i in self.ids if i not in observations]
        box=self.aircraft_box(observations,gray)
        return observations,missing,box


def labelme_document(record):
    def shape(label,points,shape_type):
        return {"label":label,"points":points,"group_id":0,"description":"",
                "shape_type":shape_type,"flags":{},"mask":None}
    x1,y1,x2,y2=record["bbox"]
    shapes=[shape("target",[[x1,y1],[x2,y2]],"rectangle")]
    shapes.extend(shape(str(item["id"]),[[round(float(v),3) for v in item["ellipse"][0]]],"point")
                  for item in sorted(record["markers"],key=lambda item:item["id"]))
    return {"version":"5.5.0","flags":{},"shapes":shapes,"imagePath":record["image"],
            "imageData":None,"imageHeight":record["height"],"imageWidth":record["width"]}


def validate_labelme_document(document):
    width,height=document["imageWidth"],document["imageHeight"]
    boxes=[s for s in document["shapes"] if s["shape_type"]=="rectangle"]
    points=[s for s in document["shapes"] if s["shape_type"]=="point"]
    if len(boxes)!=1 or boxes[0]["label"]!="target":
        raise ValueError("Expected one target rectangle")
    (x1,y1),(x2,y2)=boxes[0]["points"]
    if not (0<=x1<x2<=width and 0<=y1<y2<=height):
        raise ValueError("Invalid target rectangle")
    ids=[int(s["label"]) for s in points]
    if len(ids)!=len(set(ids)) or not set(ids)<=set(range(11)):
        raise ValueError("Duplicate or invalid point IDs")
    for shape in points:
        if len(shape["points"])!=1:
            raise ValueError("A point must have exactly one coordinate pair")
        x,y=shape["points"][0]
        if not (math.isfinite(x) and math.isfinite(y) and 0<=x<width and 0<=y<height
                and x1<=x<=x2 and y1<=y<=y2):
            raise ValueError(f"Point {shape['label']} is outside image or box")


def write_labelme_record(record,image_dir,overwrite=False):
    document=labelme_document(record)
    validate_labelme_document(document)
    path=Path(image_dir)/Path(record["image"]).with_suffix(".json")
    if path.exists() and not overwrite:
        return False
    temporary=path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(temporary,path)
    validate_labelme_document(json.loads(path.read_text(encoding="utf-8")))
    return True


def annotation_preview(image,record):
    result=image.copy()
    x1,y1,x2,y2=map(lambda v:int(round(v)),record["bbox"])
    cv2.rectangle(result,(x1,y1),(x2,y2),(0,230,0),2,cv2.LINE_AA)
    cv2.putText(result,"target",(x1+6,min(y1+25,result.shape[0]-10)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,230,0),2,cv2.LINE_AA)
    for marker in record["markers"]:
        center,axes,angle=marker["ellipse"]
        ellipse=(tuple(center),tuple(axes),angle)
        cv2.ellipse(result,ellipse,(0,210,255),1,cv2.LINE_AA)
        xy=tuple(int(round(v)) for v in center)
        cv2.drawMarker(result,xy,(0,0,255),cv2.MARKER_CROSS,9,1,cv2.LINE_AA)
        at=(xy[0]+8,xy[1]-12)
        cv2.putText(result,str(marker["id"]),at,cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(result,str(marker["id"]),at,cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),1,cv2.LINE_AA)
    return result


def annotate_sequence(image_dir,reference_path=None,report_dir=None,config_path=None,
                      limit=None,dry_run=False,preview_every=100,overwrite=False):
    """Annotate a continuous, filename-ordered sequence using a complete reference.

    The reference image must be the first image. Existing labels are preserved
    unless overwrite is explicitly requested. Previews/reports live outside the
    training image folder. The JSONL cache retains all rejection reasons.
    """
    cv2.setNumThreads(2)
    cv2.setRNGSeed(0)
    root=Path(image_dir).expanduser().resolve()
    images=sorted(p for p in root.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"})
    if not images:
        raise ValueError(f"No images: {root}")
    if limit:
        images=images[:limit]
    reference_path=Path(reference_path) if reference_path else images[0].with_suffix(".json")
    reference=json.loads(reference_path.read_text(encoding="utf-8"))
    if Path(reference["imagePath"]).name!=images[0].name:
        raise ValueError("Reference annotation must correspond to the first image")
    config=json.loads(Path(config_path).read_text(encoding="utf-8")) if config_path else {}
    tracker=AircraftMarkerTracker(reference,config=config)
    report=Path(report_dir) if report_dir else root.with_name(root.name+"_annotation_review")
    report.mkdir(parents=True,exist_ok=True)
    preview_dir=report/"previews"
    preview_dir.mkdir(exist_ok=True)
    existing={p.stem:p for p in root.glob("*.json")}
    stats=Counter()
    missing_reasons=Counter()
    review_frames=[]
    excluded_frames=[]
    started=last_progress=time.monotonic()
    cache_path=report/"annotations.jsonl"
    with cache_path.open("w",encoding="utf-8") as cache:
        for number,path in enumerate(images,1):
            image,digest=read_annotation_image(path)
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            known=(json.loads(existing[path.stem].read_text(encoding="utf-8"))
                   if path.stem in existing and not overwrite else None)
            try:
                observations,missing,box=tracker.process(gray,known)
            except AnnotationTrackingError as exc:
                excluded={"algorithm":ANNOTATION_ALGORITHM_VERSION,"frame":number,"image":path.name,
                          "image_sha256":digest,"width":image.shape[1],"height":image.shape[0],
                          "status":"excluded","reason":str(exc),"requires_visual_review":True}
                excluded_frames.append(excluded)
                cache.write(json.dumps(excluded,ensure_ascii=False,allow_nan=False)+"\n")
                stats["frames"]+=1
                stats["excluded"]+=1
                print(f"EXCLUDED {path.name}: {exc}",flush=True)
                save_image(preview_dir/(path.stem+"_EXCLUDED.jpg"),image,".jpg",[cv2.IMWRITE_JPEG_QUALITY,92])
                if len(excluded_frames)>=8 and all(e["frame"]>=number-7 for e in excluded_frames[-8:]):
                    raise AnnotationTrackingError("Eight consecutive failures: a new reference/review is required") from exc
                continue
            record={"algorithm":ANNOTATION_ALGORITHM_VERSION,"frame":number,"image":path.name,
                    "image_sha256":digest,"width":image.shape[1],"height":image.shape[0],"bbox":box,"status":"annotated",
                    "markers":[{"id":i,**observations[i]} for i in sorted(observations)],"missing":missing}
            review_ids=[item["id"] for item in record["markers"] if item["quality"]["review"]]
            uncertain=[item["id"] for item in missing if item["reason"]!="out_of_frame"]
            record["review_ids"]=sorted(set(review_ids+uncertain))
            validate_labelme_document(labelme_document(record))
            cache.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")
            stats["frames"]+=1
            stats["points"]+=len(observations)
            stats[f"frames_with_{len(observations)}_points"]+=1
            missing_reasons.update(item["reason"] for item in missing)
            if record["review_ids"]:
                review_frames.append({"frame":number,"image":path.name,"ids":record["review_ids"]})
            if preview_every and (number==1 or number%preview_every==0 or number==len(images)):
                preview=annotation_preview(image,record)
                save_image(preview_dir/(path.stem+".jpg"),preview,".jpg",[cv2.IMWRITE_JPEG_QUALITY,92])
            now=time.monotonic()
            if now-last_progress>=10 or number==len(images):
                cache.flush()
                print(f"Annotated {number}/{len(images)}; visible={len(observations)}; "
                      f"review_frames={len(review_frames)}; elapsed={now-started:.1f}s",flush=True)
                last_progress=now
    records=[json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    recovered=refine_missing_from_neighbors(records,root,config=config)
    cache_path.write_text("".join(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n" for r in records),encoding="utf-8")
    stats=Counter({"frames":len(records),"excluded":len(excluded_frames)})
    missing_reasons=Counter()
    review_frames=[]
    for record in records:
        if record.get("status")=="excluded":continue
        stats["points"]+=len(record["markers"])
        stats[f"frames_with_{len(record['markers'])}_points"]+=1
        missing_reasons.update(m["reason"] for m in record["missing"])
        if record["review_ids"]:review_frames.append({"frame":record["frame"],"image":record["image"],"ids":record["review_ids"]})
    if not dry_run:
        stats["written"]=export_annotation_cache(cache_path,root,overwrite)
    refresh_recovery_previews(records,recovered,root,report)
    summary={"algorithm":ANNOTATION_ALGORITHM_VERSION,"image_dir":str(root),"dry_run":dry_run,"config":config,
             "statistics":dict(stats),"missing_reasons":dict(missing_reasons),"review_frames":review_frames,
             "excluded_frames":excluded_frames,"recovered_points":len(recovered),
             "elapsed_seconds":round(time.monotonic()-started,2),"cache":str(cache_path)}
    (report/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (report/"excluded_frames.json").write_text(json.dumps(excluded_frames,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    excluded_names={r["image"] for r in excluded_frames}
    (report/"training_images.txt").write_text("".join(str(p)+"\n" for p in images if p.name not in excluded_names),encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k not in {"review_frames","excluded_frames"}},ensure_ascii=False),flush=True)
    return summary


def refine_missing_from_neighbors(records,image_dir,max_gap=30,config=None):
    """Re-search missed disks from nearby measured poses, using current pixels.

    Neighbor coordinates are only search seeds. Every recovered annotation must
    independently pass the detector's current-image boundary/contrast checks.
    The original measured points and explicitly excluded images are preserved.
    """
    detector=EllipseMarkerDetector(min_coverage=(config or {}).get("min_coverage",0.65))
    observed=[{m["id"]:m for m in r.get("markers",[])} for r in records]
    recovered=[]
    for index,record in enumerate(records):
        if record.get("status")=="excluded" or not record.get("missing"):continue
        gray=None
        for missing in list(record["missing"]):
            marker_id=missing["id"]
            sources=[]
            for direction in [-1,1]:
                for distance in range(1,max_gap+1):
                    other=index+direction*distance
                    if not 0<=other<len(records):break
                    if marker_id in observed[other]:
                        sources.append(other)
                        break
            measured=[]
            for other in sources:
                common=sorted(set(observed[other])&set(observed[index]))
                if len(common)<5:continue
                source=np.array([observed[other][i]["ellipse"][0] for i in common])
                target=np.array([observed[index][i]["ellipse"][0] for i in common])
                H,inliers=cv2.findHomography(source,target,cv2.RANSAC,4.0)
                if H is None or not np.isfinite(H).all() or inliers.sum()<5:continue
                prior=transform_ellipse(observed[other][marker_id]["ellipse"],H)
                x,y=prior[0];margin=max(prior[1])*0.25
                if not (-margin<=x<record["width"]+margin and -margin<=y<record["height"]+margin):continue
                if gray is None:
                    image,_=read_annotation_image(Path(image_dir)/record["image"])
                    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                result,_=detector.detect(gray,prior[0],prior,wide=True)
                if result is None:continue
                center=np.array(result["ellipse"][0])
                if any(np.linalg.norm(center-item["ellipse"][0])<0.4*(min(result["ellipse"][1])+min(item["ellipse"][1]))
                       for item in record["markers"]):continue
                if marker_id<8 and any((i<marker_id and item["ellipse"][0][0]>=center[0])
                                      or (i>marker_id and item["ellipse"][0][0]<=center[0])
                                      for i,item in observed[index].items() if i<8):continue
                result["quality"]["neighbor_search_frame"]=records[other]["frame"]
                measured.append(result)
            if not measured:continue
            if len(measured)>1 and np.linalg.norm(np.asarray(measured[0]["ellipse"][0])-measured[1]["ellipse"][0])>1.0:
                continue
            chosen=min(measured,key=lambda item:(-item["quality"]["coverage"],item["quality"]["edge_rms_px"]))
            chosen["quality"]["recovered_from_current_image"]=True
            chosen["quality"]["review"]=True
            item={"id":marker_id,**chosen}
            record["markers"].append(item)
            record["markers"].sort(key=lambda m:m["id"])
            record["missing"]=[m for m in record["missing"] if m["id"]!=marker_id]
            record["review_ids"]=sorted(set(record.get("review_ids",[])+[marker_id]))
            boundary,_=ellipse_samples(chosen["ellipse"],32)
            x1,y1=boundary.min(axis=0)-5;x2,y2=boundary.max(axis=0)+5
            box=record["bbox"]
            record["bbox"]=[round(max(0,min(box[0],x1)),2),round(max(0,min(box[1],y1)),2),
                            round(min(record["width"],max(box[2],x2)),2),round(min(record["height"],max(box[3],y2)),2)]
            # Only original measurements seed this pass, so recovered guesses
            # cannot cascade into other recovered annotations.
            recovered.append({"frame":record["frame"],"image":record["image"],"id":marker_id,
                              "previous_reason":missing["reason"],"ellipse":chosen["ellipse"],"quality":chosen["quality"]})
            validate_labelme_document(labelme_document(record))
    for record in records:record["algorithm"]=ANNOTATION_ALGORITHM_VERSION
    return recovered


def refresh_recovery_previews(records,recovered,image_dir,report_dir):
    report=Path(report_dir)
    (report/"recovered_points.json").write_text(json.dumps(recovered,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    changed={item["frame"] for item in recovered}
    for record in records:
        if record["frame"] not in changed:continue
        path=report/"previews"/(Path(record["image"]).stem+".jpg")
        if path.exists():
            image,_=read_annotation_image(Path(image_dir)/record["image"])
            save_image(path,annotation_preview(image,record),".jpg",[cv2.IMWRITE_JPEG_QUALITY,92])


def recover_annotation_cache(cache_path,image_dir):
    """Apply the evidence-based recovery stage to an existing dry-run cache."""
    path=Path(cache_path)
    records=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary_path=path.parent/"summary.json"
    summary=json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    recovered=refine_missing_from_neighbors(records,image_dir,config=summary.get("config"))
    path.write_text("".join(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n" for r in records),encoding="utf-8")
    refresh_recovery_previews(records,recovered,image_dir,path.parent)
    if summary_path.exists():
        stats=Counter({"frames":len(records)})
        missing=Counter();review=[]
        for record in records:
            if record.get("status")=="excluded":stats["excluded"]+=1;continue
            stats["points"]+=len(record["markers"])
            stats[f"frames_with_{len(record['markers'])}_points"]+=1
            missing.update(m["reason"] for m in record["missing"])
            if record["review_ids"]:review.append({"frame":record["frame"],"image":record["image"],"ids":record["review_ids"]})
        summary.update(algorithm=ANNOTATION_ALGORITHM_VERSION,statistics=dict(stats),missing_reasons=dict(missing),
                       review_frames=review,recovered_points=len(recovered))
        summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"Recovered {len(recovered)} points from current-image boundaries",flush=True)
    return recovered


def export_annotation_cache(cache_path,image_dir,overwrite=False):
    """Publish reviewed records only if their corresponding source JPEGs match."""
    root=Path(image_dir)
    records=[json.loads(line) for line in Path(cache_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for record in records:
        source=root/record["image"]
        if hashlib.sha256(source.read_bytes()).hexdigest()!=record["image_sha256"]:
            raise ValueError(f"Source changed since review: {source}")
        if record.get("status")!="excluded":
            validate_labelme_document(labelme_document(record))
    accepted=[r for r in records if r.get("status")!="excluded"]
    count=sum(write_labelme_record(record,root,overwrite) for record in accepted)
    print(f"Exported {count} Labelme files; preserved {len(accepted)-count} existing files; "
          f"excluded {len(records)-len(accepted)} images",flush=True)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["extract","annotate"],default="extract",help="视频拆帧或靶标标注")
    parser.add_argument("--images",help="需要标注的连续图片目录")
    parser.add_argument("--reference",help="第一张图的完整 0-10 点 Labelme JSON")
    parser.add_argument("--report-dir",help="质量报告与预览目录；默认放在图片目录旁边")
    parser.add_argument("--annotation-config",help="可选标注参数 JSON：min_coverage、bbox_padding")
    parser.add_argument("--limit",type=int,help="仅处理前 N 张，供算法验证")
    parser.add_argument("--dry-run",action="store_true",help="仅生成缓存和质量报告，不写图片旁的 JSON")
    parser.add_argument("--preview-every",type=int,default=100,help="每 N 张保存一张标注预览，0 禁用")
    parser.add_argument("--export-cache",help="将已复核的 JSONL 缓存导出为 Labelme JSON")
    parser.add_argument("--recover-cache",help="对已有缓存执行前后帧辅助搜索，仍须通过当前图片边缘验证")
    parser.add_argument("--overwrite",action="store_true",help="明确允许覆盖已有标注；默认保留")
    parser.add_argument("--video", default=VIDEO_PATH, help="输入视频路径")
    parser.add_argument("--output", default=OUTPUT_DIR, help="输出目录；已存在时自动添加序号")
    parser.add_argument("--every", type=int, default=FRAME_INTERVAL, help="每隔N帧保存一对，默认1")
    parser.add_argument("--format", choices=["jpg", "png"], default=IMAGE_FORMAT, help="图片格式")
    parser.add_argument("--quality", type=int, default=JPEG_QUALITY, help="JPG质量，1~100")
    args = parser.parse_args()
    if args.mode=="annotate":
        if not args.images:
            parser.error("--mode annotate requires --images")
        if args.recover_cache:
            recover_annotation_cache(args.recover_cache,args.images)
        elif args.export_cache:
            export_annotation_cache(args.export_cache,args.images,args.overwrite)
        else:
            annotate_sequence(args.images,args.reference,args.report_dir,args.annotation_config,
                              args.limit,args.dry_run,args.preview_every,args.overwrite)
    else:
        extract_and_split(args.video, args.output, args.every, args.format, args.quality)


if __name__ == "__main__":
    main()
