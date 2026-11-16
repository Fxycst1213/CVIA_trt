
import argparse
import csv
import os

import cv2
import numpy as np


CSV_FILE = "eye_to_hand_samples.csv"


def collect_samples(camera_id, output_csv):
    """采集机器人基座 XYZ 与图像像素点的对应关系。"""

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开相机: {camera_id}")

    window = "Eye-to-Hand Calibration"
    cv2.namedWindow(window)

    clicked_point = []
    frozen_image = None

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point.clear()
            clicked_point.append((x, y))

    cv2.setMouseCallback(window, on_mouse)

    file_exists = os.path.exists(output_csv)
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists or os.path.getsize(output_csv) == 0:
            writer.writerow(["x", "y", "z", "u", "v"])

        print("\n采集说明：")
        print("1. 先让机器人停止，确认 XYZ 与图像对应同一姿态。")
        print("2. 按 c 冻结图像，然后用鼠标点击对应特征点。")
        print("3. 在终端输入机器人基座坐标 X Y Z。")
        print("4. 按 q 退出采集。")
        print("坐标单位必须始终一致，例如全部使用 mm。\n")

        try:
            while True:
                if frozen_image is None:
                    ok, frame = cap.read()
                    if not ok:
                        print("读取相机图像失败")
                        break

                    display = frame.copy()
                    cv2.putText(
                        display,
                        "Press C to capture, Q to quit",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow(window, display)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("c"):
                        frozen_image = frame.copy()
                        clicked_point.clear()

                else:
                    display = frozen_image.copy()

                    if clicked_point:
                        u, v = clicked_point[0]
                        cv2.circle(display, (u, v), 6, (0, 0, 255), -1)
                        cv2.putText(
                            display, f"Pixel: ({u}, {v})",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2
                        )

                    cv2.imshow(window, display)
                    key = cv2.waitKey(20) & 0xFF

                    if key == ord("q"):
                        break
                    if key == 27:  # ESC：放弃当前帧，重新拍摄
                        frozen_image = None
                        clicked_point.clear()
                        continue

                    if clicked_point:
                        u, v = clicked_point[0]
                        print(f"\n图像像素坐标: u={u}, v={v}")
                        print("请输入与该图像点对应的机器人基座 XYZ。")

                        try:
                            values = input("X Y Z: ").strip().replace(",", " ").split()
                            if len(values) != 3:
                                print("需要输入三个数字，当前帧不保存。")
                                frozen_image = None
                                clicked_point.clear()
                                continue

                            x, y, z = map(float, values)
                        except ValueError:
                            print("坐标格式错误，当前帧不保存。")
                            frozen_image = None
                            clicked_point.clear()
                            continue

                        writer.writerow([x, y, z, u, v])
                        f.flush()

                        print(f"已保存: XYZ=({x}, {y}, {z}), UV=({u}, {v})")
                        frozen_image = None
                        clicked_point.clear()

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"采集数据已保存至: {output_csv}")


def solve_calibration(input_csv, intrinsics_file, output_dir):
    """使用 PnP 求解基座到相机的变换，并输出误差。"""

    data = np.genfromtxt(input_csv, delimiter=",", skip_header=1)

    if data.ndim == 1 and data.size:
        data = data.reshape(1, -1)

    if data.ndim != 2 or data.shape[1] != 5:
        raise ValueError("CSV 格式应为 x,y,z,u,v")

    data = data[np.isfinite(data).all(axis=1)]

    if len(data) < 6:
        raise ValueError("至少采集 6 组有效对应点；建议采集 12–20 组。")

    # 三维点：机器人基座坐标；二维点：图像像素坐标
    object_points = data[:, :3].astype(np.float64)
    image_points = data[:, 3:5].astype(np.float64)

    # 检查三维点分布
    centered = object_points - object_points.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)

    print("三维点分布奇异值:", singular_values)
    if singular_values[0] <= 1e-9:
        raise ValueError("三维点几乎没有空间分布，无法可靠标定。")
    if singular_values[-1] / singular_values[0] < 1e-3:
        print("警告：三维点接近共面或退化，请增加不同方向的采样。")

    params = np.load(intrinsics_file)
    K = np.asarray(params["K"], dtype=np.float64)
    dist = np.asarray(params["dist"], dtype=np.float64)

    if K.shape != (3, 3):
        raise ValueError("K 必须为 3x3 矩阵。")

    # RANSAC 用于降低少量错误点击的影响
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        K,
        dist,
        iterationsCount=2000,
        reprojectionError=4.0,  # 像素阈值，可按实际精度调整
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )

    if not success or inliers is None or len(inliers) < 6:
        raise RuntimeError(
            "PnP 求解失败或有效内点不足。请检查点对应关系、单位和空间分布。"
        )

    idx = inliers.ravel()
    inlier_obj = object_points[idx]
    inlier_img = image_points[idx]

    # 对 RANSAC 内点进行非线性精化
    rvec, tvec = cv2.solvePnPRefineLM(
        inlier_obj, inlier_img, K, dist, rvec, tvec
    )

    R_camera_base, _ = cv2.Rodrigues(rvec)

    T_camera_base = np.eye(4)
    T_camera_base[:3, :3] = R_camera_base
    T_camera_base[:3, 3] = tvec.ravel()

    T_base_camera = np.linalg.inv(T_camera_base)

    # 计算所有点和内点的重投影误差
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, K, dist
    )
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - image_points, axis=1)

    os.makedirs(output_dir, exist_ok=True)

    np.savetxt(
        os.path.join(output_dir, "T_camera_base.txt"),
        T_camera_base, fmt="%.10f"
    )
    np.savetxt(
        os.path.join(output_dir, "T_base_camera.txt"),
        T_base_camera, fmt="%.10f"
    )
    np.savez(
        os.path.join(output_dir, "calibration_result.npz"),
        T_camera_base=T_camera_base,
        T_base_camera=T_base_camera,
        rvec=rvec,
        tvec=tvec,
        inliers=idx,
        reprojection_errors=errors,
    )

    with open(
        os.path.join(output_dir, "reprojection_errors.csv"),
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["index", "u", "v", "error_px", "is_inlier"])
        for i, (uv, err) in enumerate(zip(image_points, errors)):
            writer.writerow([i, uv[0], uv[1], err, int(i in idx)])

    print("\n===== Eye-to-Hand 标定结果 =====")
    print(f"总样本数: {len(data)}")
    print(f"RANSAC 内点数: {len(idx)} / {len(data)}")
    print(f"内点平均重投影误差: {errors[idx].mean():.3f} px")
    print(f"内点重投影误差 RMS: {np.sqrt(np.mean(errors[idx]**2)):.3f} px")