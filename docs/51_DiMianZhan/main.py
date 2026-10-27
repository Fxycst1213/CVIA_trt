import argparse
import re
import shutil
import socket
import subprocess
import sys
import time

import cv2


def main():
    parser = argparse.ArgumentParser(description="USB 相机 → H.264 → 本机 MediaMTX → 天空端 RTSP 拉流")
    parser.add_argument("--camera", type=int, default=0, help="相机编号，默认 0")
    parser.add_argument("--width", type=int, default=1920, help="输出宽度，必须为正偶数")
    parser.add_argument("--height", type=int, default=1080, help="输出高度，必须为正偶数")
    parser.add_argument("--fps", type=int, default=30, help="目标帧率，默认 30")
    parser.add_argument("--bitrate", default="4M", help="目标码率，例如 2M、1500k")
    parser.add_argument("--rtsp-port", type=int, default=8554, help="本机 MediaMTX 端口")
    parser.add_argument("--ffmpeg", default=r"D:\72_mediamtx\ffmpeg\bin\ffmpeg.exe", help="FFmpeg 可执行文件")
    args = parser.parse_args()
    if args.camera < 0 or args.fps <= 0:
        parser.error("相机编号必须 >= 0，帧率必须 > 0")
    if any(size <= 0 or size % 2 for size in (args.width, args.height)):
        parser.error("H.264 yuv420p 编码要求宽度、高度为正偶数")
    if not 1 <= args.rtsp_port <= 65535:
        parser.error("端口必须在 1～65535 之间")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([kKmM]?)", args.bitrate)
    if not match:
        parser.error("码率格式错误，请使用 2M、1500k 或 2000000")
    bitrate = int(float(match[1]) * {"": 1, "k": 1000, "m": 1000000}[match[2].lower()])
    if bitrate <= 0:
        parser.error("码率必须大于 0")

    camera = None
    encoder = None
    try:
        ffmpeg = shutil.which(args.ffmpeg)
        if ffmpeg is None:
            raise RuntimeError(f"找不到 FFmpeg：{args.ffmpeg}，请用 --ffmpeg 指定正确路径")
        try:
            with socket.create_connection(("127.0.0.1", args.rtsp_port), timeout=2):
            # with socket.create_connection(("192.168.2.10", args.rtsp_port), timeout=2):
                pass
        except OSError as error:
            raise RuntimeError(f"本机 {args.rtsp_port} 端口未就绪，请先启动 MediaMTX 并核对端口") from error

        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
            camera = cv2.VideoCapture(args.camera, backend)
            if camera.isOpened():
                break
            camera.release()
        if not camera.isOpened():
            raise RuntimeError(f"无法打开相机 {args.camera}，请检查编号、USB 连接、占用和相机权限")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        camera.set(cv2.CAP_PROP_FPS, args.fps)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"相机报告：{camera.get(cv2.CAP_PROP_FRAME_WIDTH):g}×"
              f"{camera.get(cv2.CAP_PROP_FRAME_HEIGHT):g}，{camera.get(cv2.CAP_PROP_FPS):g} FPS")

        url = f"rtsp://127.0.0.1:{args.rtsp_port}/live"
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo", "-pixel_format", "bgr24",
            # "-video_size", f"{args.width}x{args.height}", "-framerate", str(args.fps),
            "-video_size", f"{1920}x{1080}", "-framerate", str(args.fps),
            "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-preset", "ultrafast", "-tune", "zerolatency",
            "-profile:v", "baseline", "-pix_fmt", "yuv420p",
            "-b:v", str(bitrate), "-maxrate", str(bitrate),
            "-bufsize", str(max(1, bitrate // 10)),
            "-g", str(args.fps), "-keyint_min", str(args.fps),
            "-sc_threshold", "0", "-bf", "0",
            "-flush_packets", "1", "-rw_timeout", "5000000",
            "-f", "rtsp", "-rtsp_transport", "tcp", url,
        ]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, bufsize=0)
        print(f"正在向本机发布：{url}")
        print(f"天空端拉流地址：rtsp://<电脑以太网IPv4>:{args.rtsp_port}/live")
        print("按 Ctrl+C 退出；FFmpeg 编码或发布错误会直接显示在此窗口。")
        failures = 0
        next_frame = time.monotonic()
        while True:
            if encoder.poll() is not None:
                raise RuntimeError(f"FFmpeg 已退出，退出码 {encoder.returncode}，请查看其错误信息")
            success, frame = camera.read()
            # frame = frame[:,:1920,:]
            if not success or frame is None or frame.size == 0:
                failures += 1
                if failures >= 10:
                    raise RuntimeError("相机连续读取失败 10 次，请检查连接后重新启动")
                print(f"相机读取失败，重试 {failures}/10", file=sys.stderr)
                time.sleep(0.1)
                continue
            failures = 0
            if frame.dtype != "uint8" or frame.ndim != 3 or frame.shape[2] != 3:
                raise RuntimeError("采集图像必须为 uint8、BGR、H×W×3；更换 SDK 时请转换格式")
            # if frame.shape[:2] != (args.height, args.width):
            #     frame = cv2.resize(frame, (args.width, args.height))
            remaining = memoryview(frame.tobytes())
            while remaining:
                written = encoder.stdin.write(remaining)
                if not written:
                    raise RuntimeError("FFmpeg 输入管道已关闭")
                remaining = remaining[written:]
            next_frame = max(next_frame + 1 / args.fps, time.monotonic())
            time.sleep(max(0, next_frame - time.monotonic()))
    except KeyboardInterrupt:
        print("\n正在停止……")
    except (RuntimeError, OSError, cv2.error) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    finally:
        if encoder is not None:
            try:
                encoder.stdin.close()
            except OSError:
                pass
            try:
                encoder.wait(timeout=3)
            except subprocess.TimeoutExpired:
                encoder.kill()
                encoder.wait()
        if camera is not None:
            camera.release()
        print("已释放相机和编码进程资源。")
    return 0


if __name__ == "__main__":
    # raise SystemExit(main())
    main()
