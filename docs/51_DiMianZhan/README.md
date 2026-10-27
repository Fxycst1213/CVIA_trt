# Windows 相机 RTSP 视频传输

程序只有 `main.py` 一个 Python 文件，所有逻辑都在 `main()` 中。MediaMTX 单独运行；本目录的 YAML 是它的服务配置，不是另一个程序。默认使用已安装的 `D:\72_mediamtx\ffmpeg\bin\ffmpeg.exe`。

链路：USB 相机 → OpenCV 读取 uint8/BGR 图像 → FFmpeg 编码 H.264 → 发布到本机 MediaMTX → 天空端通过网口主动拉流 → 无线链路 → 地面站。

Python 发布地址固定为本机 `rtsp://127.0.0.1:8554/live`。天空端连接电脑以太网 IP，例如 `rtsp://192.168.50.10:8554/live`。无需在程序内填入天空端 IP，电脑通过 Windows 路由把拉流连接上的视频数据发送到天空端。

## 1. Python 环境

当前已检测到 Python 3.9.13、OpenCV 4.7.0 和 NumPy 1.21.5，可直接尝试运行。新环境建议使用 Python 3.11 或 3.12，在 PowerShell 中执行：

```powershell
cd F:\51_DiMianZhan
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install opencv-python
```

OpenCV 会自动安装 NumPy；无需 pip 安装 FFmpeg 或 MediaMTX。以下命令使用当前 `python`，若采用虚拟环境，请替换为 `.\.venv\Scripts\python.exe`。

## 2. 启动 MediaMTX

打开第一个 PowerShell 窗口：

```powershell
cd F:\51_DiMianZhan
& 'D:\72_mediamtx\mediamtx.exe' '.\mediamtx.yml'
```

使用本目录配置，看到 RTSP listener 在 `:8554` 启动后保持窗口运行。本配置只开启 TCP RTSP，其他常用流媒体协议关闭。若提示地址占用，请检查是否已经运行过 MediaMTX，不要重复启动。

## 3. 启动相机发布

打开第二个 PowerShell 窗口：

```powershell
cd F:\51_DiMianZhan
python .\main.py
```

默认相机 0、1280×720、30 FPS、2 Mbps、端口 8554。自定义示例：

```powershell
python .\main.py --camera 1 --width 640 --height 480 --fps 25 --bitrate 1M --rtsp-port 8554
```

用 `python .\main.py --help` 查看参数。也可以修改 `main()` 开头各参数的 `default`，然后直接运行。FFmpeg 移动后使用 `--ffmpeg '新的路径\ffmpeg.exe'`。

宽度、高度必须为正偶数。相机不支持所请求分辨率时会缩放；输出比例不同会拉伸。帧率是目标值，实际采集或编码性能不足时不能保证达到目标帧率。程序采用 ultrafast、zerolatency、无 B 帧、一秒 GOP、约 100 ms VBV 缓冲；这不是端到端延迟保证，也不是每秒码率严格恒定。相机内部缓存以及天空端、无线链路和播放器缓存仍会影响延迟。OpenCV 的缓冲设置可能被驱动忽略。

更换端口时，要同时修改 `mediamtx.yml` 的 `rtspAddress`、程序 `--rtsp-port`、拉流 URL 和防火墙规则。

## 4. 本机验证

保持上述两个窗口运行，在第三个 PowerShell 窗口执行：

```powershell
& 'D:\72_mediamtx\ffmpeg\bin\ffplay.exe' -rtsp_transport tcp -fflags nobuffer -flags low_delay -framedrop 'rtsp://127.0.0.1:8554/live'
```

VLC 也可以：媒体 → 打开网络串流，输入该 URL；命令行可使用 `vlc --rtsp-tcp --network-caching=100 rtsp://127.0.0.1:8554/live`（VLC 在 PATH 中时）。必须选择 RTP over RTSP/TCP，缓存过小可能卡顿。服务启动但尚未发布时 `/live` 没有视频，出现 404 并不表示相机已成功发布。

## 5. 网口 IP 和天空端

用网线连接电脑和天空端，或接同一交换机。在 Windows 以太网适配器的 IPv4 属性中设置，例如：

| 设备 | IP | 子网掩码 |
|---|---|---|
| 电脑以太网 | 192.168.50.10 | 255.255.255.0 |
| 天空端以太网 | 192.168.50.20 | 255.255.255.0 |

这些是示例，请按天空端实际可配置的网段设置，避免 IP 冲突；直连不跨网段时网关、DNS 可留空。使用 `ipconfig` 确认以太网地址，尝试 `ping 192.168.50.20`。ping 被设备禁用时不一定代表 RTSP 不通。

天空端填写 `rtsp://192.168.50.10:8554/live`，选择 **RTSP over TCP**。RTSP 支持不等于同时支持 TCP 和 UDP 视频传输；若天空端只支持 RTP/UDP，需要另行调整 MediaMTX 和 UDP 防火墙配置。本次先以 TCP 验证。

电脑有 Wi-Fi 和多个网口时，应填写接天空端的以太网 IPv4，不要填 Wi-Fi 地址或 127.0.0.1。IP 在 Windows 中设置，程序不修改网卡。

## 6. Windows 防火墙

在受控直连设备网络上，将该以太网设置为“专用网络”。管理员 PowerShell 执行（实际端口/IP 变化时同步修改）：

```powershell
New-NetFirewallRule -DisplayName 'MediaMTX RTSP 8554 Sky' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8554 -Profile Private -RemoteAddress 192.168.50.20
```

TCP 模式下控制和视频复用 TCP 8554，无需额外 UDP 端口。规则仅允许该天空端；如要用另一台电脑测试，可另外为其 IP 添加同类规则。从另一台 Windows 电脑执行 `Test-NetConnection 192.168.50.10 -Port 8554`，再用 ffplay/VLC 拉取网口 URL。本配置为无密码的局域网测试配置，仅用于受控网络。

## 7. 故障和退出

- 相机打不开：关闭会议/相机软件，检查 Windows 桌面应用相机权限，尝试 `--camera 1` 或 `2`。启动时会打印相机报告的尺寸和帧率。
- 读取失败：短暂失败重试，连续 10 次失败则退出；重新连接后手动启动。
- FFmpeg 报错或断管：错误直接显示在窗口，程序退出并清理相机和编码进程。检查服务、端口、重复发布和 `libx264` 支持。
- 本机可看，天空端不可看：检查电脑网口 IP、同网段、防火墙网络类型，以及天空端是否选择 TCP。
- 卡顿/积累延迟：先降为 640×480、25 FPS、1M，并减小播放器缓存；本程序同步采集与编码，不保证驱动或链路阻塞时仍实时。rawvideo 使用配置帧率计时，实际帧率不足会影响播放时序。
- 退出：发布程序按 Ctrl+C，程序关闭管道、等待或结束 FFmpeg，并释放相机；再在 MediaMTX 窗口按 Ctrl+C。极端 USB 驱动阻塞可能让 Ctrl+C 延迟响应。

## 8. 后续板卡适配

按当前要求不拆分采集函数。以后在 `main()` 中替换 `cv2.VideoCapture`、`camera.read()` 和 `camera.release()` 对应代码即可接入 SDK，帧数据应先转换为 uint8、BGR、H×W×3。真实板卡还需要适配操作系统、CPU 架构、相机 SDK 和软/硬件编码能力；Windows 成功不能证明同一程序能直接运行在板卡上。

## 本次验证结果

2026-10-14：使用本机 MediaMTX v1.21.2 和 FFmpeg 9.0.2，在仅监听回环地址的临时 18554 端口完成测试。将测试 BGR 图像送入同一个 `main()`，成功发布 H.264 640×480 视频；ffprobe 识别成功，FFmpeg 客户端成功解码 10 帧。模拟中断、连续相机读取失败和编码进程退出的资源释放检查通过。测试服务和发布进程已停止。

实际相机 0 在本次测试中无法打开，尚未完成 USB 相机、真实网口、天空端和无线地面端联调。请连接相机、检查权限和占用，必要时更换相机编号后，按上述步骤继续验证。
