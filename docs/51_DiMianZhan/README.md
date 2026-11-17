# Orin 本机 MediaMTX：本地电脑只接收

MediaMTX 运行在 Orin。Orin 的推理程序把带解算结果的图像发布到本机 MediaMTX；天空端或本地电脑通过 Orin 的网口地址读取视频。不在本地电脑上运行 MediaMTX，也不把天空端设备 IP 当作发布服务器。

```text
ZED → TensorRT / PnP → Orin 本机 MediaMTX :8554/live → 天空端 / 无线链路 / 本地播放器
                         ├─ 视频内含关键点、位置、坐标轴、源时间戳
                         ├─ 可选 Orin HTTP :8080/status.json：启用网页时由电脑读取位姿
                         └─ 可选 UDP → 本地电脑 :9000：电脑只接收独立位姿
```

当前地址：Orin 有线口 `192.168.2.107/24`，本地电脑 `192.168.2.61`。天空端设备的地址不填入 FFmpeg 发布 URL。

## Orin 一条命令启动

在 `/home/cvia/nuaa/CVIA_trt` 中执行：

```bash
./start_groundstation.sh 192.168.2.61
```

参数 `192.168.2.61` **仅是 UDP 位姿接收电脑的地址**。视频始终发布到 Orin 回环地址 `rtsp://127.0.0.1:8554/live`，由 Orin MediaMTX 统一提供给接收端。不填参数时不发独立 UDP，只输出带位姿叠加的 RTSP 视频。

脚本自动启动 Orin 上的 MediaMTX、图像/位姿桥接服务和 `trt --preview --coordinate-frames --camera-eye right`。默认启用 `/dev/ttyUSB0` 串口，9600 波特率、8N1；新解算数据充足时按 10 Hz 发送，每个周期只取最新结果，帧校验取累加和低 8 位。所有进程在后台运行，关闭终端或断开 SSH 不影响服务。旧模式会自动切换；本项目的旧网页预览会正常停止、释放相机后被接管，不影响其他项目或独立串口控制任务。启动成功前会确认 Orin `/live` 已返回有效 RTSP 描述。串口设备不存在或无法打开时，C++ 会在 `trt.log` 中报错，视频推流继续运行。

## 相机右目与标定

现在默认读取 SDK 的 **`RIGHT_UNRECTIFIED` 原始右目**。PnP 使用同一次 SDK 打开相机后返回的 `calibration_parameters_raw.right_cam`，包括右目内参和完整的 12 个 OpenCV 畸变系数。标定随同一帧图像一起传给桥接程序；RTSP 坐标轴和投影也使用这组参数。视频标注 `Camera: RIGHT`，`trt.log` 和 `server.log` 中的 `[camera]` 可核对实际参数。推理网络输入仍是 640×640。

需要切回原始左目时可使用 `CVIA_CAMERA_EYE=left`；参数会一并切换，不混用左右目或校正/未校正内参。位姿的相机参考系也随所选镜头变化。历史离线工具与旧协议仍保留原有缺省标定；实时链路会用 SDK 实际参数覆盖缺省值。

原始图像与原始标定的对应关系见 [StereoLabs 相机标定说明](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/camera-calibration)。

## CH340 / CH341 串口驱动

本机内核为 `5.15.148-tegra`，已安装匹配头文件，但原系统缺少 CH341 驱动模块。`tools/install_ch341.sh` 从 [Linux 5.15.148 稳定版源码](https://raw.githubusercontent.com/gregkh/linux/v5.15.148/drivers/usb/serial/ch341.c) 下载驱动，校验固定 SHA256，并按 [Linux 外部模块编译方法](https://docs.kernel.org/kbuild/modules.html) 编译。准备阶段只写入 `build/drivers/ch341/5.15.148-tegra/`，当前已生成 `ch341.ko`，其 AArch64 架构和 vermagic 与本机 `usbserial` 一致。

安装时在项目目录执行一条命令，并在终端输入 sudo 密码：

```bash
./tools/install_ch341.sh
```

脚本使用已缓存的源码，安装模块到当前内核的 `updates/`，执行 `depmod`、加载驱动，并仅释放被 `usbfs` 占用的 CH340 接口后重新匹配驱动。成功后会显示 `CH340 串口已就绪：/dev/ttyUSB0`。随后需停止并重新启动 `start_groundstation.sh`，让 C++ 重新初始化串口；仅加载驱动不会使此前打开失败的进程自动重连。实际加载仍需在 sudo 安装后验证。

仅下载和编译可运行 `./tools/install_ch341.sh --prepare`，无需 sudo。脚本限定当前内核版本，升级内核后需重新匹配源码和头文件。

## MediaMTX 安装

MediaMTX 固定为官方 v1.21.3 Linux ARM64/AMD64，保存在 `tools/bin/`，由启动脚本管理，无需 sudo 或系统服务。二进制不纳入 Git。首次缺少程序时会从官方发行页下载并检查固定 SHA256；现场断网时可提前运行离线安装器：

```bash
python3 tools/install_mediamtx.py --archive /路径/mediamtx_v1.21.3_linux_arm64.tar.gz
```

## 本地电脑 / 天空端接收

天空端的 RTSP 输入配置，或本地电脑 VLC 的“打开网络串流”，填写：

```text
rtsp://192.168.2.107:8554/live
```

选择 **RTSP over TCP**。本地电脑如果安装了 FFmpeg，可用一条命令接收：

```powershell
ffplay -rtsp_transport tcp -fflags nobuffer -flags low_delay -framedrop rtsp://192.168.2.107:8554/live
```

本地电脑只做播放器/接收器，不启动本目录原来的 `main.py` 或 MediaMTX。若天空端主动拉取视频，在天空端填写 Orin 的 URL；若图传设备提供 IP 透传，本地电脑也可直接访问该 URL。具体是否支持 RTSP/TCP 和独立 IP 数据透传，需要以图传设备配置为准。

普通浏览器不能直接播放 RTSP。现在默认**只用 RTSP 显示，不启动网页服务**，Orin 的 8080 端口关闭，也不执行网页缩放或网页 JPEG 编码。RTSP 内的位姿叠加与独立 UDP 位姿继续输出；推理输入和算法不变。

需要临时开启网页时，一条命令切换：

```bash
CVIA_WEB_ENABLE=1 ./start_groundstation.sh 192.168.2.61
```

启用后网页为 `http://192.168.2.107:8080`，独立位姿 JSON 为 `http://192.168.2.107:8080/status.json`，均不需要 SSH 转发。关闭网页可使用 `CVIA_WEB_ENABLE=0`，或在未设置该环境变量时直接运行默认启动命令。启动成功判断直接检测 RTSP，不依赖网页接口。

## 独立位姿接收（可选）

如果本地电脑需要 UDP 数值，可在电脑上运行本目录的接收器，只依赖 Python 标准库：

```powershell
python .\receive_pose.py
```

它只接收 UDP 9000，不运行媒体服务器。数值在本地 `http://127.0.0.1:8081` 查看；需要保存时使用 `--csv results/pose.csv`。UDP 9000 的电脑入站规则只影响独立 UDP，视频接收不依赖电脑开放 8554 入站端口。启用 Orin 网页后，也可由电脑直接读取其 HTTP JSON，不运行任何 UDP 接收器。

位姿叠加在视频里时可随视频一起经过图传。独立 UDP 或 HTTP 需要无线链路支持对应的 IP 数据传输。UDP 不保证到达，接收器会拒绝近期乱序/重复时间戳，并标识超过一秒未更新的数据。

## 配置与停止

更新代码后，在项目目录用一条命令停止旧服务、增量编译并重新启动：

```bash
./start_groundstation.sh --stop && make -j2 && ./start_groundstation.sh 192.168.2.61
```

正常启动命令会复用同配置的现有进程；源码更新后需要上述重启步骤，才能加载新程序。先停止也可避免编译后覆盖正在运行的 `trt` 文件失败。默认仍为 1080p RTSP、关闭网页预览。

MediaMTX 配置为 `tools/mediamtx_orin.yml`，仅开放 RTSP/TCP。脚本使用的运行配置和日志在 `build/groundstation/`，包括 `mediamtx.log`、`server.log`、`trt.log` 和 `launcher.log`。停止这一组进程：

```bash
./start_groundstation.sh --stop
```

需要调整视频端口、帧率或码率时，在同一行指定环境变量；已有服务会按新配置重新启动：

```bash
CVIA_RTSP_PORT=8554 CVIA_VIDEO_FPS=15 CVIA_BITRATE=1500k ./start_groundstation.sh 192.168.2.61
```

默认 RTSP 图像为 **1920×1080、20 FPS、约 4 Mbps**。相机采集仍为 1920×1080，TensorRT 网络输入仍为 640×640；模型、预处理和位姿解算算法未改动。1080p 来自相机原图，不是将 720p 视频放大。发送端 JPEG 与 RTSP 使用同一传输宽度，框、关键点、PnP 结果保持同帧对应。

网页预览默认关闭。启用时网页 JPEG 为 1280×720，独立于 RTSP，减少额外 JPEG 编码、浏览器和网络开销。RTSP 编码在独立线程执行，只取最新画面；桥接/FFmpeg 以较低 CPU 调度优先级运行，软件编码限制两个线程。网络或编码阻塞不会等待推理线程。1080p 仍会增加 JPEG/视频编码和内存带宽开销，因此不能保证所有负载下推理帧率绝对不变。

需要降低传输带宽时，可在一行命令中切回 720p，而不更改模型输入：

```bash
CVIA_VIDEO_WIDTH=1280 CVIA_BITRATE=2M ./start_groundstation.sh 192.168.2.61
```

`CVIA_WEB_ENABLE=1 CVIA_WEB_WIDTH=1920` 可启用 1080p 网页预览；`CVIA_VIDEO_WIDTH=1920` 是 RTSP 默认值。修改后仍只需要同一个启动脚本，旧配置会自动切换。

## 位姿数据含义

每个 UDP 数据报是一条 UTF-8 JSON，`version=1`。`timestamp_ms` 是同一张相机图像的源时间戳，也会叠在视频底部。独立 UDP 和视频经过不同缓冲，接收时刻不能当成同帧匹配依据，应按源时间戳匹配。

| 字段 | 含义 |
|---|---|
| `detected` | 此帧是否检测到目标；不代表 PnP 已成功 |
| `predicted_xyz_mm` | 现有预测 XYZ，毫米 |
| `filtered_xyz_mm` | 相机系滤波/补偿后的 XYZ，外参变换的输入，毫米 |
| `camera_xyz_mm` | 相机系原始 PnP XYZ，与 `pnp_tvec_mm` 相同；当前帧 PnP 无效时为 `null` |
| `robot_base_xyz_mm` | 同帧 `combined * [filtered_xyz_mm, 1]` 的前三项，即机械臂基底 XYZ，毫米 |
| `robot_base_valid` | 是否已有可用的滤波/变换结果；首次解算前为 `false`，旧协议为 `null` |
| `pnp_valid` | 此帧 PnP 是否成功；旧 IHV 协议没有该标志，返回 `null` |
| `pnp_rvec_rad` | 当前帧 Rodrigues 旋转向量，弧度；不是欧拉角。无效时为 `null` |
| `pnp_tvec_mm` | 当前帧未经滤波的 PnP 平移，毫米。无效时为 `null` |
| `camera` | 同帧相机标定：`eye`、原始视图、图像尺寸、`intrinsics=[fx,fy,cx,cy]` 和 12 项 `distortion`；旧协议为 `null` |

网页分别显示相机系原始 PnP、相机系滤波/补偿、机械臂基底系三行 XYZ，保留两位小数，单位均为毫米。后两行是外参变换前后的同一位置：`robot_base_xyz_mm = (combined * [filtered_xyz_mm, 1])[:3]`。外参由 C++ 使用现有矩阵计算，网页不保存另一份矩阵。`filtered_xyz_mm` 包含现有卡尔曼滤波及绕 X 轴 -0.18° 的补偿；基底值为串口换轴/偏移前的结果，串口仍使用 `[base_y + 220, base_x + 250, -base_z + 90]`。这些数值对应飞机模型中央蓝色标记定义的原点。

当前帧 PnP 无效时，原始相机坐标显示“—”；已有滤波结果时，滤波与基底位置明确标为估计值。首次解算前或网页数据超过一秒未更新时，相应坐标显示“—”。网页接口及独立 UDP 均提供上述新增字段，PC 的 `receive_pose.py` 网页也显示相同坐标表。

完整 PnP 约定为 `camera_point = Rodrigues(pnp_rvec_rad) * body_point + pnp_tvec_mm`，机体轴为 X 指向机头、Y 指向右侧、Z 指向下方，原点为白色飞机机身中央蓝色标记的中心（仅作原点，不参与 PnP）。当前使用 `models/onnx/x_plane1.onnx` 的 9 个关键点，三维坐标来自 `data/plane9/points_body_mm.json`，按翼展 3445 mm 缩放。滤波输出经过现有补偿，与原始 PnP 平移分别提供，不混用。PnP 失败时不发送上一次的旋转向量冒充当前有效结果。视频中坐标轴仅在当前帧 PnP 有效时绘制，轴长 200 mm。

图像 TCP 包中的关键点与框为 `9*3+5=32` 个 float（128 字节），接收端 `tools/server.py` 必须与发送端同步使用 9 点；切换模型后需重启发送端与接收端。位姿字段和后续标定字段的格式保持原有约定。

`receive_pose.py --csv results/pose.csv` 可保存数值。图像模式沿用 IHV 握手，发送端与接收端均使用上述 9 点数据长度；IHD 纯数据格式不变。`--extended-pose` 使用 IHX 握手，在图像、关键点和 32 字节位置区之后追加 28 字节：`<I6f`，分别为有效标志、三维旋转向量和三维平移。仅在接收端支持 IHX 时启用。

`--calibrated-pose` / IHC：在 IHX 内容后追加 `<III16d>`，分别是 JPEG 宽/高、右目标志、4 个内参和 12 个畸变系数。缩小传输图像时同时缩放内参，畸变系数保持不变，推理仍使用原始帧。旧接收端不支持 IHC 时需要同步更新 `tools/server.py`。

当前一键脚本使用 `--coordinate-frames` / IHB：在 IHC 内容后追加 16 字节 `<I3f>`，依次为 `robot_base_valid` 和三维基底坐标（float、毫米）。无效时坐标区置零，由接收端输出 `null`。IHB 的有效标志表示滤波器已初始化，与当前帧 `pnp_valid` 分开；旧 IHV/IHD/IHX/IHC 解析仍受支持。更新后需重新编译并重启发送端与网页服务；地面站网页通过 `CVIA_WEB_ENABLE=1 ./start_groundstation.sh 192.168.2.61` 开启。

发布接口参照 [MediaMTX FFmpeg 发布文档](https://mediamtx.org/docs/publish/ffmpeg)；视频使用 [FFmpeg RTSP/TCP](https://www.ffmpeg.org/ffmpeg-protocols.html#rtsp) 发布到服务端，不需要 Windows 相机采集程序参与。

## Jetson 适配验证

右目补充验证：已重新编译并部署 `RIGHT_UNRECTIFIED`。真实相机与桥接程序均记录右目标定 `fx=1057.53, fy=1057.40, cx=965.01, cy=553.697`，包含 12 项原始畸变系数。13 项预览/协议测试和 7 项媒体/位姿测试通过，覆盖右目标定随帧传输及 JPEG 缩放时内参同步缩放。本机 RTSP 识别为 H.264 1920×1080、20 FPS，并实际解码 10 帧；8080 未监听。此次检查时 `eno1` 为 `NO-CARRIER`、无 IPv4，需恢复网线连接及有线地址后再做接收端验证。

仅 RTSP 模式补充验证：Orin 的 8080 端口关闭，RTSP 仍输出 H.264 1920×1080、20 FPS，并实际解码 10 帧。7 项媒体/位姿测试通过；新增测试禁止 HTTP 创建、网页缩放和 JPEG 编码调用，同时验证 RTSP 和独立 UDP 位姿仍可用。

1080p 补充验证：通过 Orin 有线地址实际识别并解码 H.264 1920×1080、20 FPS；网页预览仍独立为 1280×720。6 项媒体/位姿回归测试通过，新增测试覆盖同一帧的原生 1080p RTSP 与独立 720p 网页预览。推理模型与 640×640 输入未调整。

Orin 本机模式补充验证：MediaMTX 与真实 ZED 推理均已部署在 Orin，监听 8554。通过 `rtsp://192.168.2.107:8554/live` 识别到 H.264 1280×720、20 FPS，并实际解码 10 帧。独立位姿持续通过 Orin HTTP 输出，包含当前帧 PnP 旋转、平移及有效标志。5 项媒体/位姿回归测试和进程隔离测试通过；重复启动会复用当前服务。远端电脑或无线图传设备的最终播放仍需在接收端确认。

2026-10-27：20 项回归测试通过，覆盖 IHV/IHD 兼容、IHX 完整位姿、同帧图像/数据、无效 PnP 不带旧向量、慢编码管道、RTSP 中断重连及 UDP 接收。使用临时 MediaMTX v1.21.3 模拟地面站，真实 ZED 相机成功发布 H.264 1280×720、20 FPS，FFmpeg 实际解码 10 帧；UDP 收到有效旋转/平移及与视频源一致的时间戳。验证脱离启动终端后后台进程仍持续推流。测试服务仅监听本机回环地址，实际远端地面站、防火墙及无线链路尚未联调。

---

# Windows 相机 RTSP 视频传输（原始参考方案）

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
