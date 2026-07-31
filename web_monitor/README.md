# CVIA AGX Orin 网页控制台

控制台运行在 AGX Orin，默认监听 `0.0.0.0:8765`。同一可信局域网内的笔记本可直接访问 `http://<AGX-Orin-IP>:8765`。网页没有登录鉴权，请勿把该端口暴露到公网。

## 一键启动（不编译 C++）

先自行完成 C++ 编译，然后运行：

```bash
./web_monitor/start.sh
```

网页服务会同时维护 NOKOV 动捕 SDK 连接。第一次使用或 SDK 桥接源码更新后，先单独编译桥接程序：

```bash
cmake \
  -S web_monitor/mocap \
  -B web_monitor/mocap/build \
  -DCMAKE_BUILD_TYPE=Release

cmake --build web_monitor/mocap/build \
  --target MocapBridge \
  -j4
```

生成文件为 `web_monitor/mocap/bin/MocapBridge`。工程已包含成套的 NOKOV ARM64
头文件和 `lib/aarch64/libnokov_sdk.so`；它们来自
`/home/wts/getViedo` 中 README 所述的 NOKOV 官方 SDK，不要混用其他版本的头文件或动态库。
原始网络部署、SDK 校验和 `POSE` 协议说明已一并复制到
[`mocap/README_NOKOV_SDK_ORIN.md`](mocap/README_NOKOV_SDK_ORIN.md)；其中
`/home/wts/getViedo` 路径用于记录原始部署，当前工程运行路径以上述 `web_monitor/mocap`
目录为准。

脚本只启动网页与现有的 `./trt`，不会执行 CMake 或编译。网页先启动，随后推理开始。完成采集时，请在网页顶部点击“停止推理并保存”：

1. 网页向 `trt` 发送优雅停止请求。
2. `trt` 停止采集、推理和输出线程，并刷新临时 CSV。
3. 推理停止后网页服务继续在线，浏览器弹出会话保存面板。
4. 点击“选择位置并保存”，把 CSV 保存到当前笔记本。
5. 保存完成后，可在 AGX 终端按 `Ctrl+C` 关闭网页服务。

不要用终端 `Ctrl+C` 代替网页的停止按钮：终端中断会同时关闭推理和网页，浏览器无法继续显示保存面板。

可用环境变量修改启动项：

```bash
WEB_PORT=9000 ./web_monitor/start.sh
WEB_HOST=127.0.0.1 WEB_PORT=9000 ./web_monitor/start.sh
TRT_BINARY=/path/to/trt CONFIG_FILE=/path/to/config.json ./web_monitor/start.sh
```

如果请求的网页端口已被占用，启动脚本会自动向后查找可用端口。例如
`8765` 被占用时会尝试 `8766`，并在终端输出最终访问地址。默认最多尝试
20 个端口，可通过 `WEB_PORT_MAX_TRIES` 调整。

## 从笔记本访问

在 AGX Orin 上查询局域网地址：

```bash
hostname -I
```

例如 AGX 地址是 `192.168.1.50`，笔记本访问：

```text
http://192.168.1.50:8765
```

## PnP 临时记录与另存为

- 每次推理启动都会新建并覆盖 `web_monitor/runtime/pnp_session.csv`，不会继续向旧会话追加。
- 只记录 PnP 成功的位姿，不保存图片、检测框或关键点。
- CSV 固定为七列：`timestamp,x,y,z,rx,ry,rz`。第 1 列是时间戳，第 2–4 列是 XYZ，第 5–7 列是三个角度。
- 运行期间使用持久文件句柄，每 30 条有效位姿刷新一次；收到网页停止请求后会等待各线程退出并再次强制刷新。
- 推理仍在运行时，下载接口会返回冲突错误，避免保存尚未结束的会话。
- 网页停止推理后，顶部按钮会变为“保存本次 PnP”，因此关闭弹窗后仍可重新打开。
- 浏览器安全策略不允许普通网页强制指定笔记本目录。支持 File System Access API 的安全上下文会显示系统文件选择器；通过普通局域网 HTTP 访问时通常使用标准下载。若希望每次都弹出路径窗口，请在笔记本浏览器中开启“下载前询问每个文件的保存位置”。

`PNP_result` 不再作为运行期输出目录。网页曲线读取本次临时 CSV，预览 JPEG 和重投影 JSON 也只写入 `web_monitor/runtime`。

## 标定参数历史

- 网页每次保存配置时，会把保存前和保存后的相机内参、畸变系数、4×4 外参写入 `web_monitor/calibration_history.json`。
- 完全相同的标定不会重复记录，最多保留最近 50 个版本。
- “标定”页按时间倒序显示历史记录。选中版本并点击“载入到表单”后，还需点击底部“保存配置”才会成为下次推理使用的配置。
- `web_monitor/config.json` 和标定历史均采用临时文件加原子替换，避免断电或并发读取产生半份 JSON。

## NOKOV 动捕与位姿对照

- SDK 文件是否位于当前工作区不影响接收；运行时只要求桥接程序能加载同架构、同版本的
  `libnokov_sdk.so`，并且 Orin 到 XING SDK 服务器的网络可达。当前工程已经携带成套
  ARM64 SDK，因此启动后不依赖 `/home/wts/getViedo`。接收器默认优先使用工程内的
  `web_monitor/mocap/bin/MocapBridge`；若它不存在，会自动回退到
  `/home/wts/getViedo/XING_Linux/bin/MocapBridge`。也可以在启动时显式指定外部桥接：

  ```bash
  CVIA_MOCAP_BRIDGE=/path/to/MocapBridge ./web_monitor/start.sh
  ```

  外部桥接必须能通过自身 RPATH 或 `LD_LIBRARY_PATH` 找到配套的 NOKOV 动态库。
- 默认连接 XING/XINGYING SDK 服务器 `10.1.1.198`。本机联调时 SDK 的 `DESC`
  输出为刚体 `Tracker4`（ID 0），因此当前默认选择器为 `name:Tracker4`。可在“网络”页
  选择“按名称”或“按 ID”，也可以直接点选 SDK 已发现的刚体。点击“应用目标并连接”
  会原子保存选择并只重启动捕接收器，不中断网页或视觉推理。服务器地址和数据过期
  阈值仍随完整配置保存、重启控制台后生效。现场应始终以 API 状态中的刚体描述为准，
  名称末尾数字不等于 SDK ID。
- 网页服务启动 `MocapBridge` 并在连接退出后自动重试。XING/XINGYING 端需开启
  `SDK Enabled`，且 Orin 到服务器的 `10.1.1.0/24` 路由必须可达。
- SDK 的 XYZ 原样显示，默认单位为 mm。四元数先归一化，再转换成
  Roll-X / Pitch-Y / Yaw-Z 欧拉角，网页统一显示为度。
- `9999999` 哨兵值、非有限数值和零长度四元数会标记为无效。按照 NOKOV 部署说明，
  `params=0` 不会单独导致一帧数据被删除。
- “位姿”页按 X、Y、Z、Rx、Ry、Rz 六行并列显示视觉 PnP 与 NOKOV 动捕。
  页面不画对比曲线，也不直接计算两列差值。只有在两套坐标系完成轴向、原点、尺度和
  欧拉角约定的配准后，数值差才具有物理意义。
- “本机更新时差”是视觉状态文件修改时间与最新动捕 SDK 回调到达 Orin 的时间之差，
  用于判断两路数据的新鲜度，不等同于硬件同步误差。需要严格逐帧对齐时，应让相机、
  Orin 与 XING 主机使用统一 NTP/PTP 或硬件触发。

## 图像、相机与重投影

- 红外相机 ID 默认为 0，负责推理并允许配置曝光、白平衡和画质参数。
- 可见光相机 ID 默认为 2，只设置采集 ID、分辨率与 60 FPS，不写入曝光、白平衡或画质控制，保持设备默认成像效果。
- 两路相机目标采集帧率为 60 FPS。网页预览可设为 1–30 FPS，默认建议 15；预览 FPS 只控制 JPEG 编码和浏览器刷新，不改变相机目标帧率。
- 网页使用独立的 960×540 JPEG 预览编码；TCP 图像兼容传输仍使用 1280×720。文件夹模式的主副图为同一帧时只编码一次。
- 文件夹模式支持 JPG、JPEG、PNG、BMP，可设置循环和帧间隔。约 60 Hz 应使用 17 ms，而不是 60 ms。
- 采集、推理和输出队列均采用最新帧优先，慢速网页编码不会积压历史帧。
- 浏览器最多保留一个双图在途请求，并使用持久 Canvas 显示：后端把同批次两张 JPEG 原子发布为一个帧包，浏览器等两张都成功解码后才成组绘制；任意一张加载失败时，两块画布都继续保留上一批帧。切到后台标签页时暂停图像与位姿轮询；网页服务使用 HTTP/1.1，并忽略成功预览请求的逐条日志。
- PnP 求解成功后，C++ 使用同一组 `_p3d`、旋转/平移、内参和畸变系数调用 `cv::projectPoints`。网页在红外主图绘制七点：P0 为红色，P1–P6 为统一青色。
- 位姿页把 X、Y、Z、Rx、Ry、Rz 按轴并排显示检测与动捕的最新结果。

若采集仍低于 60 FPS，请用 `v4l2-ctl --list-formats-ext` 确认相机支持 MJPEG 1920×1080@60，并检查两路相机是否共享 USB 2.0 控制器。暗环境下自动曝光也可能延长曝光时间并降低实际帧率。

## 网络

TCP 默认关闭，不创建连接，因此不会产生旧接收端的 `connect error`。UDP 位姿上传独立配置。只有确实需要兼容旧 TCP 接收端时才在网页中重新启用 TCP。

## API

- `GET /api/config`：读取配置。
- `PUT /api/config`：校验并原子保存配置，同时维护标定历史。
- `GET /api/status`：读取图像、推理进程和临时 PnP 会话状态。
- `POST /api/inference/stop`：请求当前 `trt` 优雅停止。
- `GET /api/pnp?limit=240`：读取本次临时 CSV 尾部，供六轴曲线使用。
- `GET /api/pnp/download`：仅在推理停止后下载完整临时 CSV。
- `GET /api/calibration-history`：读取最近 50 个去重标定版本。
- `GET /api/reprojection`：读取当前帧的七点 PnP 重投影状态。
- `GET /api/pose-comparison`：读取最新视觉 PnP、NOKOV XYZ/欧拉角、数据状态及本机更新时差。
- `POST /api/mocap/target`：按 `{"mode":"name","value":"Tracker4"}` 或
  `{"mode":"id","value":0}` 保存目标刚体并立即重启动捕接收器。
- `GET /preview/pair.bin`：读取后端原子发布的同批次双图帧包（网页主路径）。
- `GET /preview/primary.jpg`、`GET /preview/secondary.jpg`：读取单张预览的兼容接口；数据同样来自双图帧包。
