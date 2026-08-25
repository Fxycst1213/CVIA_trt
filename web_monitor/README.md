# CVIA AGX Orin 网页控制台

控制台运行在 AGX Orin，默认监听 `0.0.0.0:8765`。同一可信局域网内的笔记本可直接访问 `http://<AGX-Orin-IP>:8765`。网页没有登录鉴权，请勿把该端口暴露到公网。

## 一键启动（不编译 C++）

先完成 C++ 编译：

```bash
cmake -S . -B build
cmake --build build -j2
```

然后运行：

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

NOKOV `iTimeStamp` 已确认来自独立时钟。当前使用的 `CLOCK` FrameGroup 连续采样、动态
仿射映射、短于 2 秒位姿段和离线锚点复用的完整说明见
[`mocap/README_SDK_AFFINE_TIME_SYNC.md`](mocap/README_SDK_AFFINE_TIME_SYNC.md)。

脚本只启动网页服务，不会执行 CMake、不会自动启动推理。它会校验推理程序，默认优先使用
`build/trt`，若它不存在再回退到项目根目录的 `./trt`；也可通过 `TRT_BINARY` 显式指定。
推荐工作流程：

1. 运行 `start.sh`，等待网页控制台上线。
2. 在网页修改 ONNX 模型路径、输入源、相机、标定、网络、动捕和运行参数。
3. 点击底部“保存并应用”。有未保存修改时，顶部“开始推理”按钮不可用。
4. 点击顶部“开始推理”。网页服务使用当前保存的配置启动 `trt`，并按最新动捕设置建立
   本轮 NOKOV 会话。
5. 完成采集时点击“停止推理并保存”；`trt` 会优雅停止并刷新临时 CSV。
6. 在浏览器会话面板点击“选择位置并保存”，把 CSV 保存到当前笔记本。
7. 可以继续修改配置并开始下一轮，也可以在 AGX 终端按 `Ctrl+C` 关闭网页服务。

若上一轮 PnP 临时会话尚未下载，开始新一轮前网页会要求明确确认，因为 C++ 每次启动都会
覆盖 AGX 上的临时 `pnp_session.csv`。成功下载后可直接开始下一轮。

不要用终端 `Ctrl+C` 代替网页的停止按钮：终端中断会同时关闭推理和网页，浏览器无法继续显示保存面板。

## 网页选择 ONNX 模型

“输入”页的 `ONNX 文件路径` 保存为 `model.onnx_path`。可填写相对于软件根目录的路径，
例如 `models/onnx/qdy0815.onnx`，也可填写 AGX 上的绝对路径。模型不会在推理运行中热切换；
停止当前推理、保存路径并再次点击“开始推理”后，C++ 才创建新的 TensorRT 上下文。

系统按 ONNX 位置自动推导 FP16 engine：

```text
models/onnx/example.onnx  ->  models/engine/example-fp16.engine
```

- 对应 engine 已存在时直接加载，因此客户运行包可以只带 engine 而不泄露 ONNX。
- engine 不存在但 ONNX 存在时，启动时会在推导出的 `engine` 目录构建；该目录必须可写。
- 两者均不存在时，“开始推理”会禁用并显示模型不可用，服务端也会拒绝启动，而不是进入
  TensorRT 后才报告模糊的 ONNX 解析错误。
- 关键点数量由 `calibration.model_keypoints_3d` 的数组长度决定，允许 4～256 点。单类别
  模型的输出特征数必须满足 `5 + 3×点数`（4 个框参数、1 个类别分数、每点 X/Y/置信度）。
  例如 10 点对应 35 个特征，16 点对应 53 个特征；不匹配时推理进程会明确报错并停止初始化。
- 客户保密包默认不附带 ONNX；若客户在网页选择其他模型，需要由客户自行把相应 ONNX
  或预先生成的同名 `*-fp16.engine` 放到设备上。

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

## Windows 11 / Windows 7 浏览器兼容

网页顶部“显示模式”按当前电脑单独保存，不会写入 AGX 的项目配置，因此不同电脑可以
同时使用不同模式：

- `自动`：检测到 `Windows NT 6.1` 时启用 Windows 7 兼容模式，其他系统使用
  Windows 11 模式。
- `Windows 11`：保留背景模糊、阴影、高 DPI Canvas 等完整显示效果。
- `Windows 7 兼容`：使用微软雅黑/宋体本地字体链，关闭背景模糊、复杂阴影、过渡和
  滚动吸附，并把预览及重投影 Canvas 的设备像素倍率限制为 1，减少旧系统软件合成负载。

如果 Windows 7 页面文字已经乱码，导致无法操作顶部选择框，可直接访问：

```text
http://<AGX-Orin-IP>:8765/?compat=win7
```

该选择会保存到这台电脑的浏览器 `localStorage`。恢复自动模式可访问：

```text
http://<AGX-Orin-IP>:8765/?compat=auto
```

网页服务对 HTML、CSS、JavaScript 和 SVG 显式发送 UTF-8 `Content-Type`，并附带
`Content-Language: zh-CN` 与 `X-UA-Compatible: IE=edge`。建议 Windows 7 使用该系统
可安装的最终 Chromium Edge 109；IE 和过旧的非 Chromium 浏览器不在支持范围内。

## PnP 临时记录与另存为

- 每次推理启动都会新建并覆盖 `web_monitor/runtime/pnp_session.csv`，不会继续向旧会话追加。
- 只记录 PnP 成功的位姿，不保存图片、检测框或关键点。
- CSV 兼容保留毫秒字段 `timestamp`，并增加纳秒级 `capture_timestamp_ns` 和
  `publish_timestamp_ns`；其后是 `x,y,z,rx,ry,rz` 与 `coordinate_frame=mocap`。同步使用
  采集时间，结果发布时间只用于统计“采集→推理/PnP完成”的管线延迟。离线工具仍可读取旧的
  七列 CSV；旧文件按主程序同一条 `m_result` 输出链视为动捕坐标系。
- 运行期间使用持久文件句柄，每 30 条有效位姿刷新一次；收到网页停止请求后会等待各线程退出并再次强制刷新。
- 推理仍在运行时，下载接口会返回冲突错误，避免保存尚未结束的会话。
- 网页停止推理后，顶部按钮会变为“保存本次 PnP”，因此关闭弹窗后仍可重新打开。
- 顶部“开始推理”只在配置已经保存、推理程序可用且当前没有推理进程时启用。上一轮临时
  会话尚未另存时，开始新一轮需要确认覆盖；已经通过网页下载后不会重复询问。
- 浏览器安全策略不允许普通网页强制指定笔记本目录。支持 File System Access API 的安全上下文会显示系统文件选择器；通过普通局域网 HTTP 访问时通常使用标准下载。若希望每次都弹出路径窗口，请在笔记本浏览器中开启“下载前询问每个文件的保存位置”。

`PNP_result` 不再作为运行期输出目录。网页曲线读取本次临时 CSV，预览 JPEG 和重投影 JSON 也只写入 `web_monitor/runtime`。

## 标定参数历史

`calibration.extrinsic` 的定义固定为 `T_M_C`，即从相机坐标系 C 到 NOKOV 动捕坐标系 M
的齐次变换。主程序对 `solvePnP` 输出的完整位姿执行
`T_M_O = T_M_C × T_C_O`，再把 `T_M_O` 写入实时 JSON、PNP CSV、UDP/TCP 位姿输出。因此
离线绘图直接使用 PNP CSV，不能再次应用外参，否则会造成重复变换。

- 网页每次保存配置时，会把保存前和保存后的相机内参、畸变系数、动态三维模型点集及
  4×4 外参写入 `web_monitor/calibration_history.json`。
- 完全相同的标定不会重复记录，最多保留最近 50 个版本。
- “标定”页按时间倒序显示历史记录。选中版本并点击“载入到表单”后，还需点击底部“保存并应用”；推理运行时会在下一帧前切换到新标定。
- “世界轴姿态修正”支持 `XYZ、XZY、YXZ、YZX、ZXY、ZYX` 六种固定动捕/世界轴顺序，顺序文字从左到右表示实际执行次序。例如选择 `XZY` 时，网页按 `R_new = Ry × Rz × Rx × R_old` 左乘旋转块。默认相机光心固定，所以平移列不变；输入值是相对当前表单矩阵的增量，连续应用会累积，应用后仍需点击底部“保存并应用”。
- `web_monitor/config.json` 和标定历史均采用临时文件加原子替换，避免断电或并发读取产生半份 JSON。

## 标定参数运行时热更新

相机内参、畸变系数、`calibration.model_keypoints_3d` 和 `calibration.extrinsic` 支持在推理运行期间热更新。网页完成
校验并原子保存 `config.json` 后，C++ 推理线程每 250 ms 检查一次文件版本，并在下一帧
TensorRT/PnP 推理开始前一次性应用全部标定矩阵。该过程不会重新加载 TensorRT engine、
不会关闭相机，也不会中断网页和预览流。

- 只修改外参时，保留相机坐标系中的卡尔曼滤波状态，从下一帧开始使用新 `T_M_C` 输出。
- 修改内参或畸变系数时，自动清除旧标定下的 PnP 缓存、重投影结果和卡尔曼滤波状态，
  防止新观测被旧滤波状态当作跳变异常点。
- 修改任一三维模型点时同样会清除 PnP 缓存和卡尔曼滤波状态；新坐标从下一帧参与
  `solvePnP` 和模型点重投影。
- PnP 不再以“RANSAC 有四个内点”作为唯一有效条件。运行时会去除落在同一图像亮斑上的重复
  关键点，要求至少 60% 的高置信关键点成为内点，LM 精修后的内点 RMS 重投影误差不超过
  2.5 px、最大误差不超过 4 px，并拒绝位于相机后方或近零深度的解。
- 连续帧优先以上一帧位姿作为 ITERATIVE PnP 初值，并在旋转矩阵域检查平移与旋转连续性。
  单帧大幅跳变不会进入 CSV、UDP、TCP 或网页位姿；目标确实移动到新位置时，连续三帧相互
  一致的候选会触发重新捕获和卡尔曼重置，避免永久锁死在旧位置。
- 关键点亮斑精修的搜索范围从 60 px 收紧为 30 px，并同时过滤过大轮廓，减少相邻标记误吸附。
  卡尔曼滤波只平滑已经通过几何门控的测量，不再采用“连续拒绝三帧后接受异常值”的旧逻辑。
- 输出欧拉角会在两组数学等价的 XYZ 分支中选择最接近上一帧的一组，并按 360° 连续展开，
  防止跨越 ±180° 或接近 `Ry=±90°` 时出现没有物理意义的角度竖直跳变。
- 三维模型点按 P0～Pn 顺序对应模型输出。网页支持新增、删除及修改 XYZ，点数范围为
  4～256；只改坐标可在下一帧热应用，增删点会改变 TensorRT 输出解析维度，因此保存后需
  停止并重新开始推理，但不再需要重新编译程序。
- C++ 每次读取配置后原子写入 `web_monitor/runtime/runtime_config_status.json`。网页只有在
  该文件确认的配置时间戳与当前 `config.json` 一致时，才显示“标定参数已热应用”。
- 推理运行期间保存时，若同时修改 ONNX 模型路径、相机 ID、分辨率、FPS、输入模式、网络端点或运行参数，
  网页会显示“部分生效”：标定立即应用，其余参数在停止并再次点击“开始推理”后生效，
  不需要重启网页服务。
- 配置解析失败时继续使用上一个有效标定，网页显示运行时应用失败；修正并再次保存后会
  自动重试。

## NOKOV 动捕与位姿对照

- SDK 文件是否位于当前工作区不影响接收；运行时只要求桥接程序能加载同架构、同版本的
  `libnokov_sdk.so`，并且 Orin 到 XING SDK 服务器的网络可达。当前工程已经携带成套
  ARM64 SDK，因此启动后不依赖工程目录以外的开发机路径。接收器默认使用工程内的
  `web_monitor/mocap/bin/MocapBridge`；也可以在启动时显式指定外部桥接：

  ```bash
  CVIA_MOCAP_BRIDGE=/path/to/MocapBridge ./web_monitor/start.sh
  ```

  外部桥接必须能通过自身 RPATH 或 `LD_LIBRARY_PATH` 找到配套的 NOKOV 动态库。
- 默认连接 XING/XINGYING SDK 服务器 `10.1.1.198`。本机联调时 SDK 的 `DESC`
  输出为刚体 `Tracker4`（ID 0），因此当前默认选择器为 `name:Tracker4`。可在“网络”页
  选择“按名称”或“按 ID”，也可以直接点选 SDK 已发现的刚体。点击“应用目标并连接”
  会原子保存选择并只重启动捕接收器，不中断网页或视觉推理。服务器地址和数据过期
  阈值随完整配置保存，并在下次点击“开始推理”重建动捕接收器时生效。现场应始终以 API 状态中的刚体描述为准，
  名称末尾数字不等于 SDK ID。
- 网页服务启动 `MocapBridge` 并在连接退出后自动重试。XING/XINGYING 端需开启
  `SDK Enabled`，且 Orin 到服务器的 `10.1.1.0/24` 路由必须可达。
- SDK 的 XYZ 原样显示，默认单位为 mm。四元数先归一化，再转换成
  Roll-X / Pitch-Y / Yaw-Z 欧拉角，网页统一显示为度。
- `9999999` 哨兵值、非有限数值和零长度四元数会标记为无效。按照 NOKOV 部署说明，
  `params=0` 不会单独导致一帧数据被删除。
- “位姿”页按 X、Y、Z、Rx、Ry、Rz 六行显示视觉 PnP 与已经完成时间配对的 NOKOV
  动捕。视觉列是经 `T_M_C` 变换后的动捕坐标系位姿，NOKOV 列是动捕坐标系原生位姿；
  两列的位置单位均为 mm，角度均按 Roll-X / Pitch-Y / Yaw-Z 显示为度。
- “本机更新时差”是视觉状态文件修改时间与最新动捕 SDK 回调到达 Orin 的时间之差，
  用于判断两路数据的新鲜度，不等同于硬件同步误差。需要严格逐帧对齐时，应让相机、
  Orin 与 XING 主机使用统一 NTP/PTP 或硬件触发。

## 软件时间同步

工程同时提供实时同步和离线同步，两者都使用视觉帧内嵌的采集 Unix 时间与 NOKOV 回调
到达 Orin 的 Unix 时间配对，不再把文件修改时间作为采集时间：

- 实时同步完全位于 Python 网页服务中。NOKOV 接收线程维护默认 5 秒的轻量位姿历史；
  网页读取一帧视觉 PnP 后，按其采集时间查询历史动捕。C++ 相机采集、TensorRT、PnP 和
  输出队列没有增加等待、Socket 发送、图像复制或同步锁，因此动捕迟到只会显示
  `未配对`，不会阻塞主检测。
- 默认寻找视觉时刻前后的两帧动捕。XYZ 线性插值，姿态先做四元数 SLERP，再转换为
  Roll-X / Pitch-Y / Yaw-Z 欧拉角；缺少包围帧时退化为最近邻。最近帧超过
  `sync.max_error_ms` 时拒绝配对，不复用陈旧动捕值。
- `sync.offset_ms` 会加到速率校正后的动捕时间轴，用来补偿相机/USB/解码与 NOKOV
  曝光、解算、网络/SDK 链路之间的固定延迟。网页可以在采集停止后，通过视觉/动捕运动曲线的归一化
  互相关自动估计该值，不能默认把 0 ms 当作已标定值。
- 时间轨显示的 `Δt` 是视觉采集时刻与最近动捕接收时刻（含补偿）的差；数据源年龄和旧的
  “本机更新时差”仍只用于新鲜度诊断。

配置项：

```json
"sync": {
  "enabled": true,
  "offset_ms": 0.0,
  "max_error_ms": 12.0,
  "history_ms": 5000,
  "interpolate": true,
  "auto_offset_search_ms": 1000
}
```

自动估计不直接比较两套坐标的 X/Y/Z 数值，而是分别构造平移速度模长和四元数角速度模长，
平滑后在 `±auto_offset_search_ms` 内做归一化互相关。速度模长不受坐标原点和平移轴方向影响，
归一化相关也不受固定单位比例影响，因此对残余空间标定误差不敏感。平移通道
权重为 70%，转动通道为 30%；某个通道没有足够变化时由另一个有效通道完成估计。

建议在两套设备都采集到目标时，连续做 5～10 秒带有停顿、加减速和转向的非周期平移与转动。
停止推理后先点击“自动估计 offset”，再生成离线报告。算法以 5 ms 粗搜、1 ms 精搜，并检查
相关峰值、峰值与次峰差以及搜索边界：只有中/高置信结果才会原子写回 `sync.offset_ms`；静止、
重复周期运动、采样太少或最佳峰位于边界时不会覆盖人工值，应增加有效运动或扩大搜索范围后重试。

USB 相机不支持 PTP，但当前相机采集已经绕过会丢失缓冲元数据的 OpenCV
`VideoCapture::read()`，直接通过 `VIDIOC_DQBUF` 取得 V4L2 帧时间戳。若驱动声明
`CLOCK_MONOTONIC`，程序会在出队时用 `CLOCK_REALTIME`—`CLOCK_MONOTONIC` 夹读映射，
把帧时间转换到由 PTP 驯服的 Unix 时间域；若驱动直接提供实时钟时间则原样使用。
V4L2 的 SOE/EOF 标志、时间戳来源以及采集到结果的延迟会写入实时重投影状态并显示在网页。

若驱动不给时间戳、时间戳跳变，或设备不支持直接 V4L2 MJPEG streaming，程序会自动回退
到 OpenCV，并明确显示“软件接收时间”。回退值仍可通过运动曲线互相关估计固定补偿，但精度
低于有效的 V4L2 帧时间戳。相机时间转换成立的前提是 `phc2sys` 不仅同步网卡 PHC，还把
Orin 的 `CLOCK_REALTIME` 驯服到同一 PTP 时域；只运行 `ptp4l` 而不校准系统时钟是不够的。

约 30 ms 的 TensorRT/PnP 时间不会加入采集时间。结果同时携带：

```text
capture_timestamp_ns  = V4L2 帧采集时间，用于视觉/动捕配对
publish_timestamp_ns  = 推理和 PnP 完成时间，用于延迟统计
pipeline_latency_ms   = (publish - capture) / 1e6
```

结果完成后用 `capture_timestamp_ns` 回查动捕历史并做位置/四元数插值，因此融合显示会晚约
30 ms，但视觉和动捕仍代表同一个物理时刻。NOKOV `iTimeStamp` 已确认来自独立时钟：接收线程
用 `receive_monotonic_ns` 在默认 60 秒滑动窗口内拟合（180 帧更新一次）
`T_orin = a × (T_sdk - anchor) + b`，以 `aligned_unix_ns` 消除独立时钟的长期速率漂移。
模型预热期间或速率超出保护范围时自动退回 `receive_unix_ns`；原始 SDK、收包和校正时间均写入
CSV。仿射模型只能确定时间速率和收包基线，不能单独识别曝光、解算、传输的固定业务延迟，
因此剩余固定差仍由 `offset_ms` 或运动互相关标定。

桥接程序在每个 FrameGroup 回调先输出一条不依赖刚体可见性的
`CLOCK\tframe\tsdk_ms\treceive_unix_ns\treceive_monotonic_ns`，然后才按选择器输出
`POSE`。因此即使 Tracker3 暂时完全不可见，独立时钟模型仍能预热；刚体随后只出现不到
2 秒时，可以立即使用已就绪模型映射位姿。新旧桥接兼容：没有 `CLOCK` 的旧版本仍从
`POSE` 取时钟样本。离线 CSV 保存模型就绪标记、速率和校正锚点，短位姿段不会重新估计速率。

动捕独立时钟配置位于 `mocap`：

```json
"clock_mode": "sdk_affine",
"clock_fit_window_seconds": 60.0,
"clock_fit_min_seconds": 3.0,
"clock_max_rate_ppm": 5000.0
```

网页服务的动捕状态和同步结果会返回 `clock_model`。`ready=true` 后使用校正时间轴；
`rate_ppm` 是 Orin 每个 SDK 毫秒所需的速率修正量，例如约 `+896 ppm` 与实测独立时钟漂移一致。

## 离线同步与六轴图

- 每次点击“开始推理”都会按最新动捕配置重建接收器，并新建
  `web_monitor/runtime/mocap_session.csv`。动捕回调只追加一条小型
  数值记录，使用持久缓冲句柄并每 30 行刷新；不会保存图像，也不进入检测线程。
- 停止推理后，在“位姿”页点击“生成离线报告”。只要存在有效的
  `pnp_session.csv`，即使没有启用动捕、没有生成 `mocap_session.csv`，也会生成相机检测 PNG。
  “自动估计 offset”仍需要两路数据；有动捕时可先估计 offset，再生成完整同步报告。生成时会
  冻结本次动捕记录，并按照当前 `sync` 参数重新配对，生成：

  - `synchronized_session.csv`：每个成功配对的视觉/动捕位姿、统一的
    `coordinate_frame=mocap`、动捕帧号、同步方式、`sync_error_ms`、插值系数和包围帧间隔；
  - `synchronized_camera_report.png`：相机检测数据图，从相机首次输出有效位姿的时刻开始绘制完整
    X、Y、Z、Rx、Ry、Rz；这些相机推导位姿已经由 `T_M_C` 转到动捕坐标系；
  - `synchronized_mocap_report.png`：仅在至少成功配对一帧动捕时生成；与相机图共用同一时间
    零点，只绘制成功配对的动捕坐标系原生位姿，尚未配对或中途断配时保持空档；
  - `synchronized_report.png`：仅在至少成功配对一帧动捕时生成；在同一个六轴布局内叠加相机
    与动捕曲线，只用于观察时间趋势和对齐效果；两条曲线均为动捕坐标系位姿；
  - `synchronized_summary.json`：覆盖率、最近邻/插值数量以及平均、P95、最大绝对时间差。

- 所有生成图片均为真正的 `1200×990`、8-bit、RGB、非隔行 PNG，不是把 SVG 重命名为 PNG。
  内置光栅画布只使用 Python 标准库绘制网格、文字和分段折线，不依赖 Matplotlib、Pillow、
  Cairo 或浏览器截图，因此客户机无需额外安装绘图库。
- 离线工具会对旧 CSV 应用与运行时一致的 SE(3) 连续性检查。被标记为异常的连续区间如果同时
  存在可信前帧和后帧，X/Y/Z 按纳秒时间戳做线性插值，姿态使用四元数 SLERP；原始 CSV 不会
  被改写。同步 CSV 的 `visual_interpolated=1` 可识别修复记录，图片顶部显示 `REPAIRED` 与
  `DROPPED` 数量。缺少任一边界的首尾异常帧会丢弃，不做单边外推。角度曲线继续使用最近的
  数学等价欧拉角分支。相机报告支持只有约 10～20 张图片的稀疏采集：只要相邻两张图片都有
  有效相机位姿，就按采集时间做分段线性插值并连续绘制，不再因图片间隔较大而主动断线；这只
  影响报告曲线，不会伪造额外检测帧。动捕报告的未配对区仍保持断线，不会用相机数据补动捕。
- 离线画图会进一步把相机有效位姿从原始约 60 Hz 重采样到统一 300 Hz 时间轴：X/Y/Z 按纳秒
  捕获时间线性插值，旋转使用四元数 SLERP；随后在每个 300 Hz 时刻匹配动捕并生成相机、动捕
  和合成曲线。原始 `pnp_session.csv` 与同步 CSV 仍保留真实相机帧率，不写入伪造检测帧；摘要
  中的 `camera_plot_hz`、`camera_plot_samples` 和 `camera_plot_interpolated_samples` 记录绘图频率、
  绘图总点数与新生成点数。300 Hz 插值只提高比较时间分辨率，不等同于 300 Hz 真实相机测量。
- 报告生成只允许在推理停止后执行，避免离线全量扫描和 PNG 压缩与 TensorRT 主检测争用 CPU。
  无动捕时网页只显示相机报告卡片；成功配对动捕时显示相机、动捕、合成三张图，每张图右上角
  均可直接下载 PNG。生成相机单源报告时会删除上一次遗留的动捕图和合成图，避免混用会话。
- 六轴图中的所有位姿均位于动捕坐标系：视觉 PnP 使用 `T_M_C` 转换结果，NOKOV 使用
  原生结果。绘图不假设自由落体方向，也不对 X/Y/Z 任一轴施加位移、单调性或静止约束。

也可以对已经保存的两个 CSV 使用命令行离线生成：

```bash
python3 web_monitor/offline_sync.py \
  --detection /path/to/pnp_session.csv \
  --mocap /path/to/mocap_session.csv \
  --output-dir /path/to/report \
  --estimate-offset \
  --search-ms 1000 \
  --max-error-ms 12
```

`--estimate-offset` 会在互相关通过可靠性检查后使用估计值；也可去掉它并通过
`--offset-ms` 指定人工补偿。增加 `--nearest-only` 可禁用位置/四元数插值。

## 图像、相机与重投影

- 红外相机 ID 默认为 0，负责推理并允许配置曝光、白平衡和画质参数。
- 可见光相机 ID 默认为 2，只设置采集 ID、分辨率与 60 FPS，不写入曝光、白平衡或画质控制，保持设备默认成像效果。
- 两路相机目标采集帧率为 60 FPS。网页预览可设为 1–30 FPS，默认建议 15；预览 FPS 只控制 JPEG 编码和浏览器刷新，不改变相机目标帧率。
- 网页使用独立的 960×540 JPEG 预览编码；TCP 图像兼容传输仍使用 1280×720。文件夹模式的主副图为同一帧时只编码一次。
- 文件夹模式支持 JPG、JPEG、PNG、BMP，可设置循环和帧间隔。约 60 Hz 应使用 17 ms，而不是 60 ms。
- 采集、推理和输出队列均采用最新帧优先，慢速网页编码不会积压历史帧。
- 浏览器最多保留一个双图在途请求，并使用持久 Canvas 显示：后端把同批次两张 JPEG 原子发布为一个帧包，浏览器等两张都成功解码后才成组绘制；任意一张加载失败时，两块画布都继续保留上一批帧。切到后台标签页时暂停图像与位姿轮询；网页服务使用 HTTP/1.1，并忽略成功预览请求的逐条日志。
- PnP 求解成功后，C++ 使用同一组旋转/平移、内参和畸变系数分别重投影刚体坐标系原点 `(0,0,0)` 和 `_p3d` 的模型特征点。网页在红外主图将刚体几何中心绘制为红色，将自动编号的模型特征点绘制为青色。`src/params/pose_params.hpp` 的 `MODEL_KEYPOINTS_3D` 只提供默认几何；实际点数和 XYZ 均来自网页保存的 `calibration.model_keypoints_3d`。
- 位姿页把 X、Y、Z、Rx、Ry、Rz 按轴并排显示检测与动捕的最新结果。

若采集仍低于 60 FPS，请用 `v4l2-ctl --list-formats-ext` 确认相机支持 MJPEG 1920×1080@60，并检查两路相机是否共享 USB 2.0 控制器。暗环境下自动曝光也可能延长曝光时间并降低实际帧率。

## 网络

TCP 默认关闭，不创建连接，因此不会产生旧接收端的 `connect error`。UDP 位姿上传独立配置。只有确实需要兼容旧 TCP 接收端时才在网页中重新启用 TCP。

## API

- `GET /api/config`：读取配置。
- `PUT /api/config`：校验并原子保存配置，维护标定历史，并返回热应用/下次启动生效参数分类。
- `GET /api/status`：读取图像、推理进程和临时 PnP 会话状态。
- `POST /api/inference/start`：使用当前已保存配置启动推理；可传
  `{"discard_previous_session":true}` 明确覆盖尚未另存的临时会话。
- `POST /api/inference/stop`：请求当前 `trt` 优雅停止。
- `GET /api/pnp?limit=240`：读取本次临时 CSV 尾部，供六轴曲线使用。
- `GET /api/pnp/download`：仅在推理停止后下载完整临时 CSV。
- `GET /api/calibration-history`：读取最近 50 个去重标定版本。
- `GET /api/reprojection`：读取当前帧的刚体中心与模型特征点 PnP 重投影状态；`origin` 为几何中心像素坐标，`points` 为按模型输出顺序编号的特征点。
- `GET /api/pose-comparison`：读取最新视觉 PnP、NOKOV XYZ/欧拉角、数据状态及本机更新时差。
- `GET /api/pose-comparison` 的 `synchronization` 字段：读取实时配对状态、配对后的动捕位姿、
  动捕帧号、插值系数及同步误差。
- `GET /api/sync/offline`：读取本次离线同步报告状态和统计摘要。
- `POST /api/sync/offline/estimate-offset`：推理停止后对运动曲线做归一化互相关；可靠时自动
  写回 `sync.offset_ms`，低置信结果仅返回诊断信息。
- `POST /api/sync/offline/generate`：推理停止后生成报告；无动捕时生成相机六轴 PNG，成功配对
  动捕时另生成同步 CSV、动捕 PNG 与合成 PNG。
- `GET /api/sync/offline/download.csv`：下载离线同步 CSV。
- `GET /api/sync/offline/report/camera.png`：查看或下载相机检测六轴 PNG。
- `GET /api/sync/offline/report/mocap.png`：查看或下载动捕六轴 PNG。
- `GET /api/sync/offline/report/composite.png`：查看或下载相机/动捕合成六轴 PNG。
- `GET /api/sync/offline/report.png`：合成图的短兼容入口。
- `POST /api/mocap/target`：按 `{"mode":"name","value":"Tracker4"}` 或
  `{"mode":"id","value":0}` 保存目标刚体并立即重启动捕接收器。
- `GET /preview/pair.bin`：读取后端原子发布的同批次双图帧包（网页主路径）。
- `GET /preview/primary.jpg`、`GET /preview/secondary.jpg`：读取单张预览的兼容接口；数据同样来自双图帧包。
