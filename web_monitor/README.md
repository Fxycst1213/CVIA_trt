# CVIA AGX Orin 网页控制台

控制台运行在 AGX Orin，默认监听 `0.0.0.0:8765`。同一可信局域网内的笔记本可直接访问 `http://<AGX-Orin-IP>:8765`。网页没有登录鉴权，请勿把该端口暴露到公网。

## 一键启动（不编译 C++）

先自行完成 C++ 编译，然后运行：

```bash
./web_monitor/start.sh
```

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

## 图像、相机与重投影

- 红外相机 ID 默认为 0，负责推理并允许配置曝光、白平衡和画质参数。
- 可见光相机 ID 默认为 2，只设置采集 ID、分辨率与 60 FPS，不写入曝光、白平衡或画质控制，保持设备默认成像效果。
- 两路相机目标采集帧率为 60 FPS。网页预览可设为 1–30 FPS，默认建议 30；预览 FPS 只控制 JPEG 编码和浏览器刷新，不改变相机目标帧率。
- 文件夹模式支持 JPG、JPEG、PNG、BMP，可设置循环和帧间隔。约 60 Hz 应使用 17 ms，而不是 60 ms。
- 采集、推理和输出队列均采用最新帧优先，慢速网页编码不会积压历史帧。
- PnP 求解成功后，C++ 使用同一组 `_p3d`、旋转/平移、内参和畸变系数调用 `cv::projectPoints`。网页在红外主图绘制七点：P0 为红色，P1–P6 为统一青色。
- 位姿页把 X、Y、Z、Rx、Ry、Rz 分别绘制在六个时序坐标系中。

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
- `GET /preview/primary.jpg`、`GET /preview/secondary.jpg`：读取两路网页预览。
