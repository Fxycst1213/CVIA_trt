# CVIA TensorRT Pose Inference

本项目是一个基于 C++、CUDA、TensorRT、OpenCV 和 ZED SDK 的实时位姿推理程序。程序从 ZED 相机采集图像，使用 YOLO Pose/TensorRT 模型完成推理，并通过 TCP 发送图像、关键点和位姿数据；同时可以通过 CAN 或 RS485 发送前三个浮点位姿结果。

## 功能概览

- ZED/ZED X 相机实时取流。
- TensorRT GPU 推理，默认使用 FP16 精度。
- 支持 YOLO Pose 关键点和位姿结果输出。
- TCP 客户端发送 JPEG 图像、关键点、位姿和时间戳。
- 支持 CAN 或 RS485 独立线程发送位姿结果。
- 推理、TCP 发送和通信发送使用独立队列，队列堆积时丢弃旧帧以保证实时性。
- 新增 RoboSense E1R 在线点云接入辅助文件，可配合外部 ROS2 Humble `rslidar_sdk` 发布 `/rslidar_points` 并用 RViz2 显示。

## 项目结构

```text
.
├── CMakeLists.txt                 # CMake 构建配置
├── config/
│   └── e1r_rslidar_sdk.yaml       # RoboSense E1R rslidar_sdk 配置
├── launch/
│   └── e1r_start.py               # E1R rslidar_sdk 路径式 ROS2 launch
├── rviz/
│   └── e1r_pointcloud.rviz        # /rslidar_points RViz2 显示配置
├── scripts/
│   ├── e1r_check_network.sh       # E1R 网络和抓包检查
│   └── e1r_start_rviz.sh          # Jetson 本机显示器 RViz2 启动脚本
├── docs/
│   └── README_E1R_Jetson_AGX_Orin.md
├── src/
│   ├── main.cpp                   # 程序入口和主要运行参数
│   ├── prj_detector.*             # 相机采集、推理、TCP、CAN/RS485 主流程
│   ├── params/                    # 运行参数定义
│   ├── ZEDX/                      # ZED 相机封装
│   ├── communication/             # TCP、CAN、RS485 通信
│   ├── TensorRT/                  # TensorRT 模型、logger、calibrator、工具代码
│   ├── yolov_pose/                # YOLO Pose 解码和后处理
│   ├── yolov8_detector/           # YOLOv8 检测模块
│   ├── yolov8_classifier/         # YOLOv8 分类模块
│   ├── preprocess/                # CUDA/OpenCV 预处理
│   ├── lstm/                      # LSTM 预测模块
│   └── algorithms/                # 轨迹、周期等算法模块
├── models/
│   ├── onnx/                      # ONNX 模型
│   └── engine/                    # TensorRT engine 文件
├── data/
│   ├── source/                    # 示例输入图片
│   └── result/                    # 示例输出结果
└── build/                         # CMake 构建目录
```

## 环境依赖

当前 `CMakeLists.txt` 中的路径和架构更偏向 Jetson/aarch64 环境：

- CMake 3.16 或更高版本
- C++14 编译器
- CUDA 12.6
- TensorRT
- cuDNN
- OpenCV
- ZED SDK 3.x
- Eigen3
- FFTW3
- Linux CAN SocketCAN 支持

当前硬编码配置包括：

```cmake
set(CMAKE_CUDA_ARCHITECTURES 87)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")
set(TENSORRT_INCLUDE_DIR "/usr/include/aarch64-linux-gnu/")
set(TENSORRT_LIB_DIR "/usr/lib/aarch64-linux-gnu/")
```

如果设备 CUDA 路径、GPU 架构或 TensorRT 安装路径不同，需要先修改 `CMakeLists.txt`。

## RoboSense E1R 在线点云

E1R 接入不改动当前 TensorRT/ZED 主程序。当前仓库也不是 ROS2 package，因此新增文件作为外部 `rslidar_sdk` 的配置、启动和验证辅助：

- 配置：`config/e1r_rslidar_sdk.yaml`
- 启动：`launch/e1r_start.py`
- RViz2：`rviz/e1r_pointcloud.rviz`
- 检查脚本：`scripts/e1r_check_network.sh`
- RViz2 脚本：`scripts/e1r_start_rviz.sh`

### 快速核验是否用到了 E1R

先确认工程中的 E1R 配置文件存在，并且雷达类型、端口、topic 没写错：

```bash
cd /home/wts/CVIA_trt
grep -nE "lidar_type|msop_port|difop_port|ros_frame_id|ros_send_point_cloud_topic" config/e1r_rslidar_sdk.yaml
```

期望看到：

```text
lidar_type: RSE1
msop_port: 6699
difop_port: 7788
ros_frame_id: rslidar
ros_send_point_cloud_topic: /rslidar_points
```

启动前先检查 Jetson `eno1` 是否能收到 E1R 的 UDP 包：

```bash
cd /home/wts/CVIA_trt
./scripts/e1r_check_network.sh
```

启动 `rslidar_sdk` 时使用本仓库的路径式 launch：

```bash
cd ~/e1r_ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch /home/wts/CVIA_trt/launch/e1r_start.py
```

另一个终端核验节点实际使用的配置路径：

```bash
source /opt/ros/humble/setup.bash
source ~/e1r_ros2_ws/install/setup.bash
ros2 node list
ros2 param get /rslidar_sdk/rslidar_sdk_node config_path
```

`config_path` 应指向：

```text
/home/wts/CVIA_trt/config/e1r_rslidar_sdk.yaml
```

最后检查点云是否发布出来：

```bash
ros2 topic list
ros2 topic info /rslidar_points
ros2 topic hz /rslidar_points
ros2 topic echo /rslidar_points --once | sed -n '1,25p'
```

期望 `/rslidar_points` 持续发布，频率接近实测的约 `10 Hz`，并且消息头里的 `frame_id` 为 `rslidar`。

详细网络参数、固化 `eno1` IP、抓包验证、`rslidar_sdk` 启动、`/rslidar_points` 检查和 RViz2 显示流程见：

[docs/README_E1R_Jetson_AGX_Orin.md](docs/README_E1R_Jetson_AGX_Orin.md)

## 构建

首次构建：

```bash
cmake -S . -B build
cmake --build build -j2
```

如果已经生成过 `build/`，可以直接编译：

```bash
cmake --build build -j2
```

仓库根目录下也可能存在旧的构建输出 `./trt`，推荐以后统一使用 `build/trt`。

## 运行

运行前确认：

- ZED 相机已连接并可被 ZED SDK 正常识别。
- 默认模型文件存在：`models/onnx/0128last.onnx`。
- TCP 接收端已启动，并监听 `src/main.cpp` 中配置的 IP 和端口。
- 如果启用 CAN，`can0` 已配置并处于 up 状态。
- 如果启用 RS485，串口设备路径和波特率正确。

启动程序：

```bash
./build/trt
```

如果使用根目录旧输出：

```bash
./trt
```

## 主要参数

主要运行参数在 `src/main.cpp` 中配置：

```cpp
string onnxPath = "models/onnx/0128last.onnx";

params.img = {640, 640, 3};
params.task = model::task_type::POSE;
params.dev = model::device::GPU;
params.prec = model::precision::FP16;

p_params.H = 1080;
p_params.W = 1920;
p_params.resolution = "HD1080";
p_params.cameraID = 0;

p_params.ip = "192.168.137.1";
p_params.port = 1234;
p_params.socket_mode = 0;
```

参数说明：

- `onnxPath`：推理模型路径。
- `params.img`：模型输入尺寸，默认 `640x640x3`。
- `p_params.H/W`：相机图像尺寸，默认 `1920x1080`。
- `p_params.resolution`：ZED 分辨率字符串，默认 `HD1080`。
- `p_params.cameraID`：相机 ID。
- `p_params.ip` / `p_params.port`：TCP 服务端地址。
- `p_params.socket_mode`：TCP 发送模式。

TCP 模式：

- `0`：发送 JPEG 图像、关键点、位姿和时间戳。
- `1`：只发送位姿数据和时间戳。

## CAN / RS485 配置

通信模式同样在 `src/main.cpp` 中切换。

使用 CAN：

```cpp
p_params.communication_mode = CommunicationMode::CAN;
p_params.can_interface = "can0";
p_params.can_base_id = 0x120;
```

使用 RS485：

```cpp
p_params.communication_mode = CommunicationMode::RS485;
p_params.rs485_port = "/dev/ttyUSB0";
p_params.rs485_baudrate = B57600;
```

禁用 CAN/RS485：

```cpp
p_params.communication_mode = CommunicationMode::NONE;
```

发送间隔：

```cpp
p_params.communication_send_interval_us = 150000;
```

CAN 和 RS485 当前发送格式均为文本形式：

```text
x.xxx,y.yyy,z.zzz
```

Classic CAN 单帧最多 8 字节，项目会按 8 字节切分文本载荷，并从 `can_base_id` 开始使用连续 CAN ID 发送。

## SocketCAN 示例

如果使用 CAN，需要先根据实际波特率配置接口，例如：

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up
```

检查 CAN 状态：

```bash
ip -details link show can0
```

监听 CAN 数据：

```bash
candump can0
```

## 无相机图片调试

`prj_v8detector` 中保留了 `camera_foldimages()`，可从图片文件夹读取图片进行调试。当前 `run()` 默认启动真实相机线程：

```cpp
auto t1 = std::thread(_func_camera);
// auto t2 = std::thread(_func_camera_foldimages);
```

如果没有连接 ZED 相机，可以按需改为图片文件夹模式，并确认 `camera_foldimages()` 中的图片路径存在：

```cpp
cv::String folder = "/home/cvia/yifei/images_old3/*.png";
```

注意：当前代码已经启用 ZED 初始化：

```cpp
_zed = ZEDX::GetInstance();
_zed->init(p_params.cameraID, p_params.resolution);
```

## TensorRT engine 注意事项

`models/engine/` 中包含多个 `.engine` 文件。TensorRT engine 与生成它的 GPU 型号、CUDA 版本、TensorRT 版本强相关。跨设备或跨版本使用时，可能出现如下警告或运行失败：

```text
Using an engine plan file across different models of devices is not recommended
```

如果遇到兼容性问题，建议在目标设备上用对应 ONNX 模型重新生成 engine。

## 常见问题

### 找不到 CUDA 或 nvcc

确认 CUDA 安装路径是否为 `/usr/local/cuda-12.6/`。如果不是，修改 `CMAKE_CUDA_COMPILER`。

### 找不到 TensorRT 头文件或库

确认 TensorRT 是否安装在：

```text
/usr/include/aarch64-linux-gnu/
/usr/lib/aarch64-linux-gnu/
```

如果路径不同，修改 `TENSORRT_INCLUDE_DIR` 和 `TENSORRT_LIB_DIR`。

### ZED 初始化失败

检查：

- 相机是否连接。
- ZED SDK 是否安装。
- `ZED_Diagnostic_Results.json` 中的诊断结果。
- `p_params.cameraID` 和 `p_params.resolution` 是否匹配当前设备。

### TCP 连接失败

检查：

- 接收端是否已启动。
- `p_params.ip` 和 `p_params.port` 是否正确。
- 本机与接收端网络是否可达。
- 防火墙是否阻止连接。

### CAN 初始化失败

检查：

- `can0` 是否存在。
- CAN 接口是否已经 `up`。
- 当前用户是否有 SocketCAN 操作权限。
- 线束、终端电阻和波特率是否匹配。

## 开发提示

- 切换模型、相机、TCP、CAN、RS485 参数时，优先修改 `src/main.cpp`。
- 修改通信结构体默认值时，查看 `src/params/params.hpp`。
- TCP 打包格式在 `src/communication/client.cpp`。
- CAN 文本切帧逻辑在 `src/communication/CAN.cpp`。
- RS485 文本发送逻辑在 `src/communication/RS485.cpp`。
