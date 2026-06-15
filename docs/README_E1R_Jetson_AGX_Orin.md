# RoboSense E1R on Jetson AGX Orin

本文档记录 RoboSense E1R 在 Jetson AGX Orin 上通过 ROS2 Humble、`rslidar_sdk` 和 RViz2 在线接入点云的最小操作流程。

当前仓库是普通 C++/CUDA/TensorRT 工程，不是 ROS2 package。这里新增的 `config/`、`launch/`、`rviz/` 和 `scripts/` 文件用于配合外部 `rslidar_sdk` 工作，不改动本仓库的主推理程序。

## 已验证链路

```text
E1R -> Jetson eno1 -> UDP 6699/7788 -> rslidar_sdk -> /rslidar_points -> RViz2
```

已实测 `/rslidar_points` 约 10 Hz。

## 网络参数

| 项目 | 值 |
| --- | --- |
| 设备 | Jetson AGX Orin + RoboSense E1R |
| 雷达连接网口 | `eno1` |
| 雷达 IP | `192.168.1.200` |
| Jetson 主机 IP | `192.168.1.102/16` |
| MSOP 点云端口 | `6699` |
| DIFOP 信息端口 | `7788` |
| ROS2 | Humble |
| 点云 topic | `/rslidar_points` |
| frame id | `rslidar` |
| rslidar_sdk lidar_type | `RSE1` |

## 固化 Jetson eno1 IP

先查看 `eno1` 对应的 NetworkManager 连接名：

```bash
nmcli device status
nmcli -t -f NAME,DEVICE connection show
```

如果 `eno1` 已有连接名，可以使用下面命令固化 IP。这里假设连接名保存到 `CONN`：

```bash
CONN="$(nmcli -t -f NAME,DEVICE connection show | awk -F: '$2=="eno1"{print $1; exit}')"
echo "${CONN}"

sudo nmcli connection modify "${CONN}" \
  connection.interface-name eno1 \
  ipv4.method manual \
  ipv4.addresses 192.168.1.102/16 \
  ipv4.never-default yes \
  ipv6.method disabled

sudo nmcli connection up "${CONN}"
```

检查结果：

```bash
ip addr show eno1
ip route get 192.168.1.200
```

本仓库不会修改 `/etc/NetworkManager` 或其他系统文件；以上命令需要在 Jetson 上手动执行。

## 抓包验证

使用新增脚本：

```bash
cd /home/wts/CVIA_trt
./scripts/e1r_check_network.sh
```

脚本内部会执行：

```bash
ip addr show eno1
ip route get 192.168.1.200
sudo tcpdump -i eno1 -nn "udp port 6699 or udp port 7788" -c 10
```

如果需要覆盖默认参数：

```bash
LIDAR_IFACE=eno1 LIDAR_IP=192.168.1.200 CAPTURE_COUNT=20 ./scripts/e1r_check_network.sh
```

## rslidar_sdk 配置

本仓库新增配置：

```text
config/e1r_rslidar_sdk.yaml
```

关键参数：

```yaml
common:
  msg_source: 1
  send_point_cloud_ros: true

lidar:
  - driver:
      lidar_type: RSE1
      msop_port: 6699
      difop_port: 7788
      host_address: 192.168.1.102
    ros:
      ros_frame_id: rslidar
      ros_send_point_cloud_topic: /rslidar_points
```

注意：E1R 在 `rs_driver v1.5.19` / `rslidar_sdk` 中的雷达类型必须写成 `RSE1`，不要写成 `E1R`。

## 编译 rslidar_sdk 工作区

下面假设外部 ROS2 工作区路径为 `~/e1r_ros2_ws`，且其中已经包含 `rslidar_sdk` 和 `rslidar_msg`：

```bash
cd ~/e1r_ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 启动 E1R 在线点云

优先使用本仓库新增的路径式 launch 文件。它会启动 `rslidar_sdk/rslidar_sdk_node`，并把 `config_path` 指到 `config/e1r_rslidar_sdk.yaml`：

```bash
cd ~/e1r_ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch /home/wts/CVIA_trt/launch/e1r_start.py
```

也可以直接运行节点：

```bash
cd ~/e1r_ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run rslidar_sdk rslidar_sdk_node --ros-args \
  -p config_path:=/home/wts/CVIA_trt/config/e1r_rslidar_sdk.yaml
```

如果后续把这些文件迁移到一个真正的 ROS2 bringup package，才使用 package 形式：

```bash
ros2 launch <package_name> e1r_start.py
```

当前仓库没有 `package.xml`，因此不能直接作为 `<package_name>` 被 `ros2 launch` 查找。

## 检查点云 topic

另一个终端：

```bash
source /opt/ros/humble/setup.bash
source ~/e1r_ros2_ws/install/setup.bash
ros2 topic list
ros2 topic info /rslidar_points
ros2 topic hz /rslidar_points
```

查看一帧消息头和字段：

```bash
ros2 topic echo /rslidar_points --once
```

正常情况下，`header.frame_id` 应为 `rslidar`。

## RViz2 显示

本仓库新增 RViz2 配置：

```text
rviz/e1r_pointcloud.rviz
```

该配置使用：

- Fixed Frame：`rslidar`
- Display：`PointCloud2`
- Topic：`/rslidar_points`

Jetson 本机显示器环境下可使用脚本：

```bash
cd /home/wts/CVIA_trt
./scripts/e1r_start_rviz.sh
```

手动启动命令：

```bash
export DISPLAY=:1
export XAUTHORITY=/run/user/1002/gdm/Xauthority
export XDG_RUNTIME_DIR=/run/user/1002

source /opt/ros/humble/setup.bash
source ~/e1r_ros2_ws/install/setup.bash
rviz2 -d /home/wts/CVIA_trt/rviz/e1r_pointcloud.rviz
```

## 已知问题

不要把 `rs_driver_viewer` 作为当前 Jetson + E1R 的主显示方案。它在当前 Jetson + PCL/VTK/X11 环境下会崩溃，典型调用栈包含：

```text
_XEventsQueued()
XPending()
vtkXRenderWindowInteractor::StartEventLoop()
pcl::visualization::PCLVisualizer::spinOnce()
```

推荐在线显示方案是：

```text
ROS2 Humble + rslidar_sdk + RViz2
```

## 参考

- rslidar_sdk 官方参数说明：<https://docs.ros.org/en/ros2_packages/jazzy/api/rslidar_sdk/doc/intro/02_parameter_intro.html>
- rslidar_sdk 在线雷达高级配置：<https://docs.ros.org/en/ros2_packages/jazzy/api/rslidar_sdk/doc/howto/07_online_lidar_advanced_topics.html>
- RoboSense rslidar_sdk 仓库：<https://github.com/RoboSense-LiDAR/rslidar_sdk>
