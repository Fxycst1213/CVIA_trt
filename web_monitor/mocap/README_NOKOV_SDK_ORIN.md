# 在 NVIDIA Jetson Orin 部署 NOKOV SDK 并保存动捕数据

PTP 硬件时间同步的独立部署、验证和故障排查见
[`README_PTP_TIME_SYNC.md`](README_PTP_TIME_SYNC.md)。

当前 NOKOV SDK 独立时钟到 Orin 的 `CLOCK`/`POSE` 动态仿射同步方式见
[`README_SDK_AFFINE_TIME_SYNC.md`](README_SDK_AFFINE_TIME_SYNC.md)。

本文只说明以下内容：

- 在 ARM64/AArch64 架构的 NVIDIA Jetson Orin 上部署 NOKOV SDK；
- 配置 Orin 与 XING/XINGYING SDK 服务器之间的网络；
- 编译并运行 `MocapBridge`；
- 按刚体名称或刚体 ID 接收动捕数据；
- 将完整动捕帧或指定的七维位姿保存为 TXT。

本文使用的工作目录为：

```text
/home/wts/getViedo
```

当前网络约定：

| 项目 | 值 |
|---|---|
| XING/XINGYING SDK 服务器 | `10.1.1.198` |
| Orin 动捕网段地址 | `10.1.1.199/24` |
| Orin 有线网卡 | `eno1` |
| NetworkManager 连接名称 | `orin-eno1-static` |

## 一、部署结构

完成部署后，相关文件结构如下：

```text
/home/wts/getViedo/
├── third_party/
│   └── mocap4ros2_nokov/
│       └── mocap4r2_nokov_driver/
│           └── nokov_sdk/
│               ├── include/
│               │   ├── NokovSDKCAPI.h
│               │   ├── NokovSDKClient.h
│               │   ├── NokovSDKTypes.h
│               │   └── Utility.h
│               └── lib/
│                   └── aarch64/
│                       └── libnokov_sdk.so
└── XING_Linux/
    ├── CMakeLists.txt
    ├── MocapBridge.cpp
    ├── include/
    ├── lib/
    │   └── libnokov_sdk.so
    ├── bin/
    │   └── MocapBridge
    └── build/
```

SDK 来源为 NOKOV 官方仓库：

```text
https://github.com/NOKOV-MOCAP/mocap4ros2_nokov.git
```

本次验证使用的官方仓库提交为：

```text
81d4902ab0bfec06029fbbcaf24f90e4c45ec47d
```

## 二、安装构建工具

在 Orin 上执行：

```bash
sudo apt update
sudo apt install -y git cmake build-essential
```

确认系统架构：

```bash
uname -m
```

Orin 应输出：

```text
aarch64
```

如果输出 `x86_64`，说明当前设备不是 ARM64 Orin，不能使用下面的
`lib/aarch64/libnokov_sdk.so`。

## 三、配置 Orin 动捕网络

先查看现有地址：

```bash
ip -br -4 address show eno1
```

当前 Orin 需要保留原来的：

```text
192.168.137.2/24
```

同时追加：

```text
10.1.1.199/24
```

先查看 NetworkManager 连接名称：

```bash
nmcli -g NAME,DEVICE,TYPE connection show --active
```

当前使用的连接名称是：

```text
orin-eno1-static
```

如果 `10.1.1.199/24` 尚未存在，执行：

```bash
sudo nmcli connection modify orin-eno1-static \
  +ipv4.addresses 10.1.1.199/24

sudo nmcli device reapply eno1
```

这里使用 `+ipv4.addresses`，表示追加地址，不会覆盖
`192.168.137.2/24`。

重新检查：

```bash
ip -br -4 address show eno1
ip -4 route get 10.1.1.198
```

预期结果包含：

```text
eno1  UP  192.168.137.2/24 ... 10.1.1.199/24
10.1.1.198 dev eno1 src 10.1.1.199
```

检查连通性：

```bash
ping -I 10.1.1.199 -c 4 10.1.1.198
```

必须能收到 `10.1.1.198` 的回复。如果路由仍然经过 Wi-Fi 或其他网卡，不要继续启动
SDK，应先修正地址和路由。

## 四、获取 NOKOV 官方 ARM64 SDK

进入工作目录：

```bash
cd /home/wts/getViedo
mkdir -p third_party
```

第一次部署时克隆官方仓库：

```bash
git clone --depth 1 \
  https://github.com/NOKOV-MOCAP/mocap4ros2_nokov.git \
  third_party/mocap4ros2_nokov
```

如果目录已经存在，不要再次执行 `git clone`。可用以下命令检查：

```bash
git -C third_party/mocap4ros2_nokov rev-parse HEAD
```

设置官方 SDK 路径：

```bash
NOKOV_OFFICIAL_SDK=/home/wts/getViedo/third_party/mocap4ros2_nokov/mocap4r2_nokov_driver/nokov_sdk
```

检查 ARM64 动态库：

```bash
file "$NOKOV_OFFICIAL_SDK/lib/aarch64/libnokov_sdk.so"
```

预期包含：

```text
ELF 64-bit
ARM aarch64
shared object
```

检查依赖：

```bash
ldd "$NOKOV_OFFICIAL_SDK/lib/aarch64/libnokov_sdk.so"
```

不应出现：

```text
not found
```

官方 SDK 必须成套使用以下文件：

```text
nokov_sdk/include/
nokov_sdk/lib/aarch64/libnokov_sdk.so
```

不要把其他版本的头文件与这个 ARM64 动态库混合使用。

## 五、备份原 SDK

替换前先保存当前 `XING_Linux/include` 和 `XING_Linux/lib`。

使用带时间戳的备份目录，避免覆盖以前的备份：

```bash
cd /home/wts/getViedo

SDK_BACKUP="XING_Linux/sdk_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SDK_BACKUP"

cp -a XING_Linux/include "$SDK_BACKUP/include"
cp -a XING_Linux/lib "$SDK_BACKUP/lib"

echo "SDK backup: $SDK_BACKUP"
```

当前工作区还保留了一份最初的 x86-64 SDK：

```text
XING_Linux/sdk_backup_x86_64/
```

验证备份库：

```bash
file "$SDK_BACKUP/lib/libnokov_sdk.so"
```

## 六、安装 ARM64 SDK

执行：

```bash
cd /home/wts/getViedo

NOKOV_OFFICIAL_SDK=/home/wts/getViedo/third_party/mocap4ros2_nokov/mocap4r2_nokov_driver/nokov_sdk

cp -a "$NOKOV_OFFICIAL_SDK/include/." \
  XING_Linux/include/

cp -a "$NOKOV_OFFICIAL_SDK/lib/aarch64/libnokov_sdk.so" \
  XING_Linux/lib/libnokov_sdk.so

chmod 755 XING_Linux/lib/libnokov_sdk.so
```

验证安装结果：

```bash
file XING_Linux/lib/libnokov_sdk.so
ldd XING_Linux/lib/libnokov_sdk.so
```

必须显示：

```text
ARM aarch64
```

并且 `ldd` 不能出现 `not found`。

确认复制后的库与官方源文件完全一致：

```bash
sha256sum \
  "$NOKOV_OFFICIAL_SDK/lib/aarch64/libnokov_sdk.so" \
  XING_Linux/lib/libnokov_sdk.so
```

两行 SHA-256 必须相同。本次验证的库哈希为：

```text
40a96298c636ae9b5f82a8e4532d608c4c9b6f1ce74fc4ee6111421d0c863a53
```

## 七、编译 MocapBridge

配置 Release 构建：

```bash
cd /home/wts/getViedo

cmake \
  -S XING_Linux \
  -B XING_Linux/build \
  -DCMAKE_BUILD_TYPE=Release
```

编译：

```bash
cmake --build XING_Linux/build \
  --target MocapBridge \
  -j4
```

注意，构建命令必须以 `cmake --build` 开头，不能只写：

```text
--build XING_Linux/build
```

生成文件：

```text
XING_Linux/bin/MocapBridge
```

验证程序架构：

```bash
file XING_Linux/bin/MocapBridge
```

必须显示：

```text
ARM aarch64
```

检查运行时库：

```bash
ldd XING_Linux/bin/MocapBridge
```

应看到：

```text
libnokov_sdk.so => /home/wts/getViedo/XING_Linux/lib/libnokov_sdk.so
```

程序已经设置构建 RPATH，因此不需要把 NOKOV 动态库复制到 `/usr/lib`，也不需要设置
全局 `LD_LIBRARY_PATH`。

检查命令行：

```bash
XING_Linux/bin/MocapBridge --help
```

## 八、配置 XING/XINGYING SDK 广播

在运行 XING/XINGYING 的电脑上完成：

1. 选择连接动捕交换机的网卡；
2. 确认该网卡地址为 `10.1.1.198`；
3. 打开“数据广播”设置；
4. 开启“使用 SDK”或 `SDK Enabled`；
5. 实时模式下开始运行；
6. 后处理模式下开始播放；
7. 确认目标刚体已经加载。

官方文档：

- [XINGYING 4.6 官方文档](https://xingying-docs.nokov.com/xingying/XingYing4.6-CN/)
- [数据广播说明](https://xingying-docs.nokov.com/xingying/XingYing4.2-CN/liu-shi-tu-jie-shao/san-shu-ju-guang-bo/)

## 九、测试 SDK 数据

### 按刚体名称

例如保存 `Tracker3`：

```bash
cd /home/wts/getViedo

XING_Linux/bin/MocapBridge \
  --server 10.1.1.198 \
  --tracker "name:Tracker3"
```

也可以简写为：

```bash
XING_Linux/bin/MocapBridge \
  --server 10.1.1.198 \
  --tracker "Tracker3"
```

推荐使用 `name:` 前缀，使名称选择与 ID 选择更清楚。

### 按刚体 ID

如果描述输出确认某个刚体 ID 是 0：

```bash
XING_Linux/bin/MocapBridge \
  --server 10.1.1.198 \
  --tracker id:0
```

刚体名称末尾的数字不一定等于 SDK ID。例如名称 `Tracker3` 不代表其 ID 必然是 3。
以程序输出的 `DESC` 为准：

```text
DESC	0	Tracker1
DESC	3	Tracker3
```

上例只用于说明格式，实际 ID 必须使用现场输出。

### 正常启动输出

正常时首先出现刚体描述：

```text
DESC	0	Tracker1
DESC	3	Tracker3
```

随后出现 SDK 版本：

```text
READY	2.5.47.54
```

最后持续输出：

```text
POSE	...
```

按 `Ctrl+C` 停止。

如果只有 `DESC` 和 `READY`、没有 `POSE`，检查：

- XING/XINGYING 是否正在运行或播放；
- SDK 广播是否开启；
- 刚体名称或 ID 是否正确；
- 数据是否从 `10.1.1.198` 对外广播。

## 十、保存完整动捕数据为 TXT

创建输出目录：

```bash
cd /home/wts/getViedo
mkdir -p mocap_data
```

执行：

```bash
XING_Linux/bin/MocapBridge \
  --server 10.1.1.198 \
  --tracker "name:Tracker3" \
  | tee "mocap_data/Tracker3_$(date +%Y%m%d_%H%M%S).txt"
```

该命令同时：

- 在终端显示数据；
- 将数据实时写入 TXT；
- 保存 `DESC`、`READY` 和所有 `POSE` 行。

按 `Ctrl+C` 停止保存。

每一行都会立即刷新到文件，不需要等程序结束才写入。

## 十一、只保存 POSE 行

如果不需要 `DESC` 和 `READY`：

```bash
XING_Linux/bin/MocapBridge \
  --server 10.1.1.198 \
  --tracker "name:Tracker3" \
  | awk -F '\t' '$1 == "POSE" { print; fflush() }' \
  | tee "mocap_data/Tracker3_pose_$(date +%Y%m%d_%H%M%S).txt"
```

TXT 中每一行对应一帧刚体数据。

## 十二、只保存 x,y,z,qx,qy,qz,qw

如果只需要七维位姿，并要求每行顺序固定为：

```text
x,y,z,qx,qy,qz,qw
```

执行：

```bash
XING_Linux/bin/MocapBridge \
  --server 10.1.1.198 \
  --tracker "name:Tracker3" \
  | awk -F '\t' '
      BEGIN {
        OFS=","
        print "x,y,z,qx,qy,qz,qw"
      }
      $1 == "POSE" {
        print $9,$10,$11,$12,$13,$14,$15
        fflush()
      }
    ' \
  | tee "mocap_data/Tracker3_xyz_q_$(date +%Y%m%d_%H%M%S).txt"
```

结果示例：

```text
x,y,z,qx,qy,qz,qw
-1310.90393,-843.626221,179.871338,0.101498149,-0.534910023,0.811196387,-0.213377193
-1310.70215,-843.512634,179.920441,0.101520114,-0.534821033,0.811260104,-0.213335246
```

第一行为字段名，从第二行开始每一行是一帧数据。

如果不需要字段名，删除 `awk` 中的：

```awk
BEGIN {
  OFS=","
  print "x,y,z,qx,qy,qz,qw"
}
```

并把 `OFS=","` 放入数据处理块：

```bash
XING_Linux/bin/MocapBridge \
  --server 10.1.1.198 \
  --tracker "name:Tracker3" \
  | awk -F '\t' '
      $1 == "POSE" {
        OFS=","
        print $9,$10,$11,$12,$13,$14,$15
        fflush()
      }
    ' \
  > "mocap_data/Tracker3_values_$(date +%Y%m%d_%H%M%S).txt"
```

## 十三、同时保存多个刚体

重复使用 `--tracker`：

```bash
XING_Linux/bin/MocapBridge \
  --server 10.1.1.198 \
  --tracker "name:Tracker1" \
  --tracker "name:Tracker3" \
  | tee "mocap_data/multiple_$(date +%Y%m%d_%H%M%S).txt"
```

完整 `POSE` 行中包含目标选择器、实际刚体 ID 和刚体名称，因此能够区分不同刚体。

如果多个刚体只保存七维位姿，应同时保留名称：

```bash
XING_Linux/bin/MocapBridge \
  --server 10.1.1.198 \
  --tracker "name:Tracker1" \
  --tracker "name:Tracker3" \
  | awk -F '\t' '
      BEGIN {
        OFS=","
        print "tracker_name,x,y,z,qx,qy,qz,qw"
      }
      $1 == "POSE" {
        print $4,$9,$10,$11,$12,$13,$14,$15
        fflush()
      }
    ' \
  | tee "mocap_data/multiple_xyz_q_$(date +%Y%m%d_%H%M%S).txt"
```

## 十四、POSE 字段定义

`MocapBridge` 对每个 FrameGroup 先输出一条 5 列 `CLOCK`，它不依赖目标刚体是否可见：

```text
CLOCK  frame  mocap_timestamp_ms  receive_unix_ns  receive_monotonic_ns
```

这条记录专门用于把独立 SDK 时钟动态映射到 Orin 时间域。随后，匹配到目标刚体时再输出
制表符分隔的 17 列 `POSE`：

| 列号 | 字段 | 说明 |
|---:|---|---|
| 1 | `POSE` | 记录类型 |
| 2 | `selector` | 命令行目标选择器，经过百分号编码 |
| 3 | `tracker_id` | SDK 返回的刚体 ID |
| 4 | `tracker_name` | SDK 返回的刚体名称 |
| 5 | `mocap_frame` | XING/XINGYING 动捕帧号 |
| 6 | `mocap_timestamp_ms` | SDK 帧时间戳，毫秒 |
| 7 | `receive_unix_ns` | Orin 收到回调时的 Unix 时间，纳秒 |
| 8 | `receive_monotonic_ns` | Orin 收到回调时的单调时钟，纳秒 |
| 9 | `x` | 刚体 X 坐标，默认单位毫米 |
| 10 | `y` | 刚体 Y 坐标，默认单位毫米 |
| 11 | `z` | 刚体 Z 坐标，默认单位毫米 |
| 12 | `qx` | 四元数 X |
| 13 | `qy` | 四元数 Y |
| 14 | `qz` | 四元数 Z |
| 15 | `qw` | 四元数 W |
| 16 | `mean_error` | 刚体平均解算误差 |
| 17 | `params` | SDK 返回的原始参数 |

名称和选择器可能包含空格或特殊字符，因此第 2、4 列会使用百分号编码。例如：

```text
name%3ATracker%201
```

对应：

```text
name:Tracker 1
```

## 十五、数据有效性注意事项

### 1. `9999999`

如果出现：

```text
x=9999999
y=9999999
z=9999999
```

或者四元数出现相同哨兵值，表示该帧没有有效的刚体解算结果。保存程序仍可以记录原始值，
但后续处理时应过滤这些帧。

对七维数据进行过滤的示例：

```bash
XING_Linux/bin/MocapBridge \
  --server 10.1.1.198 \
  --tracker "name:Tracker3" \
  | awk -F '\t' '
      $1 == "POSE" &&
      $9 != 9999999 &&
      $10 != 9999999 &&
      $11 != 9999999 &&
      $12 != 9999999 &&
      $13 != 9999999 &&
      $14 != 9999999 {
        OFS=","
        print $9,$10,$11,$12,$13,$14,$15
        fflush()
      }
    ' \
  > "mocap_data/Tracker3_valid_$(date +%Y%m%d_%H%M%S).txt"
```

### 2. `params=0`

官方头文件把刚体 `params` 描述为宿主定义的跟踪参数，但没有在该版本头文件中定义各位
的明确语义。NOKOV 官方 ROS2 驱动直接发布 XYZ 和四元数，并没有使用 `params & 1`
过滤刚体。

因此：

- 应原样保存 `params`；
- 不应仅凭 `params=0` 删除所有正常数值；
- 同时检查 XING/XINGYING 中刚体状态；
- 移动目标时确认 XYZ 和四元数连续更新；
- 过滤明确的 `9999999` 哨兵值。

### 3. 时间同步

如果需要对比多台设备的绝对时间，建议让 Orin 和 XING/XINGYING 电脑使用同一个 NTP
时间源。要求更高时使用 PTP 或专用同步设备。

第 7、8 列是 Orin 收到 SDK 回调时记录的本地时间，可用于分析网络接收时间和数据周期。

## 十六、长期后台保存

创建目录：

```bash
cd /home/wts/getViedo
mkdir -p mocap_data
```

启动后台保存：

```bash
DATA_FILE="/home/wts/getViedo/mocap_data/Tracker3_$(date +%Y%m%d_%H%M%S).txt"

nohup /home/wts/getViedo/XING_Linux/bin/MocapBridge \
  --server 10.1.1.198 \
  --tracker "name:Tracker3" \
  > "$DATA_FILE" \
  2> "${DATA_FILE%.txt}.err" &

echo $! > /home/wts/getViedo/mocap_data/MocapBridge.pid
echo "PID=$(cat /home/wts/getViedo/mocap_data/MocapBridge.pid)"
echo "DATA=$DATA_FILE"
```

查看数据：

```bash
tail -f "$DATA_FILE"
```

停止：

```bash
kill -INT "$(cat /home/wts/getViedo/mocap_data/MocapBridge.pid)"
```

确认进程结束：

```bash
ps -p "$(cat /home/wts/getViedo/mocap_data/MocapBridge.pid)"
```

长时间记录会持续增加 TXT 大小，应定期检查磁盘空间：

```bash
df -h /home/wts/getViedo
du -sh /home/wts/getViedo/mocap_data
```

## 十七、常见故障

### `file in wrong format`

原因：使用了 x86-64 或 ARM32 动态库。

检查：

```bash
uname -m
file XING_Linux/lib/libnokov_sdk.so
```

两者必须对应 AArch64。

### `libnokov_sdk.so => not found`

检查：

```bash
ldd XING_Linux/bin/MocapBridge
```

重新执行 CMake 配置和构建，确认 `XING_Linux/lib/libnokov_sdk.so` 存在。

临时诊断可执行：

```bash
LD_LIBRARY_PATH=/home/wts/getViedo/XING_Linux/lib \
  XING_Linux/bin/MocapBridge \
  --server 10.1.1.198 \
  --tracker "name:Tracker3"
```

正常构建不需要长期设置该环境变量。

### `XING SDK initialization failed`

依次检查：

```bash
ip -4 route get 10.1.1.198
ping -I 10.1.1.199 -c 4 10.1.1.198
```

然后确认 XING/XINGYING 的 SDK 广播已经开启。

### `XING server is not present`

网络地址可以到达，但 SDK 服务未被识别。检查：

- XING/XINGYING 是否已启动；
- 是否选择 `10.1.1.198` 对应的网卡；
- SDK 广播是否开启；
- 防火墙是否阻止局域网数据；
- Orin 与服务器是否处于 `10.1.1.0/24`。

### 能看到 `DESC` 和 `READY`，但没有 `POSE`

检查：

- XING/XINGYING 是否正在运行或播放；
- 目标刚体名称、大小写和空格是否正确；
- 使用 `DESC` 输出确认实际名称和 ID；
- 尝试同时用名称和已确认的 ID 测试。

### TXT 为空

如果使用 shell 重定向，确认桥接程序仍在运行：

```bash
pgrep -af MocapBridge
```

查看错误文件：

```bash
find mocap_data -maxdepth 1 -name '*.err' -type f -print
```

前台运行一次通常最容易看到真实错误：

```bash
XING_Linux/bin/MocapBridge \
  --server 10.1.1.198 \
  --tracker "name:Tracker3"
```

## 十八、更新 SDK

更新官方仓库前先查看当前提交：

```bash
git -C third_party/mocap4ros2_nokov rev-parse HEAD
```

更新：

```bash
git -C third_party/mocap4ros2_nokov pull --ff-only
```

每次替换 SDK 前都应：

1. 创建新的带时间戳备份；
2. 同时更新头文件和 ARM64 动态库；
3. 重新检查 `file` 和 `ldd`；
4. 重新编译 `MocapBridge`；
5. 前台验证 `DESC`、`READY` 和 `POSE`；
6. 再开始写入正式 TXT。

## 十九、快速部署命令汇总

以下命令适用于官方仓库已经克隆、网络已经配置好的情况：

```bash
cd /home/wts/getViedo

NOKOV_OFFICIAL_SDK=/home/wts/getViedo/third_party/mocap4ros2_nokov/mocap4r2_nokov_driver/nokov_sdk

file "$NOKOV_OFFICIAL_SDK/lib/aarch64/libnokov_sdk.so"

SDK_BACKUP="XING_Linux/sdk_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SDK_BACKUP"
cp -a XING_Linux/include "$SDK_BACKUP/include"
cp -a XING_Linux/lib "$SDK_BACKUP/lib"

cp -a "$NOKOV_OFFICIAL_SDK/include/." XING_Linux/include/
cp -a "$NOKOV_OFFICIAL_SDK/lib/aarch64/libnokov_sdk.so" \
  XING_Linux/lib/libnokov_sdk.so
chmod 755 XING_Linux/lib/libnokov_sdk.so

file XING_Linux/lib/libnokov_sdk.so
ldd XING_Linux/lib/libnokov_sdk.so

cmake \
  -S XING_Linux \
  -B XING_Linux/build \
  -DCMAKE_BUILD_TYPE=Release

cmake --build XING_Linux/build \
  --target MocapBridge \
  -j4

file XING_Linux/bin/MocapBridge
ldd XING_Linux/bin/MocapBridge

XING_Linux/bin/MocapBridge \
  --server 10.1.1.198 \
  --tracker "name:Tracker3"
```

确认实时数据正常后，再使用本文第十至十三节的命令保存 TXT。

## 二十、单独测量动捕与 Orin 时间戳差值

`measure_clock_offset.py` 只启动 `MocapBridge`，不会启动相机、TensorRT
检测或网页服务。默认读取 `web_monitor/config.json` 中的动捕服务器和刚体：

```bash
cd /home/wts/CVIA_trt

python3 web_monitor/mocap/measure_clock_offset.py \
  --duration 30 \
  --csv web_monitor/runtime/mocap_clock_offset.csv \
  --json web_monitor/runtime/mocap_clock_offset_summary.json
```

也可以显式指定服务器和刚体：

```bash
python3 web_monitor/mocap/measure_clock_offset.py \
  --server 10.1.1.198 \
  --tracker 'name:Tracker0' \
  --duration 30
```

主要指标的符号定义为：

```text
Orin回调时间 - SDK时间
```

正值表示 Orin 在更晚的时刻收到 SDK 回调。程序会自动检查 SDK
`iTimeStamp` 是否像 Unix/PTP 毫秒时间：

- 若两端看起来处于同一绝对时间域，报告绝对表观时差的平均值、P50、P95、
  P99、范围、漂移和抖动；
- 若 SDK 使用设备启动后的相对时间，则不把原始大数值当作绝对钟差，仅报告
  相对首帧的漂移、ppm 和可变传输抖动；
- 若差值接近负 37 秒，会提示检查 SDK 是否使用 TAI，而 Orin
  `CLOCK_REALTIME` 使用 UTC。

桥接程序的 Orin 时间是在 SDK 回调入口采集的，因此统计不会包含 Python
读取标准输出的时间。输出还单独给出“桥接回调到 Python 读到数据的附加延迟”，
用于检查采样程序自身的调度影响。

即使使用 PTP，`Orin回调时间 - SDK时间` 仍然包含动捕曝光、解算、网络传输和
SDK 分发时间，所以它是端到端表观时差，不是纯粹的 PTP 伺服误差。
