# AGX Orin 与动捕系统的 PTP 时间同步

> 当前已确认 NOKOV `iTimeStamp` 使用动捕独立时钟，并不直接服从 Windows/PTP/NTP。
> 工程实际使用的动态仿射映射见
> [`README_SDK_AFFINE_TIME_SYNC.md`](README_SDK_AFFINE_TIME_SYNC.md)。本文保留用于说明
> Orin系统钟、网卡PHC和具备PTP能力设备的部署方式。

本文说明如何在当前工程中建立和验证以下时间同步链路：

```text
PTP Grandmaster
    ├── 动捕系统的采集/帧时间
    └── AGX Orin eno1 网卡 PHC（/dev/ptp0）
                 │
                 │ phc2sys
                 ▼
          Orin CLOCK_REALTIME
                 │
                 ├── USB/V4L2 相机采集时间戳
                 └── NOKOV SDK 回调接收时间戳
```

本文的当前设备参数是：

| 项目 | 值 |
|---|---|
| Orin 有线接口 | `eno1` |
| Orin 动捕网段地址 | `10.1.1.199/24` |
| Orin PTP Hardware Clock | `/dev/ptp0` |
| 动捕服务器地址 | `10.1.1.198` |
| 动捕刚体 | `name:Tracker0` |
| Orin 系统 | Ubuntu/Jetson，ARM64 |
| linuxptp 版本 | `3.1.1` |

> 重要：网卡“支持硬件时间戳”只代表具备 PTP 能力，不代表时间已经同步。
> 必须同时满足：网络中存在 Grandmaster、`ptp4l` 已锁定该主钟、`phc2sys`
> 已把 PHC 同步到 `CLOCK_REALTIME`、动捕帧时间也来自同一个 PTP 时间域。

## 1. 当前工程使用哪些时间戳

### 1.1 USB/V4L2 相机

USB 相机本身不支持 PTP。工程直接读取 V4L2 缓冲区时间戳：

- 驱动给出 `CLOCK_REALTIME` 时直接使用；
- 驱动给出 `CLOCK_MONOTONIC` 时，在出队处将其映射到
  `CLOCK_REALTIME`；
- 推理约 30 ms 的耗时不会加到采集时间戳中。

因此，相机进入 PTP 时间域的前提是 Orin 的 `CLOCK_REALTIME` 已由
`phc2sys` 驯服。

### 1.2 NOKOV/XING 动捕

`MocapBridge` 输出两种时间：

```text
mocap_timestamp_ms   = SDK sFrameOfMocapData::iTimeStamp
receive_unix_ns      = SDK 回调进入 Orin 时的 CLOCK_REALTIME
```

`receive_unix_ns` 是 Orin 本机接收时间，不是动捕曝光时间。它包含动捕解算、
网络传输、SDK 缓冲和回调调度延迟。

现有 NOKOV SDK 头文件只把 `iTimeStamp` 描述为 `FrameGroup timestamp`，没有
说明它的 epoch、UTC/TAI 时间尺度以及是否受 PTP 驯服。因此，即使主机 PTP
已经锁定，也必须实测并向厂商确认 `iTimeStamp` 的真实含义。

### 1.3 网卡硬件接收时间戳的边界

`eno1` 支持 RX/TX 硬件时间戳，但 NOKOV SDK 自己持有网络 socket，当前 SDK
接口没有向 `MocapBridge` 暴露 `SCM_TIMESTAMPING` 硬件报文时间戳。因此：

- `ptp4l` 可以利用硬件时间戳高精度同步网卡 PHC；
- 项目可以使用同步后的 Orin `CLOCK_REALTIME`；
- 项目目前不能直接取得 NOKOV 数据包到达网卡时的 RX 硬件时间戳；
- 动捕采集到 SDK 回调之间的固定延迟仍需厂商采集时间戳或运动互相关标定。

## 2. 先确定 PTP 拓扑

推荐使用：

```text
动捕端或独立 PTP 设备 = Grandmaster
AGX Orin                = PTP Client/Slave
```

运行任何命令前，先在动捕软件或 PTP 设备中确认：

1. 已启用 IEEE 1588/PTP Grandmaster；
2. PTP domain，例如 `0`；
3. 网络传输方式：UDP/IPv4 或 Layer 2；
4. delay mechanism：E2E 或 P2P；
5. 使用的 PTP profile；
6. 动捕帧时间戳是否取自该 PTP 时钟；
7. 动捕输出的是 UTC 还是 TAI/PTP timescale。

Orin 和动捕端的 domain、传输方式、delay mechanism 和 profile 必须一致。

如果动捕端不能发送 PTP Announce/Sync，也没有独立 Grandmaster，则
`ptp4l -s` 永远不会进入 `SLAVE` 状态。此时必须增加 Grandmaster，或者确认
动捕系统能否改为从钟后再让 Orin 做主钟。不要只凭网线连接就假定 PTP 存在。

## 3. 检查物理链路和动捕网段

先检查物理链路：

```bash
ip -details link show eno1
ethtool eno1
```

必须看到类似：

```text
eno1: <BROADCAST,MULTICAST,UP,LOWER_UP>
Link detected: yes
```

以下状态表示网线、对端端口、交换机或协商仍有问题，不能继续调 PTP：

```text
NO-CARRIER
state DOWN
Link detected: no
port 1: link down
```

检查 Orin 地址：

```bash
ip -br -4 address show eno1
ip -4 route get 10.1.1.198
```

预期包含：

```text
eno1  UP  ... 10.1.1.199/24
10.1.1.198 dev eno1 src 10.1.1.199
```

如果 `10.1.1.199/24` 丢失，先查看 NetworkManager 连接名称：

```bash
nmcli -g NAME,DEVICE,TYPE connection show
```

假设连接名称是 `orin-eno1-static`，追加地址并重新应用：

```bash
sudo nmcli connection modify orin-eno1-static \
  +ipv4.addresses 10.1.1.199/24 \
  ipv4.never-default yes

sudo nmcli connection up orin-eno1-static
```

`ipv4.never-default yes` 用于防止动捕专网抢占 Wi-Fi/互联网默认路由。不要在
不知道现有连接内容的情况下用 `ipv4.addresses` 覆盖已有地址；本工程原有
`192.168.137.2/24` 时应使用 `+ipv4.addresses` 追加。

临时验证也可以使用：

```bash
sudo ip link set eno1 up
sudo ip address replace 10.1.1.199/24 dev eno1
```

`ip address` 的设置重启后会消失，正式部署应使用 NetworkManager。

检查 SDK 网络：

```bash
ping -I 10.1.1.199 -c 4 10.1.1.198
ip neigh show 10.1.1.198
```

PTP Layer 2 模式本身不依赖 IP，但 NOKOV SDK 仍需要 `10.1.1.198` 可达。

## 4. 检查硬件时间戳能力

执行：

```bash
ethtool -T eno1
ls -l /dev/ptp*
```

当前 Orin 应看到：

```text
hardware-transmit
hardware-receive
hardware-raw-clock
PTP Hardware Clock: 0
/dev/ptp0
```

如果没有 `hardware-transmit`、`hardware-receive` 或 PHC，则不能使用
`ptp4l -H` 硬件模式。

## 5. 安装 linuxptp

```bash
sudo apt update
sudo apt install linuxptp
```

检查：

```bash
ptp4l -v
phc2sys -v
pmc -v
```

## 6. 先以前台方式调通 ptp4l

### 6.1 避免 NTP 与 PTP 同时控制系统时钟

`ptp4l` 只同步 PHC 时暂时不会与 NTP 冲突，但启动 `phc2sys` 前必须避免
`systemd-timesyncd`、Chrony、NTP 和 `phc2sys` 同时修改
`CLOCK_REALTIME`。

查看当前服务：

```bash
systemctl is-active systemd-timesyncd
systemctl is-active chrony
pgrep -af 'ptp4l|phc2sys|chronyd|ntpd|systemd-timesyncd'
```

当前设备原先由 `systemd-timesyncd` 使用公网 NTP 校时。正式切换 PTP 时执行：

```bash
sudo systemctl stop systemd-timesyncd
```

调试失败、需要恢复 NTP 时执行：

```bash
sudo systemctl start systemd-timesyncd
```

### 6.2 默认 UDP/IPv4、E2E、domain 0

如果动捕 Grandmaster 使用 linuxptp 默认兼容参数，可先运行：

```bash
sudo ptp4l -i eno1 -H -s -m
```

参数含义：

| 参数 | 含义 |
|---|---|
| `-i eno1` | 使用动捕有线网卡 |
| `-H` | 使用网卡硬件时间戳 |
| `-s` | Orin 只作为 PTP 从钟，不竞争主钟 |
| `-m` | 在终端打印状态和 offset |

正常的从钟状态转换应类似：

```text
INITIALIZING to LISTENING
new foreign master
LISTENING to UNCALIBRATED
UNCALIBRATED to SLAVE
selected best master clock ...
master offset ...
```

必须最终看到端口处于 `SLAVE`，并且 `master offset` 逐步收敛。

### 6.3 显式配置 domain 和传输方式

可以创建 `/etc/linuxptp/ptp4l-cvia.conf`：

```ini
[global]
time_stamping       hardware
network_transport   UDPv4
delay_mechanism     E2E
domainNumber        0
slaveOnly           1
logging_level       6

[eno1]
```

然后运行：

```bash
sudo ptp4l -f /etc/linuxptp/ptp4l-cvia.conf -i eno1 -m
```

如果动捕端使用 Layer 2，将两端都配置为：

```ini
network_transport   L2
```

如果动捕端使用 P2P，将两端都配置为：

```ini
delay_mechanism     P2P
```

如果 Grandmaster 的 domain 不是 0，修改：

```ini
domainNumber        <动捕端的domain>
```

不要靠反复尝试猜参数，应在动捕端界面或厂商说明中确认。

## 7. 检查网络中是否真的存在 PTP 主钟

UDP/IPv4 PTP 使用 UDP 319/320 端口，可观察：

```bash
sudo tcpdump -ni eno1 'udp port 319 or udp port 320'
```

Layer 2 PTP 使用 EtherType `0x88f7`：

```bash
sudo tcpdump -ni eno1 'ether proto 0x88f7'
```

如果链路正常但完全看不到 Announce/Sync：

- 动捕端没有启动 PTP；
- 监听了错误的网卡；
- 交换机/VLAN 阻止了组播；
- Orin 与动捕端使用不同传输方式；
- 对端并不是 Grandmaster。

如果能看到报文但 `ptp4l` 始终停在 `LISTENING`，重点检查 domain、profile、
E2E/P2P 和 PTP 版本。

## 8. 用 phc2sys 同步 Orin 系统时间

只有 `ptp4l` 已进入 `SLAVE` 后，才启动：

```bash
sudo phc2sys \
  -s eno1 \
  -c CLOCK_REALTIME \
  -w \
  -m
```

含义：

```text
PTP Grandmaster
       ↓ ptp4l
eno1 PHC (/dev/ptp0)
       ↓ phc2sys
CLOCK_REALTIME
```

`-w` 会等待 `ptp4l` 就绪，并从 `ptp4l` 获取 UTC/PTP timescale 相关信息。
不要为了消除约 37 秒差异而手工写一个未经确认的 `-O 37` 或 `-O -37`；
UTC offset 方向错误会制造新的固定误差。

正常输出中 offset 应逐渐收敛并保持稳定。目标阈值取决于 Grandmaster、交换机、
链路负载和 profile；硬件时间戳局域网中通常应明显优于普通 NTP。最终验收以
`ptp4l/phc2sys` 的 offset 统计和业务时间戳实测为准。

## 9. 验证 PTP 状态

### 9.1 进程和端口状态

```bash
pgrep -af 'ptp4l|phc2sys'
sudo pmc -u -b 0 \
  'GET DEFAULT_DATA_SET' \
  'GET PORT_DATA_SET' \
  'GET TIME_STATUS_NP'
```

关注：

```text
portState                 SLAVE
gmPresent                 true
master_offset             接近 0 且稳定
grandmasterIdentity       应为预期的主钟
```

### 9.2 确认系统时间已由 PTP 驯服

```bash
date --iso-8601=ns
timedatectl status
ps -eo pid,comm,args | grep -E '[p]tp4l|[p]hc2sys|[t]imesyncd|[c]hronyd'
```

不要只看 `timedatectl` 的 `System clock synchronized` 文本来判断 PTP；
linuxptp 的核心证据是 `ptp4l` 的 `SLAVE` 状态、`phc2sys` 正在运行及其 offset。

### 9.3 对比动捕主机和 Orin 系统时间

Orin/Linux：

```bash
date +%s%3N
```

如果动捕主机是 Windows，可在 PowerShell 执行：

```powershell
[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
```

人工敲命令有几十到几百毫秒误差，只适合排查“几十秒”级问题。毫秒以下验证应以
PTP 管理数据和自动采样为准。

## 10. 验证 NOKOV SDK 时间戳

PTP 锁定后执行项目内的独立测量脚本：

```bash
cd /home/wts/CVIA_trt

python3 web_monitor/mocap/measure_clock_offset.py \
  --duration 60 \
  --csv web_monitor/runtime/mocap_clock_offset_ptp.csv \
  --json web_monitor/runtime/mocap_clock_offset_ptp_summary.json
```

脚本主要计算：

```text
Orin-SDK = Orin SDK回调 CLOCK_REALTIME - NOKOV SDK iTimeStamp
```

正确解读：

- 固定的几毫秒或几十毫秒可能包含动捕曝光、解算、网络和 SDK 分发延迟；
- 小幅抖动可能来自网络和线程调度；
- 差值持续线性增长说明两个时间戳仍不在同一个受控时钟上；
- 如果仍固定相差约 35～37 秒，应检查 UTC/TAI、epoch，以及 SDK 是否真正使用
  PTP 时钟；
- 如果两台主机系统时间已经同步，但 `iTimeStamp` 仍差几十秒，应向 NOKOV 厂商
  确认 `sFrameOfMocapData::iTimeStamp` 的来源。

现已确认动捕 `mocap_timestamp_ms` 来自独立时钟。工程用 SDK 时间增量与 Orin
`receive_monotonic_ns` 在线拟合仿射时钟模型，生成 `aligned_unix_ns` 消除独立时钟速率漂移；
模型预热或异常时才回退到 `receive_unix_ns`。剩余的曝光、解算、传输固定延迟用
`sync.offset_ms` 或视觉/动捕运动曲线互相关估计。

桥接每个 FrameGroup 都输出 `CLOCK`，即使选择的刚体完全不可见也继续训练时钟模型；
`POSE` 只承担位姿数据，不再决定模型能否预热。

## 11. 当前遇到的日志如何解释

### 11.1 `selected /dev/ptp0 as PTP clock`

```text
selected /dev/ptp0 as PTP clock
```

只说明 `ptp4l` 成功打开了 `eno1` 的 PHC，不代表已经发现主钟或同步成功。

### 11.2 `selected local clock ... as best master`

```text
selected local clock 48b02d.fffe.e7b768 as best master
```

表示最佳主时钟算法没有发现更好的外部时钟，选中了 Orin 自己。使用 `-s` 时，
Orin 又被禁止成为主钟，所以不能建立同步。

### 11.3 `assuming the grand master role` 和 slave-only 告警

```text
assuming the grand master role
master state recommended in slave only mode
defaultDS.priority1 probably misconfigured
```

这不是同步成功。它通常表示：

- 网络中没有可用 Grandmaster；
- 或 Orin 根本没有收到对端 Announce；
- 但 BMCA 又认为本地时钟应该成为主钟；
- 同时 `-s`/`slaveOnly 1` 禁止本机做主钟，配置目标发生冲突。

应检查对端 PTP 服务、domain、传输模式和链路，不应仅通过修改 `priority1` 消除告警。

### 11.4 `port 1: link down`

```text
port 1: link down
LISTENING to FAULTY
```

这是物理链路问题。检查：

```bash
ip -details link show eno1
ethtool eno1
```

必须先恢复 `LOWER_UP` 和 `Link detected: yes`。当前曾实测到：

```text
eno1: <NO-CARRIER,...> state DOWN
Link detected: no
```

同时 `10.1.1.199/24` 地址消失，去 `10.1.1.198` 的路由回退到 Wi-Fi。此状态下
PTP 和 NOKOV SDK 都无法通过 `eno1` 正常工作。

## 12. 常见故障对照表

| 现象 | 含义 | 处理 |
|---|---|---|
| 网卡支持硬件时间戳，但无 `ptp4l` | 只有能力，没有同步 | 安装并启动 linuxptp |
| `LISTENING` 一直不变 | 没收到有效 Grandmaster | 查报文、domain、profile 和对端配置 |
| 本地时钟被选为 best master | 没发现更优外部主钟 | 启用对端 GM，Orin 保持 `-s` |
| `link down` / `FAULTY` | 物理链路断开 | 查网线、端口、交换机和网卡状态 |
| 路由经 Wi-Fi | `eno1` 地址/路由丢失 | 恢复 `10.1.1.199/24`，设置 never-default |
| 只运行 `ptp4l` | 仅 PHC 同步 | 再运行 `phc2sys` 同步系统钟 |
| PTP 与 NTP 同时校系统钟 | 两个伺服器互相干扰 | 系统钟只保留一个控制源 |
| 系统钟同步但 SDK 差 35～37 秒 | SDK 时间源/UTC-TAI 未确认 | 查厂商时间戳定义和 timescale |
| 固定差不大但仍有数毫秒 | 处理/传输/SDK 延迟 | 使用采集时间戳或标定 `offset_ms` |
| 差值持续增长 | 时钟速率未锁定 | 查两端 PTP 状态，延长采样验证 ppm |

## 13. 稳定后配置为 systemd 服务

只有前台调试确认 `SLAVE` 和 offset 正常后，再配置开机启动。

示例 `/etc/systemd/system/cvia-ptp4l.service`：

```ini
[Unit]
Description=CVIA PTP client on eno1
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
ExecStart=/usr/sbin/ptp4l -f /etc/linuxptp/ptp4l-cvia.conf -i eno1 -m
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
```

示例 `/etc/systemd/system/cvia-phc2sys.service`：

```ini
[Unit]
Description=Synchronize CLOCK_REALTIME from eno1 PHC
Requires=cvia-ptp4l.service
After=cvia-ptp4l.service

[Service]
Type=simple
ExecStart=/usr/sbin/phc2sys -s eno1 -c CLOCK_REALTIME -w -m
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
```

加载并启动：

```bash
sudo systemctl daemon-reload
sudo systemctl disable --now systemd-timesyncd
sudo systemctl enable --now cvia-ptp4l.service
sudo systemctl enable --now cvia-phc2sys.service
```

查看日志：

```bash
systemctl status cvia-ptp4l.service cvia-phc2sys.service
journalctl -u cvia-ptp4l.service -u cvia-phc2sys.service -f
```

如果要撤销 PTP 系统时钟控制并恢复 NTP：

```bash
sudo systemctl disable --now cvia-phc2sys.service cvia-ptp4l.service
sudo systemctl enable --now systemd-timesyncd
```

## 14. 推荐验收流程

按以下顺序验收，不要跳步：

1. `eno1` 显示 `LOWER_UP`、`Link detected: yes`；
2. `10.1.1.199/24` 在 `eno1` 上，SDK 路由不经过 Wi-Fi；
3. 抓包能看到与动捕端配置一致的 PTP Announce/Sync；
4. `ptp4l` 从 `LISTENING` 进入 `SLAVE`；
5. `master offset` 收敛且不持续发散；
6. `phc2sys` 正在运行，系统时钟 offset 收敛；
7. 没有 NTP/Chrony 与 `phc2sys` 同时控制系统钟；
8. 动捕主机和 Orin 的 UTC Unix 时间没有几十秒差异；
9. 运行 `measure_clock_offset.py` 采样至少 60 秒；
10. 再采样 5～10 分钟检查差值漂移 ppm；
11. 若 SDK 仍有固定处理延迟，使用 `offset_ms` 或运动曲线互相关标定；
12. 最后再启动相机、检测和网页端实时同步。

完成上述流程后，PTP 负责统一时钟基准，V4L2 采集时间戳负责保留视觉采集时刻，
约 30 ms 推理延迟只影响结果发布时刻，不应改变视觉/动捕所代表的物理时刻。
