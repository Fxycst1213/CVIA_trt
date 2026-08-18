# NOKOV SDK 独立时钟到 AGX Orin 的动态仿射时间同步

本文说明当前工程如何把 NOKOV/XING SDK 的独立 FrameGroup 时钟动态映射到
AGX Orin 时间域，并将其用于 USB/V4L2 相机位姿与动捕位姿的实时、离线同步。

这套方案针对以下已确认事实：

- `sFrameOfMocapData::iTimeStamp` 来自动捕设备自己的时钟；
- 它不是 Windows 7 系统时钟，也不会因为 Windows 的 NTP 校时自动与 Orin 同步；
- SDK 时钟与 Orin 时钟存在独立的历元差和速率差；
- 原始 `Orin回调时间 - SDK时间` 会随运行时间持续增长；
- Tracker 刚体可能长时间不可见，真正与相机共同可见的时间可能不到 2 秒；
- USB/V4L2 相机不支持 PTP，但相机帧时间已经映射到 Orin Unix 时间域；
- TensorRT/PnP 推理约 30 ms，配对必须使用相机采集时间而不是推理结束时间。

本文对应的核心文件：

| 文件 | 作用 |
|---|---|
| `MocapBridge.cpp` | 每个 FrameGroup 输出 `CLOCK`，目标有效时再输出 `POSE` |
| `mocap_receiver.py` | 在线估计动态仿射模型、生成 `aligned_unix_ns`、实时匹配 |
| `offline_sync.py` | 离线恢复仿射时间轴、互相关估计 `offset_ms`、插值和作图 |
| `measure_clock_offset.py` | 不启动相机和检测，测量原始时钟差、速率和仿射残差 |
| `config.json` | 仿射模型、实时同步和匹配阈值配置 |

## 一、不要修改 Orin 系统时钟去跟随 SDK

在理想数学模型中：

```text
SDK时钟：  S(t) = αt + b
Orin时钟： O(t) = βt + c

O(t) - S(t) = (β - α)t + (c - b)
```

若令 `β = α`，差值确实只剩固定项。但不应真的把 Orin 系统时钟调成 SDK
速率，因为 Orin 的 `CLOCK_REALTIME` 同时服务于：

- V4L2 相机时间戳；
- 视觉 `capture_timestamp_ns`；
- 文件和 CSV 时间；
- Chrony/NTP；
- Windows 从 Orin 获取的系统时间；
- 系统日志、网络协议和其他应用。

SDK 的速率还会随短窗口量化、运行状态和温度变化。将整个 Orin 调快约
`1000 ppm` 会使系统时间每小时偏离真实时间约 3.6 秒，并与 Chrony 形成两个互相
对抗的时钟控制器。

当前工程保留 Orin 为公共时间基准，只创建一个软件时间变换：

```text
SDK独立时间
    │
    │ 动态仿射映射
    ▼
aligned_unix_ns（Orin时间域）
```

它在数据层面实现“两个时钟同速”，不会修改 Orin 的物理系统时钟。

## 二、完整同步链路

```text
NOKOV FrameGroup 回调
    │
    ├── iFrame
    ├── iTimeStamp                  SDK独立时钟
    ├── receive_monotonic_ns        Orin单调时钟
    └── receive_unix_ns             Orin Unix/REALTIME
              │
              ▼
       动态仿射时钟模型
       T_aligned = anchor + a × ΔT_sdk
              │
              ▼
       aligned_unix_ns
              │
              ├── 加运动互相关标定的 offset_ms
              ▼
       校正后的动捕物理时间轴
              │
              ├── 与相机 capture_timestamp_ns 配对
              ├── XYZ 线性插值
              └── 四元数 SLERP 插值
              ▼
       同一物理时刻的视觉/动捕位姿
```

视觉侧使用：

```text
capture_timestamp_ns = 相机采集时刻，用于同步
publish_timestamp_ns = 推理和 PnP 完成时刻，用于延迟诊断
pipeline_latency_ms  = (publish - capture) / 1e6
```

约 30 ms 推理时间只会使结果晚显示，不会被加到采集时间中。

## 三、CLOCK 与 POSE 协议

### 1. CLOCK

`MocapBridge` 在每个 `sFrameOfMocapData` 回调开始时记录 Orin 时间，并输出：

```text
CLOCK<TAB>frame<TAB>sdk_timestamp_ms<TAB>receive_unix_ns<TAB>receive_monotonic_ns
```

示例：

```text
CLOCK	2722459	1786608848126	1786608848129795000	123456789000
```

五列含义：

| 列 | 含义 |
|---:|---|
| 1 | 固定字符串 `CLOCK` |
| 2 | SDK FrameGroup 帧号 `iFrame` |
| 3 | SDK 独立时钟 `iTimeStamp`，毫秒 |
| 4 | 回调进入 Orin 时的 Unix 时间，纳秒 |
| 5 | 回调进入 Orin 时的单调时间，纳秒 |

`CLOCK` 不依赖刚体是否可见。只要 XING 继续发送 FrameGroup 回调，即使
`nRigidBodies=0` 或所选 Tracker 没有 `POSE`，时钟模型仍能训练。

### 2. POSE

匹配到配置中的刚体后，桥接继续输出 17 列 `POSE`：

```text
POSE selector id name frame sdk_timestamp_ms receive_unix_ns
     receive_monotonic_ns x y z qx qy qz qw mean_error params
```

`POSE` 负责位置和姿态，不再决定时钟模型能否预热。

### 3. 一帧只统计一次时钟样本

新桥接按以下顺序输出：

```text
CLOCK frame=100
POSE  frame=100
```

Python 接收器先消费 `CLOCK`。随后遇到相同 SDK 时间戳的 `POSE` 时不会重复统计。
若运行旧版、只输出 `POSE` 的桥接，接收器仍会从 `POSE` 取时钟样本，保持兼容。

### 4. 输出性能

一个 FrameGroup 中的 `CLOCK` 和全部匹配 `POSE` 在 C++ 中合并为一次管道刷新。
新增数据量只有每帧几十字节，时钟拟合位于 Python 动捕接收线程，不进入相机或
TensorRT 推理线程。

## 四、动态仿射模型

### 1. 输入样本

每个 `CLOCK` 提供：

```text
x_i = sdk_timestamp_ms_i - sdk_timestamp_ms_0
y_i = receive_monotonic_ns_i - receive_monotonic_ns_0
```

使用相对值而不是对约 `1.7×10^12 ms` 的绝对值直接相乘，可以降低浮点精度损失。

### 2. 估计速率

通过最小二乘拟合：

```text
y = intercept + ns_per_sdk_ms × x
```

速率修正量：

```text
rate_ppm
= (ns_per_sdk_ms / 1,000,000 - 1) × 1,000,000
```

例如：

```text
ns_per_sdk_ms = 1,001,000
rate_ppm      = +1000 ppm
```

表示 SDK 每增加 1 ms，Orin 实际经过约 1.001 ms。

### 3. 回调排队基线

`receive_monotonic_ns` 包含 SDK、线程调度和管道排队。排队通常只会让回调变晚，
因此模型在拟合速率后计算残差：

```text
residual_i = y_i - ns_per_sdk_ms × x_i
```

使用残差的第 10 百分位作为低排队基线，而不是直接使用平均值。这样偶发的
`+5 ms`、`+10 ms` 或更大调度延迟不会明显推后整个动捕时间轴。

### 4. MONOTONIC 到 Unix 时间域

速率拟合使用 `CLOCK_MONOTONIC`，避免 UTC 校时跳变直接破坏时间间隔。模型同时估计：

```text
realtime_minus_monotonic_ns
= median(receive_unix_ns - receive_monotonic_ns)
```

默认使用最近最多 120 个样本的中位数。

最终映射：

```text
delta_sdk_ms = sdk_timestamp_ms - anchor_sdk_ms

mapped_monotonic_ns
= anchor_monotonic_ns
  + delta_sdk_ms × ns_per_sdk_ms

aligned_unix_ns
= mapped_monotonic_ns
  + realtime_minus_monotonic_ns
```

### 5. 滑动窗口和更新频率

默认配置：

```json
"clock_fit_window_seconds": 60.0,
"clock_fit_min_seconds": 3.0,
"clock_max_rate_ppm": 5000.0
```

实现行为：

- 只保留最近 60 秒时钟样本；
- 至少需要 60 个样本；
- 样本实际跨度至少 3 秒；
- 每 180 个 `CLOCK` 更新一次模型；
- 90 Hz FrameGroup 时约每 2 秒尝试一次；
- 180 Hz FrameGroup 时约每 1 秒尝试一次；
- 速率绝对值超过 5000 ppm 时拒绝异常模型；
- 同一 SDK 时间戳的重复记录被忽略；
- SDK 时间戳回退时重置模型并重新预热。

### 6. 模型状态

未就绪：

```json
{
  "mode": "receive_fallback",
  "ready": false
}
```

就绪示例：

```json
{
  "mode": "sdk_affine",
  "ready": true,
  "rate_ppm": 900.0,
  "ns_per_sdk_ms": 1000900.0,
  "samples": 5400,
  "span_seconds": 59.99,
  "residual_p95_ms": 1.8
}
```

## 五、三个时间字段必须区分

| 字段 | 含义 | 是否用于最终配对 |
|---|---|---|
| `mocap_timestamp_ms` | SDK 原始独立时钟 | 不直接使用 |
| `receive_unix_ns` | SDK 回调到达 Orin 的时间 | 新鲜度诊断、模型回退 |
| `aligned_unix_ns` | SDK时间经仿射模型映射到Orin的时间 | 是 |

实时匹配使用：

```text
visual_capture_timestamp_ns
对比
mocap_aligned_unix_ns + offset_ms
```

不要直接把：

```text
receive_unix_ns - mocap_timestamp_ms
```

填入 `offset_ms`。它包含独立时钟历元差和累计速率差，不是物理链路固定延迟。

## 六、offset_ms 仍然必需

仿射模型可以消除：

- SDK 与 Orin 的历元差；
- SDK 独立时钟的速率误差；
- 随运行时间线性增长的偏差。

但它不能单独识别：

- 动捕曝光相对 `iTimeStamp` 的定义；
- 动捕解算延迟；
- XING 内部处理延迟；
- 网络和 SDK 分发固定延迟；
- USB 相机曝光和缓冲固定延迟。

这些剩余固定差由：

```text
sync.offset_ms
```

补偿。最可靠的软件标定方法是用视觉/动捕平移速度和角速度模长做归一化互相关。

仿射模型与互相关的职责不同：

```text
仿射模型：校正速率和时间域
offset_ms：校正物理链路固定相位
```

## 七、刚体不可见、共同可见不到 2 秒

这是当前方案重点支持的场景：

```text
阶段 A：相机有位姿，Tracker不可见
阶段 B：相机和Tracker共同有效，但不足2秒
```

推荐过程：

```text
阶段 A
    CLOCK持续输出
    clock_model提前ready
    没有POSE，不做配对

阶段 B
    POSE出现
    立即使用已就绪模型生成aligned_unix_ns
    使用历史可靠offset_ms
    对短位姿段离线匹配和插值
```

不到 2 秒通常不足以重新通过运动互相关估计可靠的新 `offset_ms`。因此应先进行一次
15～30 秒的专门标定，保存高置信 `offset_ms`，后续短任务复用该值。

若没有历史 `offset_ms`，需要延长共同可见时间、累计多个短区间，或者使用视觉和动捕
都能观测到的硬件/机械同步事件。

## 八、离线短位姿段如何保持预热结果

`mocap_session.csv` 保存：

```text
receive_unix_ns
receive_monotonic_ns
mocap_timestamp_ms
aligned_unix_ns
clock_rate_ppm
clock_model_ready
frame
tracker_id
tracker_name
valid
...
```

若 `POSE` 只有不到 2 秒，但出现前 `CLOCK` 已经训练了 60 秒，离线读取不会只用这
2 秒重新猜速率。它会：

1. 找到最后一条 `clock_model_ready=1` 的有效 POSE；
2. 取该行的 `aligned_unix_ns` 作为成熟模型锚点；
3. 取该行的 `clock_rate_ppm` 恢复 `ns_per_sdk_ms`；
4. 用同一仿射模型统一回算短段全部 POSE；
5. 再应用历史 `offset_ms`、最近邻或插值。

若读取旧 CSV，没有仿射字段，则离线程序尝试从完整 POSE 记录重新拟合；样本不足时安全
回退到 `receive_unix_ns`。

## 九、动捕位姿插值

得到校正时间轴后，对每个视觉采集时刻 `T_v` 查找两帧动捕：

```text
T_before <= T_v <= T_after
```

插值系数：

```text
alpha = (T_v - T_before) / (T_after - T_before)
```

位置：

```text
P = P_before + alpha × (P_after - P_before)
```

姿态使用四元数最短弧 SLERP。

即使名义帧率为 30 Hz 与 180 Hz，USB 相机和动捕没有硬件共触发，`alpha` 也不应写死；
每个视觉帧都单独计算。

超过匹配阈值的帧被拒绝，不复用陈旧动捕值。动捕数据空洞超过 250 ms 时，离线运动曲线
不会跨空洞计算速度或插值。

## 十、配置

`web_monitor/config.json`：

```json
{
  "mocap": {
    "enabled": true,
    "server": "10.1.1.198",
    "tracker": "name:Tracker0",
    "stale_ms": 500,
    "retry_seconds": 5,
    "clock_mode": "sdk_affine",
    "clock_fit_window_seconds": 60.0,
    "clock_fit_min_seconds": 3.0,
    "clock_max_rate_ppm": 5000.0
  },
  "sync": {
    "enabled": true,
    "offset_ms": 0.0,
    "max_error_ms": 10.0,
    "history_ms": 5000,
    "interpolate": true,
    "auto_offset_search_ms": 1000
  }
}
```

`tracker` 必须使用 XING 中实际刚体名称或 ID，例如：

```json
"tracker": "name:Tracker0"
```

或：

```json
"tracker": "name:Tracker3"
```

该选择不影响 `CLOCK` 训练，但决定哪些 `POSE` 被接收。

## 十一、构建 MocapBridge

SDK桥接源码变化后执行：

```bash
cd /home/wts/CVIA_trt

cmake \
  -S web_monitor/mocap \
  -B web_monitor/mocap/build \
  -DCMAKE_BUILD_TYPE=Release

cmake --build web_monitor/mocap/build \
  --target MocapBridge \
  -j4
```

生成：

```text
web_monitor/mocap/bin/MocapBridge
```

检查：

```bash
file web_monitor/mocap/bin/MocapBridge
ldd web_monitor/mocap/bin/MocapBridge
```

应为 ARM64/AArch64，并能找到 `libnokov_sdk.so`。

## 十二、每次运行的操作顺序

### 1. 启动 XING

在 Windows 7/XING 端：

1. 启动动捕相机和 XING；
2. 确认 FrameGroup 数据持续产生；
3. 开启 SDK 数据发送；
4. 确认服务器地址为 `10.1.1.198`；
5. Tracker 此时可以暂时不可见。

### 2. 检查 Orin 网络

```bash
ip -brief address show eno1
ping -c 5 10.1.1.198
```

### 3. 避免重复 SDK 连接

```bash
pgrep -af 'MocapBridge|measure_clock_offset.py|web_monitor/server.py'
```

正式网页服务运行时，不要同时运行另一个 `measure_clock_offset.py`。

### 4. 启动完整系统

```bash
cd /home/wts/CVIA_trt
./web_monitor/start.sh
```

只测试网页和动捕接收：

```bash
python3 web_monitor/server.py --host 0.0.0.0 --port 8765
```

### 5. 提前预热

在目标进入共同视野前提前启动：

```text
最低：5秒
推荐：30～60秒
```

90 Hz 下模型通常约 4 秒首次 ready；180 Hz 下通常约 3 秒后 ready。

### 6. 查询模型

```bash
python3 - <<'PY'
import json
import urllib.request

with urllib.request.urlopen(
    "http://127.0.0.1:8765/api/pose-comparison",
    timeout=10,
) as response:
    data = json.load(response)

mocap = data.get("mocap") or {}
sync = data.get("synchronization") or {}
model = mocap.get("clock_model") or sync.get("clock_model") or {}

print("mocap.status:", mocap.get("status"))
print("mocap.message:", mocap.get("message"))
print("mode:", model.get("mode"))
print("ready:", model.get("ready"))
print("rate_ppm:", model.get("rate_ppm"))
print("samples:", model.get("samples"))
print("span_seconds:", model.get("span_seconds"))
print("residual_p95_ms:", model.get("residual_p95_ms"))
print("sync.status:", sync.get("status"))
print("sync_error_ms:", sync.get("sync_error_ms"))
PY
```

Tracker不可见时，`mocap.status` 可以是 `waiting`，但应看到：

```text
mode: sdk_affine
ready: True
samples: 持续增加
```

### 7. 短位姿段

Tracker进入共同视野后，无需重新等待模型；确认出现有效 `POSE` 后正常采集不到 2 秒的
短数据段。短段使用历史可靠 `offset_ms`，不在本次重新做互相关标定。

### 8. 停止和离线报告

在网页点击“停止推理并保存”，然后生成离线报告。报告只配对同时存在有效视觉和动捕的区间。
前段只有视觉的数据会降低全程覆盖率，但不会被强行配到不存在的动捕位姿。

## 十三、独立诊断

更新后的诊断器优先读取 `CLOCK`，不要求目标刚体可见：

```bash
cd /home/wts/CVIA_trt

python3 web_monitor/mocap/measure_clock_offset.py \
  --server 10.1.1.198 \
  --tracker 'name:Tracker0' \
  --duration 60 \
  --print-every 900 \
  --csv web_monitor/runtime/mocap_clock_60s.csv \
  --json web_monitor/runtime/mocap_clock_60s.json
```

新桥接会显示：

```text
时间样本源: 0:FrameGroup CLOCK
```

重点看：

```text
SDK独立时钟 → Orin 仿射速率修正
仿射速率校正后的剩余回调残差
```

不要用原始：

```text
Orin回调时间 - SDK时间
```

判断最终同步是否成功。它仍可能随独立时钟运行持续增长，这是正常现象。

## 十四、验收标准

### 1. 时钟模型

建议：

```text
clock_model.ready = true
clock_model.mode = sdk_affine
abs(rate_ppm) < clock_max_rate_ppm
span_seconds >= 30，正式测试推荐接近60
residual_p95_ms < 5 ms
residual_p99_ms < 10 ms（诊断器输出）
```

### 2. 固定 offset 标定

长期标定数据建议：

```text
共同可见15～30秒
包含非周期平移、旋转、加速、减速和停顿
confidence = medium 或 high
reliable = true
at_search_boundary = false
```

### 3. 最终匹配

```text
abs(sync_error_ms) < 10 ms
离线 p95_abs_error_ms < 10 ms
共同有效区间匹配率尽量 > 95%
```

`sync_error_ms` 表示时间轴匹配残差；物理同步还应检查六轴曲线在运动开始、停止、转向处
是否对齐，以及互相关峰值是否可靠。

## 十五、实机参考结果

一次 Tracker 完全不可见的实机测试得到：

```text
测试时长：约6秒
CLOCK：540
POSE：0
```

证明无刚体位姿时 FrameGroup 时钟仍可训练。

另一次 5 秒 CLOCK 诊断得到：

```text
有效CLOCK样本：450
FrameGroup速率：约90 Hz
仿射比例 a：1.000163979
速率修正：+163.979 ppm
仿射残差P95：1.595 ms
仿射残差P99：1.934 ms
仿射残差最大值：2.599 ms
```

这些数值只是当次参考，不应写死。之前不同 XING 帧率和运行阶段测得过其他速率，动态模型
必须按当前会话重新估计。

## 十六、故障排查

### 1. `ready` 一直为 false

检查：

```bash
ping -c 5 10.1.1.198
pgrep -af MocapBridge
```

确认：

- XING 正在播放或采集；
- SDK发送已开启；
- 使用的是新构建的 `web_monitor/mocap/bin/MocapBridge`；
- FrameGroup `CLOCK` 持续到达；
- 已等待至少 4～5 秒；
- SDK时间戳持续递增；
- `clock_max_rate_ppm` 没有设置得过小。

### 2. `ready=true`，但没有位姿

时钟链路正常，检查：

- `mocap.tracker` 是否为实际名称；
- Tracker反光球是否被遮挡；
- XING是否正常解算该刚体；
- 网页 `mocap.message` 是否列出了其他已发现刚体。

### 3. `residual_p95_ms` 很大

可能原因：

- Windows/XING主机负载高；
- SDK回调线程排队；
- 网络拥塞或链路抖动；
- Orin Python服务被其他CPU任务占用；
- 同时运行多个 SDK 客户端；
- 采样窗口太短；
- XING帧率或时间戳在运行中重置。

建议先运行 60 秒诊断，并停止非必要高负载进程。

### 4. 原始 `Orin-SDK` 继续增长

这是独立时钟的预期行为。应查看：

```text
仿射速率修正 rate_ppm
仿射校正后 residual_p95_ms
最终 sync_error_ms
```

不要尝试通过修改 Orin 系统时钟速率让原始差归零。

### 5. 短段能配对，但物理曲线错位

说明速率映射可能正常，但 `offset_ms` 不适用于当前参数。检查相机曝光、分辨率、帧率、
XING帧率和解算配置是否相对长期标定发生变化。若变化，需要重新做较长共同可见标定或使用
硬件同步事件。

### 6. 离线总覆盖率很低

如果前面很长时间只有视觉，当前报告的全程覆盖率会被拉低。重点看有效匹配行的
`p95_abs_error_ms`、互相关置信度和共同有效区间，而不是仅看全程覆盖率。

## 十七、方案边界

该方案提供软件时间域映射，不会改变 NOKOV 设备内部时钟。它能够处理稳定或缓慢变化的
时钟速率差，但不能替代真正的共同曝光硬件触发。

若要求每次短于 2 秒的共同可见区间都能在没有历史 `offset_ms` 的情况下严格保证亚毫秒或
小于 10 ms 的物理同步，应考虑：

- NOKOV外部Trigger；
- 相机与动捕共同触发；
- PPS/GPIO；
- 两套系统都能记录的LED闪光或光电事件；
- 厂商提供的曝光时刻/PTP/PPS接口。

当前推荐实践是：

```text
CLOCK动态校正每次会话的时钟速率
+
一次长期标定获得固定offset_ms
+
短POSE区间只做映射、匹配和插值
```

这能在不修改 Orin 系统时钟、不影响 TensorRT 主检测效率的前提下，适配刚体长时间不可见、
最终共同有效不到 2 秒的实际场景。
