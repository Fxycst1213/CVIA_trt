# RS442 串口数据使用说明

本文档用于说明视觉程序通过串口发送位移和姿态数据的启动方式、串口参数和数据帧格式。代码中的主流程变量按 `RS442` 命名，底层串口类文件为 `src/communication/RS422.cpp` / `RS422.h`。

## 启动顺序

1. 在 TCP 接收端先启动 `server.py`，监听 C++ 程序连接。

```bash
python3 server.py
```

当前 C++ 程序 TCP 配置在 `src/main.cpp` 中：

```cpp
p_params.ip = "192.168.1.10";
p_params.port = 1234;
```

`server.py` 所在机器的 IP 和监听端口需要与这里保持一致。TCP 服务必须先启动，否则 C++ 程序启动时会连接失败。

2. 在 Orin 工作区编译并运行 C++ 程序。

```bash
cd /home/desktop/CVIA_trt
make -j 32
./trt
```

## 串口配置

当前串口配置在 `src/main.cpp` 中：

```cpp
p_params.rs442_port = "/dev/ttyUSB0";
p_params.rs442_baudrate = B115200;
```

串口参数：

| 参数 | 值 |
| --- | --- |
| 接口 | RS442/RS422 |
| 设备 | `/dev/ttyUSB0` |
| 波特率 | 115200 |
| 数据位 | 8 |
| 校验位 | 无 |
| 停止位 | 1 |
| 流控 | 无 |
| 字节序 | Big-Endian，高位在前 |

## 发送帧格式

视觉程序发送的数据帧固定为 22 字节：

| 字节偏移 | 字段 | 长度 | 说明 |
| --- | --- | --- | --- |
| 0 | Head 1 | 1 Byte | 固定 `0xEB` |
| 1 | Head 2 | 1 Byte | 固定 `0x90` |
| 2 | Len | 1 Byte | 固定 `0x12`，表示 payload 为 18 字节 |
| 3-20 | Payload | 18 Bytes | 9 个 `int16` 字段 |
| 21 | Checksum | 1 Byte | 从 `Len` 到 `Payload` 所有字节累加后取低 8 位 |

## Payload 字段定义

Payload 共 18 字节，包含 9 个 `int16`，全部使用 Big-Endian：

| Payload 偏移 | 字段 | 类型 | 单位 | 说明 |
| --- | --- | --- | --- | --- |
| 0-1 | 绝对位移 X | int16 | mm | 加油机系 |
| 2-3 | 绝对位移 Y | int16 | mm | 加油机系 |
| 4-5 | 绝对位移 Z | int16 | mm | 加油机系 |
| 6-7 | 相对位移 X | int16 | mm | 加油杆系 |
| 8-9 | 相对位移 Y | int16 | mm | 加油杆系 |
| 10-11 | 相对位移 Z | int16 | mm | 加油杆系 |
| 12-13 | 相对姿态 X | int16 | 0.01 degree | X 轴旋转 |
| 14-15 | 相对姿态 Y | int16 | 0.01 degree | Y 轴旋转 |
| 16-17 | 相对姿态 Z | int16 | 0.01 degree | Z 轴旋转 |

姿态字段发送前会乘以 100。例如 `1.23 degree` 发送为整数 `123`，即十六进制 `00 7B`。

## 示例帧

测试数据：

| 字段 | 值 |
| --- | --- |
| 绝对位移 X/Y/Z | `1000, -1000, 123` mm |
| 相对位移 X/Y/Z | `456, -456, 789` mm |
| 相对姿态 X/Y/Z | `1.23, -1.23, 45.67` degree |

对应串口帧：

```text
EB 90 12 03 E8 FC 18 00 7B 01 C8 FE 38 03 15 00 7B FF 85 11 D7 8A
```

解析：

```text
EB 90    帧头
12       Payload 长度，18 字节

03 E8    1000
FC 18    -1000
00 7B    123

01 C8    456
FE 38    -456
03 15    789

00 7B    1.23 degree
FF 85    -1.23 degree
11 D7    45.67 degree

8A       Checksum
```

## 串口测试模式

代码中保留了一段固定测试数据，默认是注释状态，不影响正常运行。

测试代码位置：

```text
src/prj_detector.cpp
```

函数：

```cpp
void prj_v8detector::rs442_loop()
```

默认真实数据发送逻辑如下：

```cpp
float float_temp_pose[9];
for (int i = 0; i < 9; ++i)
{
    float_temp_pose[i] = data_to_send[i];
}

_rs442.sendFloatArray(float_temp_pose);
```

在这段代码后面保留了固定测试值，默认全部注释：

```cpp
// float_temp_pose[0] = 1000.0f;  // absolute X, mm
// float_temp_pose[1] = -1000.0f; // absolute Y, mm
// float_temp_pose[2] = 123.0f;   // absolute Z, mm
// float_temp_pose[3] = 456.0f;   // relative X, mm
// float_temp_pose[4] = -456.0f;  // relative Y, mm
// float_temp_pose[5] = 789.0f;   // relative Z, mm
// float_temp_pose[6] = 1.23f;    // attitude X, degree
// float_temp_pose[7] = -1.23f;   // attitude Y, degree
// float_temp_pose[8] = 45.67f;   // attitude Z, degree
```

### 启用测试数据

需要测试串口协议时，取消上述 9 行注释，然后重新编译运行：

```bash
make -j2
./trt
```

启用后，串口会持续发送固定测试帧。接收端应收到：

```text
EB 90 12 03 E8 FC 18 00 7B 01 C8 FE 38 03 15 00 7B FF 85 11 D7 8A
```

如果收到的数据与上述帧一致，说明：

```text
帧头正确
长度正确
字段顺序正确
Big-Endian 字节序正确
负数 int16 补码解析正确
姿态 0.01 degree 缩放正确
checksum 正确
```

### 关闭测试数据

测试完成后，需要重新把这 9 行固定赋值注释掉，恢复真实算法数据发送：

```cpp
// float_temp_pose[0] = 1000.0f;
// ...
// float_temp_pose[8] = 45.67f;
```

然后重新编译运行：

```bash
make -j2
./trt
```

## 数据来源

串口发送线程位于 `src/prj_detector.cpp` 的 `rs442_loop()`。

算法结果由 `src/yolov_pose/pose.cpp` 填充到：

```cpp
uart_result[0..8]
```

字段对应关系：

```text
uart_result[0..2]  绝对位移 X/Y/Z
uart_result[3..5]  相对位移 X/Y/Z
uart_result[6..8]  相对姿态 X/Y/Z
```

当 PnP 解算成功时，程序会更新这 9 个字段并通过串口发送。若程序启动后还没有成功解算，初始发送值可能为 0。

## 接收端校验建议

接收端建议按以下流程解析：

1. 在串口流中查找帧头 `EB 90`。
2. 读取 `Len`，当前应为 `0x12`。
3. 继续读取 18 字节 payload。
4. 读取 1 字节 checksum。
5. 计算 `Len + Payload` 所有字节累加和，取低 8 位。
6. 若计算值等于 checksum，则本帧有效。

负数字段使用 int16 补码解析，例如：

```text
FC 18 -> -1000
FE 38 -> -456
FF 85 -> -123 -> -1.23 degree
```
