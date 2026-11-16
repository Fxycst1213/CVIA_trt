import serial
import random
import time

# ---------------- 配置，根据实际修改 ----------------
port = "COM6"        # Windows：COMx；Linux：/dev/ttyUSB0
baudrate = 9600
timeout = 0.1

send_interval = 0.1  # 发送周期 0.1s，即 10Hz
point_duration = 2.0 # 每组点持续发送 2s
# ---------------------------------------------------

ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)


def build_frame(abs_x: int, abs_y: int, abs_z: int):
    buf = bytearray(22)
    buf[0] = 0xEB
    buf[1] = 0x90
    buf[2] = 18

    # 大端打包 int16
    buf[3:5] = abs_x.to_bytes(2, byteorder='big', signed=True)
    buf[5:7] = abs_y.to_bytes(2, byteorder='big', signed=True)
    buf[7:9] = abs_z.to_bytes(2, byteorder='big', signed=True)

    # 剩余12字节置0
    buf[9:21] = b'\x00' * 12

    # 校验：buf[2] ~ buf[20]累加
    chk = 0
    for b in buf[2:21]:
        chk += b

    buf[21] = chk & 0xFF

    return buf


if __name__ == "__main__":

    # 四组基准点
    points = [
        (1200,  100, -1500),
        (1200, -100, -1500),
        (1200,  100, -1400),
        (1100,  100, -1400),
        (1100, 50, -1400),
        (1100, -50, -1400),
    ]

    try:
        while True:

            # 四组点依次循环
            for i, (base_x, base_y, base_z) in enumerate(points):

                # 每进入一组点时产生一次随机扰动
                x = base_x + random.uniform(-20, 20)
                y = base_y + random.uniform(-20, 25)
                z = base_z + random.uniform(-25, 20)

                x = int(x)
                y = int(y)
                z = int(z)

                print(
                    f"\n切换到第 {i + 1} 组："
                    f"base=({base_x}, {base_y}, {base_z}), "
                    f"send=({x}, {y}, {z})"
                )

                # 2秒 / 0.1秒 = 20帧
                send_count = int(point_duration / send_interval)

                for _ in range(send_count):
                    frame = build_frame(x, y, z)

                    ser.write(frame)

                    print(
                        f"x={x:5d}, y={y:5d}, z={z:5d} | "
                        f"{frame.hex()}"
                    )

                    time.sleep(send_interval)

    except KeyboardInterrupt:
        print("\n停止发送")

    finally:
        ser.close()