# CVIA TensorRT Project

This project runs pose inference with TensorRT/ZED and sends result data through TCP plus an optional RS485 or CAN channel.

## Build

```bash
cmake --build build -j2
```

Executable:

```bash
./build/trt
```

## Communication Mode

Only `src/main.cpp` needs to be changed when switching RS485 and CAN.

Use CAN:

```cpp
p_params.communication_mode = CommunicationMode::CAN;
p_params.can_interface = "can0";
p_params.can_base_id = 0x120;
```

Use RS485:

```cpp
p_params.communication_mode = CommunicationMode::RS485;
p_params.rs485_port = "/dev/ttyUSB0";
p_params.rs485_baudrate = B57600;
```

Disable RS485/CAN result sending:

```cpp
p_params.communication_mode = CommunicationMode::NONE;
```

## Notes

- CAN sends the same text payload format as RS485: `x.xxx,y.yyy,z.zzz`.
- Classic CAN payload is limited to 8 bytes, so long text payloads are split across consecutive CAN IDs starting from `can_base_id`.
- `communication_send_interval_us` controls the send interval for both RS485 and CAN.
