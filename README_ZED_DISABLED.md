# ZED 临时禁用说明

当前没有连接 ZED 相机，所以临时注释掉了 ZED 的启动初始化代码。这里只是临时处理，后续接回相机后可以恢复。

## 本次修改内容

1. 在 `src/prj_detector.cpp` 的 `prj_v8detector::prj_v8detector(...)` 构造函数中，注释掉了 ZED 初始化代码：

```cpp
// _zed = ZEDX::GetInstance();
// _zed->init(p_params.cameraID, p_params.resolution);
```

2. 没有修改其他 ZED 相机采集逻辑，后续仍然可以恢复使用。

3. 重新执行了编译，生成了新的 `./trt` 可执行文件。

4. 短时间运行 `./trt` 验证后，没有再出现 `ZED INIT ERROR`。

## 恢复 ZED 的方式

后续接回 ZED 相机后，把下面两行前面的 `//` 去掉即可：

```cpp
_zed = ZEDX::GetInstance();
_zed->init(p_params.cameraID, p_params.resolution);
```

然后重新编译：

```bash
make -j2
```

## TensorRT engine 警告说明

运行时出现的提示：

```text
[warn]Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
```

原因是 TensorRT 的 `.engine` 文件是和生成它时的 GPU 型号、TensorRT 版本、CUDA 环境强相关的。当前程序加载的 engine plan 文件可能不是在当前这台设备或当前这套环境上生成的，所以 TensorRT 提醒可能影响性能，严重时也可能导致运行错误。

这条警告和本次注释 ZED 初始化没有直接关系。它来自 TensorRT 加载 engine 文件的过程。

如果要消除这个警告，通常需要在当前设备上重新由 `.onnx` 生成对应的 `.engine` 文件。
