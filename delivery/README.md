# 客户运行包构建

此目录只保存交付模板和原生启动器源码；客户包由白名单脚本生成，不直接复制工作区。

## 构建

构建机需要与客户设备兼容的 AGX Orin / AArch64、CUDA、TensorRT、OpenCV、ZED SDK、
Python 3.10 和 Cython 3。首次准备本地构建环境：

```bash
python3 -m venv .venv-package
.venv-package/bin/python -m pip install 'Cython>=3,<4'
```

生成一个不可覆盖的版本化交付包：

```bash
./tools/build_customer_package.sh 1.0.0
```

输出位于：

```text
release/CVIA_Runtime_1.0.0_aarch64/
release/CVIA_Runtime_1.0.0_aarch64.tar.gz
release/CVIA_Runtime_1.0.0_aarch64.tar.gz.sha256
```

版本已存在时脚本会退出，不会覆盖已有交付物。可用 `JOBS=4` 调整构建并发，或通过
`CYTHON_PYTHON=/path/to/python` 指向另一个安装了 Cython 3 的构建解释器。

## 强制门禁

脚本只复制默认的 `qdy0815-fp16.engine` 和实际运行文件。客户可在网页填写其他 ONNX
路径；交付包仍不会自动包含任何 ONNX，客户需自行提供兼容模型或对应 engine。脚本强制检查：

- C/C++/CUDA/Python 源码、Python 字节码、ONNX、CMake 和 Makefile 不得进入交付目录；
- 核心程序、客户启动器、NOKOV 桥接和三个 Python 后端模块必须是 ARM64 原生文件并剥离符号；
- 交付物不得包含开发机 `/home/wts` 绝对路径；
- 核心及 NOKOV 桥接的直接动态库在构建机上不得缺失；
- 原生后端必须能够被 Python 3.10 导入；
- 核心程序的 `--check-assets config/config.json` 必须能按网页同一模型路径规则定位并读取
  TensorRT engine 或 ONNX；默认交付配置必须命中随包携带的 engine；
- 每个文件写入内部 `SHA256SUMS`，压缩包另写外部 SHA-256 文件。

## 保密边界

核心推理、PnP 和后端 Python 逻辑以剥离符号的原生 ARM64 代码交付，显著提高静态分析门槛，
但客户端二进制无法保证绝对不可逆向。网页的 HTML/CSS/JavaScript 必须发送给浏览器，客户
始终可以查看，因此不得在前端放置模型、密钥、授权算法或其他秘密。交付时还应配合合同保密、
授权设备绑定、复制限制和升级服务条款。

在包含 `libnokov_sdk.so` 前，交付方必须确认 NOKOV/XING SDK 合同允许向目标客户再分发。
