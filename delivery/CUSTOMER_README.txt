CVIA 视觉推理系统（客户运行版）
================================

一、适用设备

本软件包面向 NVIDIA Jetson AGX Orin / AArch64，并要求客户设备安装与 BUILD_INFO.txt
兼容的 JetPack、CUDA、TensorRT、OpenCV、ZED SDK 和 Python 3.10。TensorRT engine 与
GPU 架构及 TensorRT 版本相关，不保证能跨 JetPack/TensorRT 版本运行。

二、启动

1. 解压整个目录，不要只复制单个可执行文件。
2. 在目录中执行：

   chmod +x cvia_customer
   ./cvia_customer

3. 同一局域网电脑访问：

   http://<Jetson-IP>:8765

4. 在网页中修改配置，点击“保存并应用”，然后点击顶部“开始推理”。
5. 采集结束后使用网页的“停止推理并保存”，不要直接断电。

如端口 8765 被占用，可执行：

   ./cvia_customer --port 9000

如只允许本机访问，可执行：

   ./cvia_customer --host 127.0.0.1

三、配置与数据

- 可编辑配置：config/config.json（建议直接通过网页修改）。
- 文件夹输入：把图片放入 data/source，网页选择“文件夹”模式。
- 临时运行数据：web_monitor/runtime。
- ONNX 模型路径可在网页“输入”页填写，保存后在下一次点击“开始推理”时生效。
- 软件会把 models/onnx/example.onnx 对应到 models/engine/example-fp16.engine：已有
  engine 时直接加载；只有 ONNX 时现场构建 engine，并要求 engine 目录可写。
- 交付包默认只包含 qdy0815-fp16.engine，不包含 ONNX。改用其他模型时，请自行放入兼容的
  ONNX 或同名 FP16 engine。
- 三维模型点支持在网页新增、删除和修改 XYZ，允许 4～256 点。点数必须与模型输出一致：
  单类别模型输出特征数应为 5 + 3×点数，例如 10 点为 35、16 点为 53。
- 只修改 XYZ 可在推理运行中热应用；新增或删除点后必须停止并重新开始推理，不需要重新编译。
- 推理停止后可在网页“位姿”页生成 1200×990 RGB PNG。没有动捕数据时也能生成并预览
  相机检测图 web_monitor/runtime/synchronized_camera_report.png；至少成功配对一帧动捕时，
  还会生成 NOKOV 动捕图 synchronized_mocap_report.png 和相机/动捕合成图
  synchronized_report.png。三张图均可在网页分别下载。“自动估计 offset”仍要求动捕数据。
- 离线报告会识别相机位姿中的异常跳变，并使用异常段前后两个有效帧按时间进行插值：位置
  使用线性插值，旋转使用四元数球面插值。输出 CSV 的 visual_interpolated 列为 1 时，
  表示该帧是后处理修复帧；若异常段位于数据首尾、没有前后有效帧，则该段不会被强行补值。
- 只有约 10～20 张测试图片时也可以生成连续相机曲线；报告会在相邻有效图像位姿之间
  按采集时间做分段线性插值，不会因图片时间间隔较大而断线。动捕失配区仍保持断线。
- 绘图阶段会把约 60 Hz 相机位姿重采样到统一 300 Hz 时间轴：位置使用线性插值，旋转使用
  四元数球面插值，再逐点匹配 300 Hz 动捕用于相机图、动捕图和合成图。原始检测 CSV 仍保留
  真实相机帧，不会写入伪造检测帧；300 Hz 中间点是后处理估计值，不是新的相机测量。
- 配置和运行目录必须对启动软件的用户可写。

四、完整性校验

在软件包根目录执行：

   sha256sum -c SHA256SUMS

全部显示 OK 才表示文件未损坏或被替换。

五、停止

推荐在网页点击停止。关闭整个服务可在启动终端按 Ctrl+C；服务会先停止推理并刷新会话文件。

六、安全说明

软件包不包含 CVIA C/C++/CUDA/Python 源文件、Python 字节码、Git 历史、测试、CMake
工程和 ONNX 模型。核心程序和 Python 后端均已编译为 ARM64 原生代码并剥离符号。
任何部署到客户设备的二进制与浏览器静态资源
仍存在被分析的可能，应结合合同中的保密、授权设备、复制限制和违约条款保护知识产权。
