#include <iostream>
#include <chrono>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include "IRcamera.h"

namespace
{
constexpr unsigned int kBufferCount = 4;

uint64_t timespec_ns(const timespec &value)
{
    return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL
         + static_cast<uint64_t>(value.tv_nsec);
}

uint64_t timeval_ns(const timeval &value)
{
    return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL
         + static_cast<uint64_t>(value.tv_usec) * 1000ULL;
}

uint64_t clock_ns(clockid_t clock_id)
{
    timespec value{};
    return clock_gettime(clock_id, &value) == 0 ? timespec_ns(value) : 0;
}

bool xioctl(int fd, unsigned long request, void *argument)
{
    int result;
    do
    {
        result = ioctl(fd, request, argument);
    }
    while (result == -1 && errno == EINTR);
    return result != -1;
}
}

void IRCamera::init(const camera_params &params)
{
    _camera_name = params.name;
    if (init_v4l2(params)) return;

    LOGW("%s: direct V4L2 capture unavailable; falling back to OpenCV receive-time timestamps",
         params.name.c_str());
    init_opencv_fallback(params);
}

bool IRCamera::init_opencv_fallback(const camera_params &params)
{
    _cap.open(params.cameraID, cv::CAP_V4L2);
    if (!_cap.isOpened())
    {
        LOGE("%s INIT ERROR: Cannot open camera ID %d", params.name.c_str(), params.cameraID);
        return false;
    }

    // 2. 极其重要：设置 MJPG 编码格式以支持高帧率和高分辨率
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    _cap.set(cv::CAP_PROP_FOURCC, fourcc);
    // 只保留驱动侧最新帧，避免推理速度低于采集速度时读取历史帧。
    _cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    // 3. 解析并设置分辨率
    if (params.resolution == "HD1080")
    {
        _width = 1920;
        _height = 1080;
    }
    else
    {
        // 默认作为 HD720 处理
        _width = 1280;
        _height = 720;
    }

    // v4l2-ctl -d /dev/video0 --list-ctrls  查看相机参数
    _cap.set(cv::CAP_PROP_FRAME_WIDTH, _width);
    _cap.set(cv::CAP_PROP_FRAME_HEIGHT, _height);
    _cap.set(cv::CAP_PROP_FPS, params.cameraframe);

    // 4. 只有需要调节的相机才写入成像参数；可见光相机可保持设备默认值。
    if (params.apply_image_controls)
    {
        _cap.set(cv::CAP_PROP_AUTO_EXPOSURE, params.auto_exposure_mode);
        if (params.apply_exposure)
            _cap.set(cv::CAP_PROP_EXPOSURE, params.exposure);

        _cap.set(cv::CAP_PROP_AUTO_WB, params.auto_white_balance ? 1.0 : 0.0);
        if (params.apply_white_balance_temperature)
            _cap.set(cv::CAP_PROP_WB_TEMPERATURE, params.white_balance_temperature);

        _cap.set(cv::CAP_PROP_BRIGHTNESS, params.brightness);
        _cap.set(cv::CAP_PROP_CONTRAST, params.contrast);
        _cap.set(cv::CAP_PROP_SHARPNESS, params.sharpness);
    }

    if (params.apply_image_controls)
    {
        std::cout << "--- " << params.name << " 成像参数配置 ---" << std::endl;
        std::cout << "曝光模式: " << _cap.get(cv::CAP_PROP_AUTO_EXPOSURE)
                  << " (期望: " << params.auto_exposure_mode << ")" << std::endl;
        if (params.apply_exposure)
            std::cout << "曝光值:   " << _cap.get(cv::CAP_PROP_EXPOSURE) << " (期望: " << params.exposure << ")" << std::endl;
        std::cout << "亮度:     " << _cap.get(cv::CAP_PROP_BRIGHTNESS) << " (期望: " << params.brightness << ")" << std::endl;
        std::cout << "对比度:   " << _cap.get(cv::CAP_PROP_CONTRAST) << " (期望: " << params.contrast << ")" << std::endl;
        std::cout << "清晰度:   " << _cap.get(cv::CAP_PROP_SHARPNESS) << " (期望: " << params.sharpness << ")" << std::endl;
        std::cout << "自动白平衡: " << _cap.get(cv::CAP_PROP_AUTO_WB) << " (期望: " << (params.auto_white_balance ? 1 : 0) << ")" << std::endl;
        if (params.apply_white_balance_temperature)
            std::cout << "色温:     " << _cap.get(cv::CAP_PROP_WB_TEMPERATURE) << " (期望: " << params.white_balance_temperature << ")" << std::endl;
    }
    else
    {
        std::cout << "--- " << params.name << " 使用设备默认成像参数（未写入曝光/白平衡/画质控制）---" << std::endl;
    }

    const int active_fourcc = static_cast<int>(_cap.get(cv::CAP_PROP_FOURCC));
    char active_format[5] = {
        static_cast<char>(active_fourcc & 0xff),
        static_cast<char>((active_fourcc >> 8) & 0xff),
        static_cast<char>((active_fourcc >> 16) & 0xff),
        static_cast<char>((active_fourcc >> 24) & 0xff),
        '\0'};
    const double active_width = _cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const double active_height = _cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    const double active_fps = _cap.get(cv::CAP_PROP_FPS);

    std::cout << "--- " << params.name << " 初始化成功 ---" << std::endl;
    std::cout << "请求模式: " << _width << "x" << _height << " @ "
              << params.cameraframe << " FPS, MJPG" << std::endl;
    std::cout << "生效模式: " << active_width << "x" << active_height << " @ "
              << active_fps << " FPS, " << active_format << std::endl;
    if (active_fourcc != fourcc || active_width != _width || active_height != _height ||
        (active_fps > 0.0 && active_fps + 0.5 < params.cameraframe))
    {
        LOGW("%s: camera driver did not accept requested %dx%d@%d MJPG mode",
             params.name.c_str(), _width, _height, params.cameraframe);
    }
    return true;
}

bool IRCamera::grab_frame(IRFrame *frame)
{
    if (frame == nullptr || frame->rgb_ptr == nullptr)
        return false;

    if (_v4l2_streaming) return grab_v4l2(frame);
    return grab_opencv_fallback(frame);
}

bool IRCamera::grab_opencv_fallback(IRFrame *frame)
{
    if (!_cap.isOpened()) return false;
    cv::Mat captured;
    if (_cap.read(captured) && !captured.empty())
    {
        const uint64_t now_ns = clock_ns(CLOCK_REALTIME);
        frame->capture_timestamp_ns = now_ns;
        frame->dequeue_timestamp_ns = now_ns;
        frame->timestamp = now_ns / 1000000ULL;
        frame->driver_timestamp = false;
        frame->timestamp_start_of_exposure = false;
        if (!_logged_timestamp_fallback)
        {
            LOGW("%s: timestamps are cap.read() return time, not camera exposure time",
                 _camera_name.c_str());
            _logged_timestamp_fallback = true;
        }
        // captured 独占其像素内存，移动 Mat 头即可，避免一整帧 copyTo。
        *(frame->rgb_ptr) = std::move(captured);
        return true;
    }

    ++_failed_reads;
    if (_failed_reads == 1 || _failed_reads % 60 == 0)
    {
        LOGW("IR Camera read failed (%llu times)",
             static_cast<unsigned long long>(_failed_reads));
    }
    return false;
}

bool IRCamera::init_v4l2(const camera_params &params)
{
    if (params.resolution == "HD1080")
    {
        _width = 1920;
        _height = 1080;
    }
    else
    {
        _width = 1280;
        _height = 720;
    }

    const std::string device = "/dev/video" + std::to_string(params.cameraID);
    _video_fd = open(device.c_str(), O_RDWR | O_CLOEXEC);
    if (_video_fd < 0)
    {
        LOGW("%s: cannot open %s for direct V4L2 capture: %s",
             params.name.c_str(), device.c_str(), std::strerror(errno));
        return false;
    }

    const auto fail = [this, &params](const char *operation) {
        LOGW("%s: direct V4L2 %s failed: %s",
             params.name.c_str(), operation, std::strerror(errno));
        close_v4l2();
        return false;
    };

    v4l2_capability capability{};
    if (!xioctl(_video_fd, VIDIOC_QUERYCAP, &capability)) return fail("VIDIOC_QUERYCAP");
    const uint32_t capabilities = capability.capabilities & V4L2_CAP_DEVICE_CAPS
        ? capability.device_caps : capability.capabilities;
    if (!(capabilities & V4L2_CAP_VIDEO_CAPTURE) || !(capabilities & V4L2_CAP_STREAMING))
    {
        errno = ENOTSUP;
        return fail("capture/streaming capability check");
    }

    v4l2_format format{};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = static_cast<uint32_t>(_width);
    format.fmt.pix.height = static_cast<uint32_t>(_height);
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    format.fmt.pix.field = V4L2_FIELD_ANY;
    if (!xioctl(_video_fd, VIDIOC_S_FMT, &format)) return fail("VIDIOC_S_FMT");
    if (format.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG)
    {
        LOGW("%s: camera did not accept MJPEG for direct capture", params.name.c_str());
        close_v4l2();
        return false;
    }
    _width = static_cast<int>(format.fmt.pix.width);
    _height = static_cast<int>(format.fmt.pix.height);

    v4l2_streamparm stream_parameters{};
    stream_parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    stream_parameters.parm.capture.timeperframe.numerator = 1;
    stream_parameters.parm.capture.timeperframe.denominator =
        static_cast<uint32_t>(std::max(1, params.cameraframe));
    if (!xioctl(_video_fd, VIDIOC_S_PARM, &stream_parameters))
        LOGW("%s: VIDIOC_S_PARM failed: %s", params.name.c_str(), std::strerror(errno));

    const auto set_control = [this, &params](uint32_t id, int value, const char *name) {
        v4l2_control control{};
        control.id = id;
        control.value = value;
        if (!xioctl(_video_fd, VIDIOC_S_CTRL, &control) && errno != EINVAL)
            LOGW("%s: cannot set %s=%d: %s",
                 params.name.c_str(), name, value, std::strerror(errno));
    };
    if (params.apply_image_controls)
    {
        set_control(V4L2_CID_EXPOSURE_AUTO,
                    static_cast<int>(params.auto_exposure_mode), "auto exposure");
        if (params.apply_exposure)
            set_control(V4L2_CID_EXPOSURE_ABSOLUTE,
                        static_cast<int>(params.exposure), "exposure");
        set_control(V4L2_CID_AUTO_WHITE_BALANCE,
                    params.auto_white_balance ? 1 : 0, "auto white balance");
        if (params.apply_white_balance_temperature)
            set_control(V4L2_CID_WHITE_BALANCE_TEMPERATURE,
                        static_cast<int>(params.white_balance_temperature), "white balance");
        set_control(V4L2_CID_BRIGHTNESS, static_cast<int>(params.brightness), "brightness");
        set_control(V4L2_CID_CONTRAST, static_cast<int>(params.contrast), "contrast");
        set_control(V4L2_CID_SHARPNESS, static_cast<int>(params.sharpness), "sharpness");
    }

    v4l2_requestbuffers request{};
    request.count = kBufferCount;
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_MMAP;
    if (!xioctl(_video_fd, VIDIOC_REQBUFS, &request)) return fail("VIDIOC_REQBUFS");
    if (request.count < 2)
    {
        errno = ENOMEM;
        return fail("buffer allocation");
    }

    _mapped_buffers.resize(request.count);
    for (uint32_t index = 0; index < request.count; ++index)
    {
        v4l2_buffer buffer{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = index;
        if (!xioctl(_video_fd, VIDIOC_QUERYBUF, &buffer)) return fail("VIDIOC_QUERYBUF");
        void *address = mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE,
                             MAP_SHARED, _video_fd, buffer.m.offset);
        if (address == MAP_FAILED) return fail("mmap");
        _mapped_buffers[index] = {address, buffer.length};
        if (!xioctl(_video_fd, VIDIOC_QBUF, &buffer)) return fail("VIDIOC_QBUF");
    }

    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (!xioctl(_video_fd, VIDIOC_STREAMON, &type)) return fail("VIDIOC_STREAMON");
    _v4l2_streaming = true;
    LOG("%s: direct V4L2 MJPEG %dx%d@%d enabled; using driver capture timestamps",
        params.name.c_str(), _width, _height, params.cameraframe);
    return true;
}

bool IRCamera::grab_v4l2(IRFrame *frame)
{
    v4l2_buffer buffer{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    if (!xioctl(_video_fd, VIDIOC_DQBUF, &buffer))
    {
        ++_failed_reads;
        if (_failed_reads == 1 || _failed_reads % 60 == 0)
            LOGW("%s: VIDIOC_DQBUF failed (%llu): %s", _camera_name.c_str(),
                 static_cast<unsigned long long>(_failed_reads), std::strerror(errno));
        return false;
    }

    const uint64_t realtime_before_ns = clock_ns(CLOCK_REALTIME);
    const uint64_t monotonic_ns = clock_ns(CLOCK_MONOTONIC);
    const uint64_t realtime_after_ns = clock_ns(CLOCK_REALTIME);
    const uint64_t dequeue_ns = realtime_before_ns <= realtime_after_ns
        ? realtime_before_ns + (realtime_after_ns - realtime_before_ns) / 2
        : realtime_after_ns + (realtime_before_ns - realtime_after_ns) / 2;
    const uint64_t raw_timestamp_ns = timeval_ns(buffer.timestamp);

    camera_time::Clock timestamp_clock = camera_time::Clock::Unknown;
    if (buffer.flags & V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC)
    {
        timestamp_clock = camera_time::Clock::Monotonic;
    }
    else if ((buffer.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) == V4L2_BUF_FLAG_TIMESTAMP_UNKNOWN)
    {
        // 老驱动未声明 clock id。帧时间应非常接近 dequeue 时刻，选择更接近的时钟域。
        const uint64_t realtime_distance = raw_timestamp_ns > dequeue_ns
            ? raw_timestamp_ns - dequeue_ns : dequeue_ns - raw_timestamp_ns;
        const uint64_t monotonic_distance = raw_timestamp_ns > monotonic_ns
            ? raw_timestamp_ns - monotonic_ns : monotonic_ns - raw_timestamp_ns;
        timestamp_clock = monotonic_distance < realtime_distance
            ? camera_time::Clock::Monotonic : camera_time::Clock::Realtime;
    }
    else
    {
        timestamp_clock = camera_time::Clock::Realtime;
    }

    uint64_t capture_ns = camera_time::to_realtime_ns(
        raw_timestamp_ns, timestamp_clock,
        realtime_before_ns, monotonic_ns, realtime_after_ns);
    bool driver_timestamp = capture_ns != 0;
    const uint64_t maximum_age_ns = 10000000000ULL;
    const uint64_t capture_age_ns = capture_ns > dequeue_ns
        ? capture_ns - dequeue_ns : dequeue_ns - capture_ns;
    if (!driver_timestamp || capture_age_ns > maximum_age_ns
        || (_last_capture_timestamp_ns != 0 && capture_ns <= _last_capture_timestamp_ns))
    {
        capture_ns = dequeue_ns;
        driver_timestamp = false;
        if (!_logged_timestamp_fallback)
        {
            LOGW("%s: invalid/non-monotonic driver frame timestamp; using dequeue CLOCK_REALTIME",
                 _camera_name.c_str());
            _logged_timestamp_fallback = true;
        }
    }

    if (buffer.index >= _mapped_buffers.size()
        || buffer.bytesused == 0
        || buffer.bytesused > _mapped_buffers[buffer.index].length)
    {
        xioctl(_video_fd, VIDIOC_QBUF, &buffer);
        ++_failed_reads;
        return false;
    }
    cv::Mat encoded(1, static_cast<int>(buffer.bytesused), CV_8UC1,
                    _mapped_buffers[buffer.index].address);
    cv::Mat decoded = cv::imdecode(encoded, cv::IMREAD_COLOR);
    const bool requeued = xioctl(_video_fd, VIDIOC_QBUF, &buffer);
    if (!requeued)
        LOGE("%s: VIDIOC_QBUF after decode failed: %s", _camera_name.c_str(), std::strerror(errno));
    if (decoded.empty() || !requeued)
    {
        ++_failed_reads;
        return false;
    }

    _last_capture_timestamp_ns = capture_ns;
    frame->capture_timestamp_ns = capture_ns;
    frame->dequeue_timestamp_ns = dequeue_ns;
    frame->timestamp = capture_ns / 1000000ULL;
    frame->driver_timestamp = driver_timestamp;
    frame->timestamp_start_of_exposure =
        (buffer.flags & V4L2_BUF_FLAG_TSTAMP_SRC_MASK) == V4L2_BUF_FLAG_TSTAMP_SRC_SOE;
    if (driver_timestamp && !_logged_timestamp_mode)
    {
        LOG("%s: V4L2 timestamp clock=%s point=%s",
            _camera_name.c_str(),
            timestamp_clock == camera_time::Clock::Monotonic ? "MONOTONIC" : "REALTIME/inferred",
            frame->timestamp_start_of_exposure ? "SOE" : "EOF/unknown");
        _logged_timestamp_mode = true;
    }
    *(frame->rgb_ptr) = std::move(decoded);
    return true;
}

void IRCamera::close_v4l2()
{
    if (_video_fd >= 0 && _v4l2_streaming)
    {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl(_video_fd, VIDIOC_STREAMOFF, &type);
    }
    _v4l2_streaming = false;
    for (MappedBuffer &buffer : _mapped_buffers)
    {
        if (buffer.address && buffer.address != MAP_FAILED)
            munmap(buffer.address, buffer.length);
    }
    _mapped_buffers.clear();
    if (_video_fd >= 0) close(_video_fd);
    _video_fd = -1;
}

IRCamera::~IRCamera()
{
    close_v4l2();
    if (_cap.isOpened())
    {
        _cap.release();
    }
}
