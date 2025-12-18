#include "prj_detector.hpp"

prj_v8detector::prj_v8detector(string onnxPath, logger::Level level, model::Params params, prj_params p_params, tcp_params t_params)
{
    _worker = thread::create_worker(onnxPath, level, params);
    _timer = make_shared<timer::Timer>(logger::Level::INFO);
    _timer->init();

    _zed = ZEDX::GetInstance();
    _zed->init(p_params.cameraID, p_params.resolution);

    preprocess::init_process(p_params.H, p_params.W);

    _writeframe = new ZEDframe;
    _writeframe->rgb_ptr = new cv::Mat(p_params.H, p_params.W, CV_8UC3);
    _func_camera = std::bind(&prj_v8detector::camera, this);
    _client.init(p_params, t_params);
}

// void prj_v8detector::camera()
// {
//     while (1)
//     {
//         _timer->init();
//         _timer->start_cpu();
//         // _zed->grab_frame(_writeframe);
//         *(_writeframe->rgb_ptr) = cv::imread(("data/source/00590.png"));
//         _timer->stop_cpu<timer::Timer::ms>("ZED Grab frame");
//         _timer->start_cpu();
//         _worker->inference(*(_writeframe->rgb_ptr));
//         _timer->stop_cpu<timer::Timer::ms>("inference");
//         _timer->show();
// _client.pack_and_send(
//             *(_writeframe->rgb_ptr),
//             _worker->m_pose->m_bboxes,
//             _worker->m_pose->m_result,
//             _writeframe->timestamp);
//         // _resultframe_queue.push(
//         //     Resultframe{
//         //         _writeframe->rgb_ptr,
//         //         _worker->m_pose->m_bboxes,
//         //         _worker->m_pose->m_result});
//     }
// }

void prj_v8detector::camera()
{
    // ==========================================
    // 1. (新增) 在循环外初始化文件列表
    // ==========================================
    std::vector<cv::String> filenames;                    // 用于存放所有文件名
    cv::String folder = "/home/cvia/yifei/images3/*.png"; // 指定路径和格式 (也可以是 *.jpg)

    cv::glob(folder, filenames, false);

    if (filenames.empty())
    {
        std::cerr << "[Error] No images found in " << folder << std::endl;
        return;
    }

    std::sort(filenames.begin(), filenames.end());

    std::cout << "[Info] Found " << filenames.size() << " images." << std::endl;

    int current_idx = 0; // 当前读到第几张

    while (1)
    {
        _timer->init();
        _timer->start_cpu();

        cv::Mat img = cv::imread(filenames[current_idx]);

        // 容错：如果图片读坏了（空图），跳过或者报错
        if (img.empty())
        {
            std::cerr << "[Warning] Failed to read " << filenames[current_idx] << std::endl;
        }
        else
        {
            // 只有读成功了才赋值
            *(_writeframe->rgb_ptr) = img;
            auto now = std::chrono::system_clock::now();
            _writeframe->timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        }

        // 更新下标：指向下一张
        current_idx++;

        // 如果读完了最后一张，重置为 0，实现循环播放
        if (current_idx >= filenames.size())
        {
            break;
        }

        _timer->stop_cpu<timer::Timer::ms>("ZED Grab frame");
        _timer->start_cpu();

        // 下面保持不变...
        _worker->inference(*(_writeframe->rgb_ptr));
        _timer->stop_cpu<timer::Timer::ms>("inference");
        _timer->show();

        _client.pack_and_send(
            *(_writeframe->rgb_ptr),
            _worker->m_pose->m_bboxes,
            _worker->m_pose->m_result,
            _writeframe->timestamp);
    }
}

void prj_v8detector::run()
{
    if (!_client.send_handshake())
    {
        LOGE("Failed to handshake with server. Exiting...");
        exit(EXIT_FAILURE);
    }
    auto t1 = std::thread(_func_camera);
    t1.join();
}

prj_v8detector::~prj_v8detector()
{
    delete _writeframe->rgb_ptr;
    delete _writeframe;
    preprocess::destroy_process();
}
