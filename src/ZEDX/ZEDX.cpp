#include "ZEDX.h"
#include <string>
#include <iostream>

using namespace sl;

/*
    @brief: zed init setting and open the camera.
    @param:
    1.int ID : which camera will be used
    2.std::string resolution : picture resolution
    3. int frame: fps
    4. bool enable_fill_mode: enable depth fill mode or not
*/
void ZEDX::init(int ID, std::string resolution, int frame, bool enable_fill_mode, const std::string &eye)
{
    if (eye != "left" && eye != "right")
        throw std::invalid_argument("Camera eye must be left or right");
    _eye = eye;
    if (resolution == "HD1080")
        _init_parameters.camera_resolution = RESOLUTION::HD1080;
    else
        _init_parameters.camera_resolution = RESOLUTION::HD720;
    _init_parameters.camera_fps = frame;
    _init_parameters.depth_mode = sl::DEPTH_MODE::NONE;
    _init_parameters.input.setFromCameraID(ID);
    // Load project-local factory calibration; the SDK also checks its default settings directory.
    _init_parameters.optional_settings_path = sl::String("models/");
    _runtime_parameters.enable_fill_mode = true;
    _timer = make_shared<timer::Timer>(logger::Level::VERB);
    auto returned_state = _zedx.open(_init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS)
    {
        std::cout << "Error:" << returned_state << "." << std::endl;
        LOGE("ZED INIT ERROR");
    }

    setIntrinsic();
}

/*
    @brief: lazy mode to get the only instance of camera

*/

ZEDX *ZEDX::GetInstance()
{
    static ZEDX zedx;
    return &zedx;
}

cv::Mat ZEDX::getK() const
{
    return _K;
}

/*
    @brief: get the image data. meanwhile turn rgb data into cv::Mat format
*/
void ZEDX::grab_frame(ZEDframe *frame)
{
    sl::Mat img;
    // _timer->init();
    // _timer->start_cpu();
    if (_zedx.grab(_runtime_parameters) == ERROR_CODE::SUCCESS)
    {
        _zedx.retrieveImage(img, _eye == "right" ? sl::VIEW::RIGHT_UNRECTIFIED : sl::VIEW::LEFT_UNRECTIFIED);
        // frame->timestamp = img.timestamp.getMicroseconds();// 微秒
        frame->timestamp = img.timestamp.getMilliseconds();// 毫秒
        cv::Mat tmp = ZEDX::slMat2cvMat(img);
        cv::cvtColor(tmp, *(frame->rgb_ptr), cv::COLOR_BGRA2BGR);
        // _timer->show();
    }
    else
    {
        LOGE("ZED Grabframe ERROR");
    }
}

cv::Mat ZEDX::slMat2cvMat(sl::Mat &input)
{
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), input.getPtr<sl::uchar1>(MEM::CPU), input.getStepBytes(sl::MEM::CPU));
}

void ZEDX::close()
{
    if (_zedx.isOpened())
        _zedx.close();
}

ZEDX::~ZEDX()
{
    close();
}

/*
    @brief: set the intrinsic matrix
*/
void ZEDX::setIntrinsic()
{
    const auto info = _zedx.getCameraInformation();
    const auto &raw = info.camera_configuration.calibration_parameters_raw;
    const auto &camera = _eye == "right" ? raw.right_cam : raw.left_cam;
    _calibration.width = info.camera_configuration.resolution.width;
    _calibration.height = info.camera_configuration.resolution.height;
    _calibration.right_eye = _eye == "right";
    _calibration.intrinsics = {{camera.fx, camera.fy, camera.cx, camera.cy}};
    for (int i = 0; i < 12; ++i)
        _calibration.distortion[i] = camera.disto[i];
    _K = (cv::Mat_<double>(3, 3) << camera.fx, 0.0, camera.cx,
          0.0, camera.fy, camera.cy, 0.0, 0.0, 1.0);
    std::cerr << "[camera] eye=" << _eye << " raw " << _calibration.width << 'x' << _calibration.height
              << " fx=" << camera.fx << " fy=" << camera.fy << " cx=" << camera.cx
              << " cy=" << camera.cy << '\n';
}
