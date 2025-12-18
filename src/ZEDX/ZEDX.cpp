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
void ZEDX::init(int ID, std::string resolution, int frame, bool enable_fill_mode)
{
    if (resolution == "HD1080")
        _init_parameters.camera_resolution = RESOLUTION::HD1080;
    else
        _init_parameters.camera_resolution = RESOLUTION::HD720;
    _init_parameters.camera_fps = frame;
    _init_parameters.depth_mode = sl::DEPTH_MODE::NONE;
    _init_parameters.input.setFromCameraID(ID);
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
        _zedx.retrieveImage(img, VIEW::LEFT);
        frame->timestamp = img.timestamp.getMilliseconds();
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

ZEDX::~ZEDX()
{
    _zedx.close();
}

/*
    @brief: set the intrinsic matrix
*/
void ZEDX::setIntrinsic()
{
    CalibrationParameters calibration_params = _zedx.getCameraInformation().camera_configuration.calibration_parameters;
    // Focal length of the left eye in pixels
    double fx = (double)calibration_params.left_cam.fx;
    double fy = (double)calibration_params.left_cam.fy;
    double cx = (double)calibration_params.left_cam.cx;
    double cy = (double)calibration_params.left_cam.cy;
    _K = (cv::Mat_<double>(3, 3) << fx, 0.0, cx,
          0.0, fy, cy,
          0.0, 0.0, 1.0);
}