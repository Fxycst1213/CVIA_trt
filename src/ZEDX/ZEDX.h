//
// Created by wwwindows on 2024/3/9.
//

#include <sl/Camera.hpp>
#include <string>
#include <opencv2/opencv.hpp>
#include "logger.hpp"
#include "time.hpp"
#ifndef ZEDX_H
#define ZEDX_H
using namespace std;
using namespace sl;

typedef struct ZEDframe ZEDframe;

struct ZEDframe
{
    cv::Mat *rgb_ptr = nullptr;
    uint64_t timestamp = 0;
    uint64_t end_timestamp = 0;
};

class ZEDX
{
public:
    void init(int ID, std::string resolution, int frame = 60, bool enable_fill_mode = false);
    void grab_frame(ZEDframe *frame);
    static ZEDX *GetInstance();
    cv::Mat getK() const;
    cv::Mat slMat2cvMat(sl::Mat &input);

private:
    ZEDX() {};
    ~ZEDX();
    void setIntrinsic();

    int getOCVtype(sl::MAT_TYPE type)
    {
        int cv_type = -1;
        switch (type)
        {
        case MAT_TYPE::F32_C1:
            cv_type = CV_32FC1;
            break;
        case MAT_TYPE::F32_C2:
            cv_type = CV_32FC2;
            break;
        case MAT_TYPE::F32_C3:
            cv_type = CV_32FC3;
            break;
        case MAT_TYPE::F32_C4:
            cv_type = CV_32FC4;
            break;
        case MAT_TYPE::U8_C1:
            cv_type = CV_8UC1;
            break;
        case MAT_TYPE::U8_C2:
            cv_type = CV_8UC2;
            break;
        case MAT_TYPE::U8_C3:
            cv_type = CV_8UC3;
            break;
        case MAT_TYPE::U8_C4:
            cv_type = CV_8UC4;
            break;
        default:
            break;
        }
        return cv_type;
    }

    Camera _zedx;
    InitParameters _init_parameters;
    RuntimeParameters _runtime_parameters;
    cv::Mat _K;
    std::shared_ptr<timer::Timer> _timer;
};

#endif // ZEDX_H
