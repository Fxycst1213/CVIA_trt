#ifndef TENSORRT_PRO_YOLOV8_MAIN_CAN_H
#define TENSORRT_PRO_YOLOV8_MAIN_CAN_H

#include <cstdint>
#include <string>
#include <linux/can.h>
#include "../params/params.hpp"

class CAN
{
public:
    CAN();
    ~CAN();

    int init(const prj_params &p_params);
    bool sendFloatArray(const float arr[3]);
    bool sendDoubleArray(const double arr[3]);
    void closePort();
    void setDebug(bool enable);
    bool isOpened() const;

private:
    bool sendTextPayload(const char *payload, int length);
    bool sendFrame(uint32_t id, const uint8_t *data, uint8_t length);
    canid_t buildCanId(uint32_t id) const;

    int _fd = -1;
    bool _debug = false;
    uint32_t _base_id = 0x120;
    std::string _interface = "can0";
};

#endif // TENSORRT_PRO_YOLOV8_MAIN_CAN_H
