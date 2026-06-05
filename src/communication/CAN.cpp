#include "CAN.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <net/if.h>
#include "../logger/logger.hpp"

CAN::CAN()
{
}

CAN::~CAN()
{
    closePort();
}

int CAN::init(const prj_params &p_params)
{
    _interface = p_params.can_interface;
    _base_id = static_cast<uint32_t>(p_params.can_base_id);

    _fd = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (_fd < 0)
    {
        LOGE("创建CAN socket失败: %s", _interface.c_str());
        return -1;
    }

    struct ifreq ifr;
    std::memset(&ifr, 0, sizeof(ifr));
    std::strncpy(ifr.ifr_name, _interface.c_str(), IFNAMSIZ - 1);

    if (ioctl(_fd, SIOCGIFINDEX, &ifr) < 0)
    {
        LOGE("获取CAN接口索引失败: %s", _interface.c_str());
        closePort();
        return -1;
    }

    struct sockaddr_can addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;

    if (bind(_fd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0)
    {
        LOGE("绑定CAN接口失败: %s", _interface.c_str());
        closePort();
        return -1;
    }

    LOG("CAN初始化成功: %s base_id: 0x%X", _interface.c_str(), _base_id);
    return 0;
}

bool CAN::sendFloatArray(const float arr[3])
{
    if (_fd < 0)
    {
        LOGE("CAN未初始化");
        return false;
    }

    char buffer[128];
    std::memset(buffer, 0, sizeof(buffer));

    const int len = std::snprintf(buffer, sizeof(buffer), "%.3f,%.3f,%.3f", arr[0], arr[1], arr[2]);
    if (len <= 0 || len >= static_cast<int>(sizeof(buffer)))
    {
        LOGE("CAN文本数据格式化失败");
        return false;
    }

    return sendTextPayload(buffer, len);
}

bool CAN::sendDoubleArray(const double arr[3])
{
    if (_fd < 0)
    {
        LOGE("CAN未初始化");
        return false;
    }

    char buffer[256];
    std::memset(buffer, 0, sizeof(buffer));

    const int len = std::snprintf(buffer, sizeof(buffer), "%.3lf,%.3lf,%.3lf\n", arr[0], arr[1], arr[2]);
    if (len <= 0 || len >= static_cast<int>(sizeof(buffer)))
    {
        LOGE("CAN双精度文本数据格式化失败");
        return false;
    }

    return sendTextPayload(buffer, len);
}

bool CAN::sendTextPayload(const char *payload, int length)
{
    if (payload == nullptr || length <= 0)
    {
        LOGE("CAN发送数据为空");
        return false;
    }

    bool ok = true;
    int offset = 0;
    uint32_t chunk = 0;
    while (offset < length)
    {
        const int chunk_len = std::min(8, length - offset);
        ok = sendFrame(_base_id + chunk,
                       reinterpret_cast<const uint8_t *>(payload + offset),
                       static_cast<uint8_t>(chunk_len)) &&
             ok;
        offset += chunk_len;
        ++chunk;
    }

    if (_debug)
    {
        LOG("CAN发送文本数据%d字节: %s", length, payload);
    }
    return ok;
}

bool CAN::sendFrame(uint32_t id, const uint8_t *data, uint8_t length)
{
    if (id > CAN_EFF_MASK)
    {
        LOGE("CAN ID超过扩展帧最大值: 0x%X", id);
        return false;
    }

    if (length > 8)
    {
        LOGE("CAN单帧数据超过8字节: %d", length);
        return false;
    }

    struct can_frame frame;
    std::memset(&frame, 0, sizeof(frame));
    frame.can_id = buildCanId(id);
    frame.can_dlc = length;
    std::memcpy(frame.data, data, length);

    const ssize_t bytes_written = write(_fd, &frame, sizeof(frame));
    if (bytes_written != static_cast<ssize_t>(sizeof(frame)))
    {
        LOGE("CAN发送失败: id=0x%X dlc=%d", id, length);
        return false;
    }

    if (_debug)
    {
        LOG("CAN发送: id=0x%X dlc=%d", id, length);
    }
    return true;
}

canid_t CAN::buildCanId(uint32_t id) const
{
    canid_t can_id = static_cast<canid_t>(id);
    if (id > CAN_SFF_MASK)
    {
        can_id |= CAN_EFF_FLAG;
    }
    return can_id;
}

void CAN::closePort()
{
    if (_fd >= 0)
    {
        close(_fd);
        _fd = -1;
        LOG("CAN接口已关闭: %s", _interface.c_str());
    }
}

void CAN::setDebug(bool enable)
{
    _debug = enable;
}

bool CAN::isOpened() const
{
    return _fd >= 0;
}
