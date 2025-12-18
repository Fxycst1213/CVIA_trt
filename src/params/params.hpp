#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <string> // 1. 必须包含这个

struct prj_params
{
    // 2. 给 int 类型赋默认值，防止随机数
    int H = 0;
    int W = 0;
    std::string resolution = "HD1080"; // 3. 去掉 using namespace std，改为 std::string
    int cameraID = 0;
    std::string ip; // 默认为空字符串
    int port = 0;   // 赋默认值
    int socket_mode = 0;
};

struct tcp_params
{
    // 4. 将常量改为 static constexpr
    // 这样它们只占用一份内存，且可以用作数组长度声明
    static constexpr int Point_num = 7;
    static constexpr int KeyPoint_box = (Point_num * 3 + 5);
    static constexpr int KEYPOINTS_BUFSIZE = KeyPoint_box * 4;
    static constexpr int POSE_BUFSIZE = 7 * 8;
    static constexpr int POSE_DATE_NUM = 7;

    // 这两个是变量，不加 const
    int IMG_SIZE = 1920 * 1080 * 3;
    int DATA_SIZE = 0;
    int SOCKETSEND_SIZE = 0;
};

#endif // PARAMS_HPP