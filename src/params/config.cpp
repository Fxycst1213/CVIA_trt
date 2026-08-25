#include "config.hpp"
#include <opencv2/core.hpp>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace
{
template <typename T>
void read_if_present(const cv::FileNode &node, const char *key, T &value)
{
    const cv::FileNode item = node[key];
    if (!item.empty())
        item >> value;
}

void read_camera(const cv::FileNode &node, camera_params &camera)
{
    if (node.empty()) return;
    read_if_present(node, "name", camera.name);
    read_if_present(node, "camera_id", camera.cameraID);
    read_if_present(node, "fps", camera.cameraframe);
    read_if_present(node, "resolution", camera.resolution);
    read_if_present(node, "apply_image_controls", camera.apply_image_controls);
    read_if_present(node, "auto_exposure_mode", camera.auto_exposure_mode);
    read_if_present(node, "apply_exposure", camera.apply_exposure);
    read_if_present(node, "exposure", camera.exposure);
    read_if_present(node, "auto_white_balance", camera.auto_white_balance);
    read_if_present(node, "apply_white_balance_temperature", camera.apply_white_balance_temperature);
    read_if_present(node, "white_balance_temperature", camera.white_balance_temperature);
    read_if_present(node, "brightness", camera.brightness);
    read_if_present(node, "contrast", camera.contrast);
    read_if_present(node, "sharpness", camera.sharpness);
}

template <size_t N>
void read_array(const cv::FileNode &node, std::array<double, N> &target)
{
    if (node.empty() || node.type() != cv::FileNode::SEQ || node.size() != N) return;
    size_t index = 0;
    for (cv::FileNodeIterator it = node.begin(); it != node.end(); ++it)
        target[index++] = static_cast<double>(*it);
}

bool read_model_keypoints(const cv::FileNode &node,
                          model::pose::ModelKeypoints3D &target,
                          std::string &error)
{
    if (node.empty()) return true; // 兼容尚未写入网页点集的旧配置。
    if (node.type() != cv::FileNode::SEQ || node.size() < 4 || node.size() > 256)
    {
        error = "calibration.model_keypoints_3d must contain 4..256 XYZ points";
        return false;
    }

    model::pose::ModelKeypoints3D candidate;
    candidate.reserve(node.size() * 3);
    for (cv::FileNodeIterator point_it = node.begin(); point_it != node.end(); ++point_it)
    {
        const cv::FileNode point = *point_it;
        if (point.type() != cv::FileNode::SEQ || point.size() != 3)
        {
            error = "each calibration.model_keypoints_3d item must contain X, Y and Z";
            return false;
        }
        for (cv::FileNodeIterator coordinate_it = point.begin();
             coordinate_it != point.end(); ++coordinate_it)
        {
            const double value = static_cast<double>(*coordinate_it);
            if (!std::isfinite(value))
            {
                error = "calibration.model_keypoints_3d contains a non-finite coordinate";
                return false;
            }
            candidate.push_back(value);
        }
    }
    target.swap(candidate);
    return true;
}
}

bool load_project_config(const std::string &path, prj_params &params, std::string &error)
{
    try
    {
        cv::FileStorage file(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
        if (!file.isOpened())
        {
            error = "cannot open config: " + path;
            return false;
        }

        const cv::FileNode input = file["input"];
        read_if_present(input, "mode", params.input_mode);
        read_if_present(input, "folder_path", params.folder_path);
        read_if_present(input, "loop", params.folder_loop);
        read_if_present(input, "interval_ms", params.folder_interval_ms);

        const cv::FileNode model = file["model"];
        read_if_present(model, "onnx_path", params.onnx_model_path);

        read_camera(file["detect_camera"], params.detect_camera);
        read_camera(file["photo_camera"], params.photo_camera);

        const cv::FileNode calibration = file["calibration"];
        read_array(calibration["camera_matrix"], params.camera_matrix);
        read_array(calibration["distortion"], params.distortion);
        read_array(calibration["extrinsic"], params.extrinsic);
        if (!read_model_keypoints(calibration["model_keypoints_3d"],
                                  params.model_keypoints_3d, error))
            return false;

        const cv::FileNode network = file["network"];
        read_if_present(network, "tcp_ip", params.ip);
        read_if_present(network, "tcp_port", params.port);
        read_if_present(network, "tcp_enabled", params.enable_tcp);
        read_if_present(network, "socket_mode", params.socket_mode);
        read_if_present(network, "udp_enabled", params.enable_udp);
        read_if_present(network, "udp_ip", params.udp_ip);
        read_if_present(network, "udp_port", params.udp_port);

        const cv::FileNode runtime = file["runtime"];
        read_if_present(runtime, "frame_width", params.W);
        read_if_present(runtime, "frame_height", params.H);
        read_if_present(runtime, "preview_dir", params.preview_dir);
        read_if_present(runtime, "preview_fps", params.preview_fps);
        read_if_present(runtime, "save_pnp_results", params.save_pnp_results);
        read_if_present(runtime, "pnp_result_dir", params.pnp_result_dir);
        params.folder_interval_ms = std::max(1, params.folder_interval_ms);
        params.preview_fps = std::max(1, std::min(30, params.preview_fps));
        return true;
    }
    catch (const cv::Exception &exception)
    {
        error = exception.what();
        return false;
    }
}
