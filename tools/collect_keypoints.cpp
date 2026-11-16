#include "pose.hpp"
#include "preprocess.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

// Collect unrefined neural-network predictions without a camera, TCP, or serial.
// Manifest rows are sequence-id<TAB>image-path. The CSV is reusable by CPU tests.
int main(int argc, char **argv)
{
    if (argc < 3 || argc > 4)
    {
        std::cerr << "Usage: keypoint_collector manifest.tsv raw.csv [model.onnx]\n";
        return 2;
    }
    try
    {
        cv::setNumThreads(4);
        std::ifstream manifest(argv[1]);
        std::ofstream output(argv[2]);
        if (!manifest || !output)
            throw std::runtime_error("Cannot open manifest/output");
        model::Params params;
        params.img = {640, 640, 3};
        params.task = model::POSE;
        params.dev = model::GPU;
        params.prec = model::FP16;
        auto pose = model::pose::make_pose(argc == 4 ? argv[3] : "models/onnx/x_plane1.onnx",
                                           logger::Level::WARN, params);
        pose->init_model();
        logger::Logger quiet(logger::Level::WARN);
        output << "sequence,image,timestamp,x0,y0,x1,y1,confidence";
        for (int i = 0; i < model::pose::NUM_KEYPOINTS; ++i)
            output << ",raw_x" << i + 1 << ",raw_y" << i + 1 << ",raw_conf" << i + 1;
        output << '\n' << std::setprecision(9);
        std::string line;
        cv::Size allocated;
        int count = 0;
        while (std::getline(manifest, line))
        {
            const auto tab = line.find('\t');
            if (tab == std::string::npos)
                throw std::runtime_error("Invalid manifest row");
            const std::string sequence = line.substr(0, tab), path = line.substr(tab + 1);
            Resultframe frame{};
            frame.rgb = cv::imread(path);
            if (frame.rgb.empty())
                throw std::runtime_error("Cannot read " + path);
            if (allocated != frame.rgb.size())
            {
                preprocess::destroy_process();
                preprocess::init_process(frame.rgb.rows, frame.rgb.cols);
                allocated = frame.rgb.size();
            }
            const auto slash = path.find_last_of("/\\");
            const auto dot = path.find_last_of('.');
            frame.timestamp = std::stoull(path.substr(slash + 1, dot - slash - 1)) * 1000 / 30;
            pose->inference(frame);
            output << sequence << ',' << path << ',' << frame.timestamp;
            if (pose->m_bboxes.empty())
                output << ",0,0,0,0,0";
            else
            {
                const auto &box = pose->m_bboxes[0];
                output << ',' << box.x0 << ',' << box.y0 << ',' << box.x1 << ',' << box.y1 << ',' << box.confidence;
            }
            const auto &points = pose->raw_keypoints();
            for (int i = 0; i < model::pose::NUM_KEYPOINTS; ++i)
            {
                if (i < static_cast<int>(points.size()))
                    output << ',' << points[i].x << ',' << points[i].y << ',' << points[i].conf;
                else
                    output << ",0,0,0";
            }
            output << '\n';
            if (++count % 100 == 0)
                std::cout << "Collected " << count << " frames\n" << std::flush;
        }
        pose.reset();
        preprocess::destroy_process();
        std::cout << "Collected " << count << " frames in total\n";
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
