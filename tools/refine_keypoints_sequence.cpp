#include "keypoint_refiner.hpp"
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: keypoint_refinement_eval raw.csv refined.csv\n";
        return 2;
    }
    try
    {
        cv::setNumThreads(4);
        std::ifstream input(argv[1]);
        std::ofstream output(argv[2]);
        if (!input || !output)
            throw std::runtime_error("Cannot open input/output");
        std::string line, sequence;
        std::getline(input, line);
        output << line;
        for (int i = 0; i < model::pose::NUM_KEYPOINTS; ++i)
            output << ",ref_x" << i + 1 << ",ref_y" << i + 1 << ",ref_conf" << i + 1;
        output << ",color_matches,flow_matches,rejected_jumps,refine_ms\n" << std::setprecision(9);
        model::pose::KeypointRefiner refiner;
        int frames = 0;
        while (std::getline(input, line))
        {
            std::vector<std::string> fields;
            std::istringstream row(line);
            std::string field;
            while (std::getline(row, field, ','))
                fields.push_back(field);
            if (fields.size() != 8 + 3 * model::pose::NUM_KEYPOINTS)
                throw std::runtime_error("Invalid CSV column count");
            if (sequence != fields[0])
            {
                refiner.reset();
                sequence = fields[0];
            }
            const cv::Mat image = cv::imread(fields[1]);
            if (image.empty())
                throw std::runtime_error("Cannot read " + fields[1]);
            const float x0 = std::stof(fields[3]), y0 = std::stof(fields[4]);
            const cv::Rect2f box(x0, y0, std::stof(fields[5]) - x0, std::stof(fields[6]) - y0);
            std::vector<model::pose::keypoint> points;
            for (int i = 0; i < model::pose::NUM_KEYPOINTS; ++i)
                points.emplace_back(std::stof(fields[8 + 3*i]), std::stof(fields[9 + 3*i]), std::stof(fields[10 + 3*i]));
            const auto start = std::chrono::steady_clock::now();
            refiner.refine(image, points, box, std::stoull(fields[2]));
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
            output << line;
            for (const auto &point : points)
                output << ',' << point.x << ',' << point.y << ',' << point.conf;
            const auto &stats = refiner.stats();
            output << ',' << stats.color_matches << ',' << stats.flow_matches << ','
                   << stats.rejected_jumps << ',' << ms << '\n';
            if (++frames % 250 == 0)
                std::cout << "Refined " << frames << " frames\n" << std::flush;
        }
        std::cout << "Refined " << frames << " frames in total\n";
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
