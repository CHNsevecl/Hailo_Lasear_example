#ifndef __MYHAILO_HPP__
#define __MYHAILO_HPP__

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <hailo/hailort.hpp>

struct HailoContext {
    std::shared_ptr<hailort::VDevice> vdevice;
    std::shared_ptr<hailort::InferModel> infer_model;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<std::vector<uint8_t>> input_buffer;
    std::vector<std::vector<uint8_t>> output_buffer;
    hailort::ConfiguredInferModel configured_infer_model;
    hailort::ConfiguredInferModel::Bindings bindings;
};

struct Detection {
    std::string label;
    cv::Point upper;
    cv::Point lower;
    float score;
};

std::optional<HailoContext> Hailo_init(const std::string& hef_path);
std::optional<std::vector<Detection>> ParseDetections(HailoContext& hailo_context, int target_w, int target_h, cv::Mat& RGB_frame,const std::vector<std::string>& class_names);
cv::Mat Stream_process(const cv::Mat& BGR_frame, int target_w, int target_h);

#endif
