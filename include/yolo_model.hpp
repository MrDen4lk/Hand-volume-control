#ifndef YOLO_MODEL_HPP
#define YOLO_MODEL_HPP

#pragma once
#include "hand_model_base.hpp"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

class YoloModel : public HandModelBase {
private:
    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memory_info;

public:
    explicit YoloModel(const std::string& model_path);

    std::vector<HandPoint> detect(const cv::Mat& frame) override;
};

#endif //YOLO_MODEL_HPP
