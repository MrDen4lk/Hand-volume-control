#ifndef CONVNEXT_BASE_MODEL_HPP
#define CONVNEXT_BASE_MODEL_HPP

#pragma once
#include "hand_model_base.hpp"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

class ConvNextBaseModel : public HandModelBase {
private:
    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memory_info;

    const int HEATMAP_W = 56; // 224/4
    const int HEATMAP_H = 56;

public:
    explicit ConvNextBaseModel(const std::string& model_path);

    std::vector<HandPoint> detect(const cv::Mat& frame) override;
};

#endif //CONVNEXT_BASE_MODEL_HPP
