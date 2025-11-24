#ifndef HAND_MODEL_BASE_HPP
#define HAND_MODEL_BASE_HPP

#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct HandPoint {
    float x;
    float y;
    float confidence;
};

class HandModelBase {
public:
    // Общие константы
    static constexpr int MODEL_W = 224;
    static constexpr int MODEL_H = 224;
    static constexpr int NUM_KEYPOINTS = 21;

    // Скелет руки
    inline static const std::vector<std::pair<int, int>> SKELETON = {
        {0, 1}, {1, 2}, {2, 3}, {3, 4},       // Большой
        {0, 5}, {5, 6}, {6, 7}, {7, 8},       // Указательный
        {0, 9}, {9, 10}, {10, 11}, {11, 12},  // Средний
        {0, 13}, {13, 14}, {14, 15}, {15, 16},// Безымянный
        {0, 17}, {17, 18}, {18, 19}, {19, 20} // Мизинец
    };

    virtual ~HandModelBase() = default;

    virtual std::vector<HandPoint> detect(const cv::Mat& frame) = 0;
};

#endif //HAND_MODEL_BASE_HPP
