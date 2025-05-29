#include "model.hpp"
#include "volume_control.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

cv::Mat visualize_image(const cv::Mat& image,
                                const torch::Tensor& heatmaps, // [num_keypoints, H, W]
                                const VolumeControl& volume_control,
                                const cv::Scalar& color = cv::Scalar(0, 0, 255),
                                int radius = 4) {
    cv::Mat vis_image;
    image.copyTo(vis_image);

    int heatmap_height = heatmaps.size(1);
    int heatmap_width = heatmaps.size(2);

    int img_width = image.cols;
    int img_height = image.rows;

    float scale_x = static_cast<float>(img_width) / heatmap_width;
    float scale_y = static_cast<float>(img_height) / heatmap_height;

    auto heatmaps_cpu = heatmaps.detach().to(torch::kCPU);

    std::vector<cv::Point2f> points;

    for (int k = 0; k < heatmaps.size(0); ++k) {
        auto heatmap = heatmaps_cpu[k];

        // Argmax для x, y
        auto max_val = heatmap.max();
        auto max_idx = std::get<1>(heatmap.view(-1).max(0));

        int max_index = max_idx.item<int>();
        int y = max_index / heatmap_width;
        int x = max_index % heatmap_width;

        int draw_x = static_cast<int>(x * scale_x);
        int draw_y = static_cast<int>(y * scale_y);
        points.push_back(cv::Point2f(draw_x, draw_y));

        // Рисуем точки
        cv::circle(vis_image, cv::Point(draw_x, draw_y), radius, color, -1);
    }

    // Рисуем полоску громкости
    cv::rectangle(vis_image ,
                      cv::Point(0, 256 - 20),
                      cv::Point(256, 256),
                      cv::Scalar(50, 50, 50), // тёмно-серый фон
                      cv::FILLED);
    cv::rectangle(vis_image,
                      cv::Point(0, 256 - 20),
                      cv::Point(static_cast<int>(volume_control.get_volume(points[4], points[8]) * 256), 256),
                      cv::Scalar(0, 255, 0), // зелёная полоска
                      cv::FILLED);

    return vis_image;
}

int main() {
    // Загружаем модель
    ResNet18Heatmap model;
    torch::load(model, "../data/model/model_epoch_40.pt");
    model->to(torch::kMPS);
    model->eval();

    // Получаем frame с камеры
    cv::VideoCapture cap(0);
    cv::Mat frame;
    VolumeControl value(0, 100);

    while (true) {
        cap >> frame;

        cv::resize(frame, frame, cv::Size(256, 256));

        // Получаем предсказание модели
        cv::Mat image_float = frame.clone();
        frame.convertTo(image_float, CV_32F, 1.0 / 255.0);
        auto input_tensor = torch::from_blob(image_float.data, {1, 256, 256, 3}, torch::kFloat32)
                                .permute({0, 3, 1, 2})  // NHWC -> NCHW
                                .clone();
        input_tensor = input_tensor.to(torch::kMPS);

        torch::NoGradGuard no_grad;
        auto output = model->forward(input_tensor).to(torch::kMPS); // [1, num_keypoints, H, W]

        // удаляем batch dim
        auto heatmaps = output.squeeze(0).to(torch::kCPU).detach(); // [num_keypoints, H, W]

        // Визуализация
        frame = visualize_image(frame, heatmaps, value);
        cv::imshow("image", frame);
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    return 0;
}