#include "dataset.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

HandKeypointDataset::HandKeypointDataset(const std::string& images_dir, const std::string& csv_path)
    : images_dir_(images_dir) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + csv_path);
    }

    std::string line;
    // Пропускаем первую строку (заголовки столбцов: x0,y0,x1,y1,...)
    if (!std::getline(file, line)) {
        throw std::runtime_error("CSV file is empty or corrupted: " + csv_path);
    }

    size_t line_idx = 0;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string value;
        std::vector<float> keypoints;

        while (std::getline(ss, value, ',')) {
            try {
                float coord = std::stof(value);
                if (coord >= 0.0f) {
                    coord /= 255.0f;  // нормализация
                }
                keypoints.push_back(coord);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Warning: Invalid value in CSV at line " << line_idx + 2 << ": " << value << std::endl;
                keypoints.push_back(-1.0f);  // Fallback
            }
        }

        if (keypoints.size() % 2 != 0) {
            throw std::runtime_error("Keypoint count not even at line: " + std::to_string(line_idx + 2));
        }

        keypoints_.push_back(torch::tensor(keypoints));

        // Сформировать имя изображения: img_0.png, img_1.png, ...
        std::stringstream img_name;
        img_name << "img_" << line_idx << ".png";
        std::string path = (std::filesystem::path(images_dir_) / img_name.str()).string();

        if (!std::filesystem::exists(path)) {
            throw std::runtime_error("Image file not found: " + path);
        }

        image_paths_.push_back(path);
        line_idx++;
    }
}

torch::data::Example<> HandKeypointDataset::get(size_t index) {
    cv::Mat img = cv::imread(image_paths_[index], cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + image_paths_[index]);
    }

    cv::resize(img, img, cv::Size(256, 256)); // Убедиться в размере

    torch::Tensor img_tensor = torch::from_blob(
        img.data, {img.rows, img.cols, 3}, torch::kUInt8
    ).permute({2, 0, 1}).to(torch::kFloat32).div(255.0);  // CxHxW и нормализация

    return {img_tensor.clone(), keypoints_[index].clone()};
}

torch::optional<size_t> HandKeypointDataset::size() const {
    return image_paths_.size();
}