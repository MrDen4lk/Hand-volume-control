#include "dataset.hpp"
#include <opencv2/opencv.hpp>

HandKeypointsDataset::HandKeypointsDataset(const std::vector<std::string>& images_path,
                                           const std::vector<std::vector<float>>& keypoints_vec)
    : images(images_path), keypoints(keypoints_vec) {}

torch::data::Example<> HandKeypointsDataset::get(size_t index) {
    cv::Mat image = cv::imread(images[index]);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image.convertTo(image, CV_32F, 1.0 / 255.0);

    auto image_tensor = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kFloat);
    image_tensor = image_tensor.permute({2, 0, 1}).clone();

    auto keypoints_tensor = torch::from_blob(keypoints[index].data(), {static_cast<long>(keypoints[index].size())}, torch::kFloat).clone();

    return {image_tensor, keypoints_tensor};
}

torch::optional<size_t> HandKeypointsDataset::size() const {
    return images.size();
}
