#include "dataset.hpp"
#include <opencv2/opencv.hpp>

HandKeypointsDataset::HandKeypointsDataset(const std::vector<std::string>& images_path,
                                           const std::vector<std::vector<float>>& keypoints_vec)
    : images(images_path), keypoints(keypoints_vec) {}

torch::data::Example<> HandKeypointsDataset::get(size_t index) {

}

torch::optional<size_t> HandKeypointsDataset::size() const {
    return images.size();
}
