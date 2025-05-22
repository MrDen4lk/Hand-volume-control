#ifndef DATASET_HPP
#define DATASET_HPP

#include <torch/torch.h>
#include <vector>
#include <string>

struct HandKeypointsDataset : torch::data::Dataset<HandKeypointsDataset> {
    std::vector<std::string> images;
    std::vector<std::vector<float>> keypoints;

    HandKeypointsDataset(const std::vector<std::string>& images_path,
                         const std::vector<std::vector<float>>& keypoints_vec);

    torch::data::Example<> get(size_t index) override;
    [[nodiscard]] torch::optional<size_t> size() const override;
};

#endif //DATASET_HPP
