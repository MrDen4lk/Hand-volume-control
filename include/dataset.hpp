#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <string>
#include <torch/torch.h>

class HandKeypointDataset : public torch::data::Dataset<HandKeypointDataset> {
public:
    HandKeypointDataset(const std::string &images_dir, const std::string& csv_path);

    torch::data::Example<> get(size_t index) override;
    [[nodiscard]] torch::optional<size_t> size() const override;

private:
    std::vector<std::string> image_paths_;
    std::vector<torch::Tensor> keypoints_;
    std::string images_dir_;
};

#endif //DATASET_HPP