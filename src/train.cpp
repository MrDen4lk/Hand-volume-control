#include "dataset.hpp"
#include "model.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <torch/torch.h>

std::vector<std::string> load_image_png(const std::string& directory) {
    std::vector<std::string> images_path;

    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.path().extension() == ".png") {
            images_path.emplace_back(entry.path().string());
        }
    }

    std::sort(images_path.begin(), images_path.end());
    return images_path;
}

std::vector<std::vector<float>> load_keypoints_csv(const std::string& path_csv) {
    std::vector<std::vector<float>> keypoints;
    std::ifstream file(path_csv);
    std::string line;
    bool first_string = true;

    while (std::getline(file, line)) {
        if (first_string) {
            first_string = false;
            continue;
        }
        std::stringstream ss(line);
        std::string token;
        std::vector<float> keypoints_vec;

        while (std::getline(ss, token, ',')) {
            keypoints_vec.emplace_back(std::stof(token));
        }

        keypoints.emplace_back(keypoints_vec);
    }

    return keypoints;
}

int main() {
    torch::manual_seed(42);

    // choose device for training
    auto device = torch::Device(torch::kCPU);
    if (torch::mps::is_available()) {
        std::cout << "MPS is available" << std::endl;
        device = torch::Device(torch::kMPS);
    }

    // load datasets
    auto train_images = load_image_png("../data/archive/train/train_images");
    auto train_keypoints = load_keypoints_csv("../data/archive/train/keypoints_labels_train.csv");

    auto valid_images = load_image_png("../data/archive/valid/valid_images");
    auto valid_keypoints = load_keypoints_csv("../data/archive/valid/keypoints_labels_valid.csv");

    std::cout << "train_size: " << train_keypoints.size() << ' ' << train_images.size() << std::endl;
    std::cout << "valid_size: " << valid_keypoints.size() << ' ' << valid_images.size() << std::endl;

    // prepare dataset
    auto train_dataset = HandKeypointsDataset(train_images, train_keypoints)
                                .map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader(train_dataset,
        torch::data::samplers::RandomSampler(train_dataset.size().value()),
        torch::data::DataLoaderOptions().batch_size(128).workers(6)
        );

    auto valid_dataset = HandKeypointsDataset(valid_images, valid_keypoints)
                                .map(torch::data::transforms::Stack<>());
    auto valid_loader = torch::data::make_data_loader(valid_dataset,
        torch::data::samplers::RandomSampler(valid_dataset.size().value()),
        torch::data::DataLoaderOptions().batch_size(128).workers(6));

    // train model
    HandKeypointModel model;
    model->to(device);
    model->train();

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-4));

    const int epoch = 20;
    for (int iter = 1; iter <= epoch; iter++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        float train_loss = 0.0;
        int batch_count = 0;

        for (auto& batch : *train_loader) {
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            optimizer.zero_grad();
            auto output = model->forward(data);
            auto loss = torch::nn::functional::smooth_l1_loss(target, output);
            loss.backward();
            optimizer.step();

            train_loss += loss.item<float>();
            batch_count++;
        }

        model->eval();
        torch::NoGradGuard no_grad;
        float val_loss = 0.0;
        int val_batches = 0;
        for (auto& batch : *valid_loader) {
            auto data = batch.data.to(device);
            auto targets = batch.target.to(device);
            auto output = model->forward(data);

            val_loss += torch::nn::functional::smooth_l1_loss(output, targets).item<float>();
            val_batches++;
        }
        model->train();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

        std::cout << "Epoch " << iter
              << " | Train Loss: " << (train_loss / batch_count)
              << " | Val Loss: " << (val_loss / val_batches)
              << " | Time: " << epoch_duration << " sec" << std::endl;
    }

    // save model
    torch::save(model, "../data/keypoints_model.pt");

    return 0;
}