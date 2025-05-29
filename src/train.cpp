#include "dataset.hpp"
#include "model.hpp"
#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

torch::Tensor generate_heatmaps(const torch::Tensor& keypoints, int heatmap_height, int heatmap_width, float sigma = 1.0, torch::Device device = torch::kMPS) {
    // keypoints: [B, 42] — нормализованные координаты в диапазоне [0, 1]

    int batch_size = keypoints.size(0);
    int num_points = 21;

    // [B, 21, 2] → денормализуем координаты
    torch::Tensor keypoints_reshaped = keypoints.view({batch_size, num_points, 2}).clone();
    keypoints_reshaped.select(2, 0) *= heatmap_width;   // x
    keypoints_reshaped.select(2, 1) *= heatmap_height;  // y

    keypoints_reshaped = keypoints_reshaped.to(device);

    // meshgrid: координаты всех пикселей heatmap
    auto y_range = torch::arange(0, heatmap_height, torch::TensorOptions().device(device)).view({-1, 1}).repeat({1, heatmap_width});
    auto x_range = torch::arange(0, heatmap_width, torch::TensorOptions().device(device)).repeat({heatmap_height, 1});
    // [H, W] → [1, 1, H, W]
    auto yy = y_range;
    auto xx = x_range;

    // инициализируем heatmaps
    torch::Tensor heatmaps = torch::zeros({batch_size, num_points, heatmap_height, heatmap_width}, torch::TensorOptions().device(device));

    // проходим по батчу
    for (int b = 0; b < batch_size; ++b) {
        for (int j = 0; j < num_points; ++j) {
            float x = keypoints_reshaped[b][j][0].item<float>();
            float y = keypoints_reshaped[b][j][1].item<float>();

            // Отрицательные координаты — нет точки
            if (x < 0 || y < 0) continue;

            auto dx = xx - x;
            auto dy = yy - y;
            auto exponent = -(dx.pow(2) + dy.pow(2)) / (2 * sigma * sigma);
            heatmaps[b][j] = exponent.exp();
        }
    }

    return heatmaps;
}

void show_example(torch::data::Example<>& example) {
    torch::Tensor image_tensor = example.data;      // [3, 256, 256]
    torch::Tensor keypoints = example.target;        // [42]

    // Переводим тензор изображения в cv::Mat
    // Т.к. изображение в диапазоне [0, 1], умножим на 255
    torch::Tensor image_cpu = image_tensor.detach().cpu() * 255.0;
    image_cpu = image_cpu.to(torch::kU8).clamp(0, 255);

    // Преобразуем тензор [3, 256, 256] → OpenCV [256, 256, 3]
    image_cpu = image_cpu.permute({1, 2, 0}); // [H, W, C]
    cv::Mat img(256, 256, CV_8UC3, image_cpu.data_ptr());

    // Копия для рисования
    cv::Mat img_vis = img.clone();

    // Рисуем ключевые точки
    for (int j = 0; j < 21; ++j) {
        float x = keypoints[j * 2].item<float>() * 255.0f;
        float y = keypoints[j * 2 + 1].item<float>() * 255.0f;

        if (x >= 0 && y >= 0) {  // -1 означает "нет точки"
            cv::circle(img_vis, cv::Point((int)x, (int)y), 3, cv::Scalar(0, 0, 255), -1);
        }
    }

    // Показываем изображение
    cv::imshow("Sample", img_vis);
    cv::waitKey(0);  // Нажми любую клавишу
}

int main() {
    // paths to files
    std::string train_images_dir = "../data/archive/train/train_images";
    std::string train_csv = "../data/archive/train/keypoints_labels_train.csv";

    std::string valid_images_dir = "../data/archive/valid/valid_images";
    std::string valid_csv = "../data/archive/valid/keypoints_labels_valid.csv";

    // Гиперпараметры
    const int64_t batch_size = 256;
    const size_t num_epochs = 40;
    const double learning_rate = 1e-4;
    const double weight_decay = 1e-4;

    // Собираем dataset
    auto train_dataset = HandKeypointDataset(train_images_dir, train_csv);
    auto val_dataset = HandKeypointDataset(valid_images_dir, valid_csv);

    // Создаем dataloader
    auto train_loader = torch::data::make_data_loader(train_dataset.map(torch::data::transforms::Stack<>()),
        torch::data::samplers::RandomSampler(train_dataset.map(torch::data::transforms::Stack<>()).size().value()),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(6)
        );

    auto val_loader = torch::data::make_data_loader(
        val_dataset.map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(6));

    // Выбираем на чем обучаемся
    torch::Device device = torch::Device(torch::kCPU);
    if (torch::mps::is_available()) {
        device = torch::Device(torch::kMPS);
        std::cout << "MPS is available" << std::endl;
    } else if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "CUDA is available" << std::endl;
    }

    // инициализируем модель
    auto model = ResNet18Heatmap();
    model->to(device);

    // Потимайзер
    torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(learning_rate).weight_decay(weight_decay));

    // Обучение
    for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
        auto start = std::chrono::high_resolution_clock::now();

        model->train();
        double train_loss = 0.0;
        size_t batch_idx = 0;

        for (auto& batch : *train_loader) {
            auto images = batch.data.to(device);
            auto targets = batch.target.to(device); // shape: [B, 21, 64, 64]

            optimizer.zero_grad();

            auto outputs = model->forward(images).to(device); // [B, 21, 64, 64]

            auto gt_heatmap = generate_heatmaps(targets, 64, 64);
            auto loss = torch::mse_loss(outputs, gt_heatmap);
            loss.backward();
            optimizer.step();

            train_loss += loss.item<double>();
            batch_idx++;
        }

        model->eval();
        torch::NoGradGuard no_grad;
        double val_loss = 0.0;
        size_t val_batches = 0;

        // Валидация
        for (auto& batch : *val_loader) {
            auto images = batch.data.to(device);
            auto targets = batch.target.to(device);

            auto outputs = model->forward(images).to(device);
            auto gt_heatmap = generate_heatmaps(targets, 64, 64);
            auto loss = torch::mse_loss(outputs, gt_heatmap);
            val_loss += loss.item<double>();
            val_batches++;
        }

        // Показываем результаты обучения и время
        val_loss /= val_batches;
        train_loss /= batch_idx;
        auto end = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

        std::cout << "Epoch " << epoch
              << " | Train Loss: " << train_loss
              << " | Val Loss: " << val_loss
              << " | Time: " << epoch_duration << " sec" << std::endl;

        // Сохраняем эпоху модели
        torch::save(model, "../data/model/model_epoch_" + std::to_string(epoch) + ".pt");
    }

    return 0;
}
