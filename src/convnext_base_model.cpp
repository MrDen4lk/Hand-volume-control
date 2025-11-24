#include "convnext_base_model.hpp"
#include <string>

ConvNextBaseModel::ConvNextBaseModel(const std::string& model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "ConvnextPose"),
      session(nullptr),
      memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    Ort::SessionOptions session_options;
    session = Ort::Session(env, model_path.c_str(), session_options);
}

std::vector<HandPoint> ConvNextBaseModel::detect(const cv::Mat &resized) {
    // === Препроцессинг ===

    std::vector<float> input_tensor_values(1 * 3 * HandModelBase::MODEL_W * HandModelBase::MODEL_H);
    int pixels = HandModelBase::MODEL_H * HandModelBase::MODEL_W;
    const uint8_t* data = resized.data;

    for (int i = 0; i < pixels; i++) {
        input_tensor_values[i] = data[i * 3 + 0] / 255.0f;
        input_tensor_values[i + pixels] = data[i * 3 + 1] / 255.0f;
        input_tensor_values[i + pixels * 2] = data[i * 3 + 2] / 255.0f;
    }

    // === Инференс ===

    std::vector<int64_t> input_shape = {1, 3, HandModelBase::MODEL_H, HandModelBase::MODEL_W};

    // Тензов ONNX
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    // Имена узлов
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    // === Постпроцессинг ===
    float* output_data = output_tensors[0].GetTensorMutableData<float>();

    std::vector<HandPoint> keypoints;
    int heatmap_area = HEATMAP_W * HEATMAP_H;

    for (int k = 0; k < NUM_KEYPOINTS; k++) {
        // Указатель на начало текущего хитмапа
        const float* heatmap = output_data + (k * heatmap_area);

        // Ищем индекс максимального элемента в этом хитмапе
        int max_idx = 0;
        float max_val = -1.0f;

        for (int i = 0; i < heatmap_area; i++) {
            if (heatmap[i] > max_val) {
                max_val = heatmap[i];
                max_idx = i;
            }
        }

        // Переводим индекс обратно в x и y (внутри 56x56)
        int y_hm = max_idx / HEATMAP_H;
        int x_hm = max_idx % HEATMAP_W;

        // Масштабируем обратно к 224x224
        float x_final = x_hm * (HandModelBase::MODEL_W / HEATMAP_W);
        float y_final = y_hm * (HandModelBase::MODEL_H / HEATMAP_H);

        keypoints.push_back({x_final, y_final, max_val});
    }

    return keypoints;
}
