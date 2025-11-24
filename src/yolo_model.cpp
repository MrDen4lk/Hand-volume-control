#include "yolo_model.hpp"
#include <string>

YoloModel::YoloModel(const std::string& model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "YoloPose"),
      session(nullptr),
      memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    Ort::SessionOptions session_options;
    session = Ort::Session(env, model_path.c_str(), session_options);
}

std::vector<HandPoint> YoloModel::detect(const cv::Mat &resized) {
    // === Препроцессинг ===

    std::vector<float> input_tensor_values(1 * 3 * HandModelBase::MODEL_W * HandModelBase::MODEL_H);
    int pixels = HandModelBase::MODEL_W * HandModelBase::MODEL_H;
    const uint8_t* data = resized.data;

    for (int i = 0; i < pixels; i++) {
        input_tensor_values[i] = data[i * 3 + 0] / 255.0f;
        input_tensor_values[i + pixels] = data[i * 3 + 1] / 255.0f;
        input_tensor_values[i + pixels * 2] = data[i * 3 + 2] / 255.0f;
    }

    // === Инференс ===

    std::vector<int64_t> input_shape = {1, 3, HandModelBase::MODEL_H, HandModelBase::MODEL_W};

    // Тензор ONNX
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    // Имена узлов
    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};

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

    auto output_type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    auto output_dims = output_type_info.GetShape();

    int rows = (int)output_dims[1];    // 68 (параметры)
    int anchors = (int)output_dims[2]; // Кол-во предсказаний (столбцы)

    int best_anchor_idx = -1;
    float max_conf = 0.0f;

    // Проходим по всем предсказаниям и ищем лучшую руку
    for (int i = 0; i < anchors; i++) {
        // Индекс строки 4 (Confidence) для i-го анкора
        float confidence = output_data[4 * anchors + i];

        if (confidence > max_conf) {
            max_conf = confidence;
            best_anchor_idx = i;
        }
    }

    std::vector<HandPoint> result_points;

    // Если рука найдена, извлекаем 21 точку из лучшего анкора
    for (int k = 0; k < HandModelBase::NUM_KEYPOINTS; k++) {
        // Формула доступа: (StartRow + k*3 + Offset) * Stride + Column
        int idx_x = (5 + k * 3 + 0) * anchors + best_anchor_idx;
        int idx_y = (5 + k * 3 + 1) * anchors + best_anchor_idx;
        int idx_v = (5 + k * 3 + 2) * anchors + best_anchor_idx;

        float x = output_data[idx_x];
        float y = output_data[idx_y];
        float conf = output_data[idx_v];

        result_points.push_back({x, y, conf});
    }

    return result_points;
}
