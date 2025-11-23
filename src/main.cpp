#include "volume_control.hpp"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

// Константы модели
const int MODEL_W = 224;
const int MODEL_H = 224;
const int HEATMAP_W = 56; // 224 / 4
const int HEATMAP_H = 56;
const int NUM_KEYPOINTS = 21;
const std::vector<std::pair<int, int>> SKELETON_CONNECTIONS = {
    {0, 1}, {1, 2}, {2, 3}, {3, 4},       // Большой палец
    {0, 5}, {5, 6}, {6, 7}, {7, 8},       // Указательный
    {0, 9}, {9, 10}, {10, 11}, {11, 12},  // Средний
    {0, 13}, {13, 14}, {14, 15}, {15, 16},// Безымянный
    {0, 17}, {17, 18}, {18, 19}, {19, 20} // Мизинец
};
// Параметры нормализации ImageNet
const float MEAN[] = {0.485f, 0.456f, 0.406f};
const float STD[]  = {0.229f, 0.224f, 0.225f};

cv::Point2f smooth_p4 = {0, 0};
cv::Point2f smooth_p8 = {0, 0};
float alpha = 0.5f;

struct Point {
    float x;
    float y;
    float confidence;
};

std::vector<Point> postprocess_heatmaps(const float* output_data, int num_kpts, int hm_w, int hm_h, int stride) {
    std::vector<Point> keypoints;
    int heatmap_area = hm_w * hm_h;

    for (int k = 0; k < num_kpts; k++) {
        // Указатель на начало текущего хитмапа (сдвигаем на площадь)
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
        int y_hm = max_idx / hm_w;
        int x_hm = max_idx % hm_w;

        // Масштабируем обратно к 224x224 (умножаем на stride=4)
        // Добавляем 0.5 для центровки, если нужно, но пока просто умножим
        float x_final = x_hm * stride;
        float y_final = y_hm * stride;

        keypoints.push_back({x_final, y_final, max_val});
    }

    return keypoints;
}


int main() {
    // Инициализация onnx
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "HandVolumeControl");
    Ort::SessionOptions session_options;

    const char* model_path = "../models/hand_keypoints_convnext_base.onnx";

    Ort::Session session(nullptr);
    try {
        session = Ort::Session(env, model_path, session_options);
    } catch (const Ort::Exception& e) {
        std::cerr << "Не удалось загрузить модель! Ошибка: " << e.what() << std::endl;
        return -1;
    }

    // Размер: [B, 3, 224, 224]
    std::vector<float> input_tensor_values(1 * 3 * MODEL_H * MODEL_W);
    std::vector<int64_t> input_shape = {1, 3, MODEL_H, MODEL_W};
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Работа с камерой
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Камера не найдена!" << std::endl;
        return -1;
    }

    // для FPS
    cv::TickMeter tm;

    VolumeControl volume_ctrl(20.0f, 100.0f);
    float last_volume = -1.0f;

    cv::Mat frame;
    cv::Mat model_frame; // Отдельная картинка для модели

    while (true) {
        tm.start();

        cap >> frame;
        if (frame.empty()) { break; }

        // Преобразования
        cv::resize(frame, model_frame, cv::Size(MODEL_W, MODEL_H));
        cv::cvtColor(model_frame, model_frame, cv::COLOR_BGR2RGB);

        int pixels_count = MODEL_H * MODEL_W;
        const uint8_t* img_data = model_frame.data;

        for (int i = 0; i < pixels_count; i++) {
            // Берем пиксели R, G, B (0-255) и приводим к 0-1
            float r = img_data[i * 3 + 0] / 255.0f;
            float g = img_data[i * 3 + 1] / 255.0f;
            float b = img_data[i * 3 + 2] / 255.0f;

            // Нормализация (x - mean) / std
            input_tensor_values[i] = (r - MEAN[0]) / STD[0];                   // R канал
            input_tensor_values[i + pixels_count] = (g - MEAN[1]) / STD[1];    // G канал
            input_tensor_values[i + pixels_count * 2] = (b - MEAN[2]) / STD[2];// B канал
        }

        // Инференс
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            input_shape.data(),
            input_shape.size()
        );

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        // Постпроцессинг
        float* output_data = output_tensors[0].GetTensorMutableData<float>();

        std::vector<Point> points = postprocess_heatmaps(output_data, NUM_KEYPOINTS, HEATMAP_W, HEATMAP_H, 4);


        // Индексы: 4 (Большой палец), 8 (Указательный)
        Point p4 = points[4];
        Point p8 = points[8];

        // Работаем только если уверенность в точках высокая
        if (p4.confidence > 0.4 && p8.confidence > 0.4) {

            // 2. Конвертируем ваши точки в cv::Point2f (как требует ваш класс)
            cv::Point2f cv_p4(p4.x, p4.y);
            cv::Point2f cv_p8(p8.x, p8.y);

            // 3. Вычисляем нормализованную громкость (0.0 - 1.0)

            if (smooth_p4.x == 0) { smooth_p4 = cv::Point2f(p4.x, p4.y); }
            if (smooth_p8.x == 0) { smooth_p8 = cv::Point2f(p8.x, p8.y); }

            // Формула сглаживания: New = Prev * (1-alpha) + Curr * alpha
            smooth_p4 = smooth_p4 * (1.0f - alpha) + cv::Point2f(p4.x, p4.y) * alpha;
            smooth_p8 = smooth_p8 * (1.0f - alpha) + cv::Point2f(p8.x, p8.y) * alpha;

            // Используем smooth_p4 вместо p4 для расчета громкости
            float current_vol = volume_ctrl.get_volume(smooth_p4, smooth_p8);

            // 4. Оптимизация: меняем системную громкость, только если изменение > 1%
            // Это спасет от лагов из-за system("osascript")
            if (std::abs(current_vol - last_volume) > 0.01f) {
                volume_ctrl.set_volume(current_vol);
                last_volume = current_vol;
                // std::cout << "Volume set to: " << (int)(current_vol * 100) << "%" << std::endl;
            }

            // 5. Визуализация (Рисуем линию и бар)

            // Линия между пальцами
            cv::line(model_frame,
                     cv::Point((int)p4.x, (int)p4.y),
                     cv::Point((int)p8.x, (int)p8.y),
                     cv::Scalar(0, 255, 255), 2); // Желтая линия

            // Визуальный бар громкости сбоку
            int bar_height = 150;
            int bar_width = 20;
            int bar_x = 20;
            int bar_y = 50;

            // Рамка
            cv::rectangle(model_frame,
                          cv::Point(bar_x, bar_y),
                          cv::Point(bar_x + bar_width, bar_y + bar_height),
                          cv::Scalar(0, 255, 0), 1);

            // Заливка (уровень громкости)
            int fill_height = (int)(current_vol * bar_height);
            cv::rectangle(model_frame,
                          cv::Point(bar_x, bar_y + bar_height - fill_height),
                          cv::Point(bar_x + bar_width, bar_y + bar_height),
                          cv::Scalar(0, 255, 0), -1);

            // Текст %
            cv::putText(model_frame, std::to_string((int)(current_vol * 100)) + "%",
                        cv::Point(bar_x, bar_y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }

        // отрисовка линии между большим и указательным
        Point p1 = points[4];
        Point p2 = points[8];
        if (p1.confidence > 0.4 && p2.confidence > 0.4) {
            cv::line(model_frame,
                     cv::Point((int)p1.x, (int)p1.y),
                     cv::Point((int)p2.x, (int)p2.y),
                     cv::Scalar(255, 255, 0), // Цвет (Cyan)
                     2); // Толщина
        }

        // 2. Рисуем точки
        for (const auto& p : points) {
            if (p.confidence > 0.4) {
                cv::circle(model_frame, cv::Point((int)p.x, (int)p.y), 3, cv::Scalar(0, 0, 255), -1);
            }
        }

        // FPS
        tm.stop();
        double fps = tm.getFPS();
        tm.reset(); // Сбрасываем таймер для следующего кадра

        // Рисуем FPS в углу
        std::string fps_text = "FPS: " + std::to_string((int)fps);
        cv::putText(model_frame, fps_text, cv::Point(5, 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

        //  Отображение
        imshow("image", model_frame);
        if (cv::waitKey(1) == 27) {break;}
    }

    return 0;
}