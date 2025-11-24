#include "volume_control.hpp"
#include "hand_model_base.hpp"
#include "yolo_model.hpp"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

#include "convnext_base_model.hpp"

// Параметры нормализации ImageNet
const float MEAN[] = {0.485f, 0.456f, 0.406f};
const float STD[]  = {0.229f, 0.224f, 0.225f};

cv::Point2f smooth_p4 = {0, 0};
cv::Point2f smooth_p8 = {0, 0};
float alpha = 0.5f;

int main() {
    std::unique_ptr<HandModelBase> tracker = std::make_unique<YoloModel>("../models/hand_keypoints_yolo.onnx");
    // std::unique_ptr<HandModelBase> tracker = std::make_unique<ConvNextBaseModel>("../models/hand_keypoints_convnext_base.onnx");

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

    while (true) {

        tm.start(); // fps
        cap >> frame;
        if (frame.empty()) { break; }

        cv::Mat model_frame;
        cv::resize(frame, model_frame, cv::Size(HandModelBase::MODEL_W, HandModelBase::MODEL_H));
        cv::cvtColor(model_frame, model_frame, cv::COLOR_BGR2RGB);

        // ===  Инференс ===

        std::vector<HandPoint> points = tracker->detect(model_frame);

        // === Графика ===
        // Индексы: 4 (Большой палец), 8 (Указательный)
        HandPoint p4 = points[4];
        HandPoint p8 = points[8];

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
        HandPoint p1 = points[4];
        HandPoint p2 = points[8];
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

        tm.stop();
        double fps = tm.getFPS();
        tm.reset();

        // Рисуем FPS в углу
        std::string fps_text = "FPS: " + std::to_string((int)fps);
        cv::putText(model_frame, fps_text, cv::Point(5, 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

        // === Отображение ===
        imshow("image", model_frame);
        if (cv::waitKey(1) == 27) {break;}
    }

    return 0;
}