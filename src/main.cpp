#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>

int main(int argc, char** argv) {
    cv::VideoCapture cap(0, cv::CAP_AVFOUNDATION);

    if (!cap.isOpened()) {
        std::cerr << "Unable to open camera" << std::endl;
        return -1;
    }

    cv::UMat frame;
    auto last_time = std::chrono::high_resolution_clock::now();
    double fps = 0.0;
    bool run_webcam = true;

    while (run_webcam) {
        // get 1 frame from webcamera
        cap >> frame;

        // calculate fps
        auto now = std::chrono::high_resolution_clock::now();
        double duration_ms = std::chrono::duration<double, std::milli>(now - last_time).count();
        last_time = now;
        fps = 1000.0 / duration_ms;

        // show fps on screen
        std::string fps_text = "FPS: " + std::to_string(int(fps));
        cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                    1.0, cv::Scalar(0, 255, 0), 2);

        // show frame
        cv::imshow("camera", frame);

        // press "q" to quit
        if (cv::waitKey(30) == 27) {
            run_webcam = false;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
