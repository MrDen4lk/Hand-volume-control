#ifndef VOLUME_CONTROL_HPP
#define VOLUME_CONTROL_HPP

#include <opencv2/opencv.hpp>

class VolumeControl {
public:
    explicit VolumeControl(float min_dist = 20.0f, float max_dist = 200.0f);

    [[nodiscard]] float get_volume(const cv::Point2f& point_1, const cv::Point2f& point_2) const;

    static void set_volume(float normalizedVolume);

    static void draw_volume(const cv::Mat& frame, float volume);

private:
    float _min_dist;
    float _max_dist;
};

#endif //VOLUME_CONTROL_HPP
