#include "volume_control.hpp"
#include <string>

VolumeControl::VolumeControl(const float min_dist, const float max_dist)
    : _min_dist(min_dist), _max_dist(max_dist) {}

float VolumeControl::get_volume(const cv::Point2f &point_1, const cv::Point2f &point_2) const {
    const float distance = static_cast<float>(cv::norm(point_1 - point_2));
    const float clamped = std::min(std::max(distance, _min_dist), _max_dist);

    return (clamped - _min_dist) / (_max_dist - _min_dist);
}

void VolumeControl::set_volume(const float normalizedVolume) {
    const int volume = static_cast<int>(normalizedVolume * 100);
    // forming command to change volume
    const std::string command = "osascript -e \"set volume output volume " + std::to_string(volume) + "\"";

    // change volume
    system(command.c_str());
}


