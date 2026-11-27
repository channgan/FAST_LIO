#pragma once

#include <deque>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/imu.hpp>

namespace fast_lio
{
using ImuSample = sensor_msgs::msg::Imu::ConstSharedPtr;

// Keep the first future measurement for the next scan and for interpolation.
inline bool collect_frame_imu(std::deque<ImuSample>& queue, const rclcpp::Time& end,
                              std::deque<ImuSample>& samples, ImuSample& next)
{
    if (queue.empty() || rclcpp::Time(queue.back()->header.stamp) < end)
        return false;
    samples.clear();
    while (!queue.empty() && rclcpp::Time(queue.front()->header.stamp) <= end)
    {
        samples.push_back(queue.front());
        queue.pop_front();
    }
    next = queue.empty() ? nullptr : queue.front();
    return true;
}

inline ImuSample frame_end_imu(const ImuSample& left, const ImuSample& right,
                               const rclcpp::Time& end)
{
    if (!left)
        return nullptr;
    const rclcpp::Time before(left->header.stamp);
    if (end == before)
        return left;
    if (before > end || !right || rclcpp::Time(right->header.stamp) <= end)
        return nullptr;

    const double fraction = static_cast<double>((end - before).nanoseconds()) /
                            (rclcpp::Time(right->header.stamp) - before).nanoseconds();
    auto interpolate = [fraction](const auto& a, const auto& b) {
        geometry_msgs::msg::Vector3 v;
        v.x = a.x + fraction * (b.x - a.x);
        v.y = a.y + fraction * (b.y - a.y);
        v.z = a.z + fraction * (b.z - a.z);
        return v;
    };
    auto sample = std::make_shared<sensor_msgs::msg::Imu>();
    sample->header = left->header;
    sample->header.stamp = end;
    sample->angular_velocity = interpolate(left->angular_velocity, right->angular_velocity);
    sample->linear_acceleration = interpolate(left->linear_acceleration, right->linear_acceleration);
    return sample;
}
} // namespace fast_lio
