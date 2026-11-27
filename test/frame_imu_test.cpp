#include "frame_imu.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace
{
using fast_lio::ImuSample;

void check(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

rclcpp::Time stamp(int ns)
{
    return rclcpp::Time(1700000000, ns, RCL_ROS_TIME);
}

ImuSample sample(int ns, double value)
{
    auto imu = std::make_shared<sensor_msgs::msg::Imu>();
    imu->header.stamp = stamp(ns);
    imu->angular_velocity.x = value;
    imu->angular_velocity.y = -value;
    imu->angular_velocity.z = 2.0 * value;
    imu->linear_acceleration.x = 3.0 * value;
    imu->linear_acceleration.y = 2.0 - value;
    imu->linear_acceleration.z = -2.0 * value;
    return imu;
}

double integrated_rate(const ImuSample& a, const ImuSample& b)
{
    const double dt = (rclcpp::Time(b->header.stamp) - rclcpp::Time(a->header.stamp)).seconds();
    return dt * 0.5 * (a->angular_velocity.x + b->angular_velocity.x);
}
} // namespace

int main()
{
    const auto a = sample(94000001, 1.0);
    const auto b = sample(99000001, 6.0);
    std::deque<ImuSample> queue{a}, real;
    ImuSample right;
    const auto end = stamp(96000001);
    check(!fast_lio::collect_frame_imu(queue, end, real, right), "missing right sample");
    check(queue.size() == 1, "incomplete scan consumed IMU");
    queue.push_back(b);
    check(fast_lio::collect_frame_imu(queue, end, real, right), "bracketed frame");
    check(real.size() == 1 && real.front() == a && right == b && queue.front() == b,
          "real initialization samples and retained right endpoint");
    auto tail = fast_lio::frame_end_imu(real.back(), right, end);
    check(tail && rclcpp::Time(tail->header.stamp) == end, "interpolation timestamp");
    check(tail->angular_velocity.x == 3.0 && tail->angular_velocity.y == -3.0 &&
          tail->angular_velocity.z == 6.0, "interpolated angular velocity");
    check(tail->linear_acceleration.x == 9.0 && tail->linear_acceleration.y == -1.0 &&
          tail->linear_acceleration.z == -6.0, "interpolated acceleration");

    // A second scan can end before the next real IMU arrives.
    const auto second_end = stamp(97000001);
    check(fast_lio::collect_frame_imu(queue, second_end, real, right), "second short frame");
    check(real.empty() && right == b && queue.front() == b, "right sample retained twice");
    auto second = fast_lio::frame_end_imu(tail, right, second_end);
    check(second && second->angular_velocity.x == 4.0, "continuation from frame tail");
    check(fast_lio::collect_frame_imu(queue, stamp(99000001), real, right), "exact frame");
    check(real.size() == 1 && real.back() == b && !right && queue.empty(), "exact hit consumed once");
    auto exact = fast_lio::frame_end_imu(real.back(), right, stamp(99000001));
    check(exact == b, "reuse exact sample");
    double split = integrated_rate(a, tail) + integrated_rate(tail, second) + integrated_rate(second, exact);
    check(std::abs(split - integrated_rate(a, b)) < 1e-14, "continuous split propagation");
    check(!fast_lio::frame_end_imu(nullptr, b, end), "missing left endpoint");
    std::cout << "Passed IMU frame boundary tests\n";
}
