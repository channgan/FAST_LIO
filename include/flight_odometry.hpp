#pragma once

#include <memory>
#include <chrono>
#include <functional>
#include <nav_msgs/msg/odometry.hpp>

namespace rclcpp { class Node; }

namespace fast_lio
{
inline int64_t steady_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

class FlightOdometry
{
public:
    using TimingSink = std::function<void(int64_t scan_end, int64_t enqueued, int64_t dequeued, int64_t published)>;
    explicit FlightOdometry(rclcpp::Node& node, TimingSink timing = {});
    ~FlightOdometry();
    FlightOdometry(const FlightOdometry&) = delete;
    FlightOdometry& operator=(const FlightOdometry&) = delete;

    // Submit one corrected scan-end pose and its measured body twist.
    void submit(const nav_msgs::msg::Odometry& corrected_scan);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace fast_lio
