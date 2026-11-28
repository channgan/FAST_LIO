#include "flight_odometry.hpp"
#include <px4_msgs/msg/vehicle_attitude.hpp>
#include <px4_msgs/msg/vehicle_land_detected.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <thread>

int main(int argc, char** argv)
{
    using namespace std::chrono_literals;
    using namespace px4_msgs::msg;
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.parameter_overrides({
        rclcpp::Parameter("flight_output.rotation_fc_imu", std::vector<double>{1,0,0,0,1,0,0,0,1}),
        rclcpp::Parameter("flight_output.translation_fc_imu", std::vector<double>{0,0,0})});
    auto node = std::make_shared<rclcpp::Node>("laser_mapping", options);
    auto qos = rclcpp::SensorDataQoS();
    auto status = node->create_publisher<VehicleStatus>("/fmu/out/vehicle_status_v1", qos);
    auto land = node->create_publisher<VehicleLandDetected>("/fmu/out/vehicle_land_detected", qos);
    auto attitude = node->create_publisher<VehicleAttitude>("/fmu/out/vehicle_attitude", qos);
    auto ekf = node->create_publisher<VehicleOdometry>("/fmu/out/vehicle_odometry", qos);
    uint64_t latest_ev = 0, last_request = 0, first_divergence = 0;
    int requests = 0;
    auto ev = node->create_subscription<VehicleOdometry>("/fmu/in/vehicle_visual_odometry", qos,
        [&](VehicleOdometry::ConstSharedPtr m) { latest_ev = m->timestamp_sample; });
    auto request = node->create_subscription<std_msgs::msg::Header>("nmpc/planner/residual_land_request", 4,
        [&](std_msgs::msg::Header::ConstSharedPtr m) {
            last_request = rclcpp::Time(m->stamp).nanoseconds() / 1000;
            ++requests;
        });
    fast_lio::FlightOdometry output(*node);
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    for (int frame = 0; frame < 50; ++frame) {
        const auto stamp = node->now();
        const uint64_t us = stamp.nanoseconds() / 1000;
        VehicleStatus s;
        s.timestamp = us;
        s.nav_state = s.NAVIGATION_STATE_POSCTL;
        s.arming_state = frame < 40 ? s.ARMING_STATE_DISARMED : s.ARMING_STATE_ARMED;
        status->publish(s);
        VehicleLandDetected l;
        l.timestamp = us;
        l.landed = frame < 40;
        land->publish(l);
        VehicleAttitude a;
        a.timestamp_sample = a.timestamp = us;
        a.q = {1,0,0,0};
        attitude->publish(a);
        VehicleOdometry e;
        e.timestamp_sample = e.timestamp = us;
        e.pose_frame = e.POSE_FRAME_NED;
        e.velocity_frame = e.VELOCITY_FRAME_NED;
        e.q = a.q;
        if (frame >= 20) e.position[0] = 1.;
        if (frame == 40) first_divergence = us;
        ekf->publish(e);
        std::this_thread::sleep_for(5ms);
        executor.spin_some();
        nav_msgs::msg::Odometry scan;
        scan.header.stamp = stamp;
        scan.header.frame_id = "camera_init";
        scan.child_frame_id = "body";
        scan.pose.pose.orientation.w = 1;
        if (frame >= 40) scan.pose.pose.position.x = (frame - 39) * 1.0;
        output.submit(scan);
        std::this_thread::sleep_for(95ms);
        executor.spin_some();
        
        if (!first_divergence && requests) return 1;
    }
    const bool passed = requests > 1 && last_request >= first_divergence && latest_ev >= last_request;
    if (!passed) RCLCPP_ERROR(node->get_logger(), "requests=%d EV=%lu request=%lu", requests, latest_ev, last_request);
    return passed ? 0 : 1;
}
