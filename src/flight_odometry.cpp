#include "flight_odometry.hpp"
#include <odom_variance_bridge/position_consistency.hpp>
#include <std_msgs/msg/header.hpp>
#include "flight_odometry_geometry.hpp"

#include <px4_msgs/msg/vehicle_attitude.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rcpputils/filesystem_helper.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace fast_lio {
namespace {
using flight::BodyState;
using flight::Quaternion;
using flight::Transform;
using flight::Vector;
using Raw = nav_msgs::msg::Odometry;
using Attitude = px4_msgs::msg::VehicleAttitude;
using Ev = px4_msgs::msg::VehicleOdometry;
using Status = px4_msgs::msg::VehicleStatus;
using Clock = std::chrono::steady_clock;

constexpr uint64_t pairing_us = 50000;

uint64_t difference(uint64_t a, uint64_t b) { return a >= b ? a - b : b - a; }

Quaternion px4_q(const std::array<float, 4>& q) { return {q[0], q[1], q[2], q[3]}; }
std::array<float, 3> float_array(const Vector& v)
{
    return {static_cast<float>(v.x()), static_cast<float>(v.y()), static_cast<float>(v.z())};
}

bool body_state(const Raw& msg, const Transform& imu_from_fc, BodyState& body)
{
    if (msg.header.frame_id != "camera_init" || msg.child_frame_id != "body") return false;
    const auto& pose = msg.pose.pose;
    const Quaternion q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
    const Vector p(pose.position.x, pose.position.y, pose.position.z);
    const auto& twist = msg.twist.twist;
    const Vector v(twist.linear.x, twist.linear.y, twist.linear.z);
    const Vector w(twist.angular.x, twist.angular.y, twist.angular.z);
    if (!flight::valid(q) || !p.allFinite() || !v.allFinite() || !w.allFinite()) return false;
    Transform world_from_imu = Transform::Identity();
    world_from_imu.linear() = q.normalized().toRotationMatrix();
    world_from_imu.translation() = p;
    body = flight::at_fc(world_from_imu, v, w, imu_from_fc);
    return true;
}

uint8_t next_reset_counter(std::string ns, size_t domain)
{
    const char* ros_home = std::getenv("ROS_HOME");
    const char* home = std::getenv("HOME");
    if ((!ros_home || !*ros_home) && (!home || !*home)) throw std::runtime_error("missing ROS_HOME and HOME");
    const std::string folder = ros_home && *ros_home ? ros_home : std::string(home) + "/.ros";
    rcpputils::fs::create_directories(folder);
    std::replace(ns.begin(), ns.end(), '/', '_');
    const std::string path = folder + "/lio_ev_reset_" + std::to_string(domain) + "_" + ns;
    const std::string tmp = path + "." + std::to_string(getpid()) + ".tmp";
    // Share the previous bridge's counter file. Serialize concurrent restarts
    // and replace atomically so that a new local origin has a distinct counter.
    const int fd = ::open((path + ".lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) throw std::runtime_error("cannot open EV reset counter lock");
    struct File { int fd; ~File() { ::close(fd); } } lock{fd};
    if (::flock(fd, LOCK_EX) != 0) throw std::runtime_error("cannot lock EV reset counter");
    int old = -1;
    if (rcpputils::fs::exists(path)) {
        std::ifstream previous(path);
        if (!(previous >> old) || old < 0 || old > 255) throw std::runtime_error("invalid EV reset counter");
    }
    const auto next = static_cast<uint8_t>(old + 1);
    std::ofstream file(tmp, std::ios::trunc);
    file << static_cast<int>(next);
    file.close();
    if (!file || std::rename(tmp.c_str(), path.c_str()) != 0)
        throw std::runtime_error("cannot save EV reset counter");
    return next;
}

// A reference belongs to one DDS writer. A source replacement or clock
// rollback invalidates its session instead of silently reusing a frozen frame.
struct Reference {
    std::array<uint8_t, RMW_GID_STORAGE_SIZE> id{};
    uint64_t stamp = 0;
    bool fault = false;

    bool accept(size_t writers, const rclcpp::MessageInfo& info, uint64_t next)
    {
        if (!next || fault) return false;
        if (!writers) return false;
        const auto& gid = info.get_rmw_message_info().publisher_gid.data;
        if (writers != 1 || std::all_of(gid, gid + id.size(), [](uint8_t v) { return v == 0; }) ||
            (stamp && (next < stamp || std::memcmp(id.data(), gid, id.size()) != 0))) {
            fault = true;
            return false;
        }
        if (stamp == next) return false;
        std::copy_n(gid, id.size(), id.begin());
        stamp = next;
        return true;
    }
};

Attitude::ConstSharedPtr nearest_attitude(const std::deque<Attitude::ConstSharedPtr>& history, uint64_t sample)
{
    const auto best = std::min_element(history.begin(), history.end(), [sample](const auto& a, const auto& b) {
        return difference(a->timestamp_sample, sample) < difference(b->timestamp_sample, sample);
    });
    return best != history.end() && difference((*best)->timestamp_sample, sample) <= pairing_us ? *best : nullptr;
}

} // namespace

class FlightOdometry::Impl {
public:
    Impl(rclcpp::Node& node, TimingSink timing) : node_(node), timing_(std::move(timing))
    {
        const auto depth = node.declare_parameter<int>("flight_output.queue_depth", 32);
        if (depth < 1 || depth > 4096) throw std::invalid_argument("flight_output.queue_depth must be 1..4096");
        depth_ = static_cast<size_t>(depth);
        const auto max_age = node.declare_parameter<double>("flight_output.max_sample_age_s", .5);
        if (!std::isfinite(max_age) || max_age <= 0 || max_age > 10)
            throw std::invalid_argument("flight_output.max_sample_age_s must be in (0, 10]");
        max_age_us_ = static_cast<uint64_t>(max_age * 1e6);
        const auto rotation = node.declare_parameter<std::vector<double>>("flight_output.rotation_fc_imu", std::vector<double>{});
        const auto offset = node.declare_parameter<std::vector<double>>("flight_output.translation_fc_imu", std::vector<double>{});
        imu_from_fc_ = flight::fc_from_imu(rotation, offset).inverse();
        ns_ = node.declare_parameter<std::string>("flight_output.px4_namespace", "");
        const auto start = ns_.find_first_not_of('/');
        ns_ = start == std::string::npos ? "" : ns_.substr(start, ns_.find_last_not_of('/') - start + 1);
        const std::string fmu = ns_.empty() ? "/fmu/" : "/" + ns_ + "/fmu/";
        const auto px4_qos = rclcpp::SensorDataQoS().keep_last(depth_);
        ev_ = node.create_publisher<Ev>(fmu + "in/vehicle_visual_odometry", px4_qos);
        group_ = node.create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions options;
        options.callback_group = group_;
        status_sub_ = node.create_subscription<Status>(fmu + "out/vehicle_status_v1", px4_qos,
            [this](Status::ConstSharedPtr m, const rclcpp::MessageInfo& info) {
                std::lock_guard<std::mutex> guard(ref_mutex_);
                if (!status_id_.accept(status_sub_->get_publisher_count(), info, m->timestamp)) return;
                status_ = m;
                status_received_ = Clock::now();
            }, options);
        attitude_sub_ = node.create_subscription<Attitude>(fmu + "out/vehicle_attitude", px4_qos,
            [this](Attitude::ConstSharedPtr m, const rclcpp::MessageInfo& info) {
                if (!flight::valid(px4_q(m->q))) return;
                std::lock_guard<std::mutex> guard(ref_mutex_);
                if (!attitude_id_.accept(attitude_sub_->get_publisher_count(), info, m->timestamp_sample)) return;
                attitudes_.push_back(m);
                while (attitudes_.size() > 1 &&
                       (m->timestamp_sample - attitudes_.front()->timestamp_sample > max_age_us_ + pairing_us ||
                        attitudes_.size() > 8192))
                    attitudes_.pop_front();
            }, options);
        const std::string prefix = ns_.empty() ? "/" : "/" + ns_ + "/";
        residual_request_ = node.create_publisher<std_msgs::msg::Header>(
            prefix + "nmpc/planner/residual_land_request", 1);
        worker_ = std::thread([this] { run(); });
    }

    ~Impl()
    {
        {
            std::lock_guard<std::mutex> guard(queue_mutex_);
            stopping_ = true;
            queue_.clear();
        }
        wake_.notify_one();
        if (worker_.joinable()) worker_.join();
    }

    void submit(const Raw& m)
    {
        std::unique_lock<std::mutex> guard(queue_mutex_);
        if (stopping_) return;
        if (queue_.size() == depth_) {
            queue_.pop_front();
            ++overwrites_;
        }
        queue_.push_back(m);
        const auto enqueued = steady_ns();
        guard.unlock();
        wake_.notify_one();
        if (timing_) timing_(rclcpp::Time(m.header.stamp).nanoseconds(), enqueued, 0, 0);
    }

private:
    void run()
    {
        for (;;) {
            Raw msg;
            int64_t dequeued;
            size_t dropped;
            {
                std::unique_lock<std::mutex> guard(queue_mutex_);
                wake_.wait(guard, [this] { return stopping_ || !queue_.empty(); });
                if (stopping_) return;
                msg = std::move(queue_.front());
                queue_.pop_front();
                dequeued = steady_ns();
                dropped = overwrites_;
            }
            try {
                if (dropped != reported_overwrites_) {
                    reported_overwrites_ = dropped;
                    RCLCPP_WARN_THROTTLE(node_.get_logger(), *node_.get_clock(), 1000,
                        "Flight odometry send queue full; discarded %zu older states in this session", dropped);
                }
                int64_t published = 0;
                send(msg, published);
                if (timing_) timing_(rclcpp::Time(msg.header.stamp).nanoseconds(),
                                     0, dequeued, published);
            } catch (const std::exception& e) {
                if (rclcpp::ok())
                    RCLCPP_ERROR_THROTTLE(node_.get_logger(), *node_.get_clock(), 1000, "Flight odometry: %s", e.what());
            }
        }
    }

    bool fresh_sample(uint64_t sample, int64_t now) const
    {
        return now > 0 && sample <= static_cast<uint64_t>(now) && now - sample <= max_age_us_;
    }

    void send(const Raw& m, int64_t& published)
    {
        if (m.header.stamp.sec < 0 || m.header.stamp.nanosec >= 1000000000 || ev_fault_) return;
        const uint64_t sample = static_cast<uint64_t>(rclcpp::Time(m.header.stamp).nanoseconds()) / 1000;
        if (!sample || sample <= last_sample_) return;
        const int64_t now = node_.now().nanoseconds() / 1000;
        if (!fresh_sample(sample, now) || !ev_->get_subscription_count()) return;
        last_sample_ = sample;
        if (Clock::now() >= next_graph_check_) {
            if (node_.count_publishers(ev_->get_topic_name()) > 1) {
                ev_fault_ = true;
                RCLCPP_ERROR(node_.get_logger(), "Multiple PX4 visual odometry publishers; EV output stopped");
                return;
            }
            next_graph_check_ = Clock::now() + std::chrono::milliseconds(100);
        }
        Attitude::ConstSharedPtr attitude;
        {
            std::lock_guard<std::mutex> guard(ref_mutex_);
            if (status_id_.fault || attitude_id_.fault) {
                ev_fault_ = true;
                RCLCPP_ERROR(node_.get_logger(), "PX4 reference source or timestamp changed; restart EV on the ground");
                return;
            }
            if (!status_ || Clock::now() - status_received_ > std::chrono::seconds(2) ||
                difference(now, status_->timestamp) > 2000000) return;
            if (!ev_locked_) {
                if (status_->arming_state != Status::ARMING_STATE_DISARMED) return;
                attitude = nearest_attitude(attitudes_, sample);
            }
        }
        BodyState body;
        if (!body_state(m, imu_from_fc_, body)) return;
        if (!ev_locked_) {
            if (!attitude || body.angular_fc.norm() > .12) return;
            local_from_world_ = flight::initial_local_frd(body, px4_q(attitude->q));
            try {
                reset_counter_ = next_reset_counter(ns_, node_.get_node_base_interface()->get_context()->get_domain_id());
            } catch (const std::exception& e) {
                ev_fault_ = true;
                RCLCPP_ERROR(node_.get_logger(), "Cannot initialize EV origin: %s", e.what());
                return;
            }
            ev_locked_ = true;
            RCLCPP_INFO(node_.get_logger(), "Direct FC-IMU external odometry on %s (reset counter %u)",
                        ev_->get_topic_name(), static_cast<unsigned>(reset_counter_));
        }
        const Transform local_from_flu = local_from_world_ * body.world_from_fc;
        const Quaternion q(local_from_flu.linear() * flight::flu_frd());
        Ev out;
        const int64_t sent = node_.now().nanoseconds() / 1000;
        if (!fresh_sample(sample, sent)) return;
        out.timestamp = static_cast<uint64_t>(sent);
        out.timestamp_sample = sample;
        out.pose_frame = Ev::POSE_FRAME_FRD;
        out.velocity_frame = Ev::VELOCITY_FRAME_BODY_FRD;
        out.position = float_array(local_from_flu.translation());
        out.q = {{static_cast<float>(q.w()), static_cast<float>(q.x()), static_cast<float>(q.y()), static_cast<float>(q.z())}};
        out.velocity = float_array(flight::flu_frd() * body.linear_fc);
        out.angular_velocity = float_array(flight::flu_frd() * body.angular_fc);
        out.position_variance.fill(std::numeric_limits<float>::quiet_NaN());
        out.orientation_variance = out.velocity_variance = out.position_variance;
        out.reset_counter = reset_counter_;
        published = steady_ns();
        ev_->publish(out);
        check_residual(m, local_from_flu.translation(), local_from_flu.linear() * body.linear_fc);
    }

    void check_residual(const Raw& scan, const Vector& position, const Vector& velocity)
    {
        constexpr int64_t window = 100000000;
        const auto stamp = rclcpp::Time(scan.header.stamp).nanoseconds();
        const auto gap = stamp - previous_stamp_;
        // Split a slightly late 10 Hz scan at the exact window boundary.
        if (!previous_stamp_ || gap <= 0 || gap > window * 3 / 2) {
            consistency_ = {};
            consistency_.update_world(stamp, reset_counter_, position, velocity);
            next_boundary_ = stamp + window;
        } else {
            while (next_boundary_ <= stamp) {
                const double f = static_cast<double>(next_boundary_ - previous_stamp_) / gap;
                const auto result = consistency_.update_world(next_boundary_, reset_counter_,
                    previous_position_ + f * (position - previous_position_),
                    previous_velocity_ + f * (velocity - previous_velocity_));
                if (result.exceeds_residual_limit()) {
                    auto request = scan.header;
                    request.stamp = rclcpp::Time(next_boundary_);
                    residual_request_->publish(request);
                }
                next_boundary_ += window;
            }
            if (stamp != next_boundary_ - window)
                consistency_.update_world(stamp, reset_counter_, position, velocity);
        }
        previous_stamp_ = stamp;
        previous_position_ = position;
        previous_velocity_ = velocity;
    }

    lio_output::PositionConsistency consistency_;
    int64_t previous_stamp_{0}, next_boundary_{0};
    Vector previous_position_{Vector::Zero()}, previous_velocity_{Vector::Zero()};
    rclcpp::Publisher<std_msgs::msg::Header>::SharedPtr residual_request_;
    rclcpp::Node& node_;
    size_t depth_;
    std::string ns_;
    Transform imu_from_fc_, local_from_world_;
    bool ev_locked_ = false, ev_fault_ = false;
    uint8_t reset_counter_ = 0;
    uint64_t max_age_us_ = 0, last_sample_ = 0;
    Clock::time_point next_graph_check_{}, status_received_{};
    rclcpp::Publisher<Ev>::SharedPtr ev_;
    rclcpp::CallbackGroup::SharedPtr group_;
    rclcpp::Subscription<Attitude>::SharedPtr attitude_sub_;
    rclcpp::Subscription<Status>::SharedPtr status_sub_;
    std::mutex ref_mutex_;
    Reference status_id_, attitude_id_;
    Status::ConstSharedPtr status_;
    std::deque<Attitude::ConstSharedPtr> attitudes_;
    std::mutex queue_mutex_;
    std::condition_variable wake_;
    std::deque<Raw> queue_;
    TimingSink timing_;
    size_t overwrites_ = 0, reported_overwrites_ = 0;
    bool stopping_ = false;
    std::thread worker_;
};

FlightOdometry::FlightOdometry(rclcpp::Node& node, TimingSink timing) : impl_(new Impl(node, std::move(timing))) {}
FlightOdometry::~FlightOdometry() = default;
void FlightOdometry::submit(const nav_msgs::msg::Odometry& msg) { impl_->submit(msg); }
} // namespace fast_lio
