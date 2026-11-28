#pragma once

#include <Eigen/Geometry>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace fast_lio {
namespace flight {

using Transform = Eigen::Isometry3d;
using Vector = Eigen::Vector3d;
using Quaternion = Eigen::Quaterniond;

inline Eigen::Matrix3d flu_frd()
{
    return Vector(1., -1., -1.).asDiagonal();
}

inline bool valid(const Quaternion& q)
{
    return q.coeffs().allFinite() && std::abs(q.norm() - 1.) <= 0.01;
}

inline Transform fc_from_imu(const std::vector<double>& rotation, const std::vector<double>& translation)
{
    if (rotation.size() != 9 || translation.size() != 3)
        throw std::invalid_argument("flight_output requires a 3x3 IMU-to-FC rotation and a translation");
    Transform result = Transform::Identity();
    result.linear() = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(rotation.data());
    result.translation() = Eigen::Map<const Vector>(translation.data());
    if (!result.matrix().allFinite() || !result.linear().isUnitary(1e-5) ||
        std::abs(result.linear().determinant() - 1.) > 1e-5)
        throw std::invalid_argument("invalid IMU-to-FC mounting transform");
    return result;
}

struct BodyState {
    Transform world_from_fc;
    Vector linear_fc, angular_fc;
};

inline BodyState at_fc(const Transform& world_from_imu, const Vector& velocity_imu,
                       const Vector& rate_imu, const Transform& imu_from_fc)
{
    return {world_from_imu * imu_from_fc,
            imu_from_fc.linear().transpose() * (velocity_imu + rate_imu.cross(imu_from_fc.translation())),
            imu_from_fc.linear().transpose() * rate_imu};
}

inline Transform initial_local_frd(const BodyState& body, const Quaternion& ned_from_frd)
{
    // Local X starts at the FC's initial horizontal heading, with Z down.
    const Vector forward = ned_from_frd * Vector::UnitX();
    const double heading = std::atan2(forward.y(), forward.x());
    Transform local_from_world = Transform::Identity();
    local_from_world.linear() = (Eigen::AngleAxisd(-heading, Vector::UnitZ()) *
        ned_from_frd.normalized() * Quaternion(flu_frd() * body.world_from_fc.linear().transpose())).toRotationMatrix();
    local_from_world.translation() = -(local_from_world.linear() * body.world_from_fc.translation());
    return local_from_world;
}

}  // namespace flight
}  // namespace fast_lio
