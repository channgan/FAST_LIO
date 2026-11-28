#include "flight_odometry_geometry.hpp"

#include <cmath>
#include <stdexcept>

using namespace fast_lio::flight;

namespace {
void check(bool condition, const char* description)
{
    if (!condition) throw std::runtime_error(description);
}

void check_mount_and_local_frames()
{
    Transform imu_from_fc = Transform::Identity();
    imu_from_fc.translation() = Vector(.1, 0, 0);
    const auto body = at_fc(Transform::Identity(), Vector::Zero(), Vector(0, 0, 2), imu_from_fc);
    check(body.world_from_fc.translation().isApprox(Vector(.1, 0, 0)), "FC position includes lever arm");
    check(body.linear_fc.isApprox(Vector(0, .2, 0)), "rotation gives the FC a tangential speed");
    check((flu_frd() * body.linear_fc).isApprox(Vector(0, -.2, 0)), "PX4 body-FRD velocity");
    const Quaternion ned_from_frd(Eigen::AngleAxisd(.78, Vector::UnitZ()));
    const Transform local = initial_local_frd(body, ned_from_frd);
    check((local * body.world_from_fc.translation()).isZero(1e-12), "initial FC local position");
    check((local.linear() * body.world_from_fc.linear() * flu_frd()).isIdentity(1e-12),
          "yaw relative to initial heading");
    check((local.linear() * Vector::UnitZ()).isApprox(-Vector::UnitZ()), "up is local FRD negative Z");
}

void check_tilted_mount_and_initial_attitude()
{
    const auto imu_from_fc = Transform(Eigen::AngleAxisd(.42, Vector::UnitY()));
    Transform world_from_imu = Transform::Identity();
    const auto original = at_fc(world_from_imu, Vector::Zero(), Vector::Zero(), imu_from_fc);
    const auto pitch_roll = Eigen::AngleAxisd(.18, Vector::UnitY()) * Eigen::AngleAxisd(-.32, Vector::UnitX());
    const Quaternion ned_from_frd = Eigen::AngleAxisd(.9, Vector::UnitZ()) * pitch_roll;
    const auto frame = initial_local_frd(original, ned_from_frd);
    const Quaternion at_origin(frame.linear() * original.world_from_fc.linear() * flu_frd());
    check(at_origin.angularDistance(pitch_roll) < 1e-12, "local EV axes preserve roll and pitch");
    const Vector shift(.6, -.2, .8);
    world_from_imu.translation() = shift;
    const auto translated = at_fc(world_from_imu, Vector::Zero(), Vector::Zero(), imu_from_fc);
    check((frame * translated.world_from_fc.translation()).isApprox(frame.linear() * shift),
          "EV keeps its origin fixed across motion");
}
}

int main()
{
    check_mount_and_local_frames();
    check_tilted_mount_and_initial_attitude();
}
