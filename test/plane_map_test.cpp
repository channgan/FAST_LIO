#include "adaptive_scan.hpp"
#include "plane_map.hpp"

#include <Eigen/Geometry>
#include <stdexcept>
#include <string>
#include <vector>

using Vec = Eigen::Vector3d;
using Mat = Eigen::Matrix3d;
using fast_lio::LocalPlaneMap;
using fast_lio::ScanPatch;

void require(bool ok, const std::string& failure) {
    if (!ok) throw std::runtime_error(failure);
}

struct Scan {
    std::vector<ScanPatch> patches;
    std::vector<Vec> points;

    // A 4 x 4 square of actual supports on an axis-aligned plane.
    void add(const Vec& origin, int normal_axis, double spacing = 0.035) {
        ScanPatch p;
        p.normal = Vec::Unit(normal_axis);
        p.centroid.setZero();
        p.min_bound = p.max_bound = origin;
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b) {
                Vec point = origin;
                point[(normal_axis + 1) % 3] += a * spacing;
                point[(normal_axis + 2) % 3] += b * spacing;
                p.support_indices.push_back(points.size());
                points.push_back(point);
                p.centroid += point / 16.;
                p.min_bound = p.min_bound.cwiseMin(point);
                p.max_bound = p.max_bound.cwiseMax(point);
            }
        p.offset = -p.normal.dot(p.centroid);
        patches.push_back(p);
    }

    void integrate(LocalPlaneMap& map, const Mat& R = Mat::Identity(), const Vec& t = Vec::Zero()) {
        map.integrate(patches, [&](std::size_t i) { return points.at(i); }, R, t);
    }
};

int main() {
    // Negative cells, rotating/translating a packet over multiple world leaves,
    // exact counts and scatter after two independent integrations.
    LocalPlaneMap flat;
    Scan wall;
    wall.add(Vec(-0.55, 0.03, 0.10), 2, 0.04);
    wall.patches[0].selected_for_matching = false;  // Map integration is uncapped.
    const Mat R = Eigen::AngleAxisd(0.06, Vec::UnitZ()).toRotationMatrix();
    const Vec t(0.01, 0.01, 0.1);
    wall.integrate(flat, R, t);
    wall.integrate(flat, R, t);
    require(flat.size() == 2, "corrected supports should populate two negative world cells");
    std::size_t total = 0;
    flat.visit([&](const fast_lio::MapPlane& leaf) {
        std::vector<Vec> expected;
        for (const auto& original : wall.points) {
            Vec p = R * original + t;
            if (std::floor(p.x() / 0.5) == std::floor(leaf.points.mean.x() / 0.5)) expected.push_back(p);
        }
        require(expected.size() * 2 == leaf.points.count, "support was lost or counted twice");
        Vec mean = Vec::Zero();
        for (const auto& p : expected) mean += p / expected.size();
        require((mean - leaf.points.mean).norm() < 1e-12, "merged centroid");
        Mat scatter = Mat::Zero();
        for (const auto& p : expected) scatter += 2. * (p - mean) * (p - mean).transpose();
        require((scatter - leaf.points.scatter).norm() < 1e-12, "merged covariance cross term");
        require(std::abs(leaf.normal.dot(mean) + leaf.offset) < 1e-9, "plane offset");
        total += leaf.points.count;
    });
    require(total == 2 * wall.points.size(), "all scan supports should be mapped");
    flat.setWindow(Vec(-0.48, -1., -1.), Vec(3., 1., 1.));
    require(flat.size() == 1, "remove only fully out-of-window coarse leaves");

    // A new surface splits a coarse historical plane, and the fine leaves
    // retain their own statistics and lifetime at the sliding window boundary.
    LocalPlaneMap corner;
    Scan first;
    first.add(Vec(0.04, 0.04, 0.08), 2);
    first.integrate(corner);
    require(corner.size() == 1, "planar bootstrap");
    Scan crossing;
    crossing.add(Vec(0.32, 0.30, 0.30), 0);
    crossing.integrate(corner);
    require(corner.size() == 2, "orthogonal surfaces in one coarse cell must split");
    corner.setWindow(Vec(0.25, -1, -1), Vec(1, 1, 1));
    require(corner.size() == 1, "evict completely out-of-window fine leaf");
    corner.visit([](const fast_lio::MapPlane& p) {
        require(p.points.count == 16 && std::abs(p.normal.x()) > .99, "surviving fine geometry");
    });
    corner.setWindow(Vec(3,3,3), Vec(5,5,5));
    require(corner.size() == 0, "empty map after whole-window shift");
    first.integrate(corner);
    require(corner.size() == 0, "out-of-window supports cannot resurrect a removed index");

    // Competing surfaces in the SAME fine voxel temporarily make it unusable.
    // Coherent observations later recover the original surface without averaging
    // any of the perpendicular returns into its plane offset or covariance.
    LocalPlaneMap mixed;
    Scan one, spurious;
    one.add(Vec(.03,.03,.08), 2);
    spurious.add(Vec(.08,.03,.03), 0);
    one.integrate(mixed);
    spurious.integrate(mixed);
    require(mixed.size() == 0, "fine cell with two equal surfaces should abstain");
    for (int i = 0; i < 9; ++i) one.integrate(mixed);
    require(mixed.size() == 1, "fine cell can recover from a transient conflicting observation");
    mixed.visit([](const fast_lio::MapPlane& p) {
        require(p.points.count == 160 && std::abs(p.normal.z()) > .99999,
                "competing surface must not contaminate recovered moments");
    });

    // Shared-transform test. All four associations fit the broad local gates.
    // Three parallel observations agree on a translation, the fourth requires
    // a conflicting offset. A single wall normal leaves tangential motion free.
    LocalPlaneMap corridor;
    Scan reference, observed;
    for (int i = 0; i < 4; ++i) {
        reference.add(Vec(i + .04, .04, .25), 2);
        observed.add(Vec(i + .04, .04, i == 3 ? .13 : .34), 2);
        observed.patches.back().normal = -Vec::UnitZ();
        observed.patches.back().offset = -observed.patches.back().normal.dot(observed.patches.back().centroid);
    }
    reference.integrate(corridor);
    const auto matches = corridor.associate(observed.patches, Mat::Identity(), Vec::Zero());
    require(matches.size() == 3, "common rigid motion should reject the incompatible fourth pair");
    for (const auto& pair : matches) require(pair.scan_index < 3, "outlier matched");
    require(corridor.associate(observed.patches, Mat::Identity(), Vec(0, 3, 0)).empty(),
            "parallel planes without spatial overlap cannot match");

    // Two nearly parallel normals differ by 3 degrees. Their centimeter-scale
    // offset errors should never imply meters of tangential displacement.
    LocalPlaneMap almost_parallel;
    Scan base, tilted_base;
    base.add(Vec(.04,.04,.25), 2);
    tilted_base.add(Vec(.04,.04,.25), 2);
    base.integrate(almost_parallel);
    const Mat small_tilt = Eigen::AngleAxisd(.05, Vec::UnitY()).toRotationMatrix();
    tilted_base.integrate(almost_parallel, small_tilt, Vec(1,0,0));
    Scan slightly_noisy;
    slightly_noisy.add(Vec(.04,.04,.27), 2);
    slightly_noisy.add(Vec(1.04,.04,.22), 2);
    require(almost_parallel.associate(slightly_noisy.patches, Mat::Identity(), Vec::Zero()).size() == 2,
            "angular noise cannot invent a large lateral shift");

    // A current scan may see a historical wall as multiple smaller patches.
    Scan split;
    split.add(Vec(.04,.04,.25), 2, .015);
    split.add(Vec(.10,.10,.25), 2, .015);
    const auto many_to_one = corridor.associate(split.patches, Mat::Identity(), Vec::Zero());
    require(many_to_one.size() == 2 && many_to_one[0].target == many_to_one[1].target,
            "segmentation changes should allow many-to-one correspondences");
    const Mat tilted = Eigen::AngleAxisd(.09, Vec::UnitX()).toRotationMatrix();
    require(corridor.associate(split.patches, tilted, Vec(.01,.01,.02)).size() == 2,
            "consistent common rotation and translation with an IMU prediction error");

    LocalPlaneMap room;
    Scan axes;
    axes.add(Vec(.20,.04,.04), 0);
    axes.add(Vec(1.04,.20,.04), 1);
    axes.add(Vec(2.04,.04,.20), 2);
    axes.integrate(room);
    const Mat prior_R = Eigen::AngleAxisd(.04, Vec(1, 1, 1).normalized()).toRotationMatrix();
    const auto room_matches = room.associate(axes.patches, prior_R, Vec(.05, -.04, .03));
    require(room_matches.size() == 3, "common six-degree-of-freedom pose across independent planes");
    for (const auto& pair : room_matches)
        require(std::abs(pair.target->normal.dot(axes.patches[pair.scan_index].normal)) > .9999,
                "orthogonal surfaces must match their own orientation");

    // No mutation or self-matching during association.
    require(corridor.size() == 4, "immutable historical targets during iterative correction");
}
