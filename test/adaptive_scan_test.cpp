#include "adaptive_scan.hpp"
#include "adaptive_scan_cloud.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
struct Point
{
    double x, y, z, intensity;
};
using Cloud = std::vector<Point>;

void require(bool ok, const char* message)
{
    if (!ok)
        throw std::runtime_error(message);
}

void addFloor(Cloud& scan, int cell, double z)
{
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            scan.push_back({cell * 0.5 + 0.05 + i * 0.1, 0.05 + j * 0.1, z, double(i + j)});
}

void checkPlanarAndSparse()
{
    Cloud input;
    addFloor(input, -1, 0.1);
    input.push_back({0.01, 0.02, 0.03, 4});
    input.push_back({0.02, 0.02, 0.03, 8});
    input.push_back({0.49, 0.49, 0.49, 10});
    input.push_back({std::numeric_limits<double>::quiet_NaN(), 0, 0, 10});
    fast_lio::AdaptiveScanOptions options;
    fast_lio::AdaptiveScanResult scan;
    Cloud representatives;
    fast_lio::buildAdaptiveScan(input, options, representatives, scan);
    require(representatives.size() == 3 && scan.patches.size() == 1, "coarse wall and sparse subcells");
    require(std::abs(representatives[0].x + 0.3) < 1e-10 && representatives[0].intensity == 3,
            "representative mean");
    require(representatives[1].intensity == 6, "sparse subcell mean");
    const auto& patch = scan.patches[0];
    require(patch.support_indices.size() == 16 && patch.support_indices.front() == 0 &&
            patch.support_indices.back() == 15, "unfiltered deskewed support indices");
    require(std::abs(patch.normal.z() + 1.0) < 1e-10 && std::abs(patch.offset - 0.1) < 1e-10,
            "unit normal and offset");
    require(std::abs(patch.max_bound.y() - 0.35) < 1e-10 && std::abs(patch.min_bound.x() + 0.45) < 1e-10,
            "observed extents");
    auto legacy = fast_lio::adaptive_scan_centroids(input, options.leaf_size, options.max_flatness_ratio);
    require(legacy.size() == representatives.size() && legacy[0].z == representatives[0].z, "legacy API");
}

void checkCornerAndOutlier()
{
    Cloud input;
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
        {
            input.push_back({0.03, i * 0.06 + 0.02, j * 0.06 + 0.02, 0});
            input.push_back({i * 0.06 + 0.02, j * 0.06 + 0.02, 0.03, 0});
        }
    fast_lio::AdaptiveScanOptions options;
    fast_lio::AdaptiveScanResult scan;
    Cloud representatives;
    fast_lio::buildAdaptiveScan(input, options, representatives, scan);
    require(representatives.size() == 6 && scan.patches.size() == 4, "split corner into reliable and mixed subcells");
    std::vector<bool> used(input.size());
    for (const auto& patch : scan.patches)
        for (std::size_t i : patch.support_indices)
        {
            require(!used.at(i), "support points assigned uniquely");
            used[i] = true;
        }

    input.clear();
    for (int i = 0; i < 80; ++i)
        addFloor(input, 0, 0.1);
    input.push_back({0.02, 0.02, 0.36, 12});
    fast_lio::buildAdaptiveScan(input, options, representatives, scan);
    require(representatives.size() == 5 && scan.patches.size() == 4, "sparse off-plane return retained");
    require(std::any_of(representatives.begin(), representatives.end(), [](Point p){ return p.intensity == 12; }),
            "outlier subcell representative");
}

void checkCapAndReuse()
{
    Cloud input;
    for (int i = 0; i < 20; ++i)
    {
        addFloor(input, i, 1.1);
        addFloor(input, i, -1.1);
    }
    input.push_back({0.1, 1.5, 0.0, 9});
    fast_lio::AdaptiveScanOptions options;
    options.normal_group_cap = 2;
    fast_lio::AdaptiveScanResult scan;
    Cloud representatives;
    fast_lio::buildAdaptiveScan(input, options, representatives, scan);
    require(scan.normal_groups.size() == 1 && scan.normal_groups[0].before == 40 &&
            scan.normal_groups[0].after == 2, "antipodal walls share one quota");
    require(scan.patches.size() == 40 && representatives.size() == 3, "preserve map support and sparse return");
    std::vector<Eigen::Vector3d> kept;
    for (const auto& patch : scan.patches)
        if (patch.selected_for_matching)
            kept.push_back(patch.centroid);
    require(kept.size() == 2 && std::abs(kept[0].x() - kept[1].x()) > 9 &&
            kept[0].z() * kept[1].z() < 0, "span both walls along the corridor");

    input.clear();
    fast_lio::buildAdaptiveScan(input, options, representatives, scan);
    require(representatives.empty() && scan.patches.empty() && scan.normal_groups.empty(), "frame-local output");
}
}

int main()
{
    checkPlanarAndSparse();
    checkCornerAndOutlier();
    checkCapAndReuse();
    std::cout << "Adaptive scan regressions passed\n";
}
