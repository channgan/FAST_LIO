#pragma once

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fast_lio
{
struct AdaptiveScanOptions
{
    double leaf_size = 0.5;
    int min_plane_points = 6;
    double min_spread_ratio = 0.04;
    double max_flatness_ratio = 0.05;
    double max_residual_ratio = 0.05;
    double normal_angle_deg = 15.0;
    int normal_group_cap = 0;  // Zero keeps all representatives.
};

struct ScanPatch
{
    Eigen::Vector3d normal, centroid, min_bound, max_bound;
    double offset = 0;
    std::vector<std::size_t> support_indices;  // Into the full deskewed scan.
    bool selected_for_matching = true;
};

struct NormalGroupCounts
{
    Eigen::Vector3d normal;
    std::size_t before, after;
};

struct AdaptiveScanResult
{
    std::vector<ScanPatch> patches;  // Includes capped patches for post-update mapping.
    std::vector<NormalGroupCounts> normal_groups;
};

namespace adaptive_detail
{
template <class Point>
Eigen::Vector3d position(const Point& p)
{
    return Eigen::Vector3d(p.x, p.y, p.z);
}

// Also return the centroid for cells whose normal cannot be determined.
template <class Cloud>
bool describeCell(const Cloud& input, const std::vector<std::size_t>& indices,
                  double side, const AdaptiveScanOptions& options,
                  typename Cloud::value_type& representative, ScanPatch& patch)
{
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    double intensity_sum = 0;
    patch.min_bound = patch.max_bound = position(input[indices.front()]);
    for (std::size_t index : indices)
    {
        const auto p = position(input[index]);
        sum += p;
        intensity_sum += input[index].intensity;
        patch.min_bound = patch.min_bound.cwiseMin(p);
        patch.max_bound = patch.max_bound.cwiseMax(p);
    }
    patch.centroid = sum / static_cast<double>(indices.size());
    representative.x = patch.centroid.x();
    representative.y = patch.centroid.y();
    representative.z = patch.centroid.z();
    representative.intensity = intensity_sum / static_cast<double>(indices.size());
    if (indices.size() < static_cast<std::size_t>(options.min_plane_points))
        return false;

    Eigen::Matrix3d scatter = Eigen::Matrix3d::Zero();
    for (std::size_t index : indices)
    {
        const Eigen::Vector3d delta = position(input[index]) - patch.centroid;
        scatter.noalias() += delta * delta.transpose();
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> fit(scatter);
    if (fit.info() != Eigen::Success)
        return false;
    const auto& spread = fit.eigenvalues();
    if (spread[2] <= 0 || spread[1] <= 0 || spread[1] < options.min_spread_ratio * spread[2] ||
        spread[0] > options.max_flatness_ratio * spread[1])
        return false;
    patch.normal = fit.eigenvectors().col(0);
    if (patch.normal.dot(patch.centroid) > 0)
        patch.normal = -patch.normal;
    patch.offset = -patch.normal.dot(patch.centroid);

    // A few off-plane returns can pass the RMS check on a dense wall.
    return std::all_of(indices.begin(), indices.end(), [&](std::size_t i) {
        return std::abs(patch.normal.dot(position(input[i])) + patch.offset) <= options.max_residual_ratio * side;
    });
}

struct NormalGroup
{
    Eigen::Vector3d normal;
    std::vector<std::size_t> patch_indices;
};

// Greedy farthest-point sampling in XYZ deliberately treats the two sides of a
// corridor as different places even though their normals share one direction.
inline void limitNormalGroups(AdaptiveScanResult& scan, const AdaptiveScanOptions& options)
{
    const double cosine = std::cos(options.normal_angle_deg * std::acos(-1.0) / 180.0);
    std::vector<NormalGroup> groups;
    for (std::size_t i = 0; i < scan.patches.size(); ++i)
    {
        const auto& normal = scan.patches[i].normal;
        std::size_t best = groups.size();
        double best_dot = cosine;
        for (std::size_t g = 0; g < groups.size(); ++g)
        {
            const double dot = std::abs(normal.dot(groups[g].normal));
            if (dot >= best_dot)
            {
                best_dot = dot;
                best = g;
            }
        }
        if (best == groups.size())
            groups.push_back({normal, {}});
        groups[best].patch_indices.push_back(i);
    }
    for (const auto& group : groups)
    {
        const auto& indices = group.patch_indices;
        const std::size_t keep = options.normal_group_cap ?
            std::min(indices.size(), static_cast<std::size_t>(options.normal_group_cap)) : indices.size();
        scan.normal_groups.push_back({group.normal, indices.size(), keep});
        if (keep == indices.size())
            continue;
        std::vector<double> nearest(indices.size(), std::numeric_limits<double>::infinity());
        for (auto i : indices)
            scan.patches[i].selected_for_matching = false;

        // Voxel iteration is deterministic. Start at the first occupied cell.
        std::size_t next = 0;
        for (std::size_t n = 0; n < keep; ++n)
        {
            auto& chosen = scan.patches[indices[next]];
            chosen.selected_for_matching = true;
            nearest[next] = -1.0;
            double farthest = -1.0;
            for (std::size_t j = 0; j < indices.size(); ++j)
            {
                if (nearest[j] < 0.0)
                    continue;
                const auto& candidate = scan.patches[indices[j]].centroid;
                nearest[j] = std::min(nearest[j], (candidate - chosen.centroid).squaredNorm());
                if (nearest[j] > farthest)
                {
                    farthest = nearest[j];
                    next = j;
                }
            }
        }
    }
}
}  // namespace adaptive_detail

// Input/output accept a PCL PointCloud or a point vector. Both carry XYZ and intensity.
// Each real input point belongs to at most one reliable patch, irrespective of the cap.
template <class Cloud>
void buildAdaptiveScan(const Cloud& input, const AdaptiveScanOptions& options,
                       Cloud& representatives, AdaptiveScanResult& scan)
{
    if (!std::isfinite(options.leaf_size) || options.leaf_size <= 0 ||
        options.min_plane_points < 3 || !(options.min_spread_ratio > 0 && options.min_spread_ratio <= 1) ||
        !(options.max_flatness_ratio >= 0 && options.max_flatness_ratio < 1) ||
        !(options.max_residual_ratio > 0 && std::isfinite(options.max_residual_ratio)) ||
        !(options.normal_angle_deg > 0 && options.normal_angle_deg <= 90) || options.normal_group_cap < 0)
        throw std::invalid_argument("invalid adaptive scan options");

    using Key = std::array<double, 3>;
    using Indices = std::vector<std::size_t>;
    using Point = typename Cloud::value_type;
    std::map<Key, Indices> cells;
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        const auto p = adaptive_detail::position(input[i]);
        if (p.allFinite())
            cells[{{std::floor(p.x() / options.leaf_size), std::floor(p.y() / options.leaf_size),
                    std::floor(p.z() / options.leaf_size)}}].push_back(i);
    }

    scan.patches.clear();
    scan.normal_groups.clear();
    representatives.clear();
    representatives.reserve(cells.size());
    std::vector<std::size_t> patch_to_rep;
    const auto addPatch = [&](ScanPatch& patch, Indices& indices) {
        patch.support_indices.swap(indices);
        patch_to_rep.push_back(representatives.size());
        scan.patches.push_back(std::move(patch));
    };
    for (auto& cell : cells)
    {
        Point representative{};
        ScanPatch patch;
        if (adaptive_detail::describeCell(input, cell.second, options.leaf_size, options, representative, patch))
        {
            addPatch(patch, cell.second);
            representatives.push_back(representative);
            continue;
        }

        const auto& key = cell.first;
        const Eigen::Vector3d middle((key[0] + 0.5) * options.leaf_size,
                                     (key[1] + 0.5) * options.leaf_size,
                                     (key[2] + 0.5) * options.leaf_size);
        std::array<Indices, 8> children;
        for (auto i : cell.second)
        {
            const auto& p = input[i];
            const unsigned child = (p.x >= middle.x()) + 2 * (p.y >= middle.y()) + 4 * (p.z >= middle.z());
            children[child].push_back(i);
        }
        for (auto& child : children)
        {
            if (child.empty())
                continue;
            if (adaptive_detail::describeCell(input, child, options.leaf_size / 2, options, representative, patch))
                addPatch(patch, child);
            representatives.push_back(representative);
        }
    }

    adaptive_detail::limitNormalGroups(scan, options);
    if (options.normal_group_cap > 0)
    {
        std::size_t patch = 0, write = 0;
        for (std::size_t i = 0; i < representatives.size(); ++i)
        {
            if (patch < patch_to_rep.size() && patch_to_rep[patch] == i)
            {
                const bool keep = scan.patches[patch].selected_for_matching;
                ++patch;
                if (!keep)
                    continue;
            }
            if (write != i)
                representatives[write] = representatives[i];
            ++write;
        }
        representatives.resize(write);
    }
}
}  // namespace fast_lio
