#pragma once

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <vector>

namespace fast_lio
{
namespace detail
{
struct ScanCentroid
{
    Eigen::Vector4d sum = Eigen::Vector4d::Zero();
    std::size_t count = 0;

    template <class Point>
    void add(const Point& point)
    {
        sum += Eigen::Vector4d(point.x, point.y, point.z, point.intensity);
        ++count;
    }

    template <class Point>
    Point point() const
    {
        const Eigen::Vector4d mean = sum / static_cast<double>(count);
        Point result{};
        result.x = mean.x();
        result.y = mean.y();
        result.z = mean.z();
        result.intensity = mean.w();
        return result;
    }
};

template <class Point>
bool has_single_plane(const std::vector<const Point*>& points, const Eigen::Vector3d& center,
                      double leaf_size, double max_plane_ratio)
{
    if (points.size() < 6)
        return false;

    Eigen::Matrix3d scatter = Eigen::Matrix3d::Zero();
    for (const Point* point : points)
    {
        const Eigen::Vector3d offset = Eigen::Vector3d(point->x, point->y, point->z) - center;
        scatter.noalias() += offset * offset.transpose();
    }

    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> fit(scatter);
    if (fit.info() != Eigen::Success)
        return false;
    const auto& spread = fit.eigenvalues();  // Ascending order.
    if (spread[2] <= 0.0 || spread[1] < 0.04 * spread[2] || spread[0] > max_plane_ratio * spread[1])
        return false;

    // Keep a sparse return off the dominant plane, even when its RMS contribution is small.
    const Eigen::Vector3d normal = fit.eigenvectors().col(0);
    const double max_distance = 0.05 * leaf_size;
    return std::all_of(points.begin(), points.end(), [&](const Point* point) {
        const Eigen::Vector3d offset = Eigen::Vector3d(point->x, point->y, point->z) - center;
        return std::abs(normal.dot(offset)) <= max_distance;
    });
}
}  // namespace detail

// Return representatives in the deskewed scan frame, including areas absent from the map.
template <class Point, class Allocator>
std::vector<Point, Allocator> adaptive_scan_centroids(const std::vector<Point, Allocator>& points,
                                                       double coarse_size, double max_plane_ratio)
{
    using Key = std::array<double, 3>;
    std::map<Key, std::vector<const Point*>> cells;
    for (const Point& point : points)
    {
        if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z))
        {
            const Key key = {{std::floor(point.x / coarse_size), std::floor(point.y / coarse_size),
                              std::floor(point.z / coarse_size)}};
            cells[key].push_back(&point);
        }
    }

    std::vector<Point, Allocator> result(points.get_allocator());
    result.reserve(cells.size());
    for (const auto& cell : cells)
    {
        detail::ScanCentroid coarse;
        for (const Point* point : cell.second)
            coarse.add(*point);
        const Eigen::Vector3d center = coarse.sum.head<3>() / static_cast<double>(coarse.count);
        if (detail::has_single_plane(cell.second, center, coarse_size, max_plane_ratio))
        {
            result.push_back(coarse.point<Point>());
            continue;
        }

        const Key& key = cell.first;
        const Eigen::Vector3d middle((key[0] + 0.5) * coarse_size, (key[1] + 0.5) * coarse_size,
                                     (key[2] + 0.5) * coarse_size);
        std::array<detail::ScanCentroid, 8> children{};
        for (const Point* point : cell.second)
        {
            const unsigned child = (point->x >= middle.x()) + 2 * (point->y >= middle.y())
                                                            + 4 * (point->z >= middle.z());
            children[child].add(*point);
        }
        for (const auto& child : children)
        {
            if (child.count > 0)
                result.push_back(child.point<Point>());
        }
    }
    return result;
}
}  // namespace fast_lio
