#pragma once

#include "adaptive_scan.hpp"

namespace fast_lio
{
// Original publication-only API. Use the same representatives as the mapping frontend.
template <class Point, class Allocator>
std::vector<Point, Allocator> adaptive_scan_centroids(const std::vector<Point, Allocator>& points,
                                                       double coarse_size, double max_plane_ratio)
{
    AdaptiveScanOptions options;
    options.leaf_size = coarse_size;
    options.max_flatness_ratio = max_plane_ratio;
    AdaptiveScanResult scan;
    std::vector<Point, Allocator> representatives(points.get_allocator());
    buildAdaptiveScan(points, options, representatives, scan);
    return representatives;
}
}  // namespace fast_lio
