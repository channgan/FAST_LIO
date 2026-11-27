#pragma once

#include "adaptive_scan.hpp"
#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <functional>
#include <unordered_map>
#include <vector>

namespace fast_lio {

// Sufficient statistics in world coordinates. The extrema bound actual returns.
struct PlaneMoments {
    std::size_t count = 0;
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    Eigen::Matrix3d scatter = Eigen::Matrix3d::Zero();
    Eigen::Vector3d lower = Eigen::Vector3d::Zero(), upper = Eigen::Vector3d::Zero();
    Eigen::Vector3d reference_normal = Eigen::Vector3d::Zero();
    bool mixed_normals = false;
    void add(const Eigen::Vector3d& p, const Eigen::Vector3d& normal);
    void merge(const PlaneMoments& other);
};

struct MapPlane {
    PlaneMoments points;
    Eigen::Vector3d normal;
    double offset;
};

struct PlaneMatch {
    std::size_t scan_index;
    // The historical map is read-only for the entire iterated update.
    const MapPlane* target;
};

class LocalPlaneMap {
public:
    explicit LocalPlaneMap(double leaf_size = 0.5, const AdaptiveScanOptions& options = {});
    void setWindow(const Eigen::Vector3d& lower, const Eigen::Vector3d& upper);
    void integrate(const std::vector<ScanPatch>& scan,
                   const std::function<Eigen::Vector3d(std::size_t)>& point_at,
                   const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation);
    std::vector<PlaneMatch> associate(const std::vector<ScanPatch>& scan,
                                      const Eigen::Matrix3d& rotation,
                                      const Eigen::Vector3d& translation) const;
    std::size_t size() const { return plane_count_; }
    void visit(const std::function<void(const MapPlane&)>& visitor) const;

private:
    using Key = std::array<int, 3>;
    struct KeyHash { std::size_t operator()(const Key& key) const; };
    struct Cell {
        // Retain fine statistics so a later corner can split a previously flat cell.
        struct Octant {
            int id;
            // Incompatible returns never enter the same surface statistics.
            std::vector<PlaneMoments> surfaces;
            std::size_t other = 0;
        };
        std::vector<Octant> octants;
        std::vector<MapPlane> planes;
        bool coarse = false;
    };
    using Cells = std::unordered_map<Key, Cell, KeyHash>;
    Key keyOf(const Eigen::Vector3d& point, double edge) const;
    bool intersectsWindow(const Key& key, double edge) const;
    void mergeSurface(Cell::Octant& bin, const PlaneMoments& incoming);
    void fit(const Key& key, Cell& cell);
    void trimOctants(const Key& parent, Cell& cell);
    const MapPlane* nearest(const ScanPatch& patch, const Eigen::Matrix3d& rotation,
                            const Eigen::Vector3d& translation, double cosine,
                            double max_distance) const;
    double leaf_size_;
    AdaptiveScanOptions options_;
    Cells cells_;
    std::size_t plane_count_ = 0;
    Eigen::Vector3d lower_, upper_;
};
}  // namespace fast_lio
