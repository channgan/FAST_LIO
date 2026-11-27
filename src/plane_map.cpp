#include "plane_map.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fast_lio {
namespace {
constexpr double cos20 = 0.9396926207859084;
constexpr double cos8 = 0.9902680687415704;
constexpr double sin8 = 0.13917310096006544;

bool fitPlane(const PlaneMoments& m, double edge, const AdaptiveScanOptions& options, MapPlane& out) {
    if (m.count < std::size_t(options.min_plane_points) || m.mixed_normals) return false;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(m.scatter);
    const auto& lambda = eig.eigenvalues();
    if (eig.info() != Eigen::Success || lambda.z() <= 0.0 ||
        lambda.y() <= 0 || lambda.y() < options.min_spread_ratio * lambda.z() ||
        lambda.x() > options.max_flatness_ratio * lambda.y() ||
        lambda.x() > m.count * std::pow(options.max_residual_ratio * edge, 2)) return false;
    out.normal = eig.eigenvectors().col(0);
    if (std::abs(m.reference_normal.dot(out.normal)) < cos20) return false;
    out.offset = -out.normal.dot(m.mean);
    out.points = m;
    return true;
}

// Incrementally span the directions that differ by more than angular noise.
// Even a single real end-wall normal adds an axis in a long corridor, while
// small variations among nearly parallel normals cannot manufacture one.
void addDirection(Eigen::Matrix3d& projector, const Eigen::Vector3d& unit) {
    const Eigen::Vector3d remainder = unit - projector * unit;
    if (remainder.norm() <= sin8) return;
    const Eigen::Vector3d next = remainder.normalized();
    projector.noalias() += next * next.transpose();
}

Eigen::Vector3d solveObserved(const Eigen::Matrix3d& information, const Eigen::Vector3d& rhs,
                              const Eigen::Matrix3d& projector) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(projector * information * projector);
    Eigen::Vector3d result = Eigen::Vector3d::Zero();
    for (int i = 0; i < 3; ++i) {
        if (eig.eigenvalues()[i] <= eig.eigenvalues().maxCoeff() * 1e-9) continue;
        const auto direction = eig.eigenvectors().col(i);
        result += direction * (direction.dot(rhs) / eig.eigenvalues()[i]);
    }
    return result;
}
}  // namespace

void PlaneMoments::add(const Eigen::Vector3d& p, const Eigen::Vector3d& normal) {
    if (count == 0) {
        mean = lower = upper = p;
        reference_normal = normal;
        count = 1;
        return;
    }
    ++count;
    const Eigen::Vector3d delta = p - mean;
    mean += delta / double(count);
    scatter.noalias() += delta * (p - mean).transpose();
    lower = lower.cwiseMin(p);
    upper = upper.cwiseMax(p);
    mixed_normals |= std::abs(reference_normal.dot(normal)) < cos20;
}

void PlaneMoments::merge(const PlaneMoments& other) {
    if (!other.count) return;
    if (!count) {
        *this = other;
        return;
    }
    const auto combined = count + other.count;
    const Eigen::Vector3d delta = other.mean - mean;
    scatter += other.scatter + (double(count) * other.count / combined) * delta * delta.transpose();
    mean += (double(other.count) / combined) * delta;
    lower = lower.cwiseMin(other.lower);
    upper = upper.cwiseMax(other.upper);
    const double alignment = reference_normal.dot(other.reference_normal);
    mixed_normals |= other.mixed_normals || std::abs(alignment) < cos20;
    reference_normal = (double(count) * reference_normal +
        (alignment < 0 ? -double(other.count) : double(other.count)) * other.reference_normal).normalized();
    count = combined;
}

LocalPlaneMap::LocalPlaneMap(double leaf_size, const AdaptiveScanOptions& options)
    : leaf_size_(leaf_size), options_(options) {
    if (!std::isfinite(leaf_size) || leaf_size <= 0)
        throw std::invalid_argument("plane map leaf size must be finite and positive");
    upper_.setConstant(std::numeric_limits<double>::infinity());
    lower_ = -upper_;
}

std::size_t LocalPlaneMap::KeyHash::operator()(const Key& key) const {
    std::size_t h = 0;
    for (int x : key) h ^= std::hash<int>{}(x) + 0x9e3779b9U + (h << 6) + (h >> 2);
    return h;
}

LocalPlaneMap::Key LocalPlaneMap::keyOf(const Eigen::Vector3d& p, double edge) const {
    return {{int(std::floor(p.x() / edge)), int(std::floor(p.y() / edge)), int(std::floor(p.z() / edge))}};
}

bool LocalPlaneMap::intersectsWindow(const Key& key, double edge) const {
    for (int i = 0; i < 3; ++i) {
        if ((key[i] + 1.0) * edge <= lower_[i] || key[i] * edge > upper_[i]) return false;
    }
    return true;
}

void LocalPlaneMap::trimOctants(const Key& parent, Cell& cell) {
    auto& octants = cell.octants;
    auto outside = [&](const Cell::Octant& octant) {
        Key key;
        for (int i = 0; i < 3; ++i) key[i] = 2 * parent[i] + ((octant.id >> i) & 1);
        return !intersectsWindow(key, leaf_size_ / 2);
    };
    octants.erase(std::remove_if(octants.begin(), octants.end(), outside), octants.end());
}

void LocalPlaneMap::mergeSurface(Cell::Octant& bin, const PlaneMoments& incoming) {
    const double thickness = leaf_size_ * options_.max_residual_ratio;
    for (auto& known : bin.surfaces) {
        const Eigen::Vector3d delta = incoming.mean - known.mean;
        if (std::abs(known.reference_normal.dot(incoming.reference_normal)) >= cos20 &&
            std::abs(known.reference_normal.dot(delta)) <= thickness &&
            std::abs(incoming.reference_normal.dot(delta)) <= thickness) {
            known.merge(incoming);
            return;
        }
    }
    if (bin.surfaces.size() < 2) {
        bin.surfaces.push_back(incoming);
        return;
    }
    // Bound state at corners. Unrepresented returns count against declaring a
    // reliable dominant surface but cannot contaminate its centroid or normal.
    auto weakest = std::min_element(bin.surfaces.begin(), bin.surfaces.end(),
        [](const PlaneMoments& a, const PlaneMoments& b) { return a.count < b.count; });
    if (incoming.count > weakest->count) {
        bin.other += weakest->count;
        *weakest = incoming;
    } else {
        bin.other += incoming.count;
    }
}

void LocalPlaneMap::fit(const Key& key, Cell& cell) {
    plane_count_ -= cell.planes.size();
    cell.planes.clear();
    PlaneMoments all;
    bool single_surface = true;
    for (const auto& octant : cell.octants) {
        if (octant.surfaces.size() > 1 || octant.other) single_surface = false;
        all.merge(octant.surfaces.front());
    }
    MapPlane plane;
    cell.coarse = single_surface && fitPlane(all, leaf_size_, options_, plane);
    if (cell.coarse) {
        for (const auto& octant : cell.octants) {
            const auto& points = octant.surfaces.front();
            if (std::abs(plane.normal.dot(points.mean) + plane.offset) > leaf_size_ * options_.max_residual_ratio) {
                cell.coarse = false;
                break;
            }
        }
    }
    if (cell.coarse) {
        cell.planes.push_back(plane);
    } else {
        trimOctants(key, cell);
        for (const auto& octant : cell.octants) {
            const PlaneMoments* strongest = &octant.surfaces.front();
            std::size_t total = octant.other;
            for (const auto& surface : octant.surfaces) {
                total += surface.count;
                if (surface.count > strongest->count) strongest = &surface;
            }
            // Abstain on mixed fine cells. Repeated consistent observations can
            // recover from isolated bad associations without fusing their points.
            if (double(strongest->count) >= 0.9 * total &&
                fitPlane(*strongest, leaf_size_ / 2, options_, plane)) cell.planes.push_back(plane);
        }
    }
    plane_count_ += cell.planes.size();
}

void LocalPlaneMap::setWindow(const Eigen::Vector3d& lower, const Eigen::Vector3d& upper) {
    lower_ = lower;
    upper_ = upper;
    for (auto it = cells_.begin(); it != cells_.end();) {
        auto& cell = it->second;
        if (!intersectsWindow(it->first, leaf_size_)) {
            plane_count_ -= cell.planes.size();
            it = cells_.erase(it);
            continue;
        }
        // A coarse leaf intersecting a boundary stays visible. Split cells have
        // independently removable fine leaves, including not-yet-reliable ones.
        if (!cell.coarse) {
            const auto before = cell.octants.size();
            trimOctants(it->first, cell);
            if (cell.octants.size() != before) fit(it->first, cell);
            if (cell.octants.empty()) {
                it = cells_.erase(it);
                continue;
            }
        }
        ++it;
    }
}

void LocalPlaneMap::integrate(const std::vector<ScanPatch>& scan,
                              const std::function<Eigen::Vector3d(std::size_t)>& point_at,
                              const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation) {
    // Assign corrected supports individually. In particular a rotated scan patch
    // may straddle several world voxels. Every support lands in precisely one bin.
    std::unordered_map<Key, Cell*, KeyHash> changed;
    std::unordered_map<Key, PlaneMoments, KeyHash> bins;
    for (const auto& patch : scan) {
        bins.clear();
        const Eigen::Vector3d normal = rotation * patch.normal;
        for (auto i : patch.support_indices) {
            const Eigen::Vector3d world = rotation * point_at(i) + translation;
            const auto fine = keyOf(world, leaf_size_ / 2);
            if (intersectsWindow(fine, leaf_size_ / 2)) bins[fine].add(world, normal);
        }
        for (const auto& bin : bins) {
            Key parent;
            int id = 0;
            for (int i = 0; i < 3; ++i) {
                // Floor division also places negative odd indices in the right parent.
                parent[i] = int(std::floor(0.5 * bin.first[i]));
                id |= (bin.first[i] - 2 * parent[i]) << i;
            }
            auto& cell = cells_[parent];
            auto oct = std::find_if(cell.octants.begin(), cell.octants.end(),
                                    [id](const Cell::Octant& o) { return o.id == id; });
            if (oct == cell.octants.end()) {
                cell.octants.push_back({id, {}, 0});
                oct = cell.octants.end() - 1;
            }
            mergeSurface(*oct, bin.second);
            changed[parent] = &cell;
        }
    }
    for (auto& entry : changed) fit(entry.first, *entry.second);
}

const MapPlane* LocalPlaneMap::nearest(const ScanPatch& patch, const Eigen::Matrix3d& rotation,
                                       const Eigen::Vector3d& translation, double cosine,
                                       double max_distance) const {
    const Eigen::Vector3d center = rotation * patch.centroid + translation;
    const Eigen::Vector3d normal = rotation * patch.normal;
    // Transform the support AABB by center and half-extents. Its center need not be the centroid.
    const Eigen::Vector3d mid = rotation * ((patch.min_bound + patch.max_bound) * 0.5) + translation;
    const Eigen::Vector3d half = rotation.cwiseAbs() * ((patch.max_bound - patch.min_bound) * 0.5);
    const Eigen::Vector3d lo = mid - half, hi = mid + half;
    const double margin = leaf_size_ * 0.35;
    const auto begin = keyOf(lo.array() - margin, leaf_size_);
    const auto end = keyOf(hi.array() + margin, leaf_size_);
    const MapPlane* best = nullptr;
    double score = std::numeric_limits<double>::infinity();
    for (int x = begin[0]; x <= end[0]; ++x)
        for (int y = begin[1]; y <= end[1]; ++y)
            for (int z = begin[2]; z <= end[2]; ++z) {
                auto it = cells_.find({{x,y,z}});
                if (it == cells_.end()) continue;
                for (const auto& plane : it->second.planes) {
                    const double alignment = std::abs(normal.dot(plane.normal));
                    const double distance = std::abs(plane.normal.dot(center) + plane.offset);
                    if (alignment < cosine || distance > max_distance) continue;
                    const Eigen::Vector3d gap = (lo - plane.points.upper).cwiseMax(plane.points.lower - hi).cwiseMax(0.);
                    if (gap.norm() > margin) continue;
                    const double cost = distance / leaf_size_ + (1. - alignment) +
                        (gap.norm() + 0.05 * (center - plane.points.mean).norm()) / leaf_size_;
                    if (cost < score) { best = &plane; score = cost; }
                }
            }
    return best;
}

std::vector<PlaneMatch> LocalPlaneMap::associate(const std::vector<ScanPatch>& scan,
                                                const Eigen::Matrix3d& rotation,
                                                const Eigen::Vector3d& translation) const {
    std::vector<PlaneMatch> proposals;
    for (std::size_t i = 0; i < scan.size(); ++i) {
        if (!scan[i].selected_for_matching) continue;
        const auto* plane = nearest(scan[i], rotation, translation, cos20, 0.35 * leaf_size_);
        if (plane) proposals.push_back({i, plane});
    }
    if (proposals.size() < 2) return {};

    // Fit ONE correction to the candidate combination. Rotation comes from
    // signed normal alignment, translation from plane offsets after rotation.
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
    Eigen::Matrix3d rotation_axes = Eigen::Matrix3d::Zero(), translation_axes = Eigen::Matrix3d::Zero();
    for (const auto& pair : proposals) {
        const Eigen::Vector3d n = rotation * scan[pair.scan_index].normal;
        Eigen::Vector3d target = pair.target->normal;
        if (target.dot(n) < 0.) target = -target;
        const Eigen::Vector3d tangent = n.unitOrthogonal();
        addDirection(rotation_axes, tangent);
        addDirection(rotation_axes, n.cross(tangent));
        addDirection(translation_axes, target);
        H += Eigen::Matrix3d::Identity() - n * n.transpose();
        rhs += n.cross(target);
    }
    const Eigen::Vector3d omega = solveObserved(H, rhs, rotation_axes);
    const double angle = omega.norm();
    if (angle > 0.35) return {};
    Eigen::Matrix3d fitted_rotation = rotation;
    if (angle > 1e-8) fitted_rotation = Eigen::AngleAxisd(angle, omega / angle) * rotation;

    Eigen::Vector3d shift = Eigen::Vector3d::Zero();
    // A minority of duplicate door-frame matches may want a different offset.
    // Huber reweighting keeps that minority from shifting the shared solution
    // halfway between competing surfaces and invalidating all observations.
    for (int iteration = 0; iteration < 4; ++iteration) {
        H.setZero();
        rhs.setZero();
        for (const auto& pair : proposals) {
            const auto& target = *pair.target;
            const Eigen::Vector3d p = fitted_rotation * scan[pair.scan_index].centroid + translation;
            const double error = target.normal.dot(p) + target.offset;
            const double deviation = std::abs(error + target.normal.dot(shift));
            const double weight = iteration == 0 || deviation <= leaf_size_ * 0.06
                ? 1.0 : leaf_size_ * 0.06 / deviation;
            H += weight * target.normal * target.normal.transpose();
            rhs -= weight * target.normal * error;
        }
        const Eigen::Vector3d next = solveObserved(H, rhs, translation_axes);
        const double change = (next - shift).norm();
        shift = next;
        if (change < 1e-6) break;
    }
    if (shift.norm() > leaf_size_) return {};
    std::vector<PlaneMatch> matches;
    for (const auto& pair : proposals) {
        const auto* consistent = nearest(scan[pair.scan_index], fitted_rotation, translation + shift,
                                         cos8, leaf_size_ * 0.12);
        if (consistent) matches.push_back({pair.scan_index, consistent});
    }
    if (matches.size() < 2) matches.clear();
    return matches;
}

void LocalPlaneMap::visit(const std::function<void(const MapPlane&)>& visitor) const {
    for (const auto& cell : cells_)
        for (const auto& plane : cell.second.planes) visitor(plane);
}
}  // namespace fast_lio
