#include "constraint_selection.hpp"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <limits>

namespace fast_lio {
namespace {
using PoseRowMatrix = Eigen::Matrix<double, Eigen::Dynamic, 6>;
using Information = Eigen::Matrix<double, 6, 6>;

// Spatially order the cloud along the axis of greatest spread, subdividing each
// half in turn. Equally spaced ranks then cover occupied regions of the scan.
void spatialOrder(const MatchPositions& xyz, std::vector<int>& order, int begin, int end) {
    if (end - begin < 2) return;
    Eigen::RowVector3d low = xyz.row(order[begin]), high = low;
    for (int i = begin + 1; i < end; ++i) {
        low = low.cwiseMin(xyz.row(order[i]));
        high = high.cwiseMax(xyz.row(order[i]));
    }
    Eigen::Index axis;
    (high - low).maxCoeff(&axis);
    const int mid = begin + (end - begin) / 2;
    std::nth_element(order.begin() + begin, order.begin() + mid, order.begin() + end,
        [&](int a, int b) { return xyz(a, axis) == xyz(b, axis) ? a < b : xyz(a, axis) < xyz(b, axis); });
    spatialOrder(xyz, order, begin, mid);
    spatialOrder(xyz, order, mid, end);
}
}

std::vector<int> selectMatches(const Eigen::MatrixXd& jacobian, const MatchPositions& positions,
                               SelectionMode mode, int budget, double rotation_length,
                               SelectionStats* stats) {
    const int count = jacobian.rows();
    const bool reduce = mode != SelectionMode::All && budget < count;
    if (!reduce && !stats) return {};
    std::vector<int> selected;
    if (reduce) {
        std::vector<int> order(count);
        std::iota(order.begin(), order.end(), 0);
        spatialOrder(positions, order, 0, count);
        const int spatial_quota = mode == SelectionMode::Uniform ? budget : std::max(1, budget / 3);
        selected.reserve(budget);
        for (int i = 0; i < spatial_quota; ++i) {
            const auto rank = (2LL * i + 1) * count / (2LL * spatial_quota);
            selected.push_back(order[rank]);
        }
        if (mode == SelectionMode::Uniform && !stats) {
            std::sort(selected.begin(), selected.end());
            return selected;
        }
    }

    // One radian of error corresponds to rotation_length metres at this lever arm.
    // Scale only the scoring matrix. The filter receives the physical Jacobian.
    PoseRowMatrix scaled = jacobian.leftCols<6>();
    scaled.rightCols<3>() /= rotation_length;
    Eigen::SelfAdjointEigenSolver<Information> spectrum(scaled.transpose() * scaled);
    const auto eigenvalues = spectrum.eigenvalues().cwiseMax(0.0).eval();
    if (reduce && mode == SelectionMode::Constraint) {
        PoseRowMatrix contribution = (scaled * spectrum.eigenvectors()).array().square().matrix();
        std::array<std::vector<int>, 6> ranking;
        std::array<size_t, 6> cursor{};
        std::vector<bool> used(count, false);
        Eigen::Matrix<double, 1, 6> retained = Eigen::Matrix<double, 1, 6>::Zero();
        for (int i : selected) {
            used[i] = true;
            retained += contribution.row(i);
        }
        for (int d = 0; d < 6; ++d) {
            if (eigenvalues[d] <= 1e-12 * eigenvalues[5]) continue;
            auto& order = ranking[d];
            order.resize(count);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return contribution(a,d) == contribution(b,d) ? a < b : contribution(a,d) > contribution(b,d);
            });
        }
        while (static_cast<int>(selected.size()) < budget) {
            int direction = -1;
            double lowest = std::numeric_limits<double>::infinity();
            for (int d = 0; d < 6; ++d) {
                auto& pos = cursor[d];
                const auto& order = ranking[d];
                while (pos < order.size() && used[order[pos]]) ++pos;
                if (pos == order.size() || contribution(order[pos], d) == 0.0) continue;
                // Weak directions receive up to twice the target retention fraction.
                const double weight = 2.0 - eigenvalues[d] / eigenvalues[5];
                const double fraction = retained[d] / (weight * eigenvalues[d]);
                if (fraction < lowest) { lowest = fraction; direction = d; }
            }
            if (direction < 0) break;
            const int i = ranking[direction][cursor[direction]++];
            selected.push_back(i);
            used[i] = true;
            retained += contribution.row(i);
        }
        for (int i = 0; static_cast<int>(selected.size()) < budget; ++i)
            if (!used[i]) selected.push_back(i);
    }
    if (stats) {
        stats->weakest_all = eigenvalues[0];
        stats->weakest_selected = eigenvalues[0];
        if (reduce) {
            Information information = Information::Zero();
            for (int i : selected) information.noalias() += scaled.row(i).transpose() * scaled.row(i);
            stats->weakest_selected = std::max(0.0,
                Eigen::SelfAdjointEigenSolver<Information>(information, Eigen::EigenvaluesOnly).eigenvalues()[0]);
        }
    }
    std::sort(selected.begin(), selected.end());
    return selected;
}
}
