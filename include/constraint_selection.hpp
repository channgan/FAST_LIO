#pragma once

#include <Eigen/Core>
#include <vector>

namespace fast_lio {
enum class SelectionMode { All, Uniform, Constraint };
using MatchPositions = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

struct SelectionStats {
    double weakest_all = 0.0;
    double weakest_selected = 0.0;
};

// Empty indices mean keep every row. The same ordered indices apply to H, h and debug clouds.
std::vector<int> selectMatches(const Eigen::MatrixXd& jacobian, const MatchPositions& positions,
                               SelectionMode mode, int budget, double rotation_length,
                               SelectionStats* stats = nullptr);
}
