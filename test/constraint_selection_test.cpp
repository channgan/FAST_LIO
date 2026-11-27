#include "constraint_selection.hpp"
#include <algorithm>
#include <stdexcept>

using namespace fast_lio;
void check(bool ok) { if (!ok) throw std::runtime_error("constraint selection regression"); }

int main() {
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(144, 12);
    MatchPositions xyz(h.rows(), 3);
    for (int i = 0; i < h.rows(); ++i) {
        // A corridor with only four returns supporting translation along its axis.
        h(i, i < 4 ? 0 : 1 + i % 5) = 1;
        xyz.row(i) << i % 12, i / 12, i % 2;
    }
    SelectionStats all, constrained;
    check(selectMatches(h, xyz, SelectionMode::All, 24, 1, &all).empty());
    auto selected = selectMatches(h, xyz, SelectionMode::Constraint, 24, 1, &constrained);
    check(selected.size() == 24 && std::is_sorted(selected.begin(), selected.end()));
    check(std::adjacent_find(selected.begin(), selected.end()) == selected.end());
    check(std::count_if(selected.begin(), selected.end(), [](int i) { return i < 4; }) >= 2);
    check(all.weakest_all == 4 && constrained.weakest_selected >= 2);
    auto uniform = selectMatches(h, xyz, SelectionMode::Uniform, 24, 1);
    check(uniform.size() == selected.size());
    h.col(0).setZero();
    selectMatches(h, xyz, SelectionMode::Constraint, 24, 1, &constrained);
    check(constrained.weakest_selected == 0);
    check(selectMatches(h, xyz, SelectionMode::Constraint, 144, 1).empty());
}
