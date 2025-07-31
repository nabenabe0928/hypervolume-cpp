#include <algorithm>
#include <cassert>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

using std::vector;
namespace py = pybind11;

inline int _pack_pareto_sols(
    double* sorted_loss_values,
    int* nondominated_indices_buf,
    const int& n_objectives,
    const int end_row_index
) {
    // No consideration of duplicated Pareto solutions.
    auto value_at = [&](int i, int j)->double& {return sorted_loss_values[i * n_objectives + j];};
    std::iota(nondominated_indices_buf, nondominated_indices_buf + end_row_index, 0);
    int n_remaining = end_row_index;
    int head_row_index = 0;
    while (n_remaining > 0) {
        const int new_nondominated_index = nondominated_indices_buf[0];
        int nondominated_count = 0;
        for (int i = 1; i < n_remaining; ++i) {
            const int& idx = nondominated_indices_buf[i];
            bool is_nondominated = false;
            for (int j = 1; j < n_objectives; ++j) {
                if (value_at(idx, j) < value_at(new_nondominated_index, j)) {
                    is_nondominated = true;
                    break;
                }
            }
            if (is_nondominated) {
                nondominated_indices_buf[nondominated_count++] = idx;
            }
        }
        std::swap_ranges(
            const_cast<double*>(sorted_loss_values + (head_row_index * n_objectives)),
            const_cast<double*>(sorted_loss_values + ((head_row_index + 1) * n_objectives)),
            const_cast<double*>(sorted_loss_values + (new_nondominated_index * n_objectives))
        );
        ++head_row_index;
        n_remaining = nondominated_count;
    }
    return head_row_index; // Return the number of Pareto solutions.
}

double _compute_hypervolume(
    const double* sorted_pareto_sols,
    const double* ref_point,
    int* nondominated_indices_buf,
    const int& n_objectives,
    const int end_row_index
) {
    double hv = 0.0;
    for (int i = 0; i < end_row_index; ++i) {
        double inclusive_hv = 1.0;
        for (int j = 0; j < n_objectives; ++j) {
            inclusive_hv *= ref_point[j] - sorted_pareto_sols[i * n_objectives + j];
        }
        // The early additions of the hypervolume breaks the compatibility with the Python version.
        hv += inclusive_hv;
    }

    auto value_at = [&](int i, int j)->const double& {return sorted_pareto_sols[i * n_objectives + j];};
    if (end_row_index == 1) {
        return hv;
    } else if (end_row_index == 2) {
        double intersec = 1.0;
        for (int j = 0; j < n_objectives; ++j) {
            intersec *= ref_point[j] - std::max(value_at(0, j), value_at(1, j));
        }
        return hv - intersec;
    }
    double limited_loss_values[end_row_index * n_objectives];
    for (int i = 0; i < end_row_index - 1; ++i) {
        int end_row_index_limited = end_row_index - i - 1;
        for (int j = 0; j < end_row_index_limited; ++j) {
            for (int k = 0; k < n_objectives; ++k) {
                limited_loss_values[j * n_objectives + k] = std::max(value_at(i, k), value_at(j + i + 1, k));
            }
        }
        if (end_row_index_limited > 2) {
            end_row_index_limited = _pack_pareto_sols(
                limited_loss_values, nondominated_indices_buf, n_objectives, end_row_index_limited
            );
        }
        hv -= _compute_hypervolume(
            limited_loss_values, ref_point, nondominated_indices_buf, n_objectives, end_row_index_limited
        );
    }
    return hv;
}

double compute_hypervolume(
    const vector<vector<double>>& sorted_pareto_sols,
    const vector<double>& ref_point
) {
    assert(
        sorted_pareto_sols[0].size() == ref_point.size() &&
        "The shape unmatch between ref_point and sorted_pareto_sols."
    );
    for (int i = 0; i < (int) sorted_pareto_sols.size() - 1; ++i) {
        assert(
            sorted_pareto_sols[i][0] <= sorted_pareto_sols[i + 1][0] &&
            "sorted_pareto_sols should be sorted by the first objective."
        );
        for (int j = 0; j < ref_point.size(); ++j) {
            assert(sorted_pareto_sols[i][j] <= ref_point[j] && "The Pareto solutions must be less than ref_point.");
        }
    }
    int n_max_trials = sorted_pareto_sols.size();
    int n_objectives = sorted_pareto_sols[0].size();
    int nondominated_indices_buf[sorted_pareto_sols.size()];
    double flattened_sorted_pareto_sols[n_max_trials * n_objectives];
    for (int i = 0; i < n_max_trials; ++i) {
        for (int j = 0; j < n_objectives; ++j) {
            flattened_sorted_pareto_sols[i * n_objectives + j] = sorted_pareto_sols[i][j];
        }
    }
    return _compute_hypervolume(
        flattened_sorted_pareto_sols, ref_point.data(), nondominated_indices_buf, n_objectives, n_max_trials
    );
}

PYBIND11_MODULE(hvcpp, m) {
    m.doc() = "Compute hypervolume in C++";
    m.def(
        "compute_hypervolume",
        &compute_hypervolume,
        "Compute the hypervolume of a Pareto front",
        py::arg("sorted_pareto_sols"),
        py::arg("ref_point")
    );
}
