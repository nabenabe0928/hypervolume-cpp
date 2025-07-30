#include <algorithm>
#include <cassert>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

using std::vector;
namespace py = pybind11;

void _is_pareto_front(
    const vector<vector<double>>& sorted_loss_values,
    vector<bool>& on_front_buf,
    const int end_index = -1
) {
    // No consideration of duplications.
    const int n_trials = end_index == -1 ? sorted_loss_values.size() : end_index;
    const int n_objectives = sorted_loss_values[0].size();
    for (int i = 0; i < n_trials; ++i) {
        on_front_buf[i] = false;  // Initialize the buffer to false.
    }
    vector<int> nondominated_indices(n_trials);
    std::iota(nondominated_indices.begin(), nondominated_indices.end(), 0);
    int n_remaining = n_trials;
    while (n_remaining > 0) {
        int head = nondominated_indices[0];
        on_front_buf[head] = true;
        int nondominated_count = 0;
        for (int i = 1; i < n_remaining; ++i) {
            int idx = nondominated_indices[i];
            bool is_nondominated = false;
            for (int j = 1; j < n_objectives; ++j) {
                if (sorted_loss_values[idx][j] < sorted_loss_values[head][j]) {
                    is_nondominated = true;
                    break;
                }
            }
            if (is_nondominated) {
                nondominated_indices[nondominated_count++] = idx;
            }
        }
        n_remaining = nondominated_count;
    }
}

void _pack_pareto_sols(
    vector<vector<double>>& sorted_loss_values,
    vector<bool>& on_front_buf,
    const int end_index = -1
) {
    _is_pareto_front(sorted_loss_values, on_front_buf, end_index);
    int head_index = 0;
    for (int i = 0; i < end_index; ++i) {
        if (!on_front_buf[i]) {
            continue;
        }
        sorted_loss_values[head_index++] = sorted_loss_values[i];
    }
}

double _compute_hypervolume(
    const vector<vector<double>>& sorted_pareto_sols,
    const vector<double>& ref_point,
    const int end_index = -1
) {
    const int n_trials = end_index == -1 ? sorted_pareto_sols.size() : end_index;
    const int n_objectives = sorted_pareto_sols[0].size();
    double hv = 0.0;
    for (int i = 0; i < n_trials; ++i) {
        double inclusive_hv = 1.0;
        for (int j = 0; j < n_objectives; ++j) {
            inclusive_hv *= ref_point[j] - sorted_pareto_sols[i][j];
        }
        // The early additions of the hypervolume breaks the compatibility with the Python version.
        hv += inclusive_hv;
    }
    if (n_trials == 1) {
        return hv;
    } else if (n_trials == 2) {
        double intersec = 1.0;
        for (int j = 0; j < n_objectives; ++j) {
            intersec *= ref_point[j] - std::max(sorted_pareto_sols[0][j], sorted_pareto_sols[1][j]);
        }
        return hv - intersec;
    }
    vector<vector<double>> limited_loss_values(n_trials, vector<double>(n_objectives));
    vector<bool> on_front(n_trials, false);
    for (int i = 0; i < n_trials - 1; ++i) {
        const int end_index = n_trials - i - 1;
        for (int j = 0; j < end_index; ++j) {
            for (int k = 0; k < n_objectives; ++k) {
                limited_loss_values[j][k] = std::max(sorted_pareto_sols[i][k], sorted_pareto_sols[j + i + 1][k]);
            }
        }
        if (end_index <= 3) {
            hv -= _compute_hypervolume(limited_loss_values, ref_point, end_index);
            continue;
        }
        _pack_pareto_sols(limited_loss_values, on_front, end_index);
        const int n_pareto_sols = std::count(on_front.begin(), on_front.begin() + end_index, true);
        hv -= _compute_hypervolume(limited_loss_values, ref_point, n_pareto_sols);
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
    return _compute_hypervolume(sorted_pareto_sols, ref_point);
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
