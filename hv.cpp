#include <algorithm>
#include <cassert>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

using std::vector;
namespace py = pybind11;

int _pack_pareto_sols(
    vector<vector<double>>& sorted_loss_values,
    vector<int>& nondominated_indices_buf,
    const int end_index = -1
) {
    // No consideration of duplications.
    const int n_trials = end_index == -1 ? sorted_loss_values.size() : end_index;
    const int n_objectives = sorted_loss_values[0].size();
    std::iota(nondominated_indices_buf.begin(), nondominated_indices_buf.begin() + end_index, 0);
    int n_remaining = n_trials;
    int head_index = 0;
    while (n_remaining > 0) {
        const int new_nondominated_index = nondominated_indices_buf[0];
        int nondominated_count = 0;
        const auto& new_nondominated_sol = sorted_loss_values[new_nondominated_index];
        for (int i = 1; i < n_remaining; ++i) {
            const int& idx = nondominated_indices_buf[i];
            const auto& cand = sorted_loss_values[idx];
            bool is_nondominated = false;
            for (int j = 1; j < n_objectives; ++j) {
                if (cand[j] < new_nondominated_sol[j]) {
                    is_nondominated = true;
                    break;
                }
            }
            if (is_nondominated) {
                nondominated_indices_buf[nondominated_count++] = idx;
            }
        }
        std::swap(sorted_loss_values[head_index++], sorted_loss_values[new_nondominated_index]);
        n_remaining = nondominated_count;
    }
    return head_index;
}

double _compute_hypervolume(
    const vector<vector<double>>& sorted_pareto_sols,
    const vector<double>& ref_point,
    vector<int>& nondominated_indices_buf,
    const int end_index = -1
) {
    const int n_trials = end_index == -1 ? sorted_pareto_sols.size() : end_index;
    const int n_objectives = sorted_pareto_sols[0].size();
    double hv = 0.0;
    for (int i = 0; i < n_trials; ++i) {
        double inclusive_hv = 1.0;
        const auto& vals_i = sorted_pareto_sols[i];
        for (int j = 0; j < n_objectives; ++j) {
            inclusive_hv *= ref_point[j] - vals_i[j];
        }
        // The early additions of the hypervolume breaks the compatibility with the Python version.
        hv += inclusive_hv;
    }
    if (n_trials == 1) {
        return hv;
    } else if (n_trials == 2) {
        double intersec = 1.0;
        const auto& vals1 = sorted_pareto_sols[0], vals2 = sorted_pareto_sols[1];
        for (int j = 0; j < n_objectives; ++j) {
            const double& v1 = vals1[j], v2 = vals2[j];
            intersec *= ref_point[j] - (v1 > v2 ? v1 : v2);
        }
        return hv - intersec;
    }
    vector<vector<double>> limited_loss_values(n_trials, vector<double>(n_objectives));
    for (int i = 0; i < n_trials - 1; ++i) {
        const int end_index = n_trials - i - 1;
        const auto& vals_i = sorted_pareto_sols[i];
        for (int j = 0; j < end_index; ++j) {
            const auto& vals_j = sorted_pareto_sols[j + i + 1];
            auto& target = limited_loss_values[j];
            for (int k = 0; k < n_objectives; ++k) {
                const double& v1 = vals_i[k], v2 = vals_j[k];
                target[k] = v1 > v2 ? v1 : v2;
            }
        }
        if (end_index <= 2) {
            hv -= _compute_hypervolume(limited_loss_values, ref_point, nondominated_indices_buf, end_index);
            continue;
        }
        const int n_pareto_sols = _pack_pareto_sols(limited_loss_values, nondominated_indices_buf, end_index);
        hv -= _compute_hypervolume(limited_loss_values, ref_point, nondominated_indices_buf, n_pareto_sols);
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
    vector<int> nondominated_indices_buf(sorted_pareto_sols.size());
    return _compute_hypervolume(sorted_pareto_sols, ref_point, nondominated_indices_buf);
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
