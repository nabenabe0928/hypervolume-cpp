#include <algorithm>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

using std::vector;
namespace py = pybind11;

template <class T>
vector<T> filter_by_mask(const vector<T>& vec, const vector<bool>& mask){
    vector<T> out;
    // reserve so we don’t re‑allocate too often.
    out.reserve(std::count(mask.begin(), mask.end(), true));
    for (int i = 0; i < (int) vec.size(); ++i){
        if (mask[i]){
            out.push_back(vec[i]);
        }
    }
    return out;
}

vector<bool> _is_pareto_front(const vector<vector<double>>& sorted_loss_values) {
    // No consideration of duplications.
    int n_trials = sorted_loss_values.size();
    int n_objectives = sorted_loss_values[0].size();
    vector<bool> on_front = vector<bool>(n_trials, false);
    vector<int> nondominated_indices(n_trials);
    std::iota(nondominated_indices.begin(), nondominated_indices.end(), 0);
    int n_remaining = n_trials;
    while (n_remaining > 0) {
        int head = nondominated_indices[0];
        on_front[head] = true;
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
    return on_front;
}

double compute_hypervolume(
    const vector<vector<double>>& sorted_pareto_sols,
    const vector<double>& ref_point
) {
    int n_trials = sorted_pareto_sols.size();
    int n_objectives = sorted_pareto_sols[0].size();
    vector<double> inclusive_hvs = vector<double>(n_trials, 1.0);
    for (int i = 0; i < n_trials; ++i) {
        for (int j = 0; j < n_objectives; ++j) {
            inclusive_hvs[i] *= ref_point[j] - sorted_pareto_sols[i][j];
        }
    }
    if (n_trials == 1) {
        return inclusive_hvs[0];
    } else if (n_trials == 2) {
        double intersec = 1.0;
        for (int j = 0; j < n_objectives; ++j) {
            intersec *= ref_point[j] - std::max(sorted_pareto_sols[0][j], sorted_pareto_sols[1][j]);
        }
        return inclusive_hvs[0] + inclusive_hvs[1] - intersec;
    }
    double hv = inclusive_hvs[n_trials - 1];
    for (int i = 0; i < n_trials - 1; ++i) {
        vector<vector<double>> limited_loss_values(n_trials - i - 1, vector<double>(n_objectives));
        for (int j = i + 1; j < n_trials; ++j) {
            for (int k = 0; k < n_objectives; ++k) {
                limited_loss_values[j - i - 1][k] = std::max(sorted_pareto_sols[i][k], sorted_pareto_sols[j][k]);
            }
        }
        vector<bool> on_front = _is_pareto_front(limited_loss_values);
        vector<vector<double>> pareto_sols = filter_by_mask(limited_loss_values, on_front);
        double exclusive_hv = compute_hypervolume(pareto_sols, ref_point);
        hv += inclusive_hvs[i] - exclusive_hv;
    }
    return hv;
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
