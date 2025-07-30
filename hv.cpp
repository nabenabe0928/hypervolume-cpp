#include <algorithm>
#include <cassert>
#include <numeric>
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
#include <vector>
#include <iostream>
#include <iomanip>

using std::vector;
// namespace py = pybind11;

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

double _compute_hypervolume(
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
        double exclusive_hv = _compute_hypervolume(pareto_sols, ref_point);
        hv += inclusive_hvs[i] - exclusive_hv;
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

int main(void) {
    vector<vector<double>> sorted_pareto_sols = {{-2.025142586657607,0.18645431476942764,-0.661786464768388,0.852433334796224},{-1.4785219903674274,-0.7198442083947086,-0.4606387709597875,1.0571222262189157},{-1.4153707420504142,-0.42064532276535904,-0.3427145165267695,-0.8022772692216189},{-1.377669367957091,-0.9378250399151228,0.5150352672086598,0.5137859509122088},{-1.2002964070557762,-0.3345012358409484,-0.4749453111609562,-0.6533292325737119},{-1.0623037137261049,0.4735924306351816,-0.9194242342338032,1.5499344050175394},{-1.0128311203344238,0.3142473325952739,-0.9080240755212109,-1.4123037013352915},{-1.006017381499702,-1.2141886127877322,1.1581108735000678,0.7916626939629359},{-0.926930471578083,-0.05952535606180008,-3.2412673400690726,-1.0243876413342898},{-0.883857436201133,0.1537251059455279,0.058208718445999896,-1.142970297830623},{-0.846793718068405,-1.5148472246858646,-0.4465149520670211,0.8563987943234723},{-0.8397218421807761,-0.5993926454440222,-2.123895724309807,-0.525755021680761},{-0.7832532923362371,-0.3220615162056756,0.8135172173696698,-1.2308643164339552},{-0.7020530938773524,-0.3276621465977682,-0.39210815313215763,-1.4635149481321186},{-0.47917423784528995,-0.18565897666381712,-1.1063349740060282,-1.1962066240806708},{-0.4710383056183228,0.2320499373576363,-1.4480843414973241,-1.4074637743765552},{-0.2525681513931603,-1.2477831819648495,1.6324113039316352,-1.4301413779606327},{-0.2453881160028705,-0.7537361643574896,-0.8895144296255233,-0.8158102849654383},{-0.03471176970524331,-1.168678037619532,1.1428228145150205,0.7519330326867741},{-0.013497224737933921,-1.0577109289559004,0.822544912103189,-1.2208436499710222},{0.08704706823817122,-0.29900735046586746,0.0917607765355023,-1.9875689146008928},{0.09767609854883172,-0.7730097838554665,0.024510174258942714,0.49799829124544975},{0.2088635950047554,-1.9596701238797756,-1.3281860488984305,0.19686123586912352},{0.24196227156603412,-1.913280244657798,-1.7249178325130328,-0.5622875292409727},{0.7910319470430469,-0.9093874547947389,1.4027943109360992,-1.4018510627922809},{2.3146585666735087,-1.867265192591748,0.6862601903745135,-1.6127158711896517}};
    vector<double> ref_point = {10.0, 10.0, 10.0, 10.0};
    std::cout << std::setprecision(17) << "Out: " << compute_hypervolume(sorted_pareto_sols, ref_point) << std::endl;
    std::cout << "Ans: " << 20988.584395027025 << std::endl;
}

/*
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
*/
