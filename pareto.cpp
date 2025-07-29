#include <numeric>
#include <vector>

using std::vector;

vector<bool> is_pareto_front(const vector<vector<double>>& sorted_loss_values) {
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
