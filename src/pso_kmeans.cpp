#include "pso_kmeans.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <numeric>

PSOKMeans::PSOKMeans(int k, int n_particles, int pso_max_iter, int kmeans_max_iter,
                     double w, double c1, double c2, unsigned int seed)
    : k_(k), n_particles_(n_particles),
      pso_max_iter_(pso_max_iter), kmeans_max_iter_(kmeans_max_iter),
      w_(w), c1_(c1), c2_(c2), seed_(seed),
      global_best_fitness_(std::numeric_limits<double>::max()) {}

double PSOKMeans::squared_distance(const std::vector<double>& a,
                                   const std::vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

double PSOKMeans::fitness(const std::vector<DataRow>& data,
                          const std::vector<double>& centroids_flat) const {
    if (data.empty()) return 0.0;
    size_t dim = data[0].features.size();
    double sse = 0.0;
    for (const auto& row : data) {
        double min_d2 = std::numeric_limits<double>::max();
        for (int c = 0; c < k_; ++c) {
            double d2 = 0.0;
            for (size_t j = 0; j < dim; ++j) {
                double diff = row.features[j] - centroids_flat[c * dim + j];
                d2 += diff * diff;
            }
            if (d2 < min_d2) min_d2 = d2;
        }
        sse += min_d2;
    }
    return sse;
}

void PSOKMeans::init_particles(const std::vector<DataRow>& data,
                               std::vector<std::vector<double>>& positions,
                               std::vector<std::vector<double>>& velocities) const {
    size_t n = data.size();
    size_t dim = data[0].features.size();
    size_t pos_len = static_cast<size_t>(k_) * dim;

    std::mt19937 rng(seed_);
    std::uniform_int_distribution<int> sample_idx(0, static_cast<int>(n) - 1);

    // 计算各维度极差，用于约束初始速度幅值
    std::vector<double> col_min(dim, std::numeric_limits<double>::max());
    std::vector<double> col_max(dim, std::numeric_limits<double>::lowest());
    for (const auto& row : data) {
        for (size_t j = 0; j < dim; ++j) {
            col_min[j] = std::min(col_min[j], row.features[j]);
            col_max[j] = std::max(col_max[j], row.features[j]);
        }
    }
    double mean_range = 0.0;
    for (size_t j = 0; j < dim; ++j) mean_range += (col_max[j] - col_min[j]);
    mean_range /= static_cast<double>(dim);
    double v_max = mean_range * 0.1;

    std::uniform_real_distribution<double> vel_dist(-v_max, v_max);

    positions.assign(n_particles_, std::vector<double>(pos_len, 0.0));
    velocities.assign(n_particles_, std::vector<double>(pos_len, 0.0));

    // 每个粒子用从样本中无放回抽样的 k 个点编码初始质心
    for (int i = 0; i < n_particles_; ++i) {
        std::vector<int> picked;
        picked.reserve(k_);
        while (static_cast<int>(picked.size()) < k_) {
            int idx = sample_idx(rng);
            if (std::find(picked.begin(), picked.end(), idx) == picked.end())
                picked.push_back(idx);
        }
        for (int c = 0; c < k_; ++c)
            for (size_t j = 0; j < dim; ++j)
                positions[i][c * dim + j] = data[picked[c]].features[j];

        for (size_t j = 0; j < pos_len; ++j)
            velocities[i][j] = vel_dist(rng);
    }
}

void PSOKMeans::fit(const std::vector<DataRow>& data) {
    clusters_.clear();
    convergence_history_.clear();
    if (data.empty()) return;

    size_t dim = data[0].features.size();
    size_t pos_len = static_cast<size_t>(k_) * dim;

    std::mt19937 rng(seed_);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    // ---- 1. 粒子群初始化 ----
    std::vector<std::vector<double>> positions, velocities;
    init_particles(data, positions, velocities);

    std::vector<std::vector<double>> personal_best_pos = positions;
    std::vector<double> personal_best_fit(n_particles_);
    for (int i = 0; i < n_particles_; ++i)
        personal_best_fit[i] = fitness(data, positions[i]);

    int gbest_idx = static_cast<int>(
        std::min_element(personal_best_fit.begin(), personal_best_fit.end())
        - personal_best_fit.begin());
    std::vector<double> global_best_pos = personal_best_pos[gbest_idx];
    double global_best_fit = personal_best_fit[gbest_idx];

    convergence_history_.push_back(global_best_fit);

    // ---- 2. PSO 主循环 ----
    for (int iter = 0; iter < pso_max_iter_; ++iter) {
        // 线性递减惯性权重 w_t : w_max -> 0.4
        double current_w = w_ - (w_ - 0.4) * (static_cast<double>(iter) / pso_max_iter_);
        double r1 = u01(rng);
        double r2 = u01(rng);

        for (int i = 0; i < n_particles_; ++i) {
            for (size_t j = 0; j < pos_len; ++j) {
                velocities[i][j] = current_w * velocities[i][j]
                    + c1_ * r1 * (personal_best_pos[i][j] - positions[i][j])
                    + c2_ * r2 * (global_best_pos[j] - positions[i][j]);
                positions[i][j] += velocities[i][j];
            }

            double fit_val = fitness(data, positions[i]);
            if (fit_val < personal_best_fit[i]) {
                personal_best_fit[i] = fit_val;
                personal_best_pos[i] = positions[i];
                if (fit_val < global_best_fit) {
                    global_best_fit = fit_val;
                    global_best_pos = positions[i];
                }
            }
        }
        convergence_history_.push_back(global_best_fit);
    }
    global_best_fitness_ = global_best_fit;

    // ---- 3. 用 PSO 输出的 gbest 作为 K-Means 初始质心进行局部精细化 ----
    // 直接基于 global_best_pos 跑 Lloyd 迭代
    clusters_.assign(k_, Cluster{});
    for (int c = 0; c < k_; ++c) {
        clusters_[c].centroid.assign(dim, 0.0);
        for (size_t j = 0; j < dim; ++j)
            clusters_[c].centroid[j] = global_best_pos[c * dim + j];
    }

    for (int iter = 0; iter < kmeans_max_iter_; ++iter) {
        // 保存上一轮质心
        std::vector<std::vector<double>> prev;
        prev.reserve(k_);
        for (const auto& cl : clusters_) prev.push_back(cl.centroid);

        // 分配
        for (auto& cl : clusters_) cl.point_indices.clear();
        for (size_t p = 0; p < data.size(); ++p) {
            double min_d2 = std::numeric_limits<double>::max();
            int best = 0;
            for (int c = 0; c < k_; ++c) {
                double d2 = squared_distance(data[p].features, clusters_[c].centroid);
                if (d2 < min_d2) { min_d2 = d2; best = c; }
            }
            clusters_[best].point_indices.push_back(static_cast<int>(p));
        }

        // 更新
        for (auto& cl : clusters_) {
            if (cl.point_indices.empty()) continue;
            std::vector<double> nc(dim, 0.0);
            for (int idx : cl.point_indices)
                for (size_t j = 0; j < dim; ++j)
                    nc[j] += data[idx].features[j];
            for (size_t j = 0; j < dim; ++j)
                nc[j] /= static_cast<double>(cl.point_indices.size());
            cl.centroid = nc;
        }

        // 收敛判定
        bool converged = true;
        for (int c = 0; c < k_; ++c) {
            if (std::sqrt(squared_distance(prev[c], clusters_[c].centroid)) > 1e-6) {
                converged = false; break;
            }
        }
        if (converged) {
            std::cout << "  [PSO-KMeans] K-Means refinement converged at iter "
                      << iter + 1 << "\n";
            break;
        }
    }
}

const std::vector<Cluster>& PSOKMeans::get_clusters() const { return clusters_; }
const std::vector<double>& PSOKMeans::get_convergence_history() const {
    return convergence_history_;
}
double PSOKMeans::get_global_best_fitness() const { return global_best_fitness_; }
