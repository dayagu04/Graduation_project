#pragma once
#include <vector>
#include "csv_reader.h"
#include "kmeans.h"

class PSOKMeans {
public:
    PSOKMeans(int k,
              int n_particles = 20,
              int pso_max_iter = 30,
              int kmeans_max_iter = 100,
              double w = 0.729,
              double c1 = 1.49445,
              double c2 = 1.49445,
              unsigned int seed = 42);

    void fit(const std::vector<DataRow>& data);
    const std::vector<Cluster>& get_clusters() const;
    const std::vector<double>& get_convergence_history() const;
    double get_global_best_fitness() const;

private:
    int k_;
    int n_particles_;
    int pso_max_iter_;
    int kmeans_max_iter_;
    double w_;
    double c1_;
    double c2_;
    unsigned int seed_;

    std::vector<Cluster> clusters_;
    std::vector<double> convergence_history_;
    double global_best_fitness_;

    static double squared_distance(const std::vector<double>& a,
                                   const std::vector<double>& b);
    double fitness(const std::vector<DataRow>& data,
                   const std::vector<double>& centroids_flat) const;
    void init_particles(const std::vector<DataRow>& data,
                        std::vector<std::vector<double>>& positions,
                        std::vector<std::vector<double>>& velocities) const;
};
