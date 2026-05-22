#include "csv_reader.h"
#include "kmeans.h"
#include "pso_kmeans.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <numeric>

// ---- 归一化（Min-Max，原地修改）----
void normalize(std::vector<DataRow>& rows) {
    if (rows.empty()) return;
    size_t dim = rows[0].features.size();
    for (size_t j = 0; j < dim; ++j) {
        double mn = rows[0].features[j], mx = mn;
        for (const auto& r : rows) {
            mn = std::min(mn, r.features[j]);
            mx = std::max(mx, r.features[j]);
        }
        double range = mx - mn;
        if (range < 1e-9) continue;
        for (auto& r : rows)
            r.features[j] = (r.features[j] - mn) / range;
    }
}

// ---- SSE 计算 ----
double compute_sse(const std::vector<DataRow>& data, const std::vector<Cluster>& clusters) {
    double sse = 0.0;
    for (const auto& cl : clusters) {
        for (int idx : cl.point_indices) {
            const auto& f = data[idx].features;
            for (size_t d = 0; d < f.size(); ++d) {
                double diff = f[d] - cl.centroid[d];
                sse += diff * diff;
            }
        }
    }
    return sse;
}

// ---- 解析逗号分隔的列名字符串 ----
std::vector<std::string> parse_cols(const std::string& s) {
    std::vector<std::string> cols;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        // trim
        tok.erase(0, tok.find_first_not_of(" \t"));
        tok.erase(tok.find_last_not_of(" \t") + 1);
        if (!tok.empty()) cols.push_back(tok);
    }
    return cols;
}

void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n"
        << "  --data <path>          CSV file path (default: data/train-data.csv)\n"
        << "  --cols <c1,c2,...>     Columns to use (default: auto-detect numeric)\n"
        << "  --skip <c1,c2,...>     Columns to skip in auto-detect mode\n"
        << "  --k <int>              Number of clusters (default: 4)\n"
        << "  --iters <int>          Max K-Means iterations (default: 100)\n"
        << "  --output <path>        Output CSV path (default: output/clustering_results.csv)\n"
        << "  --no-normalize         Disable Min-Max normalization\n"
        << "  --algo <name>          Algorithm: kmeans++ (default) | pso-kmeans\n"
        << "  --pso-particles <int>  PSO swarm size (default: 20)\n"
        << "  --pso-iters <int>      PSO max iterations (default: 30)\n"
        << "  --seed <int>           Random seed for PSO (default: 42)\n";
}

int main(int argc, char* argv[]) {
    // ---- 默认参数 ----
    std::string data_path   = "data/train-data.csv";
    std::string output_path = "output/clustering_results.csv";
    std::string cols_str, skip_str;
    int k = 4, max_iters = 100;
    bool do_normalize = true;
    std::string algo = "kmeans++";
    int pso_particles = 20;
    int pso_iters = 30;
    unsigned int seed = 42;

    // ---- 解析命令行 ----
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
        else if (arg == "--data"   && i+1 < argc) data_path   = argv[++i];
        else if (arg == "--cols"   && i+1 < argc) cols_str    = argv[++i];
        else if (arg == "--skip"   && i+1 < argc) skip_str    = argv[++i];
        else if (arg == "--k"      && i+1 < argc) k           = std::stoi(argv[++i]);
        else if (arg == "--iters"  && i+1 < argc) max_iters   = std::stoi(argv[++i]);
        else if (arg == "--output" && i+1 < argc) output_path = argv[++i];
        else if (arg == "--no-normalize") do_normalize = false;
        else if (arg == "--algo"          && i+1 < argc) algo          = argv[++i];
        else if (arg == "--pso-particles" && i+1 < argc) pso_particles = std::stoi(argv[++i]);
        else if (arg == "--pso-iters"     && i+1 < argc) pso_iters     = std::stoi(argv[++i]);
        else if (arg == "--seed"          && i+1 < argc) seed          = static_cast<unsigned int>(std::stoul(argv[++i]));
        else { std::cerr << "Unknown argument: " << arg << "\n"; print_usage(argv[0]); return 1; }
    }

    auto feature_cols = parse_cols(cols_str);
    auto skip_cols    = parse_cols(skip_str);

    // ---- 加载数据 ----
    std::cout << "Loading: " << data_path << "\n";
    Dataset ds = CsvReader::load(data_path, feature_cols, skip_cols);
    if (ds.rows.empty()) {
        std::cerr << "No data loaded. Exiting.\n";
        return 1;
    }

    std::cout << "Rows: " << ds.rows.size()
              << "  Features (" << ds.feature_names.size() << "): ";
    for (size_t i = 0; i < ds.feature_names.size(); ++i)
        std::cout << ds.feature_names[i] << (i+1 < ds.feature_names.size() ? ", " : "\n");

    // ---- 归一化 ----
    if (do_normalize) {
        std::cout << "Normalizing (Min-Max)...\n";
        normalize(ds.rows);
    }

    // ---- 聚类 ----
    std::vector<Cluster> clusters;
    std::vector<double> pso_history;

    if (algo == "pso-kmeans" || algo == "pso") {
        std::cout << "Running PSO-KMeans with k=" << k
                  << " particles=" << pso_particles
                  << " pso_iters=" << pso_iters << "...\n";
        PSOKMeans pso(k, pso_particles, pso_iters, max_iters,
                      0.729, 1.49445, 1.49445, seed);
        pso.fit(ds.rows);
        clusters = pso.get_clusters();
        pso_history = pso.get_convergence_history();
    } else {
        std::cout << "Running K-Means++ with k=" << k << "...\n";
        KMeans kmeans(k, max_iters);
        kmeans.fit(ds.rows);
        clusters = kmeans.get_clusters();
    }

    double sse = compute_sse(ds.rows, clusters);

    std::cout << "\n=== Results ===\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "SSE: " << sse << "\n";
    for (size_t i = 0; i < clusters.size(); ++i) {
        std::cout << "Cluster " << i << ": " << clusters[i].point_indices.size() << " points  centroid=[";
        for (size_t d = 0; d < clusters[i].centroid.size(); ++d)
            std::cout << clusters[i].centroid[d] << (d+1 < clusters[i].centroid.size() ? ", " : "]\n");
    }

    // ---- 保存结果 ----
    std::vector<int> assignments(ds.rows.size(), -1);
    for (size_t i = 0; i < clusters.size(); ++i)
        for (int idx : clusters[i].point_indices)
            assignments[idx] = (int)i;

    std::ofstream out(output_path);
    if (!out.is_open()) {
        std::cerr << "Cannot write to: " << output_path << "\n";
        return 1;
    }
    out << "RowIndex,ClusterID\n";
    for (size_t i = 0; i < assignments.size(); ++i)
        out << i << "," << assignments[i] << "\n";

    std::cout << "Saved to: " << output_path << "\n";

    // ---- PSO 收敛曲线落盘（可选）----
    if (!pso_history.empty()) {
        std::string conv_path = output_path + ".pso_history.csv";
        std::ofstream cf(conv_path);
        if (cf.is_open()) {
            cf << "iteration,global_best_sse\n";
            for (size_t i = 0; i < pso_history.size(); ++i)
                cf << i << "," << pso_history[i] << "\n";
            std::cout << "PSO history saved to: " << conv_path << "\n";
        }
    }
    return 0;
}
