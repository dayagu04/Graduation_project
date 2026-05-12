import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances_argmin_min

class PSOKMeans(BaseEstimator, ClusterMixin):
    """
    PSO-KMeans 融合聚类算法
    使用粒子群优化寻找更优的初始聚类中心，再通过 K-Means 精细化
    """
    def __init__(self, n_clusters=3, n_particles=20, w=0.729, c1=1.49445, c2=1.49445,
                 pso_max_iter=30, kmeans_max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.n_particles = n_particles
        self.inertia_weight = w
        self.cognitive_coef = c1
        self.social_coef = c2
        self.pso_max_iter = pso_max_iter
        self.kmeans_max_iter = kmeans_max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.convergence_history_ = None  # 每代 PSO 全局最优 SSE

    def _fitness(self, X, centroids):
        """计算适应度（误差平方和）"""
        centroids = centroids.reshape((self.n_clusters, -1))
        labels, distances = pairwise_distances_argmin_min(X, centroids)
        return np.sum(distances ** 2)

    def _init_particles(self, X):
        """初始化粒子群的位置和速度"""
        n_samples, n_features = X.shape
        X_min, X_max = np.min(X, axis=0), np.max(X, axis=0)

        particles_pos = np.zeros((self.n_particles, self.n_clusters * n_features))
        for i in range(self.n_particles):
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            particles_pos[i] = X[indices].flatten()

        v_max = (X_max - X_min).mean() * 0.1
        particles_vel = np.random.uniform(-v_max, v_max,
                                         (self.n_particles, self.n_clusters * n_features))

        return particles_pos, particles_vel

    def _init_best_positions(self, X, particles_pos):
        """初始化个体最优和全局最优"""
        personal_best_pos = particles_pos.copy()
        personal_best_fit = np.array([self._fitness(X, p) for p in particles_pos])

        global_best_idx = np.argmin(personal_best_fit)
        global_best_pos = personal_best_pos[global_best_idx].copy()
        global_best_fit = personal_best_fit[global_best_idx]

        return personal_best_pos, personal_best_fit, global_best_pos, global_best_fit

    def _update_particles(self, particles_pos, particles_vel, personal_best_pos,
                         global_best_pos, iteration):
        """更新粒子速度和位置"""
        current_w = self.inertia_weight - (self.inertia_weight - 0.4) * (iteration / self.pso_max_iter)
        r1, r2 = np.random.rand(2)

        particles_vel = (current_w * particles_vel +
                        self.cognitive_coef * r1 * (personal_best_pos - particles_pos) +
                        self.social_coef * r2 * (global_best_pos - particles_pos))

        particles_pos = particles_pos + particles_vel

        return particles_pos, particles_vel

    def _update_best_positions(self, X, particles_pos, personal_best_pos, personal_best_fit,
                              global_best_pos, global_best_fit):
        """更新个体最优和全局最优"""
        for i in range(self.n_particles):
            fit_val = self._fitness(X, particles_pos[i])
            if fit_val < personal_best_fit[i]:
                personal_best_fit[i] = fit_val
                personal_best_pos[i] = particles_pos[i].copy()

                if fit_val < global_best_fit:
                    global_best_fit = fit_val
                    global_best_pos = particles_pos[i].copy()

        return personal_best_pos, personal_best_fit, global_best_pos, global_best_fit

    def fit(self, X, y=None):
        X = np.array(X)
        np.random.seed(self.random_state)
        n_features = X.shape[1]

        particles_pos, particles_vel = self._init_particles(X)
        personal_best_pos, personal_best_fit, global_best_pos, global_best_fit = \
            self._init_best_positions(X, particles_pos)

        history = [float(global_best_fit)]  # 第 0 代（初始化）

        for iteration in range(self.pso_max_iter):
            particles_pos, particles_vel = self._update_particles(
                particles_pos, particles_vel, personal_best_pos, global_best_pos, iteration
            )

            personal_best_pos, personal_best_fit, global_best_pos, global_best_fit = \
                self._update_best_positions(
                    X, particles_pos, personal_best_pos, personal_best_fit,
                    global_best_pos, global_best_fit
                )
            history.append(float(global_best_fit))

        self.convergence_history_ = np.array(history)

        from sklearn.cluster import KMeans
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            init=global_best_pos.reshape((self.n_clusters, n_features)),
            n_init=1,
            max_iter=self.kmeans_max_iter,
            random_state=self.random_state
        )

        kmeans.fit(X)
        self.cluster_centers_ = kmeans.cluster_centers_
        self.labels_ = kmeans.labels_
        self.inertia_ = kmeans.inertia_

        return self

    def predict(self, X):
        X = np.array(X)
        labels, _ = pairwise_distances_argmin_min(X, self.cluster_centers_)
        return labels
