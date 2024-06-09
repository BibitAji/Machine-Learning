import numpy as np

class GMM:
    def __init__(self, n_components=3, n_iter=100):
        self.n_components = n_components
        self.n_iter = n_iter

    def fit(self, X):
        self.n_samples, self.n_features = X.shape

        # Initialize weights, means, and covariances
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(self.n_samples, self.n_components, replace=False)]
        self.covariances = np.array([np.cov(X, rowvar=False)] * self.n_components)

        log_likelihoods = []

        for _ in range(self.n_iter):
            # E-step
            responsibilities = self._e_step(X)

            # M-step
            self._m_step(X, responsibilities)

            log_likelihood = self._compute_log_likelihood(X)
            log_likelihoods.append(log_likelihood)

        return log_likelihoods

    def _e_step(self, X):
        responsibilities = np.zeros((self.n_samples, self.n_components))

        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * self._multivariate_normal(X, self.means[k], self.covariances[k])

        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def _m_step(self, X, responsibilities):
        Nk = responsibilities.sum(axis=0)

        self.weights = Nk / self.n_samples
        self.means = (responsibilities.T @ X) / Nk[:, np.newaxis]
        self.covariances = np.zeros((self.n_components, self.n_features, self.n_features))

        for k in range(self.n_components):
            X_centered = X - self.means[k]
            self.covariances[k] = (responsibilities[:, k][:, np.newaxis] * X_centered).T @ X_centered / Nk[k]

    def _multivariate_normal(self, X, mean, cov):
        d = self.n_features
        cov_inv = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
        norm_factor = 1 / np.sqrt((2 * np.pi) ** d * det_cov)

        X_centered = X - mean
        result = np.exp(-0.5 * np.sum(X_centered @ cov_inv * X_centered, axis=1))

        return norm_factor * result

    def _compute_log_likelihood(self, X):
        log_likelihood = 0
        for k in range(self.n_components):
            log_likelihood += self.weights[k] * self._multivariate_normal(X, self.means[k], self.covariances[k])
        return np.sum(np.log(log_likelihood))

# Example usage:
if __name__ == "__main__":
    # Sample data
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])

    # Create and train GMM
    gmm = GMM(n_components=2, n_iter=100)
    log_likelihoods = gmm.fit(X)

    print("Log Likelihoods:", log_likelihoods)
