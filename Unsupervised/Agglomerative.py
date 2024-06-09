import numpy as np

class HierarchicalClustering:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters

    def fit(self, X):
        self.labels = np.arange(X.shape[0])

        while len(np.unique(self.labels)) > self.n_clusters:
            min_dist = float('inf')
            min_i, min_j = -1, -1

            for i in range(len(self.labels)):
                for j in range(i + 1, len(self.labels)):
                    if self.labels[i] != self.labels[j]:
                        dist = np.linalg.norm(X[i] - X[j])
                        if dist < min_dist:
                            min_dist = dist
                            min_i, min_j = i, j

            self._merge_clusters(min_i, min_j)

    def _merge_clusters(self, i, j):
        label_i, label_j = self.labels[i], self.labels[j]
        self.labels[self.labels == label_j] = label_i

# Example usage:
if __name__ == "__main__":
    # Sample data
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])

    # Create and train Hierarchical Clustering
    hierarchical = HierarchicalClustering(n_clusters=2)
    hierarchical.fit(X)

    # Get cluster labels
    labels = hierarchical.labels
    print("Cluster Labels:", labels)
