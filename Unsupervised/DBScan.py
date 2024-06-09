import numpy as np

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        self.labels = np.full(X.shape[0], -1)  # Initialize labels as -1 (unclassified)
        cluster_id = 0

        for i in range(X.shape[0]):
            if self.labels[i] == -1:  # Not yet classified
                if self._expand_cluster(X, i, cluster_id):
                    cluster_id += 1

    def _expand_cluster(self, X, point_idx, cluster_id):
        neighbors = self._region_query(X, point_idx)

        if len(neighbors) < self.min_samples:
            self.labels[point_idx] = -1  # Label as noise
            return False
        else:
            self.labels[point_idx] = cluster_id
            while neighbors:
                neighbor_idx = neighbors.pop()
                if self.labels[neighbor_idx] == -1:  # Previously labeled as noise
                    self.labels[neighbor_idx] = cluster_id
                if self.labels[neighbor_idx] == -1:  # Not yet classified
                    self.labels[neighbor_idx] = cluster_id
                    new_neighbors = self._region_query(X, neighbor_idx)
                    if len(new_neighbors) >= self.min_samples:
                        neighbors.extend(new_neighbors)
            return True

    def _region_query(self, X, point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return list(np.where(distances <= self.eps)[0])

# Example usage:
if __name__ == "__main__":
    # Sample data
    X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])

    # Create and train DBSCAN
    dbscan = DBSCAN(eps=3, min_samples=2)
    dbscan.fit(X)

    # Get cluster labels
    labels = dbscan.labels
    print("Cluster Labels:", labels)
