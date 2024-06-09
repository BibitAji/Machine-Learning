import numpy as np

class IsolationTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, current_depth=0):
        if current_depth >= self.max_depth or X.shape[0] <= 1:
            return None
        
        n_features = X.shape[1]
        split_feature = np.random.randint(0, n_features)
        split_value = np.random.uniform(np.min(X[:, split_feature]), np.max(X[:, split_feature]))
        
        left_mask = X[:, split_feature] < split_value
        right_mask = X[:, split_feature] >= split_value
        
        if np.all(left_mask) or np.all(right_mask):
            return None
        
        self.tree = {
            'split_feature': split_feature,
            'split_value': split_value,
            'left': IsolationTree(self.max_depth).fit(X[left_mask], current_depth + 1),
            'right': IsolationTree(self.max_depth).fit(X[right_mask], current_depth + 1)
        }
        return self.tree

    def path_length(self, X):
        if self.tree is None:
            return np.zeros(X.shape[0]) + self.max_depth
        
        split_feature = self.tree['split_feature']
        split_value = self.tree['split_value']
        
        left_mask = X[:, split_feature] < split_value
        right_mask = X[:, split_feature] >= split_value
        
        path_length = np.zeros(X.shape[0])
        path_length[left_mask] = 1 + self.tree['left'].path_length(X[left_mask])
        path_length[right_mask] = 1 + self.tree['right'].path_length(X[right_mask])
        
        return path_length


class IsolationForest:
    def __init__(self, n_estimators=100, max_samples='auto', max_depth=10):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X):
        if self.max_samples == 'auto':
            self.max_samples = min(256, X.shape[0])
        
        self.trees = []
        for _ in range(self.n_estimators):
            sample_indices = np.random.choice(X.shape[0], self.max_samples, replace=False)
            X_sample = X[sample_indices]
            tree = IsolationTree(self.max_depth)
            tree.fit(X_sample)
            self.trees.append(tree)

    def anomaly_score(self, X):
        path_lengths = np.zeros(X.shape[0])
        for tree in self.trees:
            path_lengths += tree.path_length(X)
        path_lengths /= len(self.trees)
        
        c = self._c_factor(self.max_samples)
        scores = 2 ** (-path_lengths / c)
        return scores

    def _c_factor(self, n):
        if n > 2:
            return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
        elif n == 2:
            return 1
        return 0

    def predict(self, X, threshold=0.5):
        scores = self.anomaly_score(X)
        return np.where(scores >= threshold, 1, -1)

# Example usage:
if __name__ == "__main__":
    # Sample data
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [8, 8],
        [8, 7],
        [25, 80]
    ])

    # Create and train Isolation Forest
    iso_forest = IsolationForest(n_estimators=100, max_samples='auto', max_depth=10)
    iso_forest.fit(X)

    # Get anomaly scores
    scores = iso_forest.anomaly_score(X)
    print("Anomaly Scores:", scores)

    # Predict anomalies
    predictions = iso_forest.predict(X, threshold=0.5)
    print("Predictions (1: Anomaly, -1: Normal):", predictions)
