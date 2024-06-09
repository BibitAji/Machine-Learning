import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def _distance(self, a, b):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((a - b) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(a - b))
        elif self.distance_metric == 'minkowski':
            return np.sum(np.abs(a - b) ** 3) ** (1/3)
        elif self.distance_metric == 'chebyshev':
            return np.max(np.abs(a - b))
        elif self.distance_metric == 'hamming':
            return np.sum(a != b) / len(a)
    
    def predict(self, X):
        predictions = []
        for x in X:
            distances = [self._distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        return predictions
import numpy as np

class KNNRegressor:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def _distance(self, a, b):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((a - b) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(a - b))
        elif self.distance_metric == 'minkowski':
            return np.sum(np.abs(a - b) ** 3) ** (1/3)
        elif self.distance_metric == 'chebyshev':
            return np.max(np.abs(a - b))
        elif self.distance_metric == 'hamming':
            return np.sum(a != b) / len(a)
    
    def predict(self, X):
        predictions = []
        for x in X:
            distances = [self._distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_values = [self.y_train[i] for i in k_indices]
            predictions.append(np.mean(k_nearest_values))
        return predictions

# Example usage:
if __name__ == "__main__":
    # Sample data
    X_train = np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]])
    y_train = np.array([0, 0, 0, 1, 1])
    
    X_test = np.array([[4, 5], [5, 6]])
    
    # Create and train the model
    model = KNNClassifier(k=3, distance_metric='euclidean')
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    print("Predictions:", predictions)
