import numpy as np
from collections import Counter

class BaseRandomForest:
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = self._build_tree()
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.swapaxes(tree_preds, 0, 1)

    def _build_tree(self):
        raise NotImplementedError("This method should be implemented in a derived class.")
class RandomForestClassifier(BaseRandomForest):
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2, n_features=None):
        super().__init__(n_trees, max_depth, min_samples_split, n_features)

    def _build_tree(self):
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.n_features
        )

    def predict(self, X):
        tree_preds = super().predict(X)
        return np.array([self._most_common_label(tree_pred) for tree_pred in tree_preds])

class DecisionTreeClassifier(BaseDecisionTree):
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, max_features=None):
        super().__init__(criterion, max_depth, min_samples_split)
        self.max_features = max_features

    def _find_best_split(self, X, y, n_samples, n_features):
        best_split = {"gain": -1}
        features = np.random.choice(n_features, self.max_features, replace=False)
        for feature_index in features:
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold
                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                left_data = (X[left_indices], y[left_indices])
                right_data = (X[right_indices], y[right_indices])
                gain = self._information_gain(y, left_data[1], right_data[1])
                if gain > best_split["gain"]:
                    best_split.update({
                        "feature_index": feature_index,
                        "threshold": threshold,
                        "left_data": left_data,
                        "right_data": right_data,
                        "gain": gain
                    })
        return best_split

    def _information_gain(self, parent, left_child, right_child):
        return self._impurity(parent) - (
            len(left_child) / len(parent) * self._impurity(left_child) +
            len(right_child) / len(parent) * self._impurity(right_child)
        )

    def _impurity(self, y):
        if self.criterion == "gini":
            proportions = np.bincount(y) / len(y)
            return 1 - np.sum(proportions ** 2)
        elif self.criterion == "entropy":
            proportions = np.bincount(y) / len(y)
            return -np.sum([p * np.log2(p) for p in proportions if p > 0])

    def _calculate_leaf_value(self, y):
        return np.argmax(np.bincount(y))
class RandomForestRegressor(BaseRandomForest):
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2, n_features=None):
        super().__init__(n_trees, max_depth, min_samples_split, n_features)

    def _build_tree(self):
        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.n_features
        )

    def predict(self, X):
        tree_preds = super().predict(X)
        return np.mean(tree_preds, axis=1)

class DecisionTreeRegressor(BaseDecisionTree):
    def __init__(self, criterion='mse', max_depth=None, min_samples_split=2, max_features=None):
        super().__init__(criterion, max_depth, min_samples_split)
        self.max_features = max_features

    def _find_best_split(self, X, y, n_samples, n_features):
        best_split = {"gain": -1}
        features = np.random.choice(n_features, self.max_features, replace=False)
        for feature_index in features:
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold
                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                left_data = (X[left_indices], y[left_indices])
                right_data = (X[right_indices], y[right_indices])
                gain = self._information_gain(y, left_data[1], right_data[1])
                if gain > best_split["gain"]:
                    best_split.update({
                        "feature_index": feature_index,
                        "threshold": threshold,
                        "left_data": left_data,
                        "right_data": right_data,
                        "gain": gain
                    })
        return best_split

    def _information_gain(self, parent, left_child, right_child):
        return self._mean_squared_error(parent) - (
            len(left_child) / len(parent) * self._mean_squared_error(left_child) +
            len(right_child) / len(parent) * self._mean_squared_error(right_child)
        )

    def _mean_squared_error(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def _calculate_leaf_value(self, y):
        return np.mean(y)

# Example usage:
if __name__ == "__main__":
    # Sample data for regression
    X_reg = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y_reg = np.array([1.1, 1.9, 3.0, 3.8, 5.2])
    
    # Create and train the random forest regressor
    reg = RandomForestRegressor(n_trees=10, max_depth=3)
    reg.fit(X_reg, y_reg)
    
    # Make predictions
    predictions_reg = reg.predict(X_reg)
    print("Random Forest Regression Predictions:", predictions_reg)

