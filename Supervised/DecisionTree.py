import numpy as np

class BaseDecisionTree:
    def __init__(self, criterion, max_depth=None, min_samples_split=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if n_samples >= self.min_samples_split and (self.max_depth is None or depth < self.max_depth):
            best_split = self._find_best_split(X, y, n_samples, n_features)
            if best_split["gain"] > 0:
                left_tree = self._build_tree(*best_split["left_data"], depth + 1)
                right_tree = self._build_tree(*best_split["right_data"], depth + 1)
                return Node(
                    feature_index=best_split["feature_index"],
                    threshold=best_split["threshold"],
                    left=left_tree,
                    right=right_tree
                )
        return LeafNode(self._calculate_leaf_value(y))

    def _find_best_split(self, X, y, n_samples, n_features):
        best_split = {"gain": -1}
        for feature_index in range(n_features):
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
        pass

    def _calculate_leaf_value(self, y):
        pass

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if isinstance(node, LeafNode):
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class Node:
    def __init__(self, feature_index, threshold, left, right):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right

class LeafNode:
    def __init__(self, value):
        self.value = value
      
class DecisionTreeClassifier(BaseDecisionTree):
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2):
        super().__init__(criterion, max_depth, min_samples_split)

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
class DecisionTreeRegressor(BaseDecisionTree):
    def __init__(self, criterion='mse', max_depth=None, min_samples_split=2):
        super().__init__(criterion, max_depth, min_samples_split)

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
    # Sample data for classification
    X_class = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y_class = np.array([0, 0, 1, 1, 1])
    
    # Create and train the classifier
    classifier = DecisionTreeClassifier(criterion='gini', max_depth=3)
    classifier.fit(X_class, y_class)
    
    # Make predictions
    predictions_class = classifier.predict(X_class)
    print("Classification Predictions:", predictions_class)
