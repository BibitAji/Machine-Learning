import numpy as np
from sklearn.tree import DecisionTreeClassifier

class SelfTraining:
    def __init__(self, base_classifier=DecisionTreeClassifier(), threshold=0.8, max_iter=10):
        self.base_classifier = base_classifier
        self.threshold = threshold
        self.max_iter = max_iter

    def fit(self, X, y):
        self.X = X
        self.y = y.copy()
        unlabeled_mask = (y == -1)
        labeled_mask = ~unlabeled_mask

        for iteration in range(self.max_iter):
            self.base_classifier.fit(X[labeled_mask], y[labeled_mask])
            probs = self.base_classifier.predict_proba(X[unlabeled_mask])
            max_probs = np.max(probs, axis=1)
            confident_indices = np.where(max_probs > self.threshold)[0]

            if len(confident_indices) == 0:
                break

            new_labels = self.base_classifier.predict(X[unlabeled_mask][confident_indices])
            y[unlabeled_mask][confident_indices] = new_labels
            labeled_mask[unlabeled_mask][confident_indices] = True
            unlabeled_mask[unlabeled_mask][confident_indices] = False

        self.base_classifier.fit(X, y)

    def predict(self, X):
        return self.base_classifier.predict(X)

# Example usage
if __name__ == "__main__":
    X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9]])
    y = np.array([0, 0, 1, -1, -1, -1])

    self_training = SelfTraining()
    self_training.fit(X, y)
    print("Predicted labels:", self_training.predict(X))
