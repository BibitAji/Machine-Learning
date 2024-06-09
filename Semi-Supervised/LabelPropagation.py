import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

class LabelPropagation:
    def __init__(self, gamma=20, max_iter=1000, tol=1e-3):
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        self.X = X
        self.y = y.copy()
        self.y_pred = y.copy()
        unlabeled_indices = np.where(y == -1)[0]

        W = rbf_kernel(X, X, gamma=self.gamma)
        D = np.diag(W.sum(axis=1))
        L = D - W

        for iteration in range(self.max_iter):
            y_pred_prev = self.y_pred.copy()
            self.y_pred = np.linalg.pinv(D).dot(W).dot(self.y_pred)
            self.y_pred[unlabeled_indices] = np.argmax(self.y_pred[unlabeled_indices], axis=1)

            if np.linalg.norm(self.y_pred - y_pred_prev) < self.tol:
                break

        self.y[unlabeled_indices] = np.argmax(self.y_pred[unlabeled_indices], axis=1)

    def predict(self, X):
        return self.y

# Example usage
if __name__ == "__main__":
    X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9]])
    y = np.array([0, 0, 1, -1, -1, -1])

    label_prop = LabelPropagation()
    label_prop.fit(X, y)
    print("Predicted labels:", label_prop.predict(X))
