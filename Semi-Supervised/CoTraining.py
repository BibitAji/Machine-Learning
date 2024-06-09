import numpy as np
from sklearn.tree import DecisionTreeClassifier

class CoTraining:
    def __init__(self, base_classifier_1=DecisionTreeClassifier(), base_classifier_2=DecisionTreeClassifier(), max_iter=10):
        self.base_classifier_1 = base_classifier_1
        self.base_classifier_2 = base_classifier_2
        self.max_iter = max_iter

    def fit(self, X1, X2, y):
        self.X1 = X1
        self.X2 = X2
        self.y = y.copy()
        unlabeled_mask = (y == -1)
        labeled_mask = ~unlabeled_mask

        for iteration in range(self.max_iter):
            self.base_classifier_1.fit(X1[labeled_mask], y[labeled_mask])
            self.base_classifier_2.fit(X2[labeled_mask], y[labeled_mask])

            probs_1 = self.base_classifier_1.predict_proba(X1[unlabeled_mask])
            probs_2 = self.base_classifier_2.predict_proba(X2[unlabeled_mask])
            max_probs_1 = np.max(probs_1, axis=1)
            max_probs_2 = np.max(probs_2, axis=1)

            confident_indices_1 = np.where(max_probs_1 > 0.8)[0]
            confident_indices_2 = np.where(max_probs_2 > 0.8)[0]

            if len(confident_indices_1) == 0 and len(confident_indices_2) == 0:
                break

            new_labels_1 = self.base_classifier_1.predict(X1[unlabeled_mask][confident_indices_1])
            new_labels_2 = self.base_classifier_2.predict(X2[unlabeled_mask][confident_indices_2])

            y[unlabeled_mask][confident_indices_1] = new_labels_1
            y[unlabeled_mask][confident_indices_2] = new_labels_2

            labeled_mask[unlabeled_mask][confident_indices_1] = True
            labeled_mask[unlabeled_mask][confident_indices_2] = True
            unlabeled_mask[unlabeled_mask][confident_indices_1] = False
            unlabeled_mask[unlabeled_mask][confident_indices_2] = False

        self.base_classifier_1.fit(X1, y)
        self.base_classifier_2.fit(X2, y)

    def predict(self, X1, X2):
        preds_1 = self.base_classifier_1.predict(X1)
        preds_2 = self.base_classifier_2.predict(X2)
        return np.array([max(pred1, pred2, key=list([pred1, pred2]).count) for pred1, pred2 in zip(preds_1, preds_2)])

# Example usage
if __name__ == "__main__":
    X1 = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9]])
    X2 = np.array([[2, 1], [3, 2], [4, 3], [6, 5], [7, 6], [9, 8]])
    y = np.array([0, 0, 1, -1, -1, -1])

    co_training = CoTraining()
    co_training.fit(X1, X2, y)
    print("Predicted labels:", co_training.predict(X1, X2))
