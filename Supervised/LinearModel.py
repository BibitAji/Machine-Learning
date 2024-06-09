import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
    
    def fit(self, X, y):
        # Initialize weights and bias
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        
        # Gradient Descent
        for _ in range(self.iterations):
            y_pred = self.predict(X)
            dW = (1 / self.m) * np.dot(X.T, (y_pred - y))
            db = (1 / self.m) * np.sum(y_pred - y)
            
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db
    
    def predict(self, X):
        return np.dot(X, self.W) + self.bs

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        # Initialize weights and bias
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        
        # Gradient Descent
        for _ in range(self.iterations):
            y_pred = self.sigmoid(np.dot(X, self.W) + self.b)
            dW = (1 / self.m) * np.dot(X.T, (y_pred - y))
            db = (1 / self.m) * np.sum(y_pred - y)
            
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db
    
    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.W) + self.b)
    
    def predict(self, X, threshold=0.5):
        y_pred_prob = self.predict_proba(X)
        return [1 if i > threshold else 0 for i in y_pred_prob]


# Example usage:
if __name__ == "__main__":
    # Sample data
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3  # y = 1*x1 + 2*x2 + 3
    
    # Create and train the model
    model = LinearRegression(learning_rate=0.01, iterations=1000)
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X)
    print("Predictions:", predictions)
