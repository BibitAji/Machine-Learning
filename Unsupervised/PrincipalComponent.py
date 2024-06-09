import numpy as np

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit_transform(self, X):
        # Center the data
        X_centered = X - np.mean(X, axis=0)
        
        # Compute covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort eigenvectors by eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Select the top n_components eigenvectors
        eigenvectors = eigenvectors[:, :self.n_components]
        
        # Transform the data
        X_reduced = np.dot(X_centered, eigenvectors)
        
        return X_reduced

# Example usage:
if __name__ == "__main__":
    # Sample data
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])

    # Create and apply PCA
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    print("Reduced Data:\n", X_reduced)
