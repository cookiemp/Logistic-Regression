"""
 Group Members

1.Hossaena Berhan (UGR/1447/15)

2.Hosea Shimeles (UGR/5890/15)

3.Kenawaq Birhanu (UGR/1654/15)

4.Kaleab Yohannes (UGR/6598/15)
"""

import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    # The Sigmoid Activation Function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Binary Cross-Entropy Loss Calculation
    def compute_loss(self, y_true, y_pred):
        m = len(y_true)
        # Add a small epsilon to avoid log(0) errors
        epsilon = 1e-15
        loss = - (1/m) * np.sum(y_true * np.log(y_pred + epsilon) + 
                               (1 - y_true) * np.log(1 - y_pred + epsilon))
        return loss

    # Training the model
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent Loop
        for i in range(self.num_iterations):
            # 1. Forward Pass
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # 2. Compute Loss
            loss = self.compute_loss(y, y_predicted)
            self.cost_history.append(loss)

            # 3. Backward Pass (Compute Gradients)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # 4. Update Parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 100 == 0:
                print(f"Iteration {i}: Loss {loss:.4f}")

    # Prediction method
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        # Convert probability to class label (Threshold = 0.5)
        return [1 if i > 0.5 else 0 for i in y_predicted]

# --- Main Execution Block ---

if __name__ == "__main__":
    # 1. Synthetic Data Generation
    # Features: [Feature 1, Feature 2]
    X = np.array([[2.5, 3.5], [1.1, 1.2], [4.5, 5.2], 
                  [3.1, 4.0], [5.5, 6.1], [1.5, 2.2],
                  [6.2, 5.8], [2.1, 1.8]])
    # Labels: 0 or 1
    y = np.array([0, 0, 1, 0, 1, 0, 1, 0])

    # 2. Model Initialization and Training
    model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
    model.fit(X, y)

    # 3. Output Predictions
    print("Final Predictions:", model.predict(X))

    # 4. Generate Analysis Graph (Figure 1)
    plt.figure(figsize=(8, 5))
    plt.plot(model.cost_history, color='blue')
    plt.title("Figure 1: Loss vs. Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True)
    plt.show() # This displays the graph to be saved for the report
