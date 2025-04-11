import pandas as pd
import math
import time
import numpy as np
# Import for data splitting
from sklearn.model_selection import train_test_split
# Import metrics for model evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
# Directory management
import os

# Output directory configuration
output_dir = "GNB/"
# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory '{output_dir}' ensured.")

# Data loading and preprocessing
# Load combined dataset with missing values handled
df = pd.read_csv("heart+disease/combined.data", header=None, na_values='?')
df.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
              "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
df = df.apply(pd.to_numeric, errors='coerce')
print(f"Original shape: {df.shape}")
# Remove incomplete records
df.dropna(inplace=True)
print(f"Shape after dropping NaNs: {df.shape}")

# Feature extraction and target preparation
X = df.drop('num', axis=1).values
y_original = df['num'].values
# Convert target to binary classification problem
y = np.where(y_original > 0, 1, 0)

# Data partitioning
# 80% training, 20% testing with class balance preservation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Gaussian Naive Bayes classifier implementation
class GaussianNaiveBayes:
    def __init__(self):
        self._classes = None
        self._priors = None
        self._means = None
        self._stds = None
        self._epsilon = 1e-9  # Small value to prevent division by zero

    def fit(self, X, y):
        """Calculate class parameters from training data."""
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        self._means = {}
        self._stds = {}
        self._priors = {}
        for idx, c in enumerate(self._classes):
            X_c = X[y == c]  # Extract samples for current class
            self._means[c] = X_c.mean(axis=0)
            self._stds[c] = X_c.std(axis=0) + self._epsilon
            self._priors[c] = X_c.shape[0] / float(n_samples)

    def _gaussian_pdf(self, x, mean, std):
        """Compute log of Gaussian PDF for numerical stability."""
        exponent = -((x - mean)**2) / (2 * (std**2))
        log_prob = exponent - np.log(std)
        return log_prob

    def _predict_single(self, x):
        """Predict class for a single observation using log probabilities."""
        posteriors = {}
        for idx, c in enumerate(self._classes):
            prior_log = np.log(self._priors[c])
            likelihood_log_sum = np.sum(self._gaussian_pdf(x, self._means[c], self._stds[c]))
            # Log posterior = log prior + sum of log likelihoods
            posteriors[c] = prior_log + likelihood_log_sum
        # Return most probable class
        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        """Generate predictions for multiple observations."""
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

# Model training and evaluation
print("\n--- Gaussian Naive Bayes ---")
gnb = GaussianNaiveBayes()

# Train model and measure training time
start_train_time = time.time()
gnb.fit(X_train, y_train)
end_train_time = time.time()
train_time = end_train_time - start_train_time
print(f"Training completed in {train_time:.4f} seconds.")

# Generate predictions and measure inference time
start_pred_time = time.time()
y_pred = gnb.predict(X_test)
end_pred_time = time.time()
pred_time = end_pred_time - start_pred_time
print(f"Prediction on test set completed in {pred_time:.4f} seconds.")

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on the test set: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# Performance evaluation and visualization

# Generate detailed classification metrics
print("\nClassification Report:")
report = classification_report(y_test, y_pred, target_names=["No Disease (0)", "Disease (1)"])
print(report)
# Save metrics to file
report_filepath = os.path.join(output_dir, "gnb_classification_report.txt")
try:
    with open(report_filepath, 'w') as f:
        f.write(f"Gaussian Naive Bayes Classification Report\n")
        f.write(f"Dataset: combined.data (after dropping rows with '?')\n")
        f.write(f"Evaluation: 80/20 Train/Test Split (random_state=42, stratify=y)\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)
    print(f"Classification report saved to {report_filepath}")
except Exception as e:
    print(f"Error saving classification report: {e}")


# Generate confusion matrix
print("\nConfusion Matrix (counts):")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize and save confusion matrix
try:
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Predicted No Disease (0)", "Predicted Disease (1)"],
                yticklabels=["Actual No Disease (0)", "Actual Disease (1)"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Gaussian Naive Bayes Confusion Matrix')

    # Save visualization
    cm_filepath = os.path.join(output_dir, "gnb_confusion_matrix.png")
    plt.savefig(cm_filepath, bbox_inches='tight')
    print(f"Confusion matrix plot saved to {cm_filepath}")

    plt.show()
    plt.close()
except Exception as e:
    print(f"Error generating or saving confusion matrix plot: {e}")

print("\nScript finished. GNB results saved in", output_dir)