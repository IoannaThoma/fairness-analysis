import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Generate some example data
np.random.seed(42)
y_true = np.random.randint(0, 2, 100)
y_probs = np.random.rand(100)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    np.ones_like(y_true), y_true, test_size=0.2, random_state=42
)

# Fit a logistic regression model (replace this with your model)
model = LogisticRegression()
model.fit(X_train.reshape(-1, 1), y_train)

# Get predicted probabilities
y_probs_pred = model.predict_proba(X_test.reshape(-1, 1))[:, 1]

# Calculate calibration curve and confidence intervals
prob_true, prob_pred = calibration_curve(y_test, y_probs_pred, n_bins=10)
lower_bound = np.percentile(prob_pred, 2.5, axis=0)
upper_bound = np.percentile(prob_pred, 97.5, axis=0)

# Plot the calibration curve with confidence intervals
plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
plt.fill_between(prob_pred, lower_bound, upper_bound, color='gray', alpha=0.2, label='95% CI')

# Add a diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly Calibrated')

# Customize the plot
plt.title('Calibration Curve with 95% CI')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.legend()
plt.grid(True)
plt.show()