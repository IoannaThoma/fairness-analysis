import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report

# Sample data (replace this with the dataset)
np.random.seed(42)
X = np.random.rand(1000, 5)
y = (X.sum(axis=1) > 2.5).astype(int)

# Protected attribute (replace this with your protected attribute)
protected_attribute = np.random.choice([0, 1], size=len(y))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, protected_train, protected_test = train_test_split(
    X, y, protected_attribute, test_size=0.2, random_state=42
)

# Train a logistic regression model (replace this with your model)
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 1. Confusion Matrix Disparities
conf_matrix = confusion_matrix(y_test, y_pred)

# 2. Statistical Parity
statistical_parity_disparity = np.abs(np.mean(y_pred[protected_test == 0]) - np.mean(y_pred[protected_test == 1]))

# 3. Equalized Odds
# For simplicity, let's use the ROC-AUC score for equalized odds
roc_auc_group_0 = roc_auc_score(y_test[protected_test == 0], y_pred_proba[protected_test == 0])
roc_auc_group_1 = roc_auc_score(y_test[protected_test == 1], y_pred_proba[protected_test == 1])
equalized_odds_disparity = np.abs(roc_auc_group_0 - roc_auc_group_1)

# 4. Demographic Parity
demographic_parity_disparity = np.abs(np.mean(y_pred[protected_test == 0]) - np.mean(y_pred[protected_test == 1]))

# Print or use the results as needed
print("Confusion Matrix Disparities:")
print(conf_matrix)

print("\nStatistical Parity Disparity:")
print(statistical_parity_disparity)

print("\nEqualized Odds Disparity:")
print(equalized_odds_disparity)

print("\nDemographic Parity Disparity:")
print(demographic_parity_disparity)

# Other fairness metrics and evaluations can be added as needed

# Additional metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
