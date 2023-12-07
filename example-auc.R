install.packages("pROC")
library(pROC)

# Assuming 'y_true' is the true binary outcomes and 'y_scores' is the predicted probabilities
# You need to replace 'y_true' and 'y_scores' with your actual data
# 'y_scores' should contain the predicted probabilities for the positive class

# Example data
y_true <- c(0, 1, 0, 1, 1, 0, 1)
y_scores <- c(0.1, 0.8, 0.2, 0.9, 0.7, 0.3, 0.6)

# Compute ROC curve and AUC
roc_data <- roc(y_true, y_scores)
roc_auc <- auc(roc_data)

# Plot ROC curve
plot(roc_data, main='Receiver Operating Characteristic (ROC) Curve', col='blue', lwd=2)
abline(a=0, b=1, lty=2, col='red')
legend('bottomright', legend=sprintf('AUC = %.2f', roc_auc), col='blue', lwd=2)

