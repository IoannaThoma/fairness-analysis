import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os
import pandas as pd


# Set the working directory
# path where the data is located
working_directory = '../Desktop/lung-health-dataset/LHC_data_anon_20230831'
os.chdir(working_directory)

# Load data using pandas
# Replace 'your_data_file.csv' with the actual name of your data file
data_file = 'lhcdata_mcip_anon.xlsx'
df = pd.read_excel(data_file)

# Display the first few rows of the loaded data
print(df.head())

# Filter rows based on the specified condition
# remove those who did not receive a scan although they were eligible and reffered for one
condition = ~(df['yn_eligplco'] == 1) | ~(df['yn_t00ldct'] == 0)
filtered_df = df[condition]

# Display the filtered DataFrame
print(filtered_df)

# Risk scores and outcomes 
y_true = np.array(filtered_df['yn_lc_all'])
print(y_true)
y_probs = np.array(filtered_df['plco']/100)
print(y_probs)

# Compute ROC curve and AUC

# Assuming y_true and y_scores are your true labels and predicted scores for multilabel classification
roc_auc = roc_auc_score(y_true, y_probs)

print("ROC AUC Score:", roc_auc)

from sklearn.metrics import roc_curve

# Assuming y_true and y_probs are your true labels and predicted scores
fpr, tpr, thresholds = roc_curve(y_true, y_probs)

# fpr is the False Positive Rate
print("False Positive Rate (FPR):", fpr)


# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('../../../fairness-analysis/auc-curve.pdf')

plt.show()
##################################
# adding confidence intervals in AUC 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample

# Assuming you have your true labels (y_true) and predicted probabilities (y_probs) ready

# Perform bootstrapping to calculate confidence intervals for AUC
n_bootstraps = 1000
auc_values = []

for i in range(n_bootstraps):
    indices = resample(range(len(y_true)))
    fpr_bootstrap, tpr_bootstrap, i = roc_curve(y_true[indices], y_probs[indices])
    auc_values.append(auc(fpr_bootstrap, tpr_bootstrap))

# Calculate the confidence interval (e.g., 95%)
lower_bound = np.percentile(auc_values, 2.5)
upper_bound = np.percentile(auc_values, 97.5)

# Plot the ROC curve
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f} [{:.2f}-{:.2f}])'.format(roc_auc, lower_bound, upper_bound))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Add labels and legend
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('../../../fairness-analysis/roc-curve-with-ci.pdf')

# Display the plot
plt.show()
#########################

# Calibration curves

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


# Assuming y_true and y_probs are your true labels and predicted probabilities
prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
lower_bound = np.percentile(prob_pred, 2.5, axis=0)
upper_bound = np.percentile(prob_pred, 97.5, axis=0)


# Plot the calibration curve with CIs
plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
plt.fill_between(prob_pred, lower_bound, upper_bound, color='gray', alpha=0.2, label='95% CI')
# Add a diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend()
plt.show()


#### Calibration Curve with Confidence Interval
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

# Assuming you have your true labels (y_true) and predicted probabilities (y_probs) ready

# Calculate calibration curve and confidence intervals
prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=100)
lower_bound = np.percentile(prob_pred, 2.5, axis=0)
upper_bound = np.percentile(prob_pred, 97.5, axis=0)

# Plot the calibration curve with different colors for upper and lower bounds
plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve', color='blue')
plt.fill_between(prob_pred, lower_bound, upper_bound, color='lightblue', alpha=0.2, label='95% CI', edgecolor='none')

# Add numerical values to the label
ci_label = '95% CI [{:.2f}-{:.2f}]'.format(np.percentile(lower_bound, 2.5), np.percentile(upper_bound, 97.5))

# Add labels and legend
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title(f'Calibration Curve with Confidence Interval ({ci_label})')
plt.legend(loc='upper right')
plt.savefig('../../../fairness-analysis/calib-curve-bins100-ci.pdf')

# Display the plot
plt.show()
##########################

prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=7, strategy='uniform')
lower_bound = np.percentile(prob_pred, 2.5, axis=0)
upper_bound = np.percentile(prob_pred, 97.5, axis=0)

# Plot the calibration curve
plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
plt.fill_between(prob_pred, lower_bound, upper_bound, color='gray', alpha=0.2, label='95% CI')

plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend()

plt.show()


# Assuming y_true and y_probs are your true labels and predicted probabilities

# Define equal width bins
n_bins = 7  # Adjust the number of bins as needed

# Calculate calibration curve
prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins, strategy='uniform')
lower_bound = np.percentile(prob_pred, 2.5, axis=0)
upper_bound = np.percentile(prob_pred, 97.5, axis=0)
lower_bound = prob_true - 0.05 # same result as above
upper_bound = prob_true + 0.05

# Confidence intervals (just for illustration)
confidence_intervals = 0.05 * np.ones_like(prob_true)

# Plotting
plt.errorbar(prob_pred, prob_true, yerr=confidence_intervals, fmt='o', label='Calibration Curve with 95% CI')

# Plot the calibration curve with specific axes limits
plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
plt.fill_between(prob_pred, lower_bound, upper_bound, color='gray', alpha=0.2, label='95% CI')

plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve with Equal Width Bins')
plt.legend()

# Set specific axes limits
plt.xlim(0, 0.4)  # Adjust as needed
plt.ylim(0, 0.4)  # Adjust as needed

plt.savefig('../../../fairness-analysis/seven-equal-bins-calibr.pdf')
plt.show()


# Define the number of quantiles
n_quantiles = 5  # Adjust the number of quantiles as needed

# Calculate calibration curve with quantile strategy
prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_quantiles, strategy='quantile')
lower_bound = np.percentile(prob_pred, 2.5, axis=0)
upper_bound = np.percentile(prob_pred, 97.5, axis=0)


# Plot the calibration curve with specific axes limits
plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
plt.fill_between(prob_pred, lower_bound, upper_bound, color='gray', alpha=0.2, label='95% CI')

plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve with Quantile Strategy')
plt.legend()

# Set specific axes limits
plt.xlim(0, 0.2)  # Adjust as needed
plt.ylim(0, 0.2)  # Adjust as needed


plt.savefig('../../../fairness-analysis/five-quantiles-calibr.pdf')
plt.show()


#Considering your probability distribution statistics (min: 0.000, 1st qu: 0.550, median: 1.730, mean: 3.019, 3rd qu: 3.955, max: 32.930), you might want to use a combination of the strategies mentioned. 

#    Equal Width Bins:
#        [0-5%, 5-10%, 10-15%, 15-20%, 20-25%, 25-30%, 30-35%]

#    Quantile Bins:
#       Quartiles: [0-0.55%, 0.55-1.73%, 1.73-3.019%, 3.019-3.955%, 3.955-32.93%]

#    Logarithmic Bins:
#        [0-1%, 1-10%, 10-100%]