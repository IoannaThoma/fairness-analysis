import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import RandomForestClassifier
import numpy as np

## make an example first ##

# Generate sample data
np.random.seed(42)
data = pd.DataFrame({
    'Age': np.random.randint(18, 80, 1000),
    'Sex': np.random.choice(['Male', 'Female'], 1000),
    'Target': np.random.randint(2, size=1000)
})

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Train a classifier
classifier = RandomForestClassifier()
classifier.fit(train_data[['Age']], train_data['Target'])

# Get predicted probabilities on the test set
test_data['Predicted_Prob'] = classifier.predict_proba(test_data[['Age']])[:, 1]

# Define age groups
age_bins = [18, 30, 40, 50, 60, 80]
age_labels = ['18-30', '30-40', '40-50', '50-60', '60-80']

# Add Age Group column to the test_data
test_data['Age_Group'] = pd.cut(test_data['Age'], bins=age_bins, labels=age_labels)

# Plot calibration curves for each age group and sex
plt.figure(figsize=(12, 8))

for sex in ['Male', 'Female']:
    for age_group in age_labels:
        group_data = test_data[(test_data['Sex'] == sex) & (test_data['Age_Group'] == age_group)]
        
        prob_true, prob_pred = calibration_curve(group_data['Target'], group_data['Predicted_Prob'], n_bins=10)
        
        plt.plot(prob_pred, prob_true, marker='o', label=f'{sex} - {age_group} - Brier Score: {brier_score_loss(group_data["Target"], group_data["Predicted_Prob"]):.2f}')

# Reference diagonal
plt.plot([0, 1], [0, 1], '--', color='gray', label='Reference Diagonal')

# Additional plot settings
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curves for Different Age Groups and Sex')
plt.legend()
plt.savefig('../../../fairness-analysis/calib-curve-groups-example.pdf')

plt.show()
## example ends ##




## application ##
df = pd.read_csv('dataframe_with_groups')

# Assuming you have a DataFrame df with a column 'predicted_prob' for predicted probabilities and 'actual_label' for true labels
# You also have a column 'group_variable' indicating the groups

# List of unique values in the 'group_variable' column
group_values = df['education_group'].unique()


# Create a figure and axis
fig, ax = plt.subplots()

# Iterate over each group and plot the calibration curve
for group_value in group_values:
    group_df = df[df['education_group'] == group_value].copy()  # Make a copy to avoid SettingWithCopyWarning
    y_true = np.array(group_df['yn_lc_all'])
    y_probs = np.array(group_df['plco']/100)
    # Calculate calibration curve    
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=100)

    # Plot calibration curve
    ax.plot(y_probs, y_true, label=f'Group {group_value}')

# Customise the plot
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Curves for Different Groups')
ax.legend(loc='best')
plt.savefig('../../../fairness-analysis/calib-curve-education-grp.pdf')

# Show the plot
plt.show()

#######

# List of unique values in the 'group_variable' column
group_values = df['Age_Group'].unique()

# Create a figure and axis
fig, ax = plt.subplots()

# Iterate over each group and plot the calibration curve
for group_value in group_values:
    group_df = df[df['Age_Group'] == group_value].copy()  # Make a copy to avoid SettingWithCopyWarning
    
    y_true = np.array(group_df['yn_lc_all'])
    y_probs = np.array(group_df['plco']/100)    
    # Calculate calibration curve    
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
    
    # Plot calibration curve
    ax.plot(prob_pred, prob_true, label=f'Group {group_value}')

# Customise the plot
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Curves for Different Groups')
ax.legend(loc='best')
plt.savefig('../../../fairness-analysis/calib-curve-age-grp.pdf')

# Show the plot
plt.show()
