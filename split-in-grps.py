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

# Determine min and max age values
min_age = df['age'].min()
max_age = df['age'].max()

# Define the number of age groups (adjust as needed)
num_age_groups = 5

# Create age bins
age_bins = np.linspace(min_age, max_age, num=num_age_groups + 1, dtype=int)
age_labels = [f"{age_start}-{age_end}" for age_start, age_end in zip(age_bins[:-1], age_bins[1:])]

# Add Age Group column to the data
df['Age_Group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, include_lowest=True)

# Count occurrences of each age group
age_group_counts = df['Age_Group'].value_counts()

# Display the age ranges and corresponding counts
for age_group, count in age_group_counts.items():
    print(f"Age Range for '{age_group}': {age_group}, Count: {count}")



# Determine min and max bmi values
min_bmi = df['bmi'].min()
max_bmi = df['bmi'].max()

# Define the number of bmi groups (adjust as needed)
num_bmi_groups = 5

# Create bmi bins
bmi_bins = np.linspace(min_bmi, max_bmi, num=num_bmi_groups + 1, dtype=int)
bmi_labels = [f"{bmi_start}-{bmi_end}" for bmi_start, bmi_end in zip(bmi_bins[:-1], bmi_bins[1:])]

# Add Age Group column to the data
df['BMI_Group'] = pd.cut(df['bmi'], bins=bmi_bins, labels=bmi_labels, include_lowest=True)

# Count occurrences of each bmi group
bmi_group_counts = df['BMI_Group'].value_counts()

# Display the bmi ranges and corresponding counts
for bmi_group, count in bmi_group_counts.items():
    print(f"BMI Range for '{bmi_group}': {bmi_group}, Count: {count}")

    

df['education_group'] = df['education'].apply(lambda x: 'lower education' if x <= 3 else 'higher education')
df['gender'] = df['sex_f'].map({0: 'male', 1: 'female'})


#smok_cpd
# Determine min and max smok_cpd values
min_smok_cpd = df['smok_cpd'].min()
max_smok_cpd = df['smok_cpd'].max()

# Define the number of smok_cpd groups (adjust as needed)
num_smok_cpd_groups = 10

# Create smok_cpd bins
smok_cpd_bins = np.linspace(min_smok_cpd, max_smok_cpd, num=num_smok_cpd_groups + 1, dtype=int)
smok_cpd_labels = [f"{smok_cpd_start}-{smok_cpd_end}" for smok_cpd_start, smok_cpd_end in zip(smok_cpd_bins[:-1], smok_cpd_bins[1:])]

# Add Age Group column to the data
df['Smok_CPD_Group'] = pd.cut(df['smok_cpd'], bins=smok_cpd_bins, labels=smok_cpd_labels, include_lowest=True)

# Count occurrences of each smok_cpd group
smok_cpd_group_counts = df['Smok_CPD_Group'].value_counts()

# Display the smok_cpd ranges and corresponding counts
for smok_cpd_group, count in smok_cpd_group_counts.items():
    print(f"Cigarettes-per-day Range for '{smok_cpd_group}': {smok_cpd_group}, Count: {count}")

    