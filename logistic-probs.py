import numpy as np
import math


def encode_categorical_variables(coefficients, categorical_values):
    # Ensure that the lengths of coefficients and categorical_values match
    if len(coefficients) != len(categorical_values):
        raise ValueError("Lengths of coefficients and categorical_values must match.")

    # Initialise a list to store the encoded values
    encoded_values = []

    # Iterate through each coefficient and categorical value
    for coef, cat_value in zip(coefficients, categorical_values):
        # If the categorical variable is present, multiply by 1; otherwise, multiply by 0
        encoded_value = coef if cat_value == 1 else 0
        encoded_values.append(encoded_value)

    return encoded_values

# Example coefficients for categorical variables
categorical_coefficients = [0.5, -0.8, 0.3]

# Example binary values for presence/absence of categorical variables (1 for present, 0 for absent)
categorical_values = [1, 0, 1]

# Encode categorical variables
encoded_values = encode_categorical_variables(categorical_coefficients, categorical_values)

print("Encoded Values:", encoded_values)




def calculate_continuous_variable_contribution(coef, centered_value, person_value):
    # Calculate the difference by subtracting the centered value from the person's value
    diff = person_value - centered_value

    # Multiply the difference by the beta coefficient
    contribution = coef * diff

    return contribution

# Example coefficient for a continuous variable
continuous_coef = 0.2

# Example centered value (to be entered manually)
centered_value = 50.0

# Example person's value for the continuous variable
person_value = 60.0

# Calculate the contribution
variable_contribution = calculate_continuous_variable_contribution(continuous_coef, centered_value, person_value)

print("Variable Contribution:", variable_contribution)






def calculate_smoking_intensity_contribution(coef, smoking_intensity):
    # Divide the smoking intensity by 10
    adjusted_intensity = smoking_intensity / 10.0

    # Exponentiate the adjusted intensity to the power -1
    exponentiated_intensity = math.exp(-1 * adjusted_intensity)

    # Center the exponentiated intensity by subtracting 0.4021541613
    centered_intensity = exponentiated_intensity - 0.4021541613

    # Multiply the centered intensity by the beta coefficient
    contribution = coef * centered_intensity

    return contribution

# Example coefficient for the smoking intensity variable
smoking_intensity_coef = 0.3

# Example smoking intensity value
smoking_intensity_value = 25.0

# Calculate the contribution
smoking_intensity_contribution = calculate_smoking_intensity_contribution(smoking_intensity_coef, smoking_intensity_value)

print("Smoking Intensity Contribution:", smoking_intensity_contribution)




def calculate_model_logit(coefficients, values, constant):
    # Ensure that the lengths of coefficients and values match
    if len(coefficients) != len(values):
        raise ValueError("Lengths of coefficients and values must match.")

    # Calculate the sum of the products of coefficients and values
    logit_sum = sum(coef * val for coef, val in zip(coefficients, values))

    # Add the model constant
    model_logit = logit_sum + constant

    return model_logit

# Example coefficients for all variables (categorical, continuous, smoking intensity)
all_coefficients = [0.5, -0.8, 0.3, 0.2, 0.3]

# Example values for all variables (encoded categorical, continuous, smoking intensity)
all_values = [1, 0, 1, 10.0, 25.0]

# Example model constant
model_constant = -1.0

# Calculate the model logit
resulting_logit = calculate_model_logit(all_coefficients, all_values, model_constant)

print("Model Logit:", resulting_logit)




# After obtaining the model logit, you can convert it into probabilities using the logistic function (also known as the sigmoid function). 


def calculate_probability_from_logit(logit):
    # Calculate the probability using the logistic function
    probability = math.exp(logit) / (1 + math.exp(logit))

    return probability

# The logit obtained from the logistic regression model
model_logit = -0.5 

# takes the logit and calculates the corresponding probability using the logistic function
resulting_probability = calculate_probability_from_logit(model_logit)

print("Probability:", resulting_probability)






def generate_probabilities_for_dataset(coefficients, values_matrix, constant):
    # Ensure that the number of coefficients matches the number of variables
    if len(coefficients) != values_matrix.shape[1]:
        raise ValueError("Number of coefficients must match the number of variables.")

    # Initialise an array to store the generated probabilities
    probabilities = np.zeros(values_matrix.shape[0])

    # Iterate over each row (individual) in the values matrix
    for i in range(values_matrix.shape[0]):
        # Get the variable values for the current individual
        individual_values = values_matrix[i, :]

        # Generate the probability for the current individual
        probabilities[i] = generate_probability_from_logit(coefficients, individual_values, constant)

    return probabilities

# Example coefficients for all variables (categorical, continuous, smoking intensity)
all_coefficients = [0.5, -0.8, 0.3, 0.2, 0.3]

# Example values matrix for all individuals (encoded categorical, continuous, smoking intensity)
# Each row represents an individual, and each column represents a variable
all_values_matrix = np.array([
    [1, 0, 1, 10.0, 25.0],
    [0, 1, 0, 15.0, 30.0],
    # ... add more rows for additional individuals
])

model_constant = -4.532506

# Generate probabilities for the entire dataset
generated_probabilities = generate_probabilities_for_dataset(all_coefficients, all_values_matrix, model_constant)

print("Generated Probabilities for the Dataset:")
print(generated_probabilities)




def reverse_engineer_probabilities(coefficients, covariate_values):
    # Compute the log-odds
    log_odds = np.dot(coefficients, covariate_values)

    # Compute the reverse-engineered probabilities
    probabilities = 1 / (1 + np.exp(-log_odds))

    return probabilities

# enter the actual coefficient values
coefficients = [0.0778868, 0, 0.3944778, -0.7434744, -0.466585, 0, 1.027152, -0.0812744, -0.0274194, 0.3553063, 0.4589971, 0.587185, 0.2597431, -1.822606, 0.0317321, -0.0308572]

model_constant = -4.532506

# Replace these with your actual covariate values
covariate_values = [   ]

# Compute reverse-engineered probabilities
predicted_probabilities = reverse_engineer_probabilities(coefficients, covariate_values)

print("Reverse-Engineered Probabilities:", predicted_probabilities)
