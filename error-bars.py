import matplotlib.pyplot as plt
import numpy as np

# Sample data
predicted_probs = np.linspace(0, 1, 100)
observed_probs = np.random.rand(100) * 0.1 + predicted_probs

# Confidence intervals (just for illustration)
confidence_intervals = 0.05 * np.ones_like(observed_probs)

# Plotting
plt.errorbar(predicted_probs, observed_probs, yerr=confidence_intervals, fmt='o', label='Calibration Curve with 95% CI')
plt.plot([0, 1], [0, 1], '--', color='gray', label='Reference Diagonal')

# Additional plot settings
plt.xlabel('Predicted Probabilities')
plt.ylabel('Observed Probabilities')
plt.title('Calibration Curve with Error Bars (95% CI)')
plt.legend()
plt.show()
