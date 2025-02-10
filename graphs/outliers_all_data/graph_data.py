import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data from CSV
df = pd.read_csv("weekly_outliers.csv")




df['lower_bound'] = df['rolling_mean'] - (2.2 * df['rolling_std'])
df['upper_bound'] = df['rolling_mean'] + (2.2 * df['rolling_std'])

# Plot total_metric (paid amount) over time
plt.figure(figsize=(14, 6))
plt.plot(df[time_variable], df['total_metric'], marker='o', linestyle='-', label="Total Paid Amount", color='black')

# Fill the acceptable range area (shaded region)
plt.fill_between(df[time_variable], df['lower_bound'], df['upper_bound'], color='gray', alpha=0.3, label="Acceptable Range (±2 Std Dev)")

# Highlight different outliers with different colors
outliers_z = df[df['is_outlier_z'] == True]
outliers_count = df[df['is_outlier_count'] == True]
outliers_metric = df[df['is_outlier_metric'] == True]

# Handle cases where multiple outlier conditions are met (e.g., Z-score & count)
outliers_multiple = df[(df['is_outlier_z'] & df['is_outlier_count']) |
                       (df['is_outlier_z'] & df['is_outlier_metric']) |
                       (df['is_outlier_count'] & df['is_outlier_metric'])]

# Plot outliers in different colors
plt.scatter(outliers_z[time_variable], outliers_z['total_metric'], color='red', label="Z-Score Outliers", zorder=3)
plt.scatter(outliers_count[time_variable], outliers_count['total_metric'], color='blue', label="Count-Based Outliers", zorder=3)
plt.scatter(outliers_metric[time_variable], outliers_metric['total_metric'], color='green', label="Metric-Based Outliers", zorder=3)
plt.scatter(outliers_multiple[time_variable], outliers_multiple['total_metric'], color='purple', label="Multiple Outlier Conditions", zorder=3)

# Labels and Formatting
plt.xticks(rotation=45)
plt.xlabel("Week")
plt.ylabel("Total Paid Amount")
plt.title("Weekly Paid Amount with Outliers Highlighted")
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig("weekly_paid_amount_outliers.png", bbox_inches='tight')
plt.show()
