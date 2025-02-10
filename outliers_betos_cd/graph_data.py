import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# List of CSV files
csv_files = [
    "T1A.csv",
    "T1B.csv",
    "T1C.csv",
    "T1D.csv",
    "T1E.csv",
    "T1F.csv",
    "T1G.csv",
    "T1H.csv",

]

# Plot setup
plt.figure(figsize=(14, 6))

# Define colors for CSV file lines


# Outlier colors (only labeled once)
outlier_types = {
    "is_outlier_z": ("Z-Score Outliers", "red"),
    "is_outlier_count": ("Count-Based Outliers", "blue"),
    "is_outlier_metric": ("Metric-Based Outliers", "green"),
    "multiple": ("Multiple Outlier Conditions", "purple")
}

outlier_legend_labels = set()  # To avoid duplicate legend entries

# Loop through each CSV file and plot its data
for i, csv_file in enumerate(csv_files):
    df = pd.read_csv(csv_file)

    # Ensure 'service_week' is a string for plotting
    df['service_week'] = df['service_week'].astype(str)

    # Calculate the acceptable range for Z-score method (±1.5 std dev)
    df['lower_bound'] = df['rolling_mean'] - (2 * df['rolling_std'])
    df['upper_bound'] = df['rolling_mean'] + (2 * df['rolling_std'])

    # Plot total_metric (paid amount) over time
    plt.plot(df['service_week'], df['total_metric'], marker='o', linestyle='-', label=csv_file)

    # Identify and plot outliers
    for key, (label, color) in outlier_types.items():
        if key == "multiple":
            # Identify rows that satisfy multiple outlier conditions
            outliers = df[(df["is_outlier_z"] & df["is_outlier_count"]) |
                          (df["is_outlier_z"] & df["is_outlier_metric"]) |
                          (df["is_outlier_count"] & df["is_outlier_metric"])]
        else:
            outliers = df[df[key] == True]

        if not outliers.empty:
            plt.scatter(outliers['service_week'], outliers['total_metric'], color=color, label=label if label not in outlier_legend_labels else "", zorder=3)
            outlier_legend_labels.add(label)

# Labels and Formatting
plt.xticks(rotation=45)
plt.xlabel("Week")
plt.ylabel("Total Paid Amount")
plt.title("Weekly Paid Amount with Outliers Highlighted")
plt.legend(handletextpad=1, loc='upper left', bbox_to_anchor=(0, 1))  # Moves legend out of the way
plt.grid(True)

# Save and show the plot
plt.savefig("combined_weekly_paid_amount_outliers.png", bbox_inches='tight')
plt.show()
