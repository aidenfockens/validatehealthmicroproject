#imports
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import json
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, lag, count, mean, stddev, abs as spark_abs, when, lit
from pyspark.sql.window import Window
from itertools import chain, combinations

#set up config variables
with open("./input/config.yml", "r") as file:
    config = yaml.safe_load(file)
metric_variable = next((var for var, details in config["variables"].items() if details.get("metric_variable", False)), None)
time_variable = next((var for var, details in config["variables"].items() if details.get("time_variable", False)), None)
test_variables = [var for var, details in config["variables"].items() if details.get("test_variable", False)]


# set up spark
spark = SparkSession.builder.appName("OutlierDetection").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
df = spark.read.option("header", "true").option("inferSchema", "true").csv("./input/data.csv")
df.createOrReplaceTempView("service_data")

#create pivot table for descriptions
pivot_table_df = df.select("betos_cd", "betos_desc", "pos_cd", "pos_desc", "spec_cd", "spec_desc").distinct()



def preprocess_data():
    """
    Perform preprocessing checks on the dataset:
    - Check for null values.
    - Check data types.
    - Save preprocessing results in the 'preprocessing' folder.
    """

    # Ensure preprocessing folder exists
    os.makedirs("preprocessing", exist_ok=True)

    # Check for null values
    null_counts = df.select([spark_sum(col(c).isNull().cast("int")).alias(c) for c in df.columns]).collect()[0].asDict()
    null_counts_df = pd.DataFrame(list(null_counts.items()), columns=["Column", "Null Count"])
    
    # Check data types
    dtypes = {col_name: dtype for col_name, dtype in df.dtypes}
    dtypes_df = pd.DataFrame(list(dtypes.items()), columns=["Column", "Data Type"])

    # Save results to CSV files
    null_counts_df.to_csv("preprocessing/null_counts.csv", index=False)
    dtypes_df.to_csv("preprocessing/data_types.csv", index=False)

    print("Preprocessing completed. Null counts and data types saved in 'preprocessing' folder.")




def detect_outliers(group_values=None):
    """
    Detects outliers in the dataset based on given group values.
    
    Args:
        group_values (dict or None): A dictionary where keys are test variables (betos_cd, pos_cd, spec_cd),
                                     and values are specific values to filter for.
                                     Example: {"betos_cd": "T1A", "pos_cd": "11", "spec_cd": None}

    Returns:
        Spark DataFrame: Outlier analysis results.
    """

    #filter entire df based on group values
    df_filtered = df
    if group_values:
        for var, value in group_values.items():
            if value is not None:  
                df_filtered = df_filtered.filter(col(var) == value)

    # use config variables to select columns
    df_filtered = df_filtered.select(
        col(time_variable).alias(time_variable),
        col(metric_variable).alias("total_metric"),
        *[col(var) for var in test_variables]
    )
    weekly_totals = df_filtered.groupBy(time_variable).agg(
        spark_sum("total_metric").alias("total_metric"),
        count("*").alias("num_data_points")
    )

    #calculation columns
    window_spec = Window.orderBy(time_variable)
    weekly_totals = weekly_totals.withColumn("prev_total_metric", lag("total_metric").over(window_spec))
    weekly_totals = weekly_totals.withColumn("prev_num_data_points", lag("num_data_points").over(window_spec))

    #computes a rolling mean and std over a 7 day window
    weekly_totals = weekly_totals.withColumn("rolling_mean", mean("total_metric").over(window_spec.rowsBetween(-3, 3)))
    weekly_totals = weekly_totals.withColumn("rolling_std", stddev("total_metric").over(window_spec.rowsBetween(-3, 3)))
    
    # Compute Z-score
    weekly_totals = weekly_totals.withColumn(
    "z_score",
    when(col("rolling_std") == 0, lit(0)).otherwise(
        spark_abs((col("total_metric") - col("rolling_mean")) / col("rolling_std"))
    )
    )

    #error checking
    weekly_totals = weekly_totals.withColumn(
    "rolling_std",
    when(col("rolling_mean") == 0, lit(0)).otherwise(col("rolling_std"))
    )
    weekly_totals = weekly_totals.withColumn(
    "prev_total_metric",
    when(col("prev_total_metric").isNull(), col("total_metric")).otherwise(col("prev_total_metric"))
    )
    weekly_totals = weekly_totals.withColumn(
        "prev_num_data_points",
        when(col("prev_num_data_points").isNull(), col("num_data_points")).otherwise(col("prev_num_data_points"))
    )

    #Anomaly: If Z-score is greater than 2
    weekly_totals = weekly_totals.withColumn(
    "is_outlier_z",
    when(col("rolling_std") == 0, lit(False)).otherwise(col("z_score") > 2.1)
    )

    weekly_totals = weekly_totals.withColumn(
    "mean_next_2",
    mean("total_metric").over(window_spec.rowsBetween(1, 3))
    )
    weekly_totals = weekly_totals.withColumn(
        "mean_prev_2",
        mean("total_metric").over(window_spec.rowsBetween(-3, -1))
    )



    # Anomaly: If data points are double or half of last week, with a difference of at least 600
    weekly_totals = weekly_totals.withColumn(
    "is_outlier_metric",
    ((col("total_metric") >= 3 * col("prev_total_metric")) |
     (col("total_metric") <= 0.33 * col("prev_total_metric"))) &
    (spark_abs(col("total_metric") - col("prev_total_metric")) > 5000)
    )

                                                                                                                                               
    # Anomaly: If total metric is double or half of last week, with a difference of at least 10
    weekly_totals = weekly_totals.withColumn(
    "mean_next_2",
    mean("total_metric").over(window_spec.rowsBetween(1, 3))
    )
    weekly_totals = weekly_totals.withColumn(
        "mean_prev_2",
        mean("total_metric").over(window_spec.rowsBetween(-3, -1))
    )
    weekly_totals = weekly_totals.withColumn(
    "change_point",
    (((col("mean_next_2") >= 2 * col("mean_prev_2")) |
      (col("mean_prev_2") >= 2 * col("mean_next_2"))) & 
     (spark_abs(col("mean_next_2") - col("mean_prev_2")) > 8000)) & 
    col("mean_prev_2").isNotNull() & col("mean_next_2").isNotNull()
)
    # Final outlier column (if any conditions are met)
    weekly_totals = weekly_totals.withColumn(
        "is_outlier",
        col("is_outlier_z") | col("change_point") | col("is_outlier_metric") 
    )



    return weekly_totals



def compute_anomaly_counts():
    """
    Compute anomaly counts for each unique value of each test variable. Separates beween each test_variable.
    """
    anomaly_counts_dict = {}
    
    for test_variable in test_variables:
        df_grouped = df.groupBy(time_variable, test_variable).agg(
            spark_sum("paid_amount").alias("total_metric"),
            count("*").alias("num_data_points")
        )
        
        window_spec = Window.partitionBy(test_variable).orderBy(time_variable)
        
        df_grouped = df_grouped.withColumn("prev_total_metric", lag("total_metric").over(window_spec))
        df_grouped = df_grouped.withColumn("prev_num_data_points", lag("num_data_points").over(window_spec))
        df_grouped = df_grouped.withColumn("rolling_mean", mean("total_metric").over(window_spec.rowsBetween(-3, 3)))
        df_grouped = df_grouped.withColumn("rolling_std", stddev("total_metric").over(window_spec.rowsBetween(-3, 3)))
        
        df_anomalies = df_grouped.withColumn(
            "z_score",
            when(col("rolling_std") == 0, lit(0)).otherwise(
                spark_abs((col("total_metric") - col("rolling_mean")) / col("rolling_std"))
            )
        )
        
        df_anomalies = df_anomalies.withColumn(
            "is_outlier_z",
            when(col("rolling_std") == 0, lit(False)).otherwise(col("z_score") > 2.1)
        )
        
        df_anomalies = df_anomalies.withColumn(
            "is_outlier_metric",
            ((col("total_metric") >= 2 * col("prev_total_metric")) |
             (col("total_metric") <= 0.5 * col("prev_total_metric"))) &
            (spark_abs(col("total_metric") - col("prev_total_metric")) > 5000)
        )
        
        df_anomalies = df_anomalies.withColumn(
        "mean_next_2",
        mean("total_metric").over(window_spec.rowsBetween(1, 3))
        )
        df_anomalies = df_anomalies.withColumn(
            "mean_prev_2",
            mean("total_metric").over(window_spec.rowsBetween(-3, -1))
        )
        df_anomalies = df_anomalies.withColumn(
        "change_point",
        (((col("mean_next_2") >= 2 * col("mean_prev_2")) |
        (col("mean_prev_2") >= 2 * col("mean_next_2"))) & 
        (spark_abs(col("mean_next_2") - col("mean_prev_2")) > 8000)) & 
        col("mean_prev_2").isNotNull() & col("mean_next_2").isNotNull()
    )
        
        df_anomalies = df_anomalies.withColumn(
            "is_outlier",
            col("is_outlier_z") | col("change_point") | col("is_outlier_metric")
        )
        
        anomaly_counts = (
            df_anomalies.groupBy(test_variable)
            .agg(spark_sum(col("is_outlier").cast("int")).alias("anomaly_count"))
            .collect()
        )
        
        anomaly_counts_dict[test_variable] = {row[test_variable]: row["anomaly_count"] or 0 for row in anomaly_counts}

    
    return anomaly_counts_dict


def main():
    """
    Runs outlier detection for each test variable, finds the top most anomalous value for each variable, and then 
    creates a report indicating the anomalous weeks and why they were flagged
    """

    preprocess_data()

    if not test_variables:
        print("No test variables found in config.yml. Exiting.")
        return
    
    anomaly_counts_dict = compute_anomaly_counts()
    
    top_anomalous_values = {
        test_variable: max(anomaly_counts_dict[test_variable], key=anomaly_counts_dict[test_variable].get)
        for test_variable in anomaly_counts_dict if anomaly_counts_dict[test_variable]
    }
    
    print("Top single anomalous values:",top_anomalous_values)

    outlier_summary = {}

    # Generate all possible groupings of the most anomalous values
    test_variable_subsets = list(chain.from_iterable(combinations(test_variables, r) for r in range(len(test_variables) + 1)))

    for subset in test_variable_subsets:
        group_values = {var: top_anomalous_values[var] for var in subset if var in top_anomalous_values}
        print(group_values)
        # Run detect_outliers with the subset of test variables
        result_df = detect_outliers(group_values)
        if group_values == {}:
            result_df_pd = result_df.toPandas()
        elif group_values == {'betos_cd': 'T1H', 'pos_cd': 81, 'spec_cd': '69'}:
            result_df_pd2 = result_df.toPandas()
        # Collect weeks where is_outlier is True
        outliers = result_df.filter(col("is_outlier") == True).collect()

        for row in outliers:
            week = row[time_variable]
            if week not in outlier_summary:
                outlier_summary[week] = {}

            group_key = json.dumps(group_values, sort_keys=True)

            if group_key not in outlier_summary[week]:
                outlier_summary[week][group_key] = {
                    "total_cost": row["total_metric"],
                    "anomaly_type": [],
                    "rolling_mean": row["rolling_mean"],
                    "rolling_std": row["rolling_std"],
                    "count": row["num_data_points"],
                    "prev_count": row["prev_num_data_points"],
                    "total_metric": row["total_metric"],
                    "prev_total_metric": row["prev_total_metric"],
                    "mean_prev_2": row["mean_prev_2"],
                    "mean_next_2": row["mean_next_2"]
                }

            if row["is_outlier_z"]:
                outlier_summary[week][group_key]["anomaly_type"].append("z_score")
            if row["is_outlier_metric"]:
                outlier_summary[week][group_key]["anomaly_type"].append("metric")
            if row["change_point"]:
                outlier_summary[week][group_key]["anomaly_type"].append("change")

    # Apply filtering to only keep the most specific report
    filtered_summary = {week: filter_anomalies(anomalies) for week, anomalies in outlier_summary.items()}
    make_graph(result_df_pd,result_df_pd2, filtered_summary)
    filtered_summary = dict(sorted(filtered_summary.items(), key=lambda x: -max(details["total_cost"] for details in x[1].values())))
    report = generate_report(filtered_summary)

    output_file = f"./{time_variable}/outliers_summary.txt"
    with open(output_file, "w") as f:
        f.write(report)

    
import matplotlib.pyplot as plt

def make_graph(result_df_pd, result_df_pd2, filtered_summary):
    result_df_pd['lower_bound'] = result_df_pd['rolling_mean'] - (2.1 * result_df_pd['rolling_std'])
    result_df_pd['upper_bound'] = result_df_pd['rolling_mean'] + (2.1 * result_df_pd['rolling_std'])

    # Initialize figure
    plt.figure(figsize=(14, 6))

    # Plot primary data points as black dots and connect them with a line
    plt.plot(result_df_pd[time_variable], result_df_pd['total_metric'], linestyle='-', color='black', alpha=0.5)
    plt.scatter(result_df_pd[time_variable], result_df_pd['total_metric'], color='black', label="Normal Data", alpha=1.0)

    # Plot result_df_pd2 as a simple line (without scatter points)
    plt.plot(result_df_pd2[time_variable], result_df_pd2['total_metric'], linestyle='-', color='red', alpha=0.7, label="Secondary Data")
    plt.scatter(result_df_pd2[time_variable], result_df_pd2['total_metric'], color='black', label="Normal Data", alpha=1.0)

    # Fill the acceptable range area (shaded region)
    plt.fill_between(result_df_pd[time_variable], result_df_pd['lower_bound'], result_df_pd['upper_bound'], color='black', alpha=0.1, label="Acceptable Range (±2 Std Dev)")

    # Define anomaly colors
    anomaly_colors = {
        "z_score": 'red',
        "metric": 'green',
        "change": 'blue',
        "multiple": 'purple'  # When multiple outlier conditions apply
    }

    # Track added labels for the legend
    legend_labels = {}

    # Iterate over filtered_summary to highlight anomalies
    for week, anomalies in filtered_summary.items():
        for anomaly_type in anomalies.values():
            if len(anomaly_type["anomaly_type"]) > 1:
                color = anomaly_colors["multiple"]
                label = "Multiple Outliers"
            else:
                color = anomaly_colors[anomaly_type["anomaly_type"][0]]
                label = anomaly_type["anomaly_type"][0].replace("_", " ").capitalize() + " Outliers"
                if "change" in anomaly_type["anomaly_type"]:
                    label = "Change Point"
            # Plot the anomaly points
            week_data = result_df_pd[result_df_pd[time_variable] == week]
            if not week_data.empty:
                plt.scatter(week_data[time_variable], week_data['total_metric'], color=color, label=label if label not in legend_labels else "", zorder=3)
                legend_labels[label] = color  # Ensure each label appears once in the legend

    # Ensure only unique legend labels are added
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=label)
                      for label, color in legend_labels.items()]
    legend_handles.insert(0, plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label="Entire Dataset"))
    legend_handles.insert(1, plt.Line2D([0], [0], linestyle='-', color='gray', lw=2, label="Grouping: Lab tests - other (non-Medicare fee schedule), Independent Laboratory, Clinical Laboratory"))

    # Improve X-axis readability by only labeling anomalous weeks
    outlier_ticks = list(filtered_summary.keys())
    
    plt.xticks(outlier_ticks, rotation=70, ha='right')

    if "week" in time_variable:
        plt.xlabel("Week")
    if "date" in time_variable:
        plt.xlabel("Day")
    plt.ylabel("Total Paid Amount")
    plt.title(f"Paid Amount by {time_variable} with Outliers Highlighted")
    plt.legend(handles=legend_handles)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"./{time_variable}/outliers.png", bbox_inches='tight')



def get_descriptions(group_dict):
    """
    Converts a group dictionary with codes into readable descriptions using the pivot table.
    If any value in the group_dict is None, it replaces it with 'All'.
    """

    betos_cd = group_dict.get("betos_cd")
    pos_cd = group_dict.get("pos_cd")
    spec_cd = group_dict.get("spec_cd")

    # Define a function to get the first matching row's value safely
    def get_description(code, code_column, desc_column):
        if code is None:
            return "All"
        result = pivot_table_df.filter(col(code_column) == code).select(desc_column).limit(1).collect()
        return result[0][0] if result else "Unknown"  # Return "Unknown" if not found

    # Retrieve descriptions
    betos_desc = get_description(betos_cd, "betos_cd", "betos_desc")
    pos_desc = get_description(pos_cd, "pos_cd", "pos_desc")
    spec_desc = get_description(spec_cd, "spec_cd", "spec_desc")

    return (betos_desc, pos_desc, spec_desc)



def filter_anomalies(week_anomalies):
    """
    Keeps only the most specific and highest-impact anomaly for each week.

    Args:
        week_anomalies (dict): Dictionary containing anomaly data for a week.

    Returns:
        dict: A dictionary containing only the top anomaly for the week.
    """
    if not week_anomalies:
        return {}

    # Convert dictionary keys back to tuples
    parsed_keys = [
        (eval(key), details) for key, details in week_anomalies.items()
    ]

    # Sort by:
    # - More specific groupings first (longer tuples)
    # - Higher total cost if groupings are of equal length
    parsed_keys.sort(key=lambda x: (-len(x[0]), -x[1]["total_cost"]))

    # Return only the first (top-ranked) anomaly
    top_group, top_details = parsed_keys[0]
    return {str(top_group): top_details}


def generate_report(filtered_summary):
    """
    Converts the structured JSON dictionary into a formatted, readable text report.
    """
    report_lines = []

    report_lines.append("ANOMALY REPORT SUMMARY")
    report_lines.append("=" * 50)
    report_lines.append("This report sums up total costs by week.")
    report_lines.append("")
    report_lines.append("Anomaly Types:")
    report_lines.append("  - Metric Anomaly: The week's total cost doubled or halved from the previous week.")
    report_lines.append("  - Z-Score Anomaly: The total cost for the week was over or under the rolling mean (over 7 time periods) by more than 2.1 standard deviations.")
    report_lines.append("  - Change Points: Significant change in data (the mean double/halved over a month)")
    report_lines.append("=" * 50)
    report_lines.append("\n")



    for week, anomalies in filtered_summary.items():
        report_lines.append(f"Week: {week}")
        report_lines.append("=" * 40)

        for group_key, details in anomalies.items():

            group_dict = eval(group_key.replace("'", "\""))  

            # Get readable descriptions
            descriptions = get_descriptions(group_dict)
            description_line = f"Grouping: {descriptions[0]}, {descriptions[1]}, {descriptions[2]}"
            report_lines.append(description_line)
            report_lines.append("-" * len(description_line))

            # Print each type of anomaly in a structured way
            if "metric" in details["anomaly_type"]:
                report_lines.append("Metric Anomaly:")
                report_lines.append(f"  - Total Cost: ${details['total_metric']:.2f}")
                report_lines.append(f"  - Previous Week Cost: ${details['prev_total_metric']:.2f}")

            if "change" in details["anomaly_type"]:
                if "week" in time_variable:
                    time = "Weeks"
                elif "date" in time_variable:
                    time = "Days"
                report_lines.append("Change Point:")
                report_lines.append(f"  - Total Cost: ${details['total_metric']:.2f}")
                report_lines.append(f"  - Previous Two {time} Average Cost: ${details['mean_prev_2']:.2f}")
                report_lines.append(f"  - Next Two {time} Average Cost: ${details['mean_next_2']:.2f}")
               

            if "z_score" in details["anomaly_type"]:
                z_score_value = (details["total_metric"] - details["rolling_mean"]) / details["rolling_std"]
                report_lines.append("Z-Score Anomaly:")
                report_lines.append(f"  - Total Cost: ${details['total_metric']:.2f}")
                report_lines.append(f"  - Rolling Mean: ${details['rolling_mean']:.2f}")
                report_lines.append(f"  - Standard Deviation: ${details['rolling_std']:.2f}")
                report_lines.append(f"  - Z-Score: {z_score_value:.2f}")

            report_lines.append("\n")  

        report_lines.append("\n") 

    return "\n".join(report_lines)




if __name__ == "__main__":
    main()

# Stop Spark Session
spark.stop()






