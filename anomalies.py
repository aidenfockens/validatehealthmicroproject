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

#set up pivot table
pivot_table_df = pd.read_csv("./input/codes_pivot_table.csv")




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
        col(time_variable).alias("service_week"),
        col(metric_variable).alias("total_metric"),
        *[col(var) for var in test_variables]
    )
    weekly_totals = df_filtered.groupBy("service_week").agg(
        spark_sum("total_metric").alias("total_metric"),
        count("*").alias("num_data_points")
    )

    #calculation columns
    window_spec = Window.orderBy("service_week")
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
    when(col("rolling_std") == 0, lit(False)).otherwise(col("z_score") > 2.0)
    )

    weekly_totals = weekly_totals.withColumn(
    "mean_next_3",
    mean("total_metric").over(window_spec.rowsBetween(1, 3))
    )
    weekly_totals = weekly_totals.withColumn(
        "mean_prev_3",
        mean("total_metric").over(window_spec.rowsBetween(-3, -1))
    )



    # Anomaly: If data points are double or half of last week, with a difference of at least 600
    weekly_totals = weekly_totals.withColumn(
    "is_outlier_metric",
    ((col("total_metric") >= 2 * col("prev_total_metric")) |
     (col("total_metric") <= 0.5 * col("prev_total_metric"))) &
    (spark_abs(col("total_metric") - col("prev_total_metric")) > 5000)
    )


    # Anomaly: If total metric is double or half of last week, with a difference of at least 10
    weekly_totals = weekly_totals.withColumn(
    "mean_next_3",
    mean("total_metric").over(window_spec.rowsBetween(1, 3))
    )
    weekly_totals = weekly_totals.withColumn(
        "mean_prev_3",
        mean("total_metric").over(window_spec.rowsBetween(-3, -1))
    )
    weekly_totals = weekly_totals.withColumn(
    "is_outlier_count",
    (((col("mean_next_3") >= 2 * col("mean_prev_3")) |
      (col("mean_prev_3") >= 2 * col("mean_next_3"))) & 
     (spark_abs(col("mean_next_3") - col("mean_prev_3")) > 8000)) & 
    col("mean_prev_3").isNotNull() & col("mean_next_3").isNotNull()
)
    # Final outlier column (if any conditions are met)
    weekly_totals = weekly_totals.withColumn(
        "is_outlier",
        col("is_outlier_z") | col("is_outlier_count") | col("is_outlier_metric") 
    )



    return weekly_totals



def compute_anomaly_counts():
    """
    Compute anomaly counts for each unique value of each test variable. Separates beween each test_variable.
    """
    anomaly_counts_dict = {}
    
    for test_variable in test_variables:
        df_grouped = df.groupBy("service_week", test_variable).agg(
            spark_sum("paid_amount").alias("total_metric"),
            count("*").alias("num_data_points")
        )
        
        window_spec = Window.partitionBy(test_variable).orderBy("service_week")
        
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
            when(col("rolling_std") == 0, lit(False)).otherwise(col("z_score") > 2.0)
        )
        
        df_anomalies = df_anomalies.withColumn(
            "is_outlier_metric",
            ((col("total_metric") >= 2 * col("prev_total_metric")) |
             (col("total_metric") <= 0.5 * col("prev_total_metric"))) &
            (spark_abs(col("total_metric") - col("prev_total_metric")) > 5000)
        )
        
        df_anomalies = df_anomalies.withColumn(
        "mean_next_3",
        mean("total_metric").over(window_spec.rowsBetween(1, 3))
        )
        df_anomalies = df_anomalies.withColumn(
            "mean_prev_3",
            mean("total_metric").over(window_spec.rowsBetween(-3, -1))
        )
        df_anomalies = df_anomalies.withColumn(
        "is_outlier_count",
        (((col("mean_next_3") >= 2 * col("mean_prev_3")) |
        (col("mean_prev_3") >= 2 * col("mean_next_3"))) & 
        (spark_abs(col("mean_next_3") - col("mean_prev_3")) > 8000)) & 
        col("mean_prev_3").isNotNull() & col("mean_next_3").isNotNull()
    )
        
        df_anomalies = df_anomalies.withColumn(
            "is_outlier",
            col("is_outlier_z") | col("is_outlier_count") | col("is_outlier_metric")
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

    if not test_variables:
        print("No test variables found in config.yml. Exiting.")
        return
    
    #testing out compute_anomaly_counts():
    anomaly_counts_dict = compute_anomaly_counts()
    
    top_anomalous_values = {
        test_variable: max(anomaly_counts_dict[test_variable], key=anomaly_counts_dict[test_variable].get)
        for test_variable in anomaly_counts_dict if anomaly_counts_dict[test_variable]
    }
    
    print("Top single anomalous values:",top_anomalous_values)

    # Dictionary to hold report data
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

            result_df_pd['lower_bound'] = result_df_pd['rolling_mean'] - (2.0 * result_df_pd['rolling_std'])
            result_df_pd['upper_bound'] = result_df_pd['rolling_mean'] + (2.0 * result_df_pd['rolling_std'])

            # Plot total_metric (paid amount) over time
            plt.figure(figsize=(14, 6))
            plt.plot(result_df_pd[time_variable], result_df_pd['total_metric'], marker='o', linestyle='-', label="Total Paid Amount", color='black')

            # Fill the acceptable range area (shaded region)
            plt.fill_between(result_df_pd[time_variable], result_df_pd['lower_bound'], result_df_pd['upper_bound'], color='gray', alpha=0.3, label="Acceptable Range (±2 Std Dev)")

            # Highlight different outliers with different colors
            outliers_z = result_df_pd[result_df_pd['is_outlier_z'] == True]
            outliers_count = result_df_pd[result_df_pd['is_outlier_count'] == True]
            outliers_metric = result_df_pd[result_df_pd['is_outlier_metric'] == True]

            # Handle cases where multiple outlier conditions are met (e.g., Z-score & count)
            outliers_multiple = result_df_pd[(result_df_pd['is_outlier_z'] & result_df_pd['is_outlier_count']) |
                                (result_df_pd['is_outlier_z'] & result_df_pd['is_outlier_metric']) |
                                (result_df_pd['is_outlier_count'] & result_df_pd['is_outlier_metric'])]

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
            plt.savefig("./output/weekly_paid_amount_outliers.png", bbox_inches='tight')
            plt.show()

        # Collect weeks where is_outlier is True
        outliers = result_df.filter(col("is_outlier") == True).collect()

        for row in outliers:
            week = row["service_week"]
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
                    "prev_total_metric": row["prev_total_metric"]
                }

            if row["is_outlier_z"]:
                outlier_summary[week][group_key]["anomaly_type"].append("z_score")
            if row["is_outlier_metric"]:
                outlier_summary[week][group_key]["anomaly_type"].append("metric")
            if row["is_outlier_count"]:
                outlier_summary[week][group_key]["anomaly_type"].append("count")

    # Apply filtering to only keep the most specific report
    filtered_summary = {week: filter_anomalies(anomalies) for week, anomalies in outlier_summary.items()}

    filtered_summary = dict(sorted(filtered_summary.items(), key=lambda x: -max(details["total_cost"] for details in x[1].values())))
    #generate text for the report based on anomaly and data
    report = generate_report(filtered_summary)

    # Save filtered outlier_summary dictionary to a .txt file
    output_file = "./output/filtered_outlier_summary.txt"
    with open(output_file, "w") as f:
        f.write(report)

    



def get_descriptions(group_dict):
    """
    Converts a group dictionary with codes into readable descriptions using the pivot table.
    If any value in the group_dict is None, it replaces it with 'All'.
    """

    betos_cd = group_dict.get("betos_cd")
    pos_cd = group_dict.get("pos_cd")
    spec_cd = group_dict.get("spec_cd")

    # If the grouping contains None, replace it with 'All'
    betos_desc = pivot_table_df.loc[pivot_table_df["betos_cd"] == betos_cd, "betos_desc"].values[0] if betos_cd else "All"
    pos_desc = pivot_table_df.loc[pivot_table_df["pos_cd"] == pos_cd, "pos_desc"].values[0] if pos_cd else "All"
    spec_desc = pivot_table_df.loc[pivot_table_df["spec_cd"] == spec_cd, "spec_desc"].values[0] if spec_cd else "All"

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
    report_lines.append("  - Z-Score Anomaly: The total cost for the week was over or under the rolling mean (over 7 days) by more than 2 standard deviations.")
    report_lines.append("  - Count Anomaly: The number of events for this category doubled or halved compared to the previous week.")
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

            if "count" in details["anomaly_type"]:
                report_lines.append("Count Anomaly:")
                report_lines.append(f"  - Total Cost: ${details['total_metric']:.2f}")
                report_lines.append(f"  - Previous Week Cost: ${details['prev_total_metric']:.2f}")
                report_lines.append(f"  - Events This Week: {details['count']}")
                report_lines.append(f"  - Events Last Week: {details['prev_count']}")

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


def test_variable(index):
    """
    Used for EDA, to generate the csvs I have in my other folders. Choose the index out of test_variables to generate csvs 
    for each value of that test variable.
    """

    if not test_variables:
        print("No test variables found in config.yml. Exiting.")
        return

    # Choose the test variable
    first_test_variable = test_variables[index]

    # Get all unique values of the first test variable
    unique_values = df.select(first_test_variable).distinct().rdd.flatMap(lambda x: x).collect()


    # Create output folder for the test variable results
    base_output_folder = f"outliers_{first_test_variable}"
    os.makedirs(base_output_folder, exist_ok=True)

    for value in unique_values:


        # Run outlier detection for the specific value
        result_df = detect_outliers({first_test_variable: value})

        # Save CSV
        output_csv = os.path.join(base_output_folder, f"{value}.csv")
        result_df.toPandas().to_csv(output_csv, index=False)

        print(f"Saved: {output_csv}")




if __name__ == "__main__":
    main()

# Stop Spark Session
spark.stop()






