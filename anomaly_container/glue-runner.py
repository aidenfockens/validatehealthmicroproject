#imports
import os
import yaml
import json
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, lag, count, mean, stddev, abs as spark_abs, when, lit
from pyspark.sql.window import Window
from itertools import chain, combinations
import sys
import boto3
import io

import ast


# set up spark
spark = SparkSession.builder \
    .appName("OutlierDetection") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

s3_bucket = "validatemicroproject"
input_path = f"s3://{s3_bucket}/pre_data/"
output_path =  f"s3://{s3_bucket}/output/"

s3 = boto3.client("s3")
config_obj = s3.get_object(Bucket=s3_bucket, Key="pre_data/config.yml")
config = yaml.safe_load(config_obj["Body"].read().decode("utf-8"))





metric_variable = next((var for var, details in config["variables"].items() if details.get("metric_variable", False)), None)
time_variable = next((var for var, details in config["variables"].items() if details.get("time_variable", False)), None)
test_variables = [var for var, details in config["variables"].items() if details.get("test_variable", False)]


df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path + "data.csv")
df.createOrReplaceTempView("service_data")

obj = s3.get_object(Bucket=s3_bucket, Key="pre_data/codes_pivot_table.csv")
pivot_table_df = pd.read_csv(io.BytesIO(obj["Body"].read()))





def detect_outliers(df,group_values=None):
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
    when(col("rolling_std") == 0, lit(False)).otherwise(col("z_score") > 2)
    )

    # Anomaly: If data points are double or half of last week, with a difference of at least 600
    weekly_totals = weekly_totals.withColumn(
    "is_outlier_metric",
    ((col("total_metric") >= 2 * col("prev_total_metric")) |
     (col("total_metric") <= 0.5 * col("prev_total_metric"))) &
    (spark_abs(col("total_metric") - col("prev_total_metric")) > 600)
    )


    # Anomaly: If total metric is double or half of last week, with a difference of at least 10
    weekly_totals = weekly_totals.withColumn(
        "is_outlier_count",
        ((col("num_data_points") >= 2 * col("prev_num_data_points")) |
        (col("num_data_points") <= 0.5 * col("prev_num_data_points"))) &
        ((col("num_data_points") > 10) | (col("prev_num_data_points") > 10))
    )

    # Final outlier column (if any conditions are met)
    weekly_totals = weekly_totals.withColumn(
        "is_outlier",
        col("is_outlier_z") | col("is_outlier_count") | col("is_outlier_metric")
    )

    return weekly_totals



def main():
    """
    Runs outlier detection for each test variable, finds the top most anomalous value for each variable, and then 
    creates a report indicating the anomalous weeks and why they were flagged
    """

    if not test_variables:
        print("No test variables found in config.yml. Exiting.")
        return

    # Find the most anomalous value for each test variable
    top_anomalous_values = {}
    for test_variable in test_variables:

        unique_values = [row[test_variable] for row in df.select(test_variable).distinct().collect()]

        outlier_counts = []

        for value in unique_values:
            #call my detect_outliers function for every value in every test_variable, count the outliers
            result_df = detect_outliers(df,{test_variable: value})
            outlier_count = result_df.filter(col("is_outlier") == True).count()
            outlier_counts.append((value, outlier_count))

        # Keep only the most anomalous value for each test variable
        if outlier_counts:
            top_values = sorted(outlier_counts, key=lambda x: x[1], reverse=True)[:1]
            top_anomalous_values[test_variable] = top_values[0][0]  # Only keep the value, not the count



    # Dictionary to hold report data
    outlier_summary = {}

    # Generate all possible groupings of the most anomalous values
    test_variable_subsets = list(chain.from_iterable(combinations(test_variables, r) for r in range(len(test_variables) + 1)))

    for subset in test_variable_subsets:
        group_values = {var: top_anomalous_values[var] for var in subset if var in top_anomalous_values}


        # Run detect_outliers with the subset of test variables
        result_df = detect_outliers(df,group_values)

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

    #generate text for the report based on anomaly and data
    report = generate_report(filtered_summary)

    output_buffer = io.BytesIO()
    output_buffer.write(report.encode('utf-8'))
    output_buffer.seek(0)
    PREFIX = "output/"  # Ensure correct prefix
    BUCKET_NAME = s3_bucket  # Use the defined bucket
    output_key = PREFIX + "filtered_outlier_summary.txt"
    s3.put_object(Bucket=BUCKET_NAME, Key=output_key, Body=output_buffer, ContentType='text/plain')


pivot_table_dict = pivot_table_df.toPandas().set_index("betos_cd").to_dict(orient="index")

def get_descriptions(group_dict):
    betos_cd = group_dict.get("betos_cd")
    pos_cd = group_dict.get("pos_cd")
    spec_cd = group_dict.get("spec_cd")

    betos_desc = pivot_table_dict.get(betos_cd, {}).get("betos_desc", "All")
    pos_desc = pivot_table_dict.get(pos_cd, {}).get("pos_desc", "All")
    spec_desc = pivot_table_dict.get(spec_cd, {}).get("spec_desc", "All")

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
    parsed_keys = [(ast.literal_eval(key), details) for key, details in week_anomalies.items()]

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

            group_dict = json.loads(group_key)

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


if __name__ == "__main__":
    main()

# Stop Spark Session
spark.stop()


