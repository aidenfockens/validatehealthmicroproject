#imports
import os
import yaml
import json
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, lag, count, mean, stddev, abs as spark_abs, when, lit
from pyspark.sql.window import Window
from itertools import chain, combinations

#set up config variables
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)
metric_variable = next((var for var, details in config["variables"].items() if details.get("metric_variable", False)), None)
time_variable = next((var for var, details in config["variables"].items() if details.get("time_variable", False)), None)
test_variables = [var for var, details in config["variables"].items() if details.get("test_variable", False)]


# set up spark
spark = SparkSession.builder.appName("OutlierDetection").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
df = spark.read.option("header", "true").option("inferSchema", "true").csv("data.csv")
df.createOrReplaceTempView("service_data")

#set up pivot table
pivot_table_df = pd.read_csv("codes_pivot_table.csv")




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



def compute_anomaly_counts():
    """
    Compute anomaly counts for each unique value of each test variable. But separates beween each test_variable
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
            when(col("rolling_std") == 0, lit(False)).otherwise(col("z_score") > 2)
        )
        
        df_anomalies = df_anomalies.withColumn(
            "is_outlier_metric",
            ((col("total_metric") >= 2 * col("prev_total_metric")) |
             (col("total_metric") <= 0.5 * col("prev_total_metric"))) &
            (spark_abs(col("total_metric") - col("prev_total_metric")) > 600)
        )
        
        df_anomalies = df_anomalies.withColumn(
            "is_outlier_count",
            ((col("num_data_points") >= 2 * col("prev_num_data_points")) |
             (col("num_data_points") <= 0.5 * col("prev_num_data_points"))) &
            ((col("num_data_points") > 10) | (col("prev_num_data_points") > 10))
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


def compute_anomalous_groupings():
    """
    Compute anomaly counts for all unique combinations of test variables using groupBy("service_week", *test_variables)
    and return the top 5 most anomalous groupings. Does them all together, unlike compute_anomaly_counts
    """
    df_grouped = df.groupBy("service_week", *test_variables).agg(
        spark_sum("paid_amount").alias("total_metric"),
        count("*").alias("num_data_points")
    )
    
    window_spec = Window.partitionBy(*test_variables).orderBy("service_week")
    
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
        when(col("rolling_std") == 0, lit(False)).otherwise(col("z_score") > 2)
    )
    
    df_anomalies = df_anomalies.withColumn(
        "is_outlier_metric",
        ((col("total_metric") >= 2 * col("prev_total_metric")) |
         (col("total_metric") <= 0.5 * col("prev_total_metric"))) &
        (spark_abs(col("total_metric") - col("prev_total_metric")) > 600)
    )
    
    df_anomalies = df_anomalies.withColumn(
        "is_outlier_count",
        ((col("num_data_points") >= 2 * col("prev_num_data_points")) |
         (col("num_data_points") <= 0.5 * col("prev_num_data_points"))) &
        ((col("num_data_points") > 10) | (col("prev_num_data_points") > 10))
    )
    
    df_anomalies = df_anomalies.withColumn(
        "is_outlier",
        col("is_outlier_z") | col("is_outlier_count") | col("is_outlier_metric")
    )
    print("Total number of unique groupings:", df_anomalies.select(*test_variables).distinct().count())


    anomaly_counts = (
        df_anomalies.groupBy(*test_variables)
        .agg(spark_sum(col("is_outlier").cast("int")).alias("anomaly_count"))
        .orderBy(col("anomaly_count").desc())
        .limit(5)
        .collect()
    )
    
    return [{var: row[var] for var in test_variables} for row in anomaly_counts]










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

    #testing out compute_anomalous_groupings():
    top_anomalous_groupings = compute_anomalous_groupings()
    print("Top 5 anomalous groupings:", top_anomalous_groupings)
    

    

    # Dictionary to hold report data
    outlier_summary = {}

    # Generate all possible groupings of the most anomalous values
    test_variable_subsets = list(chain.from_iterable(combinations(test_variables, r) for r in range(len(test_variables) + 1)))

    for subset in test_variable_subsets:
        group_values = {var: top_anomalous_values[var] for var in subset if var in top_anomalous_values}


        # Run detect_outliers with the subset of test variables
        result_df = detect_outliers(group_values)

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
    output_file = "filtered_outlier_summary.txt"
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






