# Databricks notebook source
# Upgrade protobuf to match the FE client's generated code (v6.x)
%pip uninstall -y protobuf googleapis-common-protos
%pip install "protobuf==6.31.1" "googleapis-common-protos>=1.63.0"

# (If you installed FE in a separate cell, keep it too)
%pip install databricks-feature-engineering

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
import mlflow
from pyspark.sql import functions as F
from mlflow.tracking.client import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from pyspark.sql.functions import when, col, udf
import pandas as pd

# COMMAND ----------

fe = FeatureEngineeringClient()

# COMMAND ----------

model_uri = 'models:/leap.ai.delivery_delay_model/12'
model = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

model

# COMMAND ----------

model.classes_

# COMMAND ----------

model_name = 'leap.ai.delivery_delay_model'

# Helper function
def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

masterbill_id = spark.table("leap.silver.silver_masterbill").select("masterbill_internal_id")

batch_input_df = masterbill_id.join(spark.table("leap.ai.feature_masterbill"), on="masterbill_internal_id", how="inner")

# Ensure get_latest_model_version and model_name are defined in the current kernel/session
latest_model_version = get_latest_model_version(model_name)

# predictions_df = fe.score_batch(model_uri=f"models:/{model_name}/{latest_model_version}", df=batch_input_df)

# COMMAND ----------

features_df = batch_input_df.toPandas()

# COMMAND ----------

output_df = model.predict(features_df.drop("masterbill_internal_id", axis=1))
output_df

# COMMAND ----------

output_pd = pd.DataFrame({
    "masterbill_internal_id": features_df["masterbill_internal_id"].values,
    "delivery_delay": output_df
})

display(output_pd)

# COMMAND ----------

proba_df = model.predict_proba(features_df.drop("masterbill_internal_id", axis=1))
proba_df

# COMMAND ----------

proba_pd = pd.DataFrame({
    "masterbill_internal_id": features_df["masterbill_internal_id"].values,
    "risk_factor": proba_df[:, 0] * 100
})

display(proba_pd)

# COMMAND ----------

predictions_df = features_df.merge(output_pd, on="masterbill_internal_id", how="inner").merge(proba_pd, on="masterbill_internal_id", how="inner")

display(predictions_df)

# COMMAND ----------

predictions_df_spark = spark.createDataFrame(predictions_df)

predictions_df_spark = predictions_df_spark.withColumn(
    "risk_level",
    when((col("risk_factor") >= 0) & (col("risk_factor") <= 50), "Normal")
    .when((col("risk_factor") >= 51) & (col("risk_factor") <= 70), "Low")
    .when((col("risk_factor") >= 71) & (col("risk_factor") <= 90), "Medium")
    .when((col("risk_factor") >= 91) & (col("risk_factor") <= 100), "High")
    .otherwise(None)
)

display(predictions_df_spark)

# COMMAND ----------

# load_id, carrier_name, and mode columns
fact_load_df = spark.table("leap.gold.fact_load").select("masterbill_internal_id", "load_id", "carrier_name", "mode")
predictions_df_spark = predictions_df_spark.join(
    fact_load_df,
    on="masterbill_internal_id",
    how="left"
)

# COMMAND ----------

# Load events and keep only Pickup/Drop
events_df = spark.table("leap.silver.silver_event")

events_pick_drop = (
    events_df
    .filter(F.col("event_type").isin("Pickup", "Drop"))
    .withColumn(
        "location",
        F.concat_ws(", ",
            F.col("city"),
            F.col("state"),
            F.col("country_code"),
            F.col("postal_code")
        )
    )
    .select("master_bill_internal_id", "event_type", "location")
)

# Pivot to Source/Destination
locations_df = (
    events_pick_drop
    .groupBy("master_bill_internal_id")
    .pivot("event_type", ["Pickup", "Drop"])
    .agg(F.first("location", ignorenulls=True))
    .withColumnRenamed("Pickup", "Source")
    .withColumnRenamed("Drop", "Destination")
)

# Join to predictions_df using respective columns with explicit condition
predictions_df_spark = (
    predictions_df_spark.join(
        locations_df,
        predictions_df_spark["masterbill_internal_id"] == locations_df["master_bill_internal_id"],
        how="left"
    )
    .drop("master_bill_internal_id")
    .dropDuplicates()
)

display(predictions_df_spark)

# COMMAND ----------

predictions_df_spark.write.mode("overwrite").option("mergeSchema", "true").saveAsTable("leap.ai.delivery_delay_predictions")