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

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql import functions as F

import mlflow
import mlflow.sklearn
from mlflow.tracking.client import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# COMMAND ----------

fe = FeatureEngineeringClient()

# COMMAND ----------

# create the output (labels) table
silver_activity = spark.table("leap.silver.silver_activity")

# Compute delay, excluding rows with null dates
delivery_delay = (
    silver_activity
    .filter(F.col("completion_date").isNotNull() & F.col("planned_date").isNotNull())
    .groupBy("masterbill_internal_id")
    .agg(
        F.max("completion_date").alias("max_completion_date"),
        F.max("planned_date").alias("max_planned_date")
    )
    .withColumn(
        "delivery_delay",
        F.when(F.col("max_completion_date") > F.col("max_planned_date"), F.lit('Delayed'))
         .otherwise(F.lit('Not Delayed'))
    )
    .select("masterbill_internal_id", "delivery_delay")
)

display(delivery_delay)

# COMMAND ----------

# Save as table
delivery_delay.write.mode("overwrite").saveAsTable("leap.ai.delivery_delay_labels")

# COMMAND ----------

# create the features table
features = spark.table("leap.ai.feature_masterbill")

# COMMAND ----------

data = features.join(delivery_delay, on="masterbill_internal_id", how="inner")

data = data.withColumn("distance_km", col("distance_km").cast("double"))
data = data.withColumn("margin_value", col("margin_value").cast("double"))
data = data.withColumn("hazardous_material_flag", col("hazardous_material_flag").cast("boolean"))

display(data)

# COMMAND ----------

def load_data(joined_df):
    training_pd = joined_df.drop("masterbill_internal_id").toPandas()
    X = training_pd.drop("delivery_delay", axis=1)
    y = training_pd["delivery_delay"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_test = y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test

# Create the train and test datasets
X_train, X_test, y_train, y_test = load_data(data)

# COMMAND ----------

X_train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC **Training the model**

# COMMAND ----------

# Configure MLflow client to access models in Unity Catalog
mlflow.set_registry_uri("databricks-uc")

model_name = "leap.ai.delivery_delay_model"

client = MlflowClient()

# COMMAND ----------

mlflow.sklearn.autolog()

## fit and log model
with mlflow.start_run() as run:

    rf = RandomForestClassifier(max_depth=3, n_estimators=20, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rf.predict_proba(X_test)

    mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("test_f1_score", f1_score(y_test, y_pred, pos_label="Delayed"))

    input_example = X_train.iloc[[0]]

    mlflow.sklearn.log_model(
        sk_model=rf,
        artifact_path="delivery_delay_prediction",
        registered_model_name=model_name,
        input_example=input_example
    )

# COMMAND ----------

model_name