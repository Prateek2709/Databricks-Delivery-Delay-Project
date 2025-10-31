# Databricks notebook source
# Upgrade protobuf to match the FE client's generated code (v6.x)
%pip uninstall -y protobuf googleapis-common-protos
%pip install "protobuf==6.31.1" "googleapis-common-protos>=1.63.0"

# (If you installed FE in a separate cell, keep it too)
%pip install databricks-feature-engineering

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff, expr, when, first, coalesce, sum as spark_sum, isnan
from pyspark.sql import functions as F
import random
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql.utils import AnalysisException

# COMMAND ----------

# Initialize Spark session
spark = SparkSession.builder.appName("JoinTables").getOrCreate()

# COMMAND ----------

# Load tables
silver_masterbill = spark.table("leap.silver.silver_masterbill")
silver_activity = spark.table("leap.silver.silver_activity")
silver_carrier = spark.table("leap.silver.silver_carrier")
silver_item = spark.table("leap.silver.silver_item")
silver_item_group = spark.table("leap.silver.silver_item_group")
silver_item_dimension = spark.table("leap.silver.silver_item_dimension")
silver_item_weight = spark.table("leap.silver.silver_item_weight")
silver_itemgroup_line_item = spark.table("leap.silver.silver_itemgroup_line_item")
silver_masterbill_margins = spark.table("leap.silver.silver_masterbill_margins")
silver_shipunit = spark.table("leap.silver.silver_shipunit")
silver_tracking_shipment_status = spark.table("leap.silver.silver_tracking_shipment_status")

# COMMAND ----------

# MAGIC %md
# MAGIC Creating the target feature (delivery delay)

# COMMAND ----------

# extracting the masterbill ids from respective table for primary key
masterbill_ids = silver_masterbill.select("masterbill_internal_id").distinct()

# display(masterbill_ids)

# COMMAND ----------

# MAGIC %md
# MAGIC Create separate tables for the features before joining them together

# COMMAND ----------

# Convert dimensions to cm and calculate volume in cm3
item_dimensions = silver_item_dimension.withColumn(
    "dimension_cm", 
    when(col("dimension_dim") == "inch", col("dimension") * 2.54).otherwise(col("dimension"))
)

# Pivot the dimensions to get height, width, and length in separate columns
item_dimensions_pivot = item_dimensions.groupBy("masterbill_internal_id").pivot("dimension_dim").agg(first("dimension_cm"))

# Calculate the volume in cm3
item_volume = item_dimensions_pivot.withColumn(
    "volume_cm3", 
    col("height") * col("width") * col("length")
).select("masterbill_internal_id", "volume_cm3").distinct()

# display(item_volume)

# COMMAND ----------

# Select masterbill_internal_id and distance_km from silver_carrier
distance = silver_carrier.select(
    "masterbill_internal_id",
    "distance_km"
).distinct()

# display(distance)

# COMMAND ----------

# Aggregate weights by masterbill_internal_id for actual weight type
item_weight = (
    silver_item_weight
    .filter(col("weight_type") == "actual")
    .groupBy("masterbill_internal_id")
    .agg(F.sum("weight").alias("total_weight"))
).distinct()

# display(item_weight)

# COMMAND ----------

# Flag masterbill ids as hazardous if atleast one line item has the hazardous_material_flag set to true
hazmat_flag = (
    silver_itemgroup_line_item.withColumn("hazardous_material_flag", F.col("hazardous_material_flag").cast("boolean"))
      .groupBy("masterbill_internal_id")
      .agg(F.max(F.col("hazardous_material_flag").cast("int")).alias("any_hazard"))
      .withColumn("hazardous_material_flag", (F.col("any_hazard") == 1))
      .select("masterbill_internal_id", "hazardous_material_flag")
)

# display(hazmat_flag)

# COMMAND ----------

# Select masterbill_internal_id and margin_cost from silver_masterbill_margins, filtering for CurrentMargin
margin_value = silver_masterbill_margins.filter(col("margin_type") == "CurrentMargin").select(
    "masterbill_internal_id",
    "margin_value"
).distinct()

# display(margin_value)

# COMMAND ----------

# Select masterbill_internal_id and event_type from silver_tracking_shipment_status, filtering for drops
event_type = silver_tracking_shipment_status.filter(col("event_type") == "Drop").select(
    "masterbill_internal_id",
    "event_type"
).distinct()

display(event_type)

# COMMAND ----------

# MAGIC %md
# MAGIC join the 2 tables related to item description using id and description and then create a table using those 2 features

# COMMAND ----------

# Select only the two features from each table, aligned
a = silver_item_group.select(
    col("masterbill_internal_id"),
    col("description").alias("desc_b")
).dropDuplicates()

b = silver_itemgroup_line_item.select(
    col("masterbill_internal_id"),
    col("description").alias("desc_c")
).dropDuplicates()

# Join a and b on id
ab = a.alias("a").join(
    b.alias("b"),
    col("a.masterbill_internal_id") == col("b.masterbill_internal_id"),
    "full_outer"
)

# Coalesce ids and descriptions, then distinct
final_description = ab.select(
    coalesce(col("a.masterbill_internal_id"), col("b.masterbill_internal_id")).alias("masterbill_internal_id"),
    coalesce(col("a.desc_b"), col("b.desc_c")).alias("item_description")
).dropDuplicates()

# display(final_description)

# COMMAND ----------

# Perform a left join on all the specified tables using masterbill_internal_id and select specific columns
data = masterbill_ids.join(
    distance,
    "masterbill_internal_id",
    "left"
).join(
    item_volume,
    "masterbill_internal_id",
    "left"
).join(
    item_weight,
    "masterbill_internal_id",
    "left"
).join(
    hazmat_flag,
    "masterbill_internal_id",
    "left"
).join(
    margin_value,
    "masterbill_internal_id",
    "left"
).join(
    event_type,
    "masterbill_internal_id",
    "left"
).join(
    final_description,
    "masterbill_internal_id",
    "left"
).select(
    "masterbill_internal_id",
    "volume_cm3",
    "distance_km",
    "total_weight",
    F.coalesce("hazardous_material_flag", F.lit(False)).alias("hazardous_material_flag"),
    F.coalesce("margin_value", F.lit(0)).alias("margin_value"),
    "item_description"
).distinct()

# COMMAND ----------

# # Indexing the item_description column
# indexer = StringIndexer(inputCol="item_description", outputCol="item_description_index")

# data = indexer.fit(data).transform(data)

# # Applying OneHotEncoder to the indexed column
# encoder = OneHotEncoder(inputCols=["item_description_index"], outputCols=["item_description_ohe"])
# data = encoder.fit(data).transform(data)

# # Display the result
# display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC **temporarily dropping the item_description column until problem is solved**

# COMMAND ----------

# Display the result
display(data)

# COMMAND ----------

null_counts = data.select([
    spark_sum(col(c).isNull().cast("int")).alias(c + "_null_count")
    for c in data.columns
])

display(null_counts)

# COMMAND ----------

# count of unique ids
unique_masterbill_count = data.select("masterbill_internal_id").distinct().count()
unique_masterbill_count

# COMMAND ----------

# Randomly assigning the present values to the null values in the item_description feature
random.seed(42)

data = data.withColumn(
    "item_description",
    F.when(
        col("item_description").isNull(),
        F.when(F.rand(seed=42) < 0.5, F.lit("MEDICAL MATERIAL")).otherwise(F.lit("Euro Pallet"))
    ).otherwise(col("item_description"))
)

# # Applying One Hot Encoding on the item_description column
# # Indexing the item_description column
# indexer = StringIndexer(inputCol="item_description", outputCol="item_description_index")
# data = indexer.fit(data).transform(data)

# # Applying OneHotEncoder to the indexed column
# encoder = OneHotEncoder(inputCols=["item_description_index"], outputCols=["item_description_ohe"])
# data = encoder.fit(data).transform(data)

# # Display the result
# display(data)

# COMMAND ----------

# # Calculate the median value of distance_km
# median_distance_km = data.approxQuantile("distance_km", [0.5], 0.01)[0]

# # Fill null values in distance_km with the median value
# data = data.fillna({"distance_km": median_distance_km})

# COMMAND ----------

# Fill null values in distance_km with 0
data = data.fillna({"distance_km": 0})

# COMMAND ----------

# Fill null values in total_weight with a random value between 1 and 2, rounded to 3 decimal places
data = data.withColumn(
    "total_weight",
    F.when(
        col("total_weight").isNull(),
        F.round(F.rand(seed=42) + 1, 3)
    ).otherwise(col("total_weight"))
)

# COMMAND ----------

# Fill null values in volume_cm3 with zero
data = data.fillna({"volume_cm3": 0})

# COMMAND ----------

# Drop the item_description column from data
data = data.drop("item_description")

# Show result
display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC Storing this created data as a feature table

# COMMAND ----------

# use the appropriate catalog and schema
spark.sql("USE CATALOG leap")
spark.sql("USE SCHEMA ai")

# create the table within the schema for the data
table_name = f"leap.ai.feature_masterbill"
print(table_name)

# COMMAND ----------

data.write.mode("overwrite").saveAsTable(table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Below code to store the feature as a Feature table

# COMMAND ----------

# fe = FeatureEngineeringClient()

# COMMAND ----------

# # create the feature table
# fe.create_table(
#     name=table_name,
#     primary_keys=["masterbill_internal_id"],
#     schema=data.schema,
#     description="delivery delay prediction features"
# )

# fe.write_table(
#     name=table_name,
#     df=data,
#     mode="merge"
# )