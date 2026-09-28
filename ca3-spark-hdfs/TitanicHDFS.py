from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, when

# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("Titanic_Analysis_Karimi") \
    .getOrCreate()

# 2. Load Data from HDFS
# Based on your terminal output, the file is at /karimi/Titanic-Dataset.csv
hdfs_path = "hdfs://172.18.32.200:9000/karimi/Titanic-Dataset.csv"

print(f"Loading data from: {hdfs_path} ...")
df = spark.read.csv(hdfs_path, header=True, inferSchema=True)

# 3. Calculate Survival Rate by Sex (Question 3.2 Part 1)
print("\n--- Survival Rate by Sex ---")
# Group by Sex, calculate average of 'Survived' (0 or 1)
# Since Survived is 0 or 1, the average * 100 gives the percentage.
sex_survival = df.groupBy("Sex").agg(
    (avg("Survived") * 100).alias("Survival_Rate_Percent"),
    count("PassengerId").alias("Total_Passengers")
)
sex_survival.show()

# 4. Calculate Survival Rate by Pclass (Question 3.2 Part 2)
print("\n--- Survival Rate by Ticket Class (Pclass) ---")
class_survival = df.groupBy("Pclass").agg(
    (avg("Survived") * 100).alias("Survival_Rate_Percent"),
    count("PassengerId").alias("Total_Passengers")
).orderBy("Pclass")
class_survival.show()

# 5. Average Age for Survivors vs Non-Survivors (Question 3.2 Part 3)
print("\n--- Average Age: Survivors vs Non-Survivors ---")
# 0 = No (Deceased), 1 = Yes (Survived)
age_analysis = df.groupBy("Survived").agg(
    avg("Age").alias("Average_Age"),
    count("Age").alias("Count_With_Age_Data") # Just to see how many had valid ages
)
age_analysis.show()

# ==========================================
# PART 3.3 & 3.4: Machine Learning Pipeline
# ==========================================
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os

print("\n--- Starting Machine Learning Pipeline ---")

# 1. Prepare Data (Handle Nulls)
# We drop rows with missing values in the columns we care about
required_columns = ["Pclass", "Sex", "Age", "Fare", "Embarked", "Survived"]
data = df.select(required_columns).dropna()

# 2. Convert Categorical Columns to Numbers (StringIndexer)
# Sex: male/female -> 0/1
sex_indexer = StringIndexer(inputCol="Sex", outputCol="SexIndex")
data = sex_indexer.fit(data).transform(data)

# Embarked: S/C/Q -> 0/1/2
embarked_indexer = StringIndexer(inputCol="Embarked", outputCol="EmbarkedIndex")
data = embarked_indexer.fit(data).transform(data)

# 3. Assemble Features
# Combine all input columns into a single vector column named "features"
feature_cols = ["Pclass", "SexIndex", "Age", "Fare", "EmbarkedIndex"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
final_data = assembler.transform(data)

# 4. Split Data (80% Train, 20% Test)
train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)

print(f"Training Dataset Count: {train_data.count()}")
print(f"Test Dataset Count: {test_data.count()}")

# 5. Train Logistic Regression Model
lr = LogisticRegression(featuresCol="features", labelCol="Survived")
lr_model = lr.fit(train_data)

# 6. Make Predictions
predictions = lr_model.transform(test_data)

# 7. Evaluate Accuracy
evaluator = MulticlassClassificationEvaluator(
    labelCol="Survived", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"\nModel Accuracy on Test Data: {accuracy:.4f}")

# ==========================================
# PART 3.4: Save Result to HDFS
# ==========================================

# Step A: Write to a LOCAL file first
local_filename = "accuracy.txt"
with open(local_filename, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}")

print(f"Accuracy saved locally to {local_filename}")

# Step B: Upload this local file to your HDFS folder
# We use os.system to run the shell command from Python
# Note: Using the path /karimi/ based on your previous output
hdfs_dest = "/karimi/accuracy.txt"
upload_cmd = f"hdfs dfs -put -f {local_filename} {hdfs_dest}"

print(f"Uploading to HDFS: {upload_cmd}")
exit_code = os.system(upload_cmd)

if exit_code == 0:
    print("Successfully uploaded accuracy.txt to HDFS!")
else:
    print("Error uploading to HDFS. Please check permissions.")
