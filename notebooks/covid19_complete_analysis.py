# 1. INISIALISASI SPARK DAN IMPORT LIBRARY

import findspark
findspark.init()  # Auto-detect Spark

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, avg, sum, desc, round
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Inisialisasi Spark dengan Hadoop support
spark = SparkSession.builder \
    .appName("COVID19-Analysis") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

print(f"Spark version: {spark.version}")

# 2. LOADING DATA DARI HDFS
# Asumsikan data COVID-19 sudah diupload ke HDFS
# Anda bisa menyesuaikan path jika berbeda

hdfs_path = "hdfs://localhost:9000/user/covid19/"
covid_df = spark.read.csv(f"{hdfs_path}covid19_countries_data.csv", header=True, inferSchema=True)

# Jika belum di-upload ke HDFS, gunakan path lokal
# covid_df = spark.read.json("data/covid19_countries_data.json", header=True, inferSchema=True)

# Tampilkan info data
print("Schema:")
covid_df.printSchema()
print(f"Total negara: {covid_df.count()}")

# 3. DATA PREPROCESSING

# Pilih kolom yang akan digunakan untuk analisis
selected_cols = [
    "country", "continent", "population", 
    "cases", "deaths", "recovered", "active",
    "casesPerOneMillion", "deathsPerOneMillion", 
    "tests", "testsPerOneMillion"
]

# Filter kolom yang ada di dataset
available_cols = [col for col in selected_cols if col in covid_df.columns]
covid_clean_df = covid_df.select(available_cols)

# Handle missing values
print("Jumlah missing values per kolom:")
for column in covid_clean_df.columns:
    missing_count = covid_clean_df.filter(col(column).isNull()).count()
    print(f"{column}: {missing_count}")

# Filter negara dengan populasi > 0 dan cases > 100 (untuk analisis yang lebih bermakna)
covid_filtered_df = covid_clean_df.filter(
    (col("population") > 0) & 
    (col("cases") > 100)
)

# Tambahkan kolom turunan yang berguna
covid_analysis_df = covid_filtered_df.withColumn(
    "mortality_rate", round((col("deaths") / col("cases") * 100), 2)
).withColumn(
    "infection_rate", round((col("cases") / col("population") * 100), 2)
).withColumn(
    "recovery_rate", round((col("recovered") / col("cases") * 100), 2)
)

# 4. ANALISIS CLUSTERING (K-MEANS)

# Pilih fitur untuk clustering
cluster_features = [
    "casesPerOneMillion", "deathsPerOneMillion", 
    "testsPerOneMillion", "mortality_rate", "infection_rate"
]

# Pastikan semua kolom fitur ada di dataframe
cluster_features = [f for f in cluster_features if f in covid_analysis_df.columns]

# Persiapkan VectorAssembler untuk menggabungkan fitur
assembler = VectorAssembler(inputCols=cluster_features, outputCol="raw_features")

# Standarisasi fitur (penting untuk K-Means)
scaler = StandardScaler(inputCol="raw_features", outputCol="features")

# Pipeline untuk preprocessing
pipeline = Pipeline(stages=[assembler, scaler])
preprocessed_df = pipeline.fit(covid_analysis_df).transform(covid_analysis_df)

# Mencari jumlah klaster optimal dengan Elbow Method
wcss = []
silhouette_scores = []
k_values = range(2, 11)  # Coba K dari 2 sampai 10

for k in k_values:
    kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=k, seed=42)
    model = kmeans.fit(preprocessed_df)
    predictions = model.transform(preprocessed_df)
    
    # Hitung Within-Cluster Sum of Squares (WCSS)
    wcss.append(model.summary.trainingCost)
    
    # Hitung Silhouette Score (jika jumlah cluster > 1)
    evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="cluster")
    silhouette = evaluator.evaluate(predictions)
    silhouette_scores.append(silhouette)
    
    print(f"K={k}, WCSS={model.summary.trainingCost:.4f}, Silhouette Score={silhouette:.4f}")

# Optimal K berdasarkan Elbow Method dan Silhouette Score
# Visualisasi menggunakan Pandas & Matplotlib
results_pd = pd.DataFrame({
    "K": list(k_values),
    "WCSS": wcss,
    "Silhouette": silhouette_scores
})

# Simpan untuk visualisasi
results_pd.to_csv("elbow_method_results.csv", index=False)

# Terapkan K-Means dengan K optimal (misalnya, k=3 berdasarkan analisis Elbow Method)
optimal_k = 3  # Silakan sesuaikan setelah melihat hasil Elbow Method
kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=optimal_k, seed=42)
model = kmeans.fit(preprocessed_df)
clustered_df = model.transform(preprocessed_df)

# Evaluasi model clustering
evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="cluster")
silhouette = evaluator.evaluate(clustered_df)
print(f"Silhouette Score dengan k={optimal_k}: {silhouette:.4f}")

# Analisis cluster
cluster_stats = clustered_df.groupBy("cluster").agg(
    count("*").alias("count"),
    avg("casesPerOneMillion").alias("avg_cases_per_million"),
    avg("deathsPerOneMillion").alias("avg_deaths_per_million"),
    avg("mortality_rate").alias("avg_mortality_rate"),
    avg("infection_rate").alias("avg_infection_rate")
)

# Convert ke Pandas untuk visualisasi
cluster_stats_pd = cluster_stats.toPandas()
print("Statistik per cluster:")
print(cluster_stats_pd)

# 5. ANALISIS REGRESI LINEAR

# Pilih fitur untuk model regresi
regression_features = [
    "population", "casesPerOneMillion", "testsPerOneMillion", 
    "infection_rate", "recovery_rate"
]

# Pastikan semua kolom fitur ada di dataframe
regression_features = [f for f in regression_features if f in covid_analysis_df.columns]

# Target: mortality_rate
# Persiapkan data untuk regresi
reg_assembler = VectorAssembler(inputCols=regression_features, outputCol="features")
regression_data = reg_assembler.transform(covid_analysis_df)

# Split data: training (80%) dan testing (20%)
train_data, test_data = regression_data.randomSplit([0.8, 0.2], seed=42)

# Bangun model regresi linear
lr = LinearRegression(
    featuresCol="features", 
    labelCol="mortality_rate",
    maxIter=10,
    regParam=0.1
)

# Train model
lr_model = lr.fit(train_data)

# Evaluasi model pada data testing
predictions = lr_model.transform(test_data)

# Hitung metrik evaluasi
evaluator = RegressionEvaluator(labelCol="mortality_rate", predictionCol="prediction")
rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

print(f"Model Regresi Linear untuk Mortality Rate")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Coefficient: {lr_model.coefficients}")
print(f"Intercept: {lr_model.intercept}")

# Tampilkan feature importance
feature_importance = [(feature, coefficient) for feature, coefficient in zip(regression_features, lr_model.coefficients)]
feature_importance_sorted = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)
print("Feature Importance:")
for feature, importance in feature_importance_sorted:
    print(f"{feature}: {importance:.4f}")

# 6. EXPORT DATA UNTUK VISUALISASI DI LUAR SPARK

# Ekspor data cluster untuk visualisasi
cluster_data = clustered_df.select("country", "continent", "cluster", *cluster_features).toPandas()
cluster_data.to_csv("covid_clusters.csv", index=False)

# Ekspor data prediksi regresi untuk visualisasi
prediction_data = predictions.select(
    "country", "mortality_rate", "prediction", *regression_features
).toPandas()
prediction_data.to_csv("covid_mortality_predictions.csv", index=False)

# 7. TUTUP SPARK SESSION
spark.stop()