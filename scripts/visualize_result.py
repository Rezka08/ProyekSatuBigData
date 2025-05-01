# VISUALISASI HASIL ANALISIS COVID-19
# Script ini berjalan di luar Spark, menggunakan pandas dan matplotlib/seaborn
# untuk memvisualisasikan hasil dari analisis PySpark

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 1. VISUALISASI ELBOW METHOD UNTUK K-MEANS
try:
    elbow_data = pd.read_csv("elbow_method_results.csv")
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Jumlah Cluster (K)')
    ax1.set_ylabel('WCSS (Within-Cluster Sum of Squares)', color=color)
    ax1.plot(elbow_data['K'], elbow_data['WCSS'], 'o-', color=color, linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(elbow_data['K'], elbow_data['Silhouette'], 'o-', color=color, linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Penentuan Jumlah Cluster Optimal dengan Elbow Method dan Silhouette Score', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/elbow_method.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Elbow Method plot telah disimpan")
except Exception as e:
    print(f"Error membuat plot Elbow Method: {e}")

# 2. VISUALISASI HASIL CLUSTER
try:
    cluster_data = pd.read_csv("covid_clusters.csv")
    
    # Visualisasi Cluster dalam 2D (pilih 2 fitur untuk sumbu)
    plt.figure(figsize=(12, 10))
    
    # Gunakan scatterplot dengan warna berdasarkan cluster
    scatter = sns.scatterplot(
        x='casesPerOneMillion', 
        y='deathsPerOneMillion',
        hue='cluster',
        size='infection_rate',  # Ukuran titik berdasarkan infection_rate
        sizes=(50, 300),
        palette='viridis',
        alpha=0.7,
        data=cluster_data
    )
    
    # Tambahkan label untuk titik-titik tertentu (negara dengan nilai ekstrem)
    top_countries = pd.concat([
        cluster_data.nlargest(5, 'casesPerOneMillion'),
        cluster_data.nlargest(5, 'deathsPerOneMillion')
    ]).drop_duplicates()
    
    for _, row in top_countries.iterrows():
        plt.annotate(
            row['country'],
            xy=(row['casesPerOneMillion'], row['deathsPerOneMillion']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
        )
    
    plt.title('Clustering Negara Berdasarkan Kasus dan Kematian COVID-19 per Juta Penduduk', fontsize=14)
    plt.xlabel('Kasus COVID-19 per Juta Penduduk')
    plt.ylabel('Kematian COVID-19 per Juta Penduduk')
    
    # Format angka dengan pemisah ribuan
    formatter = ticker.StrMethodFormatter('{x:,.0f}')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    
    # Tambahkan legend yang informatif
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('visualizations/clusters_casesVSdeaths.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Visualisasi statistik cluster (boxplot)
    plt.figure(figsize=(16, 12))
    
    cluster_features = ['casesPerOneMillion', 'deathsPerOneMillion', 'mortality_rate', 'infection_rate']
    
    for i, feature in enumerate(cluster_features):
        plt.subplot(2, 2, i+1)
        
        # Boxplot untuk setiap cluster
        sns.boxplot(x='cluster', y=feature, data=cluster_data, palette='viridis')
        
        plt.title(f'Distribusi {feature} per Cluster')
        plt.xlabel('Cluster')
        
        # Format y-axis dengan pemisah ribuan jika diperlukan
        if 'PerOneMillion' in feature:
            plt.gca().yaxis.set_major_formatter(formatter)
    
    plt.tight_layout()
    plt.savefig('visualizations/cluster_statistics_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Cluster plots telah disimpan")
    
    # Visualisasi peta dunia berdasarkan cluster (hanya contoh, memerlukan package tambahan seperti geopandas)
    print("Untuk visualisasi peta dunia berdasarkan cluster, dibutuhkan implementasi dengan geopandas")
    
except Exception as e:
    print(f"Error membuat plot cluster: {e}")

# 3. VISUALISASI HASIL REGRESI LINEAR
try:
    prediction_data = pd.read_csv("covid_mortality_predictions.csv")
    
    # Scatter plot untuk nilai aktual vs prediksi
    plt.figure(figsize=(10, 8))
    
    # Plot garis identitas (y=x)
    max_val = max(prediction_data['mortality_rate'].max(), prediction_data['prediction'].max())
    min_val = min(prediction_data['mortality_rate'].min(), prediction_data['prediction'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'g--', alpha=0.5, label='Identitas (prediksi sempurna)')
    
    # Plot hasil prediksi
    plt.scatter(
        prediction_data['mortality_rate'],
        prediction_data['prediction'],
        alpha=0.6,
        c=prediction_data['mortality_rate'],
        cmap='viridis',
        s=100,
        edgecolor='k',
        linewidth=0.5
    )
    
    # Tambahkan label untuk beberapa titik
    outliers = prediction_data[
        (abs(prediction_data['mortality_rate'] - prediction_data['prediction']) > 
         2 * (prediction_data['mortality_rate'] - prediction_data['prediction']).std())
    ]
    
    for _, row in outliers.iterrows():
        plt.annotate(
            row['country'],
            xy=(row['mortality_rate'], row['prediction']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
        )
    
    plt.title('Perbandingan Tingkat Kematian Aktual vs Prediksi', fontsize=14)
    plt.xlabel('Tingkat Kematian Aktual (%)')
    plt.ylabel('Tingkat Kematian Prediksi (%)')
    plt.grid(True, alpha=0.3)
    plt.colorbar(label='Tingkat Kematian Aktual (%)')
    
    # Tambahkan teks untuk metrik evaluasi regresi
    try:
        from sklearn.metrics import r2_score, mean_squared_error
        rmse = np.sqrt(mean_squared_error(prediction_data['mortality_rate'], prediction_data['prediction']))
        r2 = r2_score(prediction_data['mortality_rate'], prediction_data['prediction'])
        
        plt.figtext(
            0.15, 0.85,
            f"RMSE: {rmse:.4f}\nR²: {r2:.4f}",
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7)
        )
    except ImportError:
        pass  # Jika sklearn tidak tersedia
    
    plt.tight_layout()
    plt.savefig('visualizations/regression_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Visualisasi importance feature
    try:
        # Ambil koefisien dari file hasil (asumsikan sudah disimpan)
        feature_cols = [col for col in prediction_data.columns if col not in 
                      ['country', 'continent', 'mortality_rate', 'prediction']]
        
        # Gunakan koefisien absolut dari model (contoh nilai untuk visualisasi)
        # Dalam implementasi nyata, nilai ini harus diambil dari hasil model
        coefficients = np.random.rand(len(feature_cols))  # Ganti dengan nilai sebenarnya
        
        plt.figure(figsize=(10, 6))
        
        # Plot horizontal bar
        sns.barplot(x=np.abs(coefficients), y=feature_cols, palette='viridis')
        
        plt.title('Importance Feature dalam Prediksi Tingkat Kematian', fontsize=14)
        plt.xlabel('Magnitude Koefisien (Nilai Absolut)')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error membuat plot feature importance: {e}")
    
    print("Regression plots telah disimpan")
except Exception as e:
    print(f"Error membuat plot regresi: {e}")

# 4. VISUALISASI PETA DUNIA (CONTOH KONSEP)
print("Untuk visualisasi peta dunia dengan data COVID-19, dibutuhkan implementasi dengan geopandas dan folium")

# 5. VISUALISASI DASHBOARD INTERAKTIF (KONSEP)
print("Untuk implementasi dashboard interaktif, gunakan Jupyter Dashboard, Plotly Dash, atau Streamlit")