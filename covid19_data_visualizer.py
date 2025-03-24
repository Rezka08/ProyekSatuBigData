import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_processed_data(filename):
    """Memuat data yang telah diproses."""
    try:
        return pd.read_csv(f"processed_data/{filename}.csv")
    except FileNotFoundError:
        print(f"File {filename}.csv tidak ditemukan.")
        return None

def visualize_continent_data(continent_data):
    """Membuat visualisasi data benua."""
    if continent_data is None:
        return
        
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # 1. Visualisasi kasus per benua
    plt.figure(figsize=(12, 6))
    sns.barplot(x='continent', y='cases', data=continent_data.sort_values('cases', ascending=False))
    plt.title('Total Kasus COVID-19 per Benua')
    plt.xlabel('Benua')
    plt.ylabel('Total Kasus')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/covid19_cases_by_continent.png')
    plt.close()
    
    # 2. Visualisasi tingkat kematian per benua
    plt.figure(figsize=(12, 6))
    sns.barplot(x='continent', y='mortality_rate', data=continent_data.sort_values('mortality_rate', ascending=False))
    plt.title('Tingkat Kematian COVID-19 per Benua (%)')
    plt.xlabel('Benua')
    plt.ylabel('Tingkat Kematian (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/covid19_mortality_by_continent.png')
    plt.close()
    
    print("Visualisasi data benua selesai!")

def visualize_countries_data(countries_data):
    """Membuat visualisasi data negara."""
    if countries_data is None:
        return
        
    # Visualisasi 10 negara dengan kasus terbanyak
    top_10_countries = countries_data.sort_values('cases', ascending=False).head(10)
    plt.figure(figsize=(14, 7))
    sns.barplot(x='country', y='cases', data=top_10_countries)
    plt.title('10 Negara dengan Kasus COVID-19 Terbanyak')
    plt.xlabel('Negara')
    plt.ylabel('Total Kasus')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/covid19_top10_countries.png')
    plt.close()
    
    # Visualisasi 10 negara dengan tingkat kematian tertinggi (min. 1000 kasus)
    countries_with_significant_cases = countries_data[countries_data['cases'] >= 1000].copy()
    countries_with_significant_cases['mortality_rate'] = (countries_with_significant_cases['deaths'] / countries_with_significant_cases['cases'] * 100).round(2)
    top_10_mortality = countries_with_significant_cases.sort_values('mortality_rate', ascending=False).head(10)
    
    plt.figure(figsize=(14, 7))
    sns.barplot(x='country', y='mortality_rate', data=top_10_mortality)
    plt.title('10 Negara dengan Tingkat Kematian COVID-19 Tertinggi (min. 1000 kasus)')
    plt.xlabel('Negara')
    plt.ylabel('Tingkat Kematian (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/covid19_top10_mortality_rate.png')
    plt.close()
    
    print("Visualisasi data negara selesai!")

def visualize_timeline_data(timeline_data):
    """Membuat visualisasi data timeline."""
    if timeline_data is None:
        return
        
    plt.figure(figsize=(14, 8))
    plt.plot(timeline_data['date'], timeline_data['cases'], marker='o', label='Kasus')
    plt.plot(timeline_data['date'], timeline_data['deaths'], marker='x', label='Kematian')
    plt.plot(timeline_data['date'], timeline_data['recovered'], marker='^', label='Sembuh')
    plt.title('Tren COVID-19 Global (30 Hari Terakhir)')
    plt.xlabel('Tanggal')
    plt.ylabel('Jumlah')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/covid19_global_trend.png')
    plt.close()
    
    # Visualisasi daily new cases (derivatif dari data kumulatif)
    timeline_data['daily_cases'] = timeline_data['cases'].diff()
    timeline_data['daily_deaths'] = timeline_data['deaths'].diff()
    
    plt.figure(figsize=(14, 8))
    plt.bar(timeline_data['date'][1:], timeline_data['daily_cases'][1:], color='skyblue', label='Kasus Harian')
    plt.plot(timeline_data['date'][1:], timeline_data['daily_deaths'][1:], color='red', marker='x', label='Kematian Harian')
    plt.title('Kasus dan Kematian Harian COVID-19 Global')
    plt.xlabel('Tanggal')
    plt.ylabel('Jumlah')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/covid19_daily_cases_deaths.png')
    plt.close()
    
    print("Visualisasi data timeline selesai!")

def create_all_visualizations():
    """Membuat semua visualisasi."""
    print("Memulai pembuatan visualisasi...")
    
    # 1. Data benua
    continent_data = load_processed_data("covid19_continent_summary")
    visualize_continent_data(continent_data)
    
    # 2. Data negara
    countries_data = load_processed_data("covid19_countries_clean")
    visualize_countries_data(countries_data)
    
    # 3. Data timeline
    timeline_data = load_processed_data("covid19_global_timeline")
    visualize_timeline_data(timeline_data)
    
    print("Pembuatan visualisasi selesai!")

if __name__ == "__main__":
    create_all_visualizations()