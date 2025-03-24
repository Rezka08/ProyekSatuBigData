import requests
import pandas as pd
import json
from datetime import datetime
import os

# Buat direktori untuk menyimpan data jika belum ada
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

def fetch_global_data():
    """Mengambil data COVID-19 global."""
    url = "https://disease.sh/v3/covid-19/all"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        # Tambahkan timestamp
        data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return data
    else:
        print(f"Error mengambil data global: {response.status_code}")
        return None

def fetch_countries_data():
    """Mengambil data COVID-19 untuk semua negara."""
    url = "https://disease.sh/v3/covid-19/countries"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error mengambil data negara: {response.status_code}")
        return None

def fetch_historical_data(days=30):
    """Mengambil data historis global."""
    url = f"https://disease.sh/v3/covid-19/historical/all?lastdays={days}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error mengambil data historis: {response.status_code}")
        return None

def fetch_country_historical_data(country, days=30):
    """Mengambil data historis untuk negara tertentu."""
    url = f"https://disease.sh/v3/covid-19/historical/{country}?lastdays={days}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error mengambil data historis {country}: {response.status_code}")
        return None

def save_data(data, filename, format_type='json'):
    """Menyimpan data ke file."""
    filepath = f"data/{filename}"
    
    if format_type == 'json':
        with open(f"{filepath}.json", 'w') as f:
            json.dump(data, f, indent=4)
    elif format_type == 'csv' and isinstance(data, pd.DataFrame):
        data.to_csv(f"{filepath}.csv", index=False)
    
    print(f"Data disimpan ke {filepath}.{format_type}")

def collect_all_data():
    """Mengumpulkan semua data COVID-19."""
    print("Memulai pengumpulan data COVID-19...")
    
    # 1. Data global
    global_data = fetch_global_data()
    if global_data:
        save_data(global_data, "covid19_global_data")
    
    # 2. Data per negara
    countries_data = fetch_countries_data()
    if countries_data:
        # Simpan sebagai JSON
        save_data(countries_data, "covid19_countries_data")
        
        # Konversi ke DataFrame untuk CSV
        countries_df = pd.json_normalize(countries_data)
        save_data(countries_df, "covid19_countries_data", 'csv')
        
        print(f"Berhasil mengambil data dari {len(countries_data)} negara")
    
    # 3. Data historis global
    historical_data = fetch_historical_data()
    if historical_data:
        save_data(historical_data, "covid19_historical_global")
    
    # 4. Data historis Indonesia
    indonesia_historical = fetch_country_historical_data("indonesia")
    if indonesia_historical:
        save_data(indonesia_historical, "covid19_historical_indonesia")
    
    # 5. Data telah selesai dibuat
    print("Pengumpulan data selesai!")
    return global_data, countries_data, historical_data, indonesia_historical

if __name__ == "__main__":
    collect_all_data()