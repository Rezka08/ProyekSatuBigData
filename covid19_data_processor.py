import pandas as pd
import json
from datetime import datetime
import os

def load_countries_data():
    """Memuat data negara dari file CSV."""
    try:
        return pd.read_csv("data/covid19_countries_data.csv")
    except FileNotFoundError:
        print("File data negara tidak ditemukan.")
        return None

def load_historical_data():
    """Memuat data historis dari file JSON."""
    try:
        with open("data/covid19_historical_global.json", 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("File data historis tidak ditemukan.")
        return None

def preprocess_countries_data(df):
    """Preprocessing data negara."""
    if df is None:
        return None
        
    # 1. Pilih kolom yang relevan
    selected_columns = ['country', 'cases', 'todayCases', 'deaths', 'todayDeaths', 
                        'recovered', 'active', 'critical', 'casesPerOneMillion', 
                        'deathsPerOneMillion', 'tests', 'testsPerOneMillion', 'population',
                        'continent', 'oneCasePerPeople', 'oneDeathPerPeople', 'oneTestPerPeople']
    
    # Pastikan semua kolom yang dipilih ada di DataFrame
    available_columns = [col for col in selected_columns if col in df.columns]
    clean_df = df[available_columns].copy()
    
    # 2. Kelompokkan data berdasarkan benua
    continent_data = clean_df.groupby('continent').agg({
        'cases': 'sum',
        'deaths': 'sum',
        'recovered': 'sum',
        'active': 'sum',
        'population': 'sum'
    }).reset_index()
    
    # 3. Hitung persentase infeksi dan kematian
    continent_data['infection_rate'] = (continent_data['cases'] / continent_data['population'] * 100).round(2)
    continent_data['mortality_rate'] = (continent_data['deaths'] / continent_data['cases'] * 100).round(2)
    
    # 4. Tambahkan informasi waktu pengambilan data
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clean_df['data_collected_at'] = timestamp
    continent_data['data_collected_at'] = timestamp
    
    return clean_df, continent_data

def preprocess_historical_data(historical_data):
    """Preprocessing data historis."""
    if historical_data is None:
        return None
        
    timeline_df = pd.DataFrame()
    
    for category, data in historical_data.items():
        if isinstance(data, dict):  # Pastikan data berupa dictionary
            for date, value in data.items():
                # Konversi format tanggal MM/DD/YY ke YYYY-MM-DD
                try:
                    date_obj = datetime.strptime(date, '%m/%d/%y')
                    formatted_date = date_obj.strftime('%Y-%m-%d')
                    
                    # Tambahkan ke DataFrame
                    if formatted_date not in timeline_df.index:
                        timeline_df.loc[formatted_date, category] = value
                    else:
                        timeline_df.at[formatted_date, category] = value
                except ValueError:
                    print(f"Format tanggal tidak valid: {date}")
    
    # Reset index untuk menjadikan tanggal sebagai kolom
    timeline_df = timeline_df.reset_index().rename(columns={'index': 'date'})
    return timeline_df

def save_processed_data(df, filename):
    """Menyimpan data yang telah diproses."""
    if df is None:
        return
        
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')
        
    filepath = f"processed_data/{filename}"
    df.to_csv(f"{filepath}.csv", index=False)
    print(f"Data yang diproses disimpan ke {filepath}.csv")

def process_all_data():
    """Melakukan preprocessing pada semua data."""
    print("Memulai preprocessing data...")
    
    # 1. Proses data negara
    countries_df = load_countries_data()
    if countries_df is not None:
        clean_df, continent_data = preprocess_countries_data(countries_df)
        save_processed_data(clean_df, "covid19_countries_clean")
        save_processed_data(continent_data, "covid19_continent_summary")
    
    # 2. Proses data historis
    historical_data = load_historical_data()
    if historical_data is not None:
        timeline_df = preprocess_historical_data(historical_data)
        save_processed_data(timeline_df, "covid19_global_timeline")
    
    print("Preprocessing data selesai!")
    return clean_df, continent_data, timeline_df

if __name__ == "__main__":
    process_all_data()