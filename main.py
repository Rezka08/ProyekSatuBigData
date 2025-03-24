import os
import time
from covid19_data_collector import collect_all_data
from covid19_data_processor import process_all_data
from covid19_data_visualizer import create_all_visualizations

def create_directories():
    """Membuat direktori yang diperlukan jika belum ada."""
    directories = ['data', 'processed_data', 'visualizations']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Direktori {directory} dibuat.")

def main():
    """Fungsi utama untuk menjalankan seluruh pipeline pengumpulan dan analisis data."""
    print("=" * 50)
    print("PROYEK PENGUMPULAN DATA COVID-19 DENGAN API")
    print("=" * 50)
    
    # Buat direktori yang diperlukan
    create_directories()
    
    # 1. Kumpulkan data dari API
    print("\n[LANGKAH 1] Mengumpulkan data dari API...")
    start_time = time.time()
    global_data, countries_data, historical_data, indonesia_historical = collect_all_data()
    collection_time = time.time() - start_time
    print(f"Pengumpulan data selesai dalam {collection_time:.2f} detik.")
    
    # 2. Preprocessing data
    print("\n[LANGKAH 2] Melakukan preprocessing data...")
    start_time = time.time()
    clean_df, continent_data, timeline_df = process_all_data()
    processing_time = time.time() - start_time
    print(f"Preprocessing data selesai dalam {processing_time:.2f} detik.")
    
    # 3. Membuat visualisasi
    print("\n[LANGKAH 3] Membuat visualisasi data...")
    start_time = time.time()
    create_all_visualizations()
    visualization_time = time.time() - start_time
    print(f"Pembuatan visualisasi selesai dalam {visualization_time:.2f} detik.")
    
    print("\n" + "=" * 50)
    print("PROSES SELESAI!")
    print("Hasil pengumpulan data tersimpan di folder 'data/'")
    print("Hasil preprocessing tersimpan di folder 'processed_data/'")
    print("Visualisasi tersimpan di folder 'visualizations/'")
    print("=" * 50)

if __name__ == "__main__":
    main()