import pandas as pd
import numpy as np
# Grafik ve test kütüphanelerine artık ihtiyacımız yok
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import io 

# --- 1. Örnek Veri Seti Oluşturma ---
# Kodu test edebilmeniz için örnek veriyi burada bırakıyorum.
# Kendi dosyanızı kullanırken bu bloğu yorum satırı yapın veya silin.
data = """marka,yas,kilometre,motor_gucu,fiyat
Marka A,5,120000,110,350000
Marka B,2,45000,130,600000
Marka A,7,180000,110,280000
Marka C,3,60000,150,750000
Marka B,1,15000,130,680000
Marka A,10,250000,90,180000
Marka C,4,80000,150,700000
Marka B,6,150000,120,450000
Marka A,3,50000,110,420000
Marka C,1,10000,160,850000
Marka A,8,200000,90,210000
Marka B,4,90000,120,500000
Marka C,2,30000,160,820000
Marka A,6,140000,110,330000
Marka B,5,110000,130,510000
"""
df = pd.read_csv(io.StringIO(data))

# --- KENDİ DOSYANIZI KULLANMAK İÇİN BU SATIRI AÇIN ---
# df = pd.read_csv("arabalar.csv")
# ----------------------------------------------------


print("--- Veri Seti Yüklendi ---")
# print(df.head()) # İsteğe bağlı

# --- 2. Veri Ön İşleme ---
# Sütun adlarınızın (yas, kilometre, motor_gucu, fiyat) CSV dosyasıyla
# eşleştiğinden emin olun.
try:
    df['yas'] = df['yas'].astype(int)
    df['kilometre'] = df['kilometre'].astype(int)
    df['motor_gucu'] = df['motor_gucu'].astype(int)
    df['fiyat'] = df['fiyat'].astype(int)
    df['marka'] = df['marka'].astype(str)
except KeyError as e:
    print(f"HATA: CSV dosyanızda '{e}' sütunu bulunamadı.")
    print("Lütfen koddaki veya CSV dosyasındaki sütun adlarını kontrol edin.")
    exit() # Hata varsa programı durdur

# --- 3. Modelin Eğitimi (Tüm Veri Setiyle) ---
# Artık modeli test etmeyeceğimiz, sadece tahmin için kullanacağımız için
# tüm veri setini (train/test ayırmadan) eğitim için kullanıyoruz.
X = df.drop('fiyat', axis=1)
y = df['fiyat']

# Kategorik ve sayısal sütunları belirleme
categorical_features = ['marka']
numerical_features = ['yas', 'kilometre', 'motor_gucu']

# Kategorik verileri One-Hot Encoding ile dönüştürecek 'preprocessor'
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features), 
        ('num', 'passthrough', numerical_features)
    ])

# Pipeline oluşturma
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, min_samples_leaf=1))
])

# Modeli tüm veri setiyle eğitme
print("Model eğitiliyor...")
model_pipeline.fit(X, y)
print("Model eğitildi ve tahmin için hazır.\n")

# --- 4. İnteraktif Tahmin Uygulaması ---
print("--- Araba Fiyat Tahmin Uygulaması ---")
print("(Çıkmak için marka sorulduğunda 'çık' yazmanız yeterli)")

# Kullanıcıya hangi markaları bildiğimizi gösterelim
known_brands = df['marka'].unique()
print(f"\nModelin eğitim verisinde tanığı markalar: {', '.join(known_brands)}")

while True:
    try:
        # --- KULLANICIDAN GİRDİLERİ ALMA ---
        user_marka = input("\nAracın Markası (örn: Marka A): ")
        if user_marka.lower() == 'çık':
            print("Uygulamadan çıkılıyor...")
            break
        
        # Kullanıcı verisetinde olmayan bir marka girerse uyaralım
        if user_marka not in known_brands:
            print(f"** UYARI: '{user_marka}' markası eğitim setinde bulunmuyor.")
            print("   Model, bu markayı 'bilinmeyen' olarak kabul edip tahmin yapacak (tahmin doğruluğu düşebilir).")

        user_yas = int(input("Aracın Yaşı (örn: 5): "))
        user_km = int(input("Aracın Kilometresi (örn: 120000): "))
        user_guc = int(input("Aracın Motor Gücü (Beygir) (örn: 110): "))

        # --- TAHMİN İÇİN VERİ ÇERÇEVESİ OLUŞTURMA ---
        # Model, bizden bir DataFrame bekler
        yeni_arac_verisi = pd.DataFrame({
            'marka': [user_marka],
            'yas': [user_yas],
            'kilometre': [user_km],
            'motor_gucu': [user_guc]
        })

        # --- TAHMİN YAPMA ---
        tahmini_fiyat = model_pipeline.predict(yeni_arac_verisi)

        # --- SONUCU GÖSTERME ---
        print("\n" + "="*30)
        print("     TAHMİN SONUCU")
        print("="*30)
        print(f"Girdiğiniz Özellikler: ")
        print(f"  - Marka: {user_marka}")
        print(f"  - Yaş: {user_yas}")
        print(f"  - Kilometre: {user_km:,} km")
        print(f"  - Motor Gücü: {user_guc} HP")
        print("\n----------------------------------")
        print(f"  TAHMİNİ FİYAT: {tahmini_fiyat[0]:,.2f} TL")
        print("----------------------------------")

    except ValueError:
        print("\n! HATA: Yaş, Kilometre ve Motor Gücü için lütfen sayısal bir değer girin.")
        print("   Lütfen tekrar deneyin.")
    except Exception as e:
        print(f"\n! BEKLENMEYEN BİR HATA OLUŞTU: {e}")
        print("   Lütfen tekrar deneyin.")