import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import io # Örnek veri setini okumak için eklendi

# --- 1. Örnek Veri Seti Oluşturma ---
# DÜZELTME 1: "arabalar.csv" dosyası bende olmadığı için kodun çalışması
# amacıyla bir örnek veri seti (data) oluşturuldu ve io.StringIO ile okundu.
# Siz kendi kodunuzda bu bloğu silip alttaki orijinal satırınızı kullanabilirsiniz.

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
# Orijinal satırınız:
# df = pd.read_csv("arabalar.csv")

# Örnek veriyle çalışan satır:
df = pd.read_csv(io.StringIO(data))


print("--- Örnek Veri Seti (İlk 5 Satır) ---")
print(df.head())
print("\n")

# --- DÜZELTME 2: Sütun Adı Tutarlılığı ---
# 'araba_yasi' yerine 'yas' olarak değiştirildi.
# Eğer CSV dosyanızda sütun adı 'araba_yasi' ise, aşağıdaki 'yas' olan
# her yeri 'araba_yasi' olarak değiştirmelisiniz.
# Ben, kodunuzun çoğunda 'yas' geçtiği için bunu standart kabul ettim.
df['yas'] = df['yas'].astype(int)
df['kilometre'] = df['kilometre'].astype(int)
df['motor_gucu'] = df['motor_gucu'].astype(int)
df['fiyat'] = df['fiyat'].astype(int)
df['marka'] = df['marka'].astype(str)


# --- 2. Veri Ön İşleme (Data Preprocessing) ---
X = df.drop('fiyat', axis=1)
y = df['fiyat']

# Kategorik ve sayısal sütunları belirleme
categorical_features = ['marka']
# 'yas' sütun adının yukarıdaki .astype() ile eşleştiğinden emin olundu.
numerical_features = ['yas', 'kilometre', 'motor_gucu']

# 'preprocessor' oluşturma
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

# --- 3. Veri Setini Ayırma (Train/Test Split) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Model Seçimi ve Eğitimi (RandomForestRegressor) ---
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, min_samples_leaf=1))
])

print("Model eğitiliyor...")
model_pipeline.fit(X_train, y_train)
print("Model eğitildi.\n")

# --- 5. Tahmin Yapma ---
y_pred = model_pipeline.predict(X_test)

# --- 6. Hata Metriklerini Hesaplama (MSE, R2) ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"--- Model Değerlendirme Metrikleri ---")
print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"R-squared (R2) Skoru: {r2:.2f}")
print("\n")

# --- 7. Gerçek ve Tahmin Edilen Fiyatları Karşılaştıran Grafik ---
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k', label='Tahminler', s=100)

min_val = min(y_test.min(), y_pred.min()) * 0.95
max_val = max(y_test.max(), y_pred.max()) * 1.05
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Mükemmel Tahmin Çizgisi')

plt.title('Araba Fiyat Tahmini: Gerçek vs. Tahmin', fontsize=16)
plt.xlabel('Gerçek Fiyatlar (y_test)', fontsize=12)
plt.ylabel('Tahmin Edilen Fiyatlar (y_pred)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# --- 8. Örnek Tahmin ---
print("--- Yeni Bir Araç İçin Örnek Tahmin ---")
# Sütun adının 'yas' olduğundan emin olundu
yeni_arac_verisi = pd.DataFrame({
    'marka': ['Marka B'],
    'yas': [3],
    'kilometre': [50000],
    'motor_gucu': [130]
})
tahmini_fiyat = model_pipeline.predict(yeni_arac_verisi)
print(f"Yeni aracın özellikleri: {yeni_arac_verisi.iloc[0].to_dict()}")
print(f"Tahmini Fiyat: {tahmini_fiyat[0]:,.2f} TL")