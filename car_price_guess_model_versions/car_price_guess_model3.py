# ============================================================
# 🚗 Araba Fiyat Tahmini (Linear Regression)
# ============================================================

# 1️⃣ Gerekli Kütüphaneler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 2️⃣ CSV Dosyasını Yükle
# 📂 arabalar.csv dosyan Python dosyasıyla aynı klasörde olmalı
df = pd.read_csv("arabalar.csv")

# 3️⃣ Veri Tiplerini Kontrol Et ve Düzelt
print("📊 Veri Tipleri (dönüştürmeden önce):")
print(df.dtypes)
print()

# Gerekirse türleri düzelt
df['araba_yasi'] = df['araba_yasi'].astype(int)
df['kilometre'] = df['kilometre'].astype(int)
df['motor_gucu'] = df['motor_gucu'].astype(int)
df['fiyat'] = df['fiyat'].astype(int)
df['marka'] = df['marka'].astype(str)

print("✅ Veri Tipleri (dönüştürmeden sonra):")
print(df.dtypes)
print()
# ====================================================================
# 🚀 YENİ ADIM: 3.5 - AYKIRI DEĞERLERİ BULMA VE TEMİZLEME
# ====================================================================
print("\n🔍 Veri Seti Analizi (Temizlemeden Önce):")
# .describe() komutu ile min, max, ortalama gibi değerlere bakıyoruz
# .apply(...) kısmı sayıların daha okunaklı (örn: 1,000,000) görünmesi için
print(df.describe().apply(lambda s: s.apply('{:,.0f}'.format)))
print()

# --- Aykırı Değerleri Filtrele ---
# ÖNEMLİ: Bu eşik değerlerini .describe() çıktısına göre sen belirlemelisin.
# Bunlar, o '1e6' hatasını çözmek için sadece birer örnek:

orijinal_veri_sayisi = len(df)

# Örneğin, fiyatı 4 Milyon TL'den yüksek veya 100.000 TL'den düşük arabaları "aykırı" kabul et
df = df[(df['fiyat'] < 4000000) & (df['fiyat'] > 100000)]

# Örneğin, kilometresi 800.000'den yüksek arabaları "aykırı" kabul et
df = df[df['kilometre'] < 800000]

# Örneğin, 25 yaşından büyük arabaları "aykırı" kabul et
df = df[df['araba_yasi'] < 25]


temizlenmis_veri_sayisi = len(df)
print(f"🧹 Temizlik yapıldı: {orijinal_veri_sayisi - temizlenmis_veri_sayisi} adet aykırı değer çıkarıldı.")
print(f"Kalan veri sayısı: {temizlenmis_veri_sayisi}\n")

# ====================================================================
# (Kodunun kalanı buradan itibaren aynı şekilde devam ediyor)
# ====================================================================




# 4️⃣ Kategorik Değişkeni Sayısala Çevir
df = pd.get_dummies(df, columns=['marka'], drop_first=True)

# 5️⃣ Özellik (X) ve hedef (y)
X = df.drop('fiyat', axis=1)
y = df['fiyat']

# 6️⃣ Eğitim ve Test Verisi

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 7️⃣ Model Oluşturma ve Eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# 8️⃣ Tahminler
y_pred = model.predict(X_test)

# 9️⃣ Değerlendirme Metrikleri
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Sonuçları ---")
print("Gerçek Değerler :", list(y_test.values))
print("Tahmin Değerleri:", [round(x) for x in y_pred])
print(f"\nMean Squared Error (MSE): {mse:.2f}") 
print(f"R-Kare (R²): {r2:.4f}")

# 🔹 10️⃣ Grafikler
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label='Tahminler')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='45° Doğru')
plt.xlabel("Gerçek Fiyatlar")
plt.ylabel("Tahmin Edilen Fiyatlar")
plt.title("Model Tahmin Doğruluğu")
plt.legend()
plt.grid(True)
plt.show()

# 🔹 Hata dağılımı
errors = y_test - y_pred
plt.figure(figsize=(8,6))
plt.hist(errors, bins=20, color='orange', edgecolor='black')
plt.xlabel("Hata (Gerçek - Tahmin)")
plt.ylabel("Frekans")
plt.title("Tahmin Hatalarının Dağılımı")
plt.grid(axis='y')
plt.show()
