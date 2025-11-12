# ============================================================
# 🚗 Araba Fiyat Tahmini (Linear Regression)
# ============================================================

# 1️⃣ Gerekli Kütüphaneler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 2️⃣ Örnek Veri Seti
data = {
    'marka': [
        'BMW', 'Audi', 'BMW', 'Mercedes', 'Audi', 'Mercedes', 'BMW', 'Audi', 'Mercedes', 'BMW',
        'BMW', 'Audi', 'Mercedes', 'BMW', 'Audi', 'Mercedes', 'BMW', 'Audi', 'Mercedes', 'BMW',
        'BMW', 'Audi', 'Mercedes', 'BMW', 'Audi', 'Mercedes', 'BMW', 'Audi', 'Mercedes', 'BMW',
        'BMW', 'Audi', 'Mercedes', 'BMW', 'Audi', 'Mercedes', 'BMW', 'Audi', 'Mercedes', 'BMW',
        'BMW', 'Audi', 'Mercedes', 'BMW', 'Audi', 'Mercedes', 'BMW', 'Audi', 'Mercedes', 'BMW'
    ],
    'araba_yasi': [
        3,5,2,7,4,6,1,3,5,2,
        4,6,3,2,5,7,1,4,6,3,
        2,5,7,3,4,6,1,2,5,3,
        4,6,2,3,5,7,1,4,6,3,
        2,5,7,3,4,6,1,2,5,3
    ],
    'kilometre': [
        30000,50000,20000,70000,40000,60000,15000,35000,50000,22000,
        32000,61000,25000,21000,48000,69000,16000,36000,52000,23000,
        31000,50000,70000,30000,41000,61000,17000,22000,49000,24000,
        33000,62000,21000,32000,45000,68000,18000,37000,53000,23000,
        30000,50000,70000,31000,42000,62000,16000,21000,48000,24000
    ],
    'motor_gucu': [
        150,160,140,200,155,180,130,160,190,145,
        150,170,135,145,160,200,130,155,180,140,
        150,160,200,135,155,180,130,140,160,145,
        150,170,135,145,160,200,130,155,180,140,
        150,160,200,135,155,180,130,140,160,145
    ],
    'fiyat': [
        350000,300000,370000,280000,320000,290000,380000,310000,295000,365000,
        340000,285000,375000,360000,305000,280000,385000,315000,290000,370000,
        355000,300000,280000,365000,325000,290000,380000,360000,295000,370000,
        345000,285000,375000,355000,310000,280000,385000,320000,295000,370000,
        350000,300000,280000,365000,325000,290000,380000,360000,295000,370000
    ]
}
df = pd.read_csv("arabalar.csv")

df = pd.DataFrame(data)

# 3️⃣ Kategorik Değişkeni Sayısala Çevir
df = pd.get_dummies(df, columns=['marka'], drop_first=True)

# 4️⃣ Özellik (X) ve hedef (y)
X = df.drop('fiyat', axis=1)
y = df['fiyat']

# 5️⃣ Eğitim ve Test Verisi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 6️⃣ Model Oluşturma ve Eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# 7️⃣ Tahminler
y_pred = model.predict(X_test)

# 8️⃣ Değerlendirme Metrikleri
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Sonuçları ---")
print("Gerçek Değerler :", list(y_test.values))
print("Tahmin Değerleri:", [round(x) for x in y_pred])
print(f"\nMean Squared Error (MSE): {mse:.2f}")
print(f"R-Kare (R²): {r2:.4f}")

# 9️⃣ Grafikler
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label='Tahminler')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='45° Doğru')
plt.xlabel("Gerçek Fiyatlar")
plt.ylabel("Tahmin Edilen Fiyatlar")
plt.title("Model Tahmin Doğruluğu")
plt.legend()
plt.grid(True)
plt.show()

# Hata dağılımı
errors = y_test - y_pred
plt.figure(figsize=(8,6))
plt.hist(errors, bins=20, color='orange', edgecolor='black')
plt.xlabel("Hata (Gerçek - Tahmin)")
plt.ylabel("Frekans")
plt.title("Tahmin Hatalarının Dağılımı")
plt.grid(axis='y')
plt.show()
