import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# --- 1. Veri Setini Yükleme ---
try:
    df = pd.read_csv("arabalar.csv")
except FileNotFoundError:
    print("HATA: 'arabalar.csv' dosyası bulunamadı.")
    df = pd.DataFrame()

if not df.empty:
    print("--- Veri Seti (İlk 5 Satır) ---")
    print(df.head(), "\n")

    print(f"Veri temizlenmeden önceki satır sayısı: {len(df)}")
    df = df.dropna()
    print(f"Veri temizlendikten sonraki satır sayısı: {len(df)}\n")


    # --- 2. Veri Ön İşleme ---
    try:
        gerekli_sutunlar = {'marka', 'model', 'araba_yasi', 'kilometre', 'motor_gucu', 'fiyat'}
        eksik = gerekli_sutunlar - set(df.columns)
        if eksik:
            raise KeyError(f"CSV dosyanızda şu sütunlar eksik: {', '.join(eksik)}")

        df['marka'] = df['marka'].astype(str)
        df['model'] = df['model'].astype(str)
        df['araba_yasi'] = df['araba_yasi'].astype(int)
        df['kilometre'] = df['kilometre'].astype(int)
        df['motor_gucu'] = df['motor_gucu'].astype(int)
        df['fiyat'] = df['fiyat'].astype(int)

    except KeyError as e:
        print(f"HATA: {e}")
    except Exception as e:
        print(f"HATA: Veri dönüştürme hatası: {e}")

    X = df.drop('fiyat', axis=1)
    y = df['fiyat']

    categorical_features = ['marka', 'model']
    numerical_features = ['araba_yasi', 'kilometre', 'motor_gucu']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10))
    ])

    print("Model eğitiliyor...")
    model_pipeline.fit(X_train, y_train)
    print("✅ Model eğitildi.\n")

    y_pred = model_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("📊 --- Model Değerlendirme ---")
    print(f"R2 Skoru: {r2:.4f}")
    print(f"RMSE: {rmse:,.0f} TL\n")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', s=70)
    min_val = min(y_test.min(), y_pred.min()) * 0.9
    max_val = max(y_test.max(), y_pred.max()) * 1.1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Mükemmel Tahmin (y=x)")
    plt.title("Gerçek vs Tahmin Edilen Araba Fiyatları", fontsize=14)
    plt.xlabel("Gerçek Fiyatlar (TL)")
    plt.ylabel("Tahmin Edilen Fiyatlar (TL)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 6. Kullanıcıdan Veri Alarak Tahmin Yap ---
    print("\n🚗 Şimdi kendi verinizi girin:")
    try:
        marka_input = input("Marka: ").strip()
        model_input = input("Model: ").strip()
        yas_input = int(input("Araba yaşı: "))
        km_input = int(input("Kilometre: "))
        guc_input = int(input("Motor gücü: "))

        # Büyük/küçük harf farkını kaldır
        marka_input = marka_input.lower()
        model_input = model_input.lower()

        mevcut_markalar = [m.lower() for m in df['marka'].unique()]
        mevcut_modeller = [m.lower() for m in df['model'].unique()]

        if marka_input not in mevcut_markalar:
            raise ValueError(f"'{marka_input}' markası veri setinde yok. Mevcut markalar: {', '.join(df['marka'].unique()[:10])}...")

        if model_input not in mevcut_modeller:
            raise ValueError(f"'{model_input}' modeli veri setinde yok. Lütfen geçerli bir model girin.")

        yeni_veri = pd.DataFrame({
            'marka': [marka_input],
            'model': [model_input],
            'araba_yasi': [yas_input],
            'kilometre': [km_input],
            'motor_gucu': [guc_input]
        })

        tahmin = model_pipeline.predict(yeni_veri)
        print(f"\n💰 Tahmini Araba Fiyatı: {tahmin[0]:,.0f} TL")

    except ValueError as e:
        print(f"HATA: {e}")
    except Exception as e:
        print(f"HATA: Tahmin sırasında bir hata oluştu -> {e}")

else:
    print("CSV dosyası okunamadığı için işlem yapılamadı.")
