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
import tkinter as tk  # YENİ EKLENDİ: GUI kütüphanesi

warnings.filterwarnings('ignore')

# --- 1. Veri Setini Yükleme ---
try:
    df = pd.read_csv("arabalar.csv")
except FileNotFoundError:
    print("HATA: 'arabalar.csv' dosyası bulunamadı.")
    df = pd.DataFrame()

# YENİ EKLENDİ: df boş değilse tüm işlemleri bu bloğun içinde yap
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

        # Veriyi normalize et
        df['marka'] = df['marka'].astype(str).str.lower()
        df['model'] = df['model'].astype(str).str.lower()
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

    # --- 3. Tahmin ve Değerlendirme ---
    y_pred = model_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("📊 --- Model Değerlendirme ---")
    print(f"R2 Skoru: {r2:.4f}")
    print(f"RMSE: {rmse:,.0f} TL\n")

    # --- 4. Grafik (Orijinal kodunuzdaki gibi) ---
    plt.ion()
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
    plt.show(block=False)

    # --- 5. Güvenli Sayı Girişi Fonksiyonu (Artık GUI içinde kullanılıyor) ---
    # Orijinal 'get_int' fonksiyonuna artık gerek yok.

    # --- 6. YENİ BÖLÜM: Grafik Arayüz ile Tahmin ---

    # GUI fonksiyonunun erişebilmesi için mevcut marka/modelleri al
    mevcut_markalar = [m.lower() for m in df['marka'].unique()]
    mevcut_modeller = [m.lower() for m in df['model'].unique()]

    # Butona tıklandığında çalışacak fonksiyon
    def gui_tahmin_yap():
        try:
            # 1. GUI'deki giriş kutularından verileri al
            marka_input = marka_entry.get().strip().lower()
            model_input = model_entry.get().strip().lower()

            # 2. Sayısal verileri al ve doğrula
            try:
                yas_input = int(yas_entry.get())
                km_input = int(km_entry.get())
                guc_input = int(guc_entry.get())
            except ValueError:
                # Sayısal olmayan bir giriş varsa hata ver
                sonuc_label.config(text="HATA: Yaş, KM ve Güç sayı olmalı!", fg="red")
                return

            # 3. Kategorik verileri doğrula (Orijinal kodunuzdaki gibi)
            if marka_input not in mevcut_markalar:
                sonuc_label.config(text=f"HATA: '{marka_input}' markası bilinmiyor.", fg="red")
                return
            if model_input not in mevcut_modeller:
                sonuc_label.config(text=f"HATA: '{model_input}' modeli bilinmiyor.", fg="red")
                return

            # 4. Tahmin için DataFrame oluştur
            yeni_veri = pd.DataFrame({
                'marka': [marka_input],
                'model': [model_input],
                'araba_yasi': [yas_input],
                'kilometre': [km_input],
                'motor_gucu': [guc_input]
            })

            # 5. Modeli kullanarak tahmini yap
            tahmin = model_pipeline.predict(yeni_veri)
            
            # 6. Sonucu arayüzdeki etikete yaz
            sonuc_label.config(text=f"💰 Tahmini Fiyat: {tahmin[0]:,.0f} TL", fg="blue")

        except Exception as e:
            # Beklenmedik bir hata olursa etikete yaz
            sonuc_label.config(text=f"HATA: {e}", fg="red")

    # --- Tkinter Arayüz Kurulumu ---
    root = tk.Tk()
    root.title("Araba Fiyat Tahmin Aracı")

    # Arayüz elemanlarını (widget) oluştur ve yerleştir (grid sistemi)
    tk.Label(root, text="Marka:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    marka_entry = tk.Entry(root, width=30)
    marka_entry.grid(row=0, column=1, padx=10, pady=5)

    tk.Label(root, text="Model:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    model_entry = tk.Entry(root, width=30)
    model_entry.grid(row=1, column=1, padx=10, pady=5)

    tk.Label(root, text="Araba Yaşı:").grid(row=2, column=0, padx=10, pady=5, sticky="e")
    yas_entry = tk.Entry(root, width=30)
    yas_entry.grid(row=2, column=1, padx=10, pady=5)

    tk.Label(root, text="Kilometre:").grid(row=3, column=0, padx=10, pady=5, sticky="e")
    km_entry = tk.Entry(root, width=30)
    km_entry.grid(row=3, column=1, padx=10, pady=5)

    tk.Label(root, text="Motor Gücü:").grid(row=4, column=0, padx=10, pady=5, sticky="e")
    guc_entry = tk.Entry(root, width=30)
    guc_entry.grid(row=4, column=1, padx=10, pady=5)

    # Tahmin butonu
    tahmin_butonu = tk.Button(root, text="Fiyatı Tahmin Et", command=gui_tahmin_yap, 
                                bg="#4CAF50", fg="white", font=('Arial', 12, 'bold'))
    tahmin_butonu.grid(row=5, column=0, columnspan=2, pady=15, padx=10, ipadx=10, ipady=5)

    # Sonucun gösterileceği etiket
    sonuc_label = tk.Label(root, text="", font=('Arial', 14, 'bold'))
    sonuc_label.grid(row=6, column=0, columnspan=2, pady=10)

    # GUI'yi başlat ve ekranda tut
    root.mainloop()

# YENİ EKLENDİ: df boş ise bu mesaj gösterilecek
else:
    print("CSV dosyası okunamadığı için işlem yapılamadı.")