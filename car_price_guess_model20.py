import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import warnings

warnings.filterwarnings('ignore')

# --- Veri Seti ---
try:
    df = pd.read_csv("arabalar.csv")
except FileNotFoundError:
    print("HATA: 'arabalar.csv' dosyası bulunamadı.")
    root_check = tk.Tk()
    root_check.withdraw() 
    messagebox.showerror("Hatalı Dosya", "HATA: 'arabalar.csv' dosyası bulunamadı. Program kapatılacak.")
    root_check.destroy()
    df = pd.DataFrame()

if not df.empty:
    df = df.dropna()
    df['marka'] = df['marka'].astype(str).str.lower()
    df['model'] = df['model'].astype(str).str.lower()
    df['araba_yasi'] = df['araba_yasi'].astype(int)
    df['kilometre'] = df['kilometre'].astype(int)
    df['motor_gucu'] = df['motor_gucu'].astype(int)
    df['fiyat'] = df['fiyat'].astype(int)

    X = df.drop('fiyat', axis=1)
    y = df['fiyat']

    categorical_features = ['marka', 'model']
    numerical_features = ['araba_yasi', 'kilometre', 'motor_gucu']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features) 
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10))
    ])

    print("Model eğitiliyor...")
    model_pipeline.fit(X_train, y_train)
    print("✅ Model eğitildi.\n")

    # --- ÖZELLİK ÖNEMLERİNİ KONTROL ET ---
    print("--- Model Özellik Önemleri ---")
    try:
        regressor = model_pipeline.named_steps['regressor']
        preprocessor = model_pipeline.named_steps['preprocessor']
        feature_names = preprocessor.get_feature_names_out()
        importances = regressor.feature_importances_
        feature_importance_df = pd.DataFrame({'Özellik': feature_names, 'Önem': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Önem', ascending=False)
        
        print("Model için en önemli 20 özellik:")
        print(feature_importance_df.head(20).to_string()) 
        
        print("\n----------------------------------")
        km_importance = feature_importance_df[feature_importance_df['Özellik'].str.contains('kilometre')]
        print("Kilometrenin Özel Önemi:")
        print(km_importance)
        print("----------------------------------\n")
        
    except Exception as e:
        print(f"Özellik önemleri alınırken hata oluştu: {e}\n")
    # --- BİTİŞ: ÖZELLİK ÖNEMLERİ ---

    print("Model performansı hesaplanıyor...")
    y_pred_test = model_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_test)
    print(f"Model R2: {r2:.4f}, RMSE: {rmse:,.0f} TL")

    # <--- DEĞİŞİKLİK 1: Hata grafiği için hataları burada hesapla
    errors_test = y_test - y_pred_test

    mevcut_markalar = list(df['marka'].unique())
    mevcut_modeller = list(df['model'].unique())

    # --- GUI ---
    root = tk.Tk()
    root.title("Araba Fiyat Tahmini")

    root.grid_columnconfigure(1, weight=1)
    # <--- DEĞİŞİKLİK 2: Grafik alanı için satır numarasını 8'e güncelledim (araya buton eklendi)
    root.grid_rowconfigure(8, weight=1) # 7 yerine 8 oldu

    labels = ["Marka:", "Model:", "Araba Yaşı:", "Kilometre:", "Motor Gücü:"]
    entries = []
    
    for i, text in enumerate(labels):
        tk.Label(root, text=text).grid(row=i, column=0, sticky="e", padx=5, pady=2)
        entry = tk.Entry(root, width=30)
        entry.grid(row=i, column=1, padx=5, pady=2, sticky="ew")
        entries.append(entry)

    marka_entry, model_entry, yas_entry, km_entry, guc_entry = entries

    tk.Button(root, text="Tahmin Et", command=lambda: tahmin_yap(), bg="green", fg="white", font=("Arial", 12, "bold")).grid(row=5, column=0, columnspan=2, pady=10, sticky="ew")

    # <--- DEĞİŞİKLİK 3: Grafik fonksiyonunu iki grafiği de çizecek şekilde güncelledim
    def guncel_grafik_ciz(grafik_tipi='her_ikisi'):
        # 1. Grafik (Scatter)
        ax1.clear()
        ax1.scatter(y_test, y_pred_test, alpha=0.6, edgecolors='k', s=50, label='Test Verisi')

        all_plot_values = list(y_test) + list(y_pred_test)
        min_val = min(all_plot_values) * 0.9
        max_val = max(all_plot_values) * 1.1

        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Mükemmel Tahmin (y=x)")
        
        ax1.set_title("Gerçek vs Tahmin Edilen Fiyatlar")
        ax1.set_xlabel("Gerçek Fiyatlar (TL)")
        ax1.set_ylabel("Tahmin Edilen Fiyatlar (TL)")
        ax1.set_xlim(min_val, max_val)
        ax1.set_ylim(min_val, max_val)
        ax1.legend()
        ax1.grid(True)
        
        # 2. Grafik (Histogram)
        ax2.clear()
        ax2.hist(errors_test, bins=20, color='orange', edgecolor='black')
        ax2.set_title("Tahmin Hatalarının Dağılımı")
        ax2.set_xlabel("Hata (Gerçek - Tahmin)")
        ax2.set_ylabel("Frekans")
        ax2.grid(axis='y')

        # Genel Düzenleme
        fig.tight_layout()
        canvas.draw()


    # <--- DEĞİŞİKLİK 4: Matplotlib Canvas'ı 2 sütunlu (yan yana) olacak şekilde ayarla
    # figsize'ı (10, 4) yaparak 2 grafiğe yer açtım
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), dpi=100)
    
    toolbar_frame = tk.Frame(root)
    # Toolbar'ın satırını 7'ye aldım
    toolbar_frame.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(5,0)) 
    canvas = FigureCanvasTkAgg(fig, master=root) 
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()
    
    # Canvas'ın (grafik alanı) satırını 8'e aldım
    canvas.get_tk_widget().grid(row=8, column=0, columnspan=2, pady=(0, 10), sticky="nsew") 

    metrics_label = tk.Label(root, text=f"Model R²: {r2:.4f}, RMSE: {rmse:,.0f} TL", font=("Arial", 10, "bold"))
    # Metrik etiketinin satırını 9'a aldım
    metrics_label.grid(row=9, column=0, columnspan=2, pady=5, sticky="ew") 


    def tahmin_yap():
        marka = marka_entry.get().strip().lower()
        model = model_entry.get().strip().lower()
        try:
            yas = int(yas_entry.get())
            km = int(km_entry.get())
            guc = int(guc_entry.get())
        except ValueError:
            messagebox.showerror("HATA", "Lütfen Yaş, KM ve Güç alanlarına geçerli sayılar girin!")
            return

        if marka not in mevcut_markalar:
            messagebox.showwarning("Uyarı", f"'{marka}' markası eğitim verisinde yok. Model en yakın tahmini yapmaya çalışacak.")
        if model not in mevcut_modeller:
            messagebox.showwarning("Uyarı", f"'{model}' modeli eğitim verisinde yok. Model en yakın tahmini yapmaya çalışacak.")

        yeni_veri = pd.DataFrame({
            'marka': [marka],
            'model': [model],
            'araba_yasi': [yas],
            'kilometre': [km],
            'motor_gucu': [guc]
        })

        tahmin = model_pipeline.predict(yeni_veri)[0]

        messagebox.showinfo("Tahmin Sonucu", 
                            f"Tahmini Araba Fiyatı: {tahmin:,.0f} TL\n\n"
                            f"(Modelin genel R² skoru: {r2:.4f})\n"
                            f"(Modelin genel RMSE'si: {rmse:,.0f} TL)")

        # Not: Grafik fonksiyonu (guncel_grafik_ciz) zaten test verisini çiziyor,
        # yeni tahmini grafiğe eklemiyor. Bu sizin orijinal mantığınızdı,
        # bu yüzden o kısmı değiştirmedim.
        guncel_grafik_ciz()

    # Program başlarken ilk grafikleri çiz
    guncel_grafik_ciz()
    
    root.mainloop()

else:
    print("CSV dosyası okunamadığı için işlem yapılamadı.")