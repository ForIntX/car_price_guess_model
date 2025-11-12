import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# StandardScaler eklendi
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
    # GUI hatası için root'u erken başlat
    root_check = tk.Tk()
    root_check.withdraw() # Ana pencereyi gizle
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

    # 'passthrough' yerine StandardScaler() kullanıyoruz
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features) # Sayısal verileri ölçeklendir
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

    mevcut_markalar = list(df['marka'].unique())
    mevcut_modeller = list(df['model'].unique())

    # --- GUI ---
    root = tk.Tk()
    root.title("Araba Fiyat Tahmini")

    # DÜZENLEME: Pencere yeniden boyutlandırma ayarları
    # 1. Sütun (Entry'ler) yatayda genişlesin
    root.grid_columnconfigure(1, weight=1)
    # 7. Satır (Grafik) dikeyde genişlesin
    root.grid_rowconfigure(7, weight=1)

    labels = ["Marka:", "Model:", "Araba Yaşı:", "Kilometre:", "Motor Gücü:"]
    entries = []
    
    for i, text in enumerate(labels):
        tk.Label(root, text=text).grid(row=i, column=0, sticky="e", padx=5, pady=2)
        entry = tk.Entry(root, width=30)
        # DÜZENLEME: Giriş kutuları yatayda genişlesin
        entry.grid(row=i, column=1, padx=5, pady=2, sticky="ew")
        entries.append(entry)

    marka_entry, model_entry, yas_entry, km_entry, guc_entry = entries

    # DÜZENLEME: Buton yatayda genişlesin
    tk.Button(root, text="Tahmin Et", command=lambda: tahmin_yap(), bg="green", fg="white", font=("Arial", 12, "bold")).grid(row=5, column=0, columnspan=2, pady=10, sticky="ew")

    def guncel_grafik_ciz():
        ax.clear()
        ax.scatter(y_test, y_pred_test, alpha=0.6, edgecolors='k', s=50, label='Test Verisi')

        all_plot_values = list(y_test) + list(y_pred_test)
        
        min_val = min(all_plot_values) * 0.9
        max_val = max(all_plot_values) * 1.1

        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Mükemmel Tahmin (y=x)")
        
        ax.set_title("Gerçek vs Tahmin Edilen Fiyatlar")
        ax.set_xlabel("Gerçek Fiyatlar (TL)")
        ax.set_ylabel("Tahmin Edilen Fiyatlar (TL)")
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.legend()
        ax.grid(True)
        
        # DÜZENLEME: Grafiğin sıkışmaması için layout'u ayarla
        fig.tight_layout()
        canvas.draw()


    # --- Matplotlib Canvas ---
    fig, ax = plt.subplots(figsize=(6,4), dpi=100)
    
    toolbar_frame = tk.Frame(root)
    # DÜZENLEME: Toolbar yatayda genişlesin
    toolbar_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(5,0))
    canvas = FigureCanvasTkAgg(fig, master=root) 
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()
    
    # DÜZENLEME: Canvas her yöne genişlesin (en önemli değişiklik)
    canvas.get_tk_widget().grid(row=7, column=0, columnspan=2, pady=(0, 10), sticky="nsew")

    metrics_label = tk.Label(root, text=f"Model R²: {r2:.4f}, RMSE: {rmse:,.0f} TL", font=("Arial", 10, "bold"))
    # DÜZENLEME: Metrik etiketi yatayda genişlesin
    metrics_label.grid(row=8, column=0, columnspan=2, pady=5, sticky="ew")


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
            # return # Kapatıldı - yine de tahmin denesin
        if model not in mevcut_modeller:
            messagebox.showwarning("Uyarı", f"'{model}' modeli eğitim verisinde yok. Model en yakın tahmini yapmaya çalışacak.")
            # return # Kapatıldı - yine de tahmin denesin

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

        guncel_grafik_ciz()

    # Program başlarken ilk grafiği çiz
    guncel_grafik_ciz()
    
    root.mainloop()

else:
    print("CSV dosyası okunamadığı için işlem yapılamadı.")