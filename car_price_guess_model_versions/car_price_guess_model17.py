import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# YENİ: StandardScaler eklendi
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
    messagebox.showerror("Hatalı Dosya", "HATA: 'arabalar.csv' dosyası bulunamadı. Program kapatılacak.")
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

    # GÜNCELLENDİ: 'passthrough' yerine StandardScaler() kullanıyoruz
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

    # --- YENİ: ÖZELLİK ÖNEMLERİNİ KONTROL ET ---
    print("--- Model Özellik Önemleri ---")
    try:
        # Pipeline'dan eğitilmiş regressor'ü ve ön işlemciyi al
        regressor = model_pipeline.named_steps['regressor']
        preprocessor = model_pipeline.named_steps['preprocessor']
        
        # Ön işlemciden özellik adlarını al (OneHotEncoder ve StandardScaler dahil)
        feature_names = preprocessor.get_feature_names_out()
        
        # Önemleri ve adları birleştir
        importances = regressor.feature_importances_
        feature_importance_df = pd.DataFrame({'Özellik': feature_names, 'Önem': importances})
        
        # Önem sırasına göre sırala
        feature_importance_df = feature_importance_df.sort_values(by='Önem', ascending=False)
        
        # En önemli ilk 20 özelliği ve 'kilometre'yi göster
        print("Model için en önemli 20 özellik:")
        print(feature_importance_df.head(20).to_string()) 
        
        print("\n----------------------------------")
        # 'kilometre' içeren satırı bul ve göster
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

    labels = ["Marka:", "Model:", "Araba Yaşı:", "Kilometre:", "Motor Gücü:"]
    entries = []
    
    for i, text in enumerate(labels):
        tk.Label(root, text=text).grid(row=i, column=0, sticky="e", padx=5, pady=2)
        entry = tk.Entry(root, width=30)
        entry.grid(row=i, column=1, padx=5, pady=2)
        entries.append(entry)

    marka_entry, model_entry, yas_entry, km_entry, guc_entry = entries

    tk.Button(root, text="Tahmin Et", command=lambda: tahmin_yap(), bg="green", fg="white", font=("Arial", 12, "bold")).grid(row=5, column=0, columnspan=2, pady=10)

    def guncel_grafik_ciz(tahmin_cizgisi=None):
        ax.clear()
        ax.scatter(y_test, y_pred_test, alpha=0.6, edgecolors='k', s=50, label='Test Verisi')

        all_plot_values = list(y_test) + list(y_pred_test)
        
        if tahmin_cizgisi is not None:
            ax.axhline(y=tahmin_cizgisi, color='red', linestyle='--', lw=2, 
                         label=f"Sizin Tahmininiz ({tahmin_cizgisi:,.0f} TL)")
            all_plot_values.append(tahmin_cizgisi)

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
        canvas.draw()


    # --- Matplotlib Canvas ---
    fig, ax = plt.subplots(figsize=(6,4), dpi=100)
    
    toolbar_frame = tk.Frame(root)
    toolbar_frame.grid(row=6, column=0, columnspan=2, sticky="w", pady=(5,0))
    canvas = FigureCanvasTkAgg(fig, master=root) 
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()
    
    canvas.get_tk_widget().grid(row=7, column=0, columnspan=2, pady=(0, 10))

    metrics_label = tk.Label(root, text=f"Model R²: {r2:.4f}, RMSE: {rmse:,.0f} TL", font=("Arial", 10, "bold"))
    metrics_label.grid(row=8, column=0, columnspan=2, pady=5)


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
            messagebox.showerror("HATA", f"'{marka}' markası veri setinde yok!")
            return
        if model not in mevcut_modeller:
            messagebox.showerror("HATA", f"'{model}' modeli veri setinde yok!")
            return

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

        guncel_grafik_ciz(tahmin_cizgisi=tahmin)

    # Program başlarken ilk grafiği çiz
    guncel_grafik_ciz()
    
    root.mainloop()

else:
    print("CSV dosyası okunamadığı için işlem yapılamadı.")