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
    print(f"Modelin Genel R2 Skoru: {r2:.4f}, RMSE Skoru: {rmse:,.0f} TL")

    errors_test = y_test - y_pred_test

    mevcut_markalar = list(df['marka'].unique())
    mevcut_modeller = list(df['model'].unique())

    # --- GUI ---
    root = tk.Tk()
    root.title("Araba Fiyat Tahmini")

    root.grid_columnconfigure(1, weight=1)
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

        fig.tight_layout()
        canvas.draw()


    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), dpi=100)
    
    toolbar_frame = tk.Frame(root)
    toolbar_frame.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(5,0)) 
    canvas = FigureCanvasTkAgg(fig, master=root) 
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()
    
    canvas.get_tk_widget().grid(row=8, column=0, columnspan=2, pady=(0, 10), sticky="nsew") 

    metrics_label = tk.Label(root, text=f"Modelin Genel R² skoru: {r2:.4f}, RMSE Skoru: {rmse:,.0f} TL", font=("Arial", 10, "bold"))
    metrics_label.grid(row=9, column=0, columnspan=2, pady=5, sticky="ew") 


    ### -----------------------------------------------------------
    ### DEĞİŞİKLİK: 'tahmin_yap' FONKSİYONU GÜNCELLENDİ
    ### (TAHMİN ARALIĞI HESAPLAMASI EKLENDİ)
    ### -----------------------------------------------------------
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
            messagebox.showwarning("Uyarı", f"'{model}' modeli eğitim verisinde yok. Model en yakın tahmini yapmaya çalışak.")

        yeni_veri = pd.DataFrame({
            'marka': [marka],
            'model': [model],
            'araba_yasi': [yas],
            'kilometre': [km],
            'motor_gucu': [guc]
        })

        # --- YENİ EKLENEN BÖLÜM: TAHMİN ARALIĞI ---

        # 1. Pipeline'ın içinden asıl model (regressor) ve önişlemciyi (preprocessor) çek
        regressor = model_pipeline.named_steps['regressor']
        preprocessor = model_pipeline.named_steps['preprocessor']
        
        # 2. Yeni veriyi manuel olarak önişlemciden geçir (scale/encode et)
        yeni_veri_processed = preprocessor.transform(yeni_veri)

        # 3. Modeldeki (n_estimators=100) 100 ağacın her birinden tek tek tahmin al
        tree_predictions = [tree.predict(yeni_veri_processed) for tree in regressor.estimators_]
        
        # 4. Bu 100 tahmini numpy array'e çevir
        tree_predictions = np.array(tree_predictions)

        # 5. Bu 100 tahminin ortalamasını, standart sapmasını ve %90'lık aralığını hesapla
        
        # Ana tahmin (100 ağacın ortalaması)
        tahmin = np.mean(tree_predictions)
        
        # Standart sapma (ağaçların ne kadar yayıldığını gösterir - düşükse iyidir)
        belirsizlik_std = np.std(tree_predictions)

        # %90 Güven Aralığı (5. yüzdelik ve 95. yüzdelik dilim)
        alt_limit = np.percentile(tree_predictions, 5)
        ust_limit = np.percentile(tree_predictions, 95)
        
        # --- BİTİŞ: YENİ EKLENEN BÖLÜM ---


        # Mesaj kutusunu yeni bilgilerle güncelle
        messagebox.showinfo("Tahmin Sonucu", 
                            f"Tahmini Fiyat: {tahmin:,.0f} TL\n\n"
                            f"Model Belirsizliği (Std Sapma): {belirsizlik_std:,.0f} TL\n"
                            f"%90 Tahmin Aralığı: {alt_limit:,.0f} TL - {ust_limit:,.0f} TL\n"
                            f"\n-------------------------------------------------------\n"
                            f"(Modelin genel R² skoru: {r2:.4f})\n"
                            f"(Modelin genel RMSE'si: {rmse:,.0f} TL)")


        # Grafikler test seti üzerinden çizilir, yeni tahminle güncellenmez
        guncel_grafik_ciz()

    # Program başlarken ilk grafikleri çiz
    guncel_grafik_ciz()
    
    root.mainloop()

else:
    print("CSV dosyası okunamadığı için işlem yapılamadı.")