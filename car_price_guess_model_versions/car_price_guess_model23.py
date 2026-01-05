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
from tkinter import ttk  # Dropdown (Combobox) için gerekli
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
    # Verileri küçük harfe çeviriyoruz
    df['marka'] = df['marka'].astype(str).str.lower()
    df['model'] = df['model'].astype(str).str.lower()
    df['araba_yasi'] = df['araba_yasi'].astype(int)
    df['kilometre'] = df['kilometre'].astype(int)
    df['motor_gucu'] = df['motor_gucu'].astype(int)
    df['fiyat'] = df['fiyat'].astype(int)

    # --- ÖNEMLİ: Maksimum Yaşı Hesapla ---
    max_yas_verisi = int(df['araba_yasi'].max())
    # -------------------------------------

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

    print("Model performansı hesaplanıyor...")
    y_pred_test = model_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_test)
    print(f"Modelin Genel R2 Skoru: {r2:.4f}, RMSE Skoru: {rmse:,.0f} TL")

    errors_test = y_test - y_pred_test
    mevcut_markalar = sorted(df['marka'].unique().tolist())
    
    # --- GUI ---
    root = tk.Tk()
    root.title("Araba Fiyat Tahmini")

    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(8, weight=1)

    # --- 1. MARKA (DROPDOWN) ---
    tk.Label(root, text="Marka:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
    marka_combo = ttk.Combobox(root, values=mevcut_markalar, state="readonly", width=27)
    marka_combo.grid(row=0, column=1, padx=5, pady=2, sticky="ew")

    # --- 2. MODEL (DROPDOWN) ---
    tk.Label(root, text="Model:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
    model_combo = ttk.Combobox(root, state="readonly", width=27)
    model_combo.grid(row=1, column=1, padx=5, pady=2, sticky="ew")

    def marka_secildi(event):
        secilen_marka = marka_combo.get()
        ilgili_modeller = sorted(df[df['marka'] == secilen_marka]['model'].unique().tolist())
        model_combo['values'] = ilgili_modeller
        model_combo.set('') 

    marka_combo.bind("<<ComboboxSelected>>", marka_secildi)

    # --- 3. ARABA YAŞI (SLIDER / SÜRGÜ) ---
    tk.Label(root, text="Araba Yaşı:").grid(row=2, column=0, sticky="e", padx=5, pady=2)
    
    # tk.Scale widget'ı kullanıyoruz
    yas_scale = tk.Scale(root, 
                         from_=0, 
                         to=max_yas_verisi,  # CSV'deki maksimum yaş buraya geliyor
                         orient=tk.HORIZONTAL, 
                         resolution=1, # 1'er 1'er artacak
                         label="Yaş Seçin")
    yas_scale.grid(row=2, column=1, padx=5, pady=2, sticky="ew")

    # --- 4. DİĞER GİRİŞLER (Kalanlar: KM ve Güç) ---
    entries = []
    labels_num = ["Kilometre:", "Motor Gücü:"]
    
    # Döngü artık 3. satırdan başlayacak
    for i, text in enumerate(labels_num):
        row_idx = i + 3 
        tk.Label(root, text=text).grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        entry = tk.Entry(root, width=30)
        entry.grid(row=row_idx, column=1, padx=5, pady=2, sticky="ew")
        entries.append(entry)

    km_entry, guc_entry = entries

    tk.Button(root, text="Tahmin Et", command=lambda: tahmin_yap(), bg="green", fg="white", font=("Arial", 12, "bold")).grid(row=5, column=0, columnspan=2, pady=10, sticky="ew")

    def guncel_grafik_ciz():
        ax1.clear()
        ax1.scatter(y_test, y_pred_test, alpha=0.6, edgecolors='k', s=50, label='Test Verisi')
        all_plot_values = list(y_test) + list(y_pred_test)
        min_val = min(all_plot_values) * 0.9
        max_val = max(all_plot_values) * 1.1
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Mükemmel Tahmin (y=x)")
        ax1.set_title("Gerçek vs Tahmin")
        ax1.set_xlabel("Gerçek Fiyat")
        ax1.set_ylabel("Tahmin")
        ax1.set_xlim(min_val, max_val)
        ax1.set_ylim(min_val, max_val)
        ax1.legend()
        ax1.grid(True)
        
        ax2.clear()
        ax2.hist(errors_test, bins=20, color='orange', edgecolor='black')
        ax2.set_title("Hata Dağılımı")
        ax2.set_xlabel("Hata")
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

    def tahmin_yap():
        marka = marka_combo.get()
        model = model_combo.get()
        
        if not marka or not model:
            messagebox.showwarning("Eksik Bilgi", "Lütfen bir Marka ve Model seçiniz.")
            return

        try:
            # Yaşı artık Slider'dan alıyoruz (yas_scale.get())
            yas = yas_scale.get() 
            km = int(km_entry.get())
            guc = int(guc_entry.get())
        except ValueError:
            messagebox.showerror("HATA", "Lütfen KM ve Güç alanlarına geçerli sayılar girin!")
            return

        yeni_veri = pd.DataFrame({
            'marka': [marka],
            'model': [model],
            'araba_yasi': [yas],
            'kilometre': [km],
            'motor_gucu': [guc]
        })

        # --- TAHMİN ARALIĞI ---
        regressor = model_pipeline.named_steps['regressor']
        preprocessor = model_pipeline.named_steps['preprocessor']
        
        yeni_veri_processed = preprocessor.transform(yeni_veri)
        tree_predictions = [tree.predict(yeni_veri_processed) for tree in regressor.estimators_]
        tree_predictions = np.array(tree_predictions)

        tahmin = np.mean(tree_predictions)
        belirsizlik_std = np.std(tree_predictions)
        alt_limit = np.percentile(tree_predictions, 5)
        ust_limit = np.percentile(tree_predictions, 95)
        
        messagebox.showinfo("Tahmin Sonucu", 
                            f"Tahmini Fiyat: {tahmin:,.0f} TL\n\n"
                            f"Model Belirsizliği (Std Sapma): {belirsizlik_std:,.0f} TL\n"
                            f"%90 Tahmin Aralığı: {alt_limit:,.0f} TL - {ust_limit:,.0f} TL\n"
                            f"\n-------------------------------------------------------\n"
                            f"(Modelin genel R² skoru: {r2:.4f})\n"
                            f"(Modelin genel RMSE'si: {rmse:,.0f} TL)")

        guncel_grafik_ciz()

    guncel_grafik_ciz()
    root.mainloop()

else:
    print("CSV dosyası okunamadığı için işlem yapılamadı.")