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
from tkinter import ttk 
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

    global_max_yas = int(df['araba_yasi'].max())
    butun_motor_gucleri = sorted(df['motor_gucu'].unique().tolist())

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

    y_pred_test = model_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_test)
    errors_test = y_test - y_pred_test
    mevcut_markalar = sorted(df['marka'].unique().tolist())
    
    # --- GUI ---
    root = tk.Tk()
    root.title("Araba Fiyat Tahmini (Full Algoritmik: KM + Yaş)")

    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(8, weight=1)

    tk.Label(root, text="Marka:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
    marka_combo = ttk.Combobox(root, values=mevcut_markalar, state="readonly", width=27)
    marka_combo.grid(row=0, column=1, padx=5, pady=2, sticky="ew")

    tk.Label(root, text="Model:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
    model_combo = ttk.Combobox(root, state="readonly", width=27)
    model_combo.grid(row=1, column=1, padx=5, pady=2, sticky="ew")

    tk.Label(root, text="Araba Yaşı:").grid(row=2, column=0, sticky="e", padx=5, pady=2)
    yas_scale = tk.Scale(root, from_=0, to=global_max_yas, orient=tk.HORIZONTAL, resolution=1, label=f"Yaş Seçin")
    yas_scale.grid(row=2, column=1, padx=5, pady=2, sticky="ew")

    tk.Label(root, text="Kilometre:").grid(row=3, column=0, sticky="e", padx=5, pady=2)
    km_entry = tk.Entry(root, width=30)
    km_entry.grid(row=3, column=1, padx=5, pady=2, sticky="ew")

    tk.Label(root, text="Motor Gücü (HP):").grid(row=4, column=0, sticky="e", padx=5, pady=2)
    guc_combo = ttk.Combobox(root, state="readonly", width=27)
    guc_combo.grid(row=4, column=1, padx=5, pady=2, sticky="ew")

    def marka_secildi(event):
        secilen_marka = marka_combo.get()
        ilgili_modeller = sorted(df[df['marka'] == secilen_marka]['model'].unique().tolist())
        model_combo['values'] = ilgili_modeller
        model_combo.set('') 
        yas_scale.set(0)
        guc_combo.set('')

    def model_secildi(event):
        secilen_marka = marka_combo.get()
        secilen_model = model_combo.get()
        if secilen_marka and secilen_model:
            filtreli_df = df[(df['marka'] == secilen_marka) & (df['model'] == secilen_model)]
            if not filtreli_df.empty:
                max_yas_local = int(filtreli_df['araba_yasi'].max())
                yas_scale.configure(to=max_yas_local, label=f"Yaş Seçin (Max: {max_yas_local})")
                yas_scale.set(0)
                mevcut_gucler = sorted(filtreli_df['motor_gucu'].unique().tolist())
                guc_combo['values'] = mevcut_gucler
                if len(mevcut_gucler) > 0:
                    guc_combo.current(0)
            else:
                yas_scale.configure(to=global_max_yas, label="Yaş Seçin")
                guc_combo['values'] = butun_motor_gucleri

    marka_combo.bind("<<ComboboxSelected>>", marka_secildi)
    model_combo.bind("<<ComboboxSelected>>", model_secildi)

    tk.Button(root, text="Tahmin Et", command=lambda: tahmin_yap(), bg="purple", fg="white", font=("Arial", 12, "bold")).grid(row=5, column=0, columnspan=2, pady=10, sticky="ew")

    def guncel_grafik_ciz(sonuc_fiyat=None):
        ax1.clear()
        ax1.scatter(y_test, y_pred_test, alpha=0.5, s=10, label='Model Test Verisi')
        if sonuc_fiyat:
            ax1.axhline(y=sonuc_fiyat, color='r', linestyle='-', linewidth=2, label='Hesaplanan Fiyat')
        
        ax1.set_title("Model Performansı ve Tahmin")
        ax1.set_xlabel("Gerçek Fiyatlar")
        ax1.set_ylabel("Tahminler")
        ax1.legend()
        ax1.grid(True)
        
        ax2.clear()
        ax2.hist(errors_test, bins=20, color='gray', edgecolor='black')
        ax2.set_title("Genel Hata Dağılımı")
        
        fig.tight_layout()
        canvas.draw()

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), dpi=100)
    toolbar_frame = tk.Frame(root)
    toolbar_frame.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(5,0)) 
    canvas = FigureCanvasTkAgg(fig, master=root) 
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()
    canvas.get_tk_widget().grid(row=8, column=0, columnspan=2, pady=(0, 10), sticky="nsew") 

    metrics_label = tk.Label(root, text=f"Model R²: {r2:.4f} | Algoritma: %50 KM düşüşü + Yıllık %3 Yaş Değer Kaybı", font=("Arial", 9))
    metrics_label.grid(row=9, column=0, columnspan=2, pady=5, sticky="ew") 

    def tahmin_yap():
        marka = marka_combo.get()
        model = model_combo.get()
        guc_str = guc_combo.get()
        
        if not marka or not model or not guc_str:
            messagebox.showwarning("Eksik", "Lütfen tüm seçimleri yapın.")
            return

        try:
            yas = yas_scale.get() 
            km = int(km_entry.get())
            guc = int(guc_str)
        except ValueError:
            messagebox.showerror("HATA", "KM sayısı geçersiz.")
            return

        # --- ADIM 1: SIFIR ARAÇ (0 YAŞ, 0 KM) BAZ FİYATI BUL ---
        # Yapay zekaya bu arabanın fabrikadan yeni çıktığını varsaydırıyoruz
        veri_sifir = pd.DataFrame({
            'marka': [marka],
            'model': [model],
            'araba_yasi': [0], # YAŞ = 0
            'kilometre': [0],  # KM = 0
            'motor_gucu': [guc]
        })

        regressor = model_pipeline.named_steps['regressor']
        preprocessor = model_pipeline.named_steps['preprocessor']
        
        veri_processed = preprocessor.transform(veri_sifir)
        tree_preds = [tree.predict(veri_processed) for tree in regressor.estimators_]
        sifir_fiyat = np.mean(tree_preds)

        # --- ADIM 2: KM ALGORİTMASI ---
        fiyat_km_sonrasi = 0
        km_aciklama = ""

        if km <= 500000:
            # 0-500k arası oran orantı ile %50'ye kadar düşüş
            kayip_orani = (km / 500000) * 0.50
            dusen_miktar = sifir_fiyat * kayip_orani
            fiyat_km_sonrasi = sifir_fiyat - dusen_miktar
            km_aciklama = f"KM Kaybı (%{kayip_orani*100:.1f}): -{dusen_miktar:,.0f} TL"
        else:
            # 500k üstü algoritması
            fiyat_500k = sifir_fiyat * 0.50 # Önce %50 düş
            kalan_km = km - 500000
            ekstra_dusus = kalan_km * (10000 / 50000) # Her 50k km'de 10.000 TL
            fiyat_km_sonrasi = fiyat_500k - ekstra_dusus
            km_aciklama = f"KM Kaybı (>500k): -{(sifir_fiyat - fiyat_km_sonrasi):,.0f} TL"

        if fiyat_km_sonrasi < 0: fiyat_km_sonrasi = 0

        # --- ADIM 3: YAŞ ALGORİTMASI (%3 YILLIK) ---
        # Formül: Fiyat = Fiyat * (0.97 üssü Yaş)
        # Her sene %3 değer kaybederse, geriye %97'si kalır.
        
        yas_carpan = 0.97 ** yas
        son_fiyat = fiyat_km_sonrasi * yas_carpan
        
        yas_kaybi_miktar = fiyat_km_sonrasi - son_fiyat
        
        # --- SONUÇ GÖSTERİMİ ---
        messagebox.showinfo("Hesaplama Detayı", 
                            f"Fabrika Çıkış (0 Km/0 Yaş) Tahmini: {sifir_fiyat:,.0f} TL\n"
                            f"-----------------------------------\n"
                            f"{km_aciklama}\n"
                            f"KM Sonrası Ara Toplam: {fiyat_km_sonrasi:,.0f} TL\n"
                            f"-----------------------------------\n"
                            f"Yaş: {yas}\n"
                            f"Yaş Değer Kaybı (Yıllık %3): -{yas_kaybi_miktar:,.0f} TL\n"
                            f"-----------------------------------\n"
                            f"SONUÇ FİYAT: {son_fiyat:,.0f} TL")

        guncel_grafik_ciz(sonuc_fiyat=son_fiyat)

    guncel_grafik_ciz()
    root.mainloop()

else:
    print("Veri okunamadı.")