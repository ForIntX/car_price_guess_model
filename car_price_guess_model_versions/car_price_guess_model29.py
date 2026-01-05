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
    root.title("Akıllı Araba Fiyatlandırma Sistemi")

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

    tk.Button(root, text="Fiyat Hesapla", command=lambda: tahmin_yap(), bg="#2c3e50", fg="white", font=("Arial", 12, "bold")).grid(row=5, column=0, columnspan=2, pady=10, sticky="ew")

    def guncel_grafik_ciz(sonuc_fiyat=None):
        ax1.clear()
        ax1.scatter(y_test, y_pred_test, alpha=0.5, s=10, label='Piyasa Verileri')
        if sonuc_fiyat:
            ax1.axhline(y=sonuc_fiyat, color='red', linestyle='-', linewidth=2, label='Tahmin Edilen')
        
        ax1.set_title("Piyasa Analizi")
        ax1.set_xlabel("Gerçek Fiyatlar")
        ax1.set_ylabel("Tahminler")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.clear()
        ax2.hist(errors_test, bins=20, color='#3498db', edgecolor='black')
        ax2.set_title("Hata Payı Dağılımı")
        
        fig.tight_layout()
        canvas.draw()

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), dpi=100)
    toolbar_frame = tk.Frame(root)
    toolbar_frame.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(5,0)) 
    canvas = FigureCanvasTkAgg(fig, master=root) 
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()
    canvas.get_tk_widget().grid(row=8, column=0, columnspan=2, pady=(0, 10), sticky="nsew") 

    metrics_label = tk.Label(root, text=f"Model Doğruluğu (R²): {r2:.4f} | Algoritma: Kademeli Yaş + Segmentli KM Düşüşü", font=("Arial", 9))
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

        # 1. BAZ FİYAT (0 KM, 0 YAŞ)
        veri_sifir = pd.DataFrame({
            'marka': [marka], 'model': [model], 'araba_yasi': [0], 'kilometre': [0], 'motor_gucu': [guc]
        })
        regressor = model_pipeline.named_steps['regressor']
        preprocessor = model_pipeline.named_steps['preprocessor']
        veri_processed = preprocessor.transform(veri_sifir)
        tree_preds = [tree.predict(veri_processed) for tree in regressor.estimators_]
        sifir_fiyat = np.mean(tree_preds)

        # --- GELİŞMİŞ ALGORİTMA BAŞLANGICI ---
        mevcut_fiyat = sifir_fiyat
        
        # A) YAŞ ETKİSİ (KADEMELİ DÜŞÜŞ)
        # 1. Yıl: Çok sert düşüş (%15)
        # 2-4. Yıl: Orta sert düşüş (%10)
        # 5+ Yıl: Yavaş düşüş (%6)
        
        for yl in range(1, yas + 1):
            if yl == 1:
                oran = 0.85 # %15 kayıp
            elif 2 <= yl <= 4:
                oran = 0.90 # %10 kayıp
            else:
                oran = 0.94 # %6 kayıp (Yaşlandıkça değer kaybı azalır)
            mevcut_fiyat *= oran
            
        fiyat_yas_sonrasi = mevcut_fiyat
        yas_kaybi = sifir_fiyat - fiyat_yas_sonrasi

        # B) KM ETKİSİ (SEGMENTLİ DÜŞÜŞ)
        # Düz %5 yerine, KM arttıkça etkisi azalan bir yapı.
        # 0-100k km: Her 10k km'de %1.5 düşüş (Araba yeniyken km çok etkiler)
        # 100k+ km: Her 10k km'de %0.8 düşüş (Araba eskiyince km o kadar korkutmaz)
        
        # Kaç tane 10.000 km var?
        on_binlikler = km / 10000
        
        if km <= 100000:
            # Sadece 1. segment
            km_carpan = 0.985 ** on_binlikler # %1.5 düşüş
        else:
            # İlk 100k için sabit düşüşü uygula
            ilk_100k_carpan = 0.985 ** 10 # 10 tane 10binlik
            
            # Kalan km için daha az düşüş uygula
            kalan_on_binlikler = (km - 100000) / 10000
            ikinci_segment_carpan = 0.992 ** kalan_on_binlikler # %0.8 düşüş
            
            km_carpan = ilk_100k_carpan * ikinci_segment_carpan

        son_fiyat = fiyat_yas_sonrasi * km_carpan
        km_kaybi = fiyat_yas_sonrasi - son_fiyat

        # --- SONUÇ ---
        messagebox.showinfo("Akıllı Fiyat Analizi", 
                            f"Fabrika Çıkış Değeri: {sifir_fiyat:,.0f} TL\n"
                            f"---------------------------------------\n"
                            f"YAŞ ETKİSİ ({yas} Yaş):\n"
                            f"- {yas_kaybi:,.0f} TL değer kaybı.\n"
                            f"(İlk yıl sert, sonraki yıllar yumuşak düşüş)\n"
                            f"---------------------------------------\n"
                            f"KM ETKİSİ ({km:,} KM):\n"
                            f"- {km_kaybi:,.0f} TL değer kaybı.\n"
                            f"(Düşük km'de yüksek etki, yüksek km'de azalan etki)\n"
                            f"---------------------------------------\n"
                            f"TAHMİNİ SATIŞ FİYATI: {son_fiyat:,.0f} TL")

        guncel_grafik_ciz(sonuc_fiyat=son_fiyat)

    guncel_grafik_ciz()
    root.mainloop()

else:
    print("Veri okunamadı.")