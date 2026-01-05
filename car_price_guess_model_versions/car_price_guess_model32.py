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
    
    # KM GİRİŞİ FORMATLAMA İÇİN FONKSİYONLAR
    def format_km(event=None):
        """Kullanıcı yazı yazarken binlik ayracı ekler"""
        text = km_var.get().replace('.', '').replace(',', '') # Önce temizle
        if text.isdigit():
            formatted = f"{int(text):,}".replace(',', '.') # Python virgül koyar, biz noktaya çeviririz
            km_var.set(formatted)
            km_entry.icursor(tk.END) # İmleci sona taşı

    km_var = tk.StringVar()
    km_var.trace("w", lambda *args: format_km()) # Her tuşa basıldığında çalışır
    
    km_entry = tk.Entry(root, textvariable=km_var, width=30)
    km_entry.grid(row=3, column=1, padx=5, pady=2, sticky="ew")
    tk.Label(root, text="(Örn: 100.000)", font=("Arial", 8, "italic"), fg="gray").grid(row=3, column=2, sticky="w")


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

    tk.Button(root, text="Fiyat Hesapla", command=lambda: tahmin_yap(), bg="#16a085", fg="white", font=("Arial", 12, "bold")).grid(row=5, column=0, columnspan=2, pady=10, sticky="ew")

    def guncel_grafik_ciz(sonuc_fiyat=None):
        ax1.clear()
        ax1.scatter(y_test, y_pred_test, alpha=0.5, s=10, label='Piyasa')
        if sonuc_fiyat:
            ax1.axhline(y=sonuc_fiyat, color='red', linestyle='-', linewidth=2, label='Tahmin')
        
        ax1.set_title("Analiz Sonucu")
        ax1.set_xlabel("Piyasa Fiyatı")
        ax1.set_ylabel("Tahmin")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.clear()
        ax2.hist(errors_test, bins=20, color='gray', edgecolor='black')
        ax2.set_title("Hata Dağılımı")
        
        fig.tight_layout()
        canvas.draw()

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), dpi=100)
    toolbar_frame = tk.Frame(root)
    toolbar_frame.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(5,0)) 
    canvas = FigureCanvasTkAgg(fig, master=root) 
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()
    canvas.get_tk_widget().grid(row=8, column=0, columnspan=2, pady=(0, 10), sticky="nsew") 

    metrics_label = tk.Label(root, text=f"R²: {r2:.4f} ", font=("Arial", 9))
    metrics_label.grid(row=9, column=0, columnspan=2, pady=5, sticky="ew") 

    def tahmin_yap():
        marka = marka_combo.get()
        model = model_combo.get()
        guc_str = guc_combo.get()
        km_text = km_var.get()
        
        if not marka or not model or not guc_str:
            messagebox.showwarning("Eksik", "Lütfen tüm seçimleri yapın.")
            return

        try:
            yas = yas_scale.get() 
            # KM'yi temizleyip integer'a çeviriyoruz (Noktaları sil)
            km_clean = km_text.replace('.', '').replace(',', '')
            if not km_clean: km_clean = "0"
            km = int(km_clean)
            
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

        # --- YENİ YAŞ ALGORİTMASI (%1 SABİT DÜŞÜŞ) ---
        # Formül: Fiyat * (0.99 ^ Yaş)
        yas_carpan = 0.99 ** yas
        
        fiyat_yas_sonrasi = sifir_fiyat * yas_carpan
        yas_kaybi = sifir_fiyat - fiyat_yas_sonrasi

        # --- KM ALGORİTMASI (Aynı Kaldı) ---
        
        # 1. Aşama: 0 - 100k KM 
        if km <= 100000:
            units = km / 10000
            km_carpan = 0.985 ** units
            
        # 2. Aşama: 100k - 1M KM 
        elif km <= 1000000:
            ilk_100k_carpan = 0.985 ** 10 
            kalan_units = (km - 100000) / 10000
            km_carpan = ilk_100k_carpan * (0.992 ** kalan_units)
            
        # 3. Aşama: 1M - 2M KM 
        elif km <= 2000000:
            ilk_100k_carpan = 0.985 ** 10
            ikinci_900k_carpan = 0.992 ** 90 
            
            fazla_km = km - 1000000
            milyon_unit = fazla_km / 1000000
            ucuncu_carpan = 0.9995 ** milyon_unit
            
            km_carpan = ilk_100k_carpan * ikinci_900k_carpan * ucuncu_carpan
            
        # 4. Aşama: 2M KM Sonrası (Sabit)
        else:
            ilk_100k_carpan = 0.985 ** 10
            ikinci_900k_carpan = 0.992 ** 90
            ucuncu_carpan = 0.9995 ** 1 
            
            km_carpan = ilk_100k_carpan * ikinci_900k_carpan * ucuncu_carpan

        son_fiyat = fiyat_yas_sonrasi * km_carpan
        km_kaybi = fiyat_yas_sonrasi - son_fiyat
        
        ekstra_not = ""
        if km > 2000000:
            ekstra_not = "\n(NOT: 2 Milyon KM sınırı aşıldı, KM düşüşü durdu.)"

        # --- SONUÇ ---
        messagebox.showinfo("Fiyat Analizi", 
                            f"Fabrika Çıkış Değeri: {sifir_fiyat:,.0f} TL\n"
                            f"---------------------------------------\n"
                            f"YAŞ ({yas}): -{yas_kaybi:,.0f} TL\n"
                            f"(Her Yıl Sabit %1 Kayıp)\n"
                            f"---------------------------------------\n"
                            f"KM ({km:,}): -{km_kaybi:,.0f} TL\n"
                            f"{ekstra_not}\n"
                            f"---------------------------------------\n"
                            f"TAHMİNİ SATIŞ FİYATI: {son_fiyat:,.0f} TL")

        guncel_grafik_ciz(sonuc_fiyat=son_fiyat)

    guncel_grafik_ciz()
    root.mainloop()

else:
    print("Veri okunamadı.")