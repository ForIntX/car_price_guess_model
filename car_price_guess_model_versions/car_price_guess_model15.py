import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings

warnings.filterwarnings('ignore')

# --- Veri Seti ---
try:
    df = pd.read_csv("arabalar.csv")
except FileNotFoundError:
    print("HATA: 'arabalar.csv' dosyası bulunamadı.")
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
            ('num', 'passthrough', numerical_features)
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10))
    ])

    model_pipeline.fit(X_train, y_train)

    # --- GUI ---
    root = tk.Tk()
    root.title("Araba Fiyat Tahmini")

    # Form etiketleri
    labels = ["Marka:", "Model:", "Araba Yaşı:", "Kilometre:", "Motor Gücü:"]
    for i, text in enumerate(labels):
        tk.Label(root, text=text).grid(row=i, column=0, sticky="e")

    marka_entry = tk.Entry(root)
    model_entry = tk.Entry(root)
    yas_entry = tk.Entry(root)
    km_entry = tk.Entry(root)
    guc_entry = tk.Entry(root)

    entries = [marka_entry, model_entry, yas_entry, km_entry, guc_entry]
    for i, entry in enumerate(entries):
        entry.grid(row=i, column=1)

    # --- Matplotlib Canvas ---
    fig, ax = plt.subplots(figsize=(6,4), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=6, column=0, columnspan=2, pady=10)

    def tahmin_yap():
        marka = marka_entry.get().strip().lower()
        model = model_entry.get().strip().lower()
        try:
            yas = int(yas_entry.get())
            km = int(km_entry.get())
            guc = int(guc_entry.get())
        except ValueError:
            messagebox.showerror("HATA", "Lütfen geçerli sayılar girin!")
            return

        if marka not in df['marka'].unique():
            messagebox.showerror("HATA", f"'{marka}' markası veri setinde yok!")
            return
        if model not in df['model'].unique():
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
        messagebox.showinfo("Tahmin Sonucu", f"Tahmini Araba Fiyatı: {tahmin:,.0f} TL")

        # --- Grafik Güncelle ---
        y_pred = model_pipeline.predict(X_test)
        ax.clear()
        ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', s=50)
        # Kullanıcının tahmini kırmızı nokta olarak gösterilsin
        ax.scatter([tahmin], [tahmin], color='red', s=100, label="Kendi Tahmininiz", marker='X')
        min_val = min(y_test.min(), y_pred.min(), tahmin) * 0.9
        max_val = max(y_test.max(), y_pred.max(), tahmin) * 1.1
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Mükemmel Tahmin (y=x)")
        ax.set_title("Gerçek vs Tahmin Edilen Fiyatlar")
        ax.set_xlabel("Gerçek Fiyatlar (TL)")
        ax.set_ylabel("Tahmin Edilen Fiyatlar (TL)")
        ax.legend()
        ax.grid(True)
        canvas.draw()

    tk.Button(root, text="Tahmin Et", command=tahmin_yap, bg="green", fg="white").grid(row=5, column=0, columnspan=2, pady=10)

    root.mainloop()
