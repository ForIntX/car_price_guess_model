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

    tk.Label(root, text="Marka:").grid(row=0, column=0, sticky="e")
    tk.Label(root, text="Model:").grid(row=1, column=0, sticky="e")
    tk.Label(root, text="Araba Yaşı:").grid(row=2, column=0, sticky="e")
    tk.Label(root, text="Kilometre:").grid(row=3, column=0, sticky="e")
    tk.Label(root, text="Motor Gücü:").grid(row=4, column=0, sticky="e")

    marka_entry = tk.Entry(root)
    model_entry = tk.Entry(root)
    yas_entry = tk.Entry(root)
    km_entry = tk.Entry(root)
    guc_entry = tk.Entry(root)

    marka_entry.grid(row=0, column=1)
    model_entry.grid(row=1, column=1)
    yas_entry.grid(row=2, column=1)
    km_entry.grid(row=3, column=1)
    guc_entry.grid(row=4, column=1)

    # --- Tahmin Fonksiyonu ---
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

        # --- Grafik ---
        y_pred = model_pipeline.predict(X_test)
        fig = plt.Figure(figsize=(6,5), dpi=100)
        ax = fig.add_subplot(111)
        ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', s=50)
        min_val = min(y_test.min(), y_pred.min()) * 0.9
        max_val = max(y_test.max(), y_pred.max()) * 1.1
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Mükemmel Tahmin (y=x)")
        ax.set_title("Gerçek vs Tahmin Edilen Fiyatlar")
        ax.set_xlabel("Gerçek Fiyatlar (TL)")
        ax.set_ylabel("Tahmin Edilen Fiyatlar (TL)")
        ax.legend()
        ax.grid(True)

        # Tkinter içinde göster
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(row=6, column=0, columnspan=2, pady=10)

    tk.Button(root, text="Tahmin Et", command=tahmin_yap, bg="green", fg="white").grid(row=5, column=0, columnspan=2, pady=10)

    root.mainloop()
