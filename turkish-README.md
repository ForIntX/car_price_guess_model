# car_price_guess_model
---

# 🚗 Yapay Zeka ile Araba Fiyat Tahmin Projesi

> Bu proje, Yapay Zeka dersi kapsamında geliştirilmiş olup, kullanıcının girdiği araba özelliklerine dayanarak **arabanın tahmini piyasa fiyatını** hesaplayan bir yapay zeka modelini içermektedir.

---

## 🎯 Projenin Amacı

Projenin temel amacı, kullanıcının girdiği **marka, model, yaş, kilometre, motor gücü** gibi araba bilgilerini kullanarak, bir yapay zeka modeli aracılığıyla **araba fiyat tahmini** yapmaktır.

---

## 👥 Ekip Üyeleri

* **Muhammet Burak Akkaş** – Ekip Lideri, Yapay Zeka Modeli Geliştirme & GUI
* **Doğanay Yıldız** – Grafik Oluşturma
* **Gürkan Özdemir** – Hata Hesaplama
* **Berkay Berber** – Veri Analizi
---


## 📈 Projenin Gelişim Süreci

### 1. Başlangıç Noktası

Proje ilk olarak **basit bir doğrusal regresyon modeli** oluşturma fikriyle başladı. Amaç, kullanıcıdan alınan temel araba bilgilerine dayanarak fiyat tahmini yapabilmekti. İlk veri analizleri sırasında, veri setinde eksik ve düzensiz bilgiler olduğunu fark ettik.

### 2. Karşılaşılan Zorluklar ve Çözümler

Süreç boyunca bazı zorluklarla karşılaştık ve bunları ekip olarak çözdük:

* **Düşük Model Performansı:** İlk denemelerde R² skorumuz oldukça düşüktü. Berkay’ın yaptığı detaylı veri analizi sayesinde **kategorik verileri One-Hot-Encoding ile modele dahil etme** kararı aldık.
* **Hata Analizi:** Gürkan, modelin en çok nerede yanıldığını tespit etti. Bu sayede modelin zayıf noktalarını belirleyebildik.
* **Görselleştirme:** Doğanay, modelin sonuçlarını ve hatalarını **grafiklerle görselleştirdi**, böylece performans değerlendirmesi daha anlaşılır hâle geldi.
* **Model Geliştirme:** Muhammet Burak, elde edilen veriler ve analizler ışığında modeli **RandomForest gibi daha güçlü bir algoritma** ile güncelledi ve GUI üzerinden kullanıcı dostu bir arayüz oluşturdu.

### 3. Öğrendiklerimiz

Bu proje sayesinde şunları deneyimledik:

* Veri temizleme ve analiz etmenin model başarısındaki kritik önemi
* Model performansını doğru metriklerle (RMSE, R²) ölçmenin değeri
* Sonuçları görselleştirerek anlaşılır hâle getirmenin gücü
* Tüm parçaları birleştirip çalışan bir uygulama ve güçlü bir yapay zeka modeli oluşturmanın önemi

---


## 💻 Kullanılan Teknolojiler

Projede Python ekosistemi ve aşağıdaki kütüphaneler kullanılmıştır:

* **contourpy** – 1.3.3
* **cycler** – 0.12.1
* **fonttools** – 4.60.1
* **joblib** – 1.5.2
* **kiwisolver** – 1.4.9
* **matplotlib** – 3.10.7
* **numpy** – 2.3.4
* **packaging** – 25.0
* **pandas** – 2.3.3
* **pillow** – 12.0.0
* **pip** – 24.0
* **pyparsing** – 3.2.5
* **python-dateutil** – 2.9.0.post0
* **pytz** – 2025.2
* **scikit-learn** – 1.7.2
* **scipy** – 1.16.3
* **six** – 1.17.0
* **threadpoolctl** – 3.6.0
* **tzdata** – 2025.2
* **tkinter**

---

## 🚀 Projeyi Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için şu adımları izleyin:

1. Repoyu klonlayın ve proje klasörüne girin:

```bash
git clone [REPO_URL]
cd [PROJE_KLASORU_ADI]
```

2. Sanal ortam (venv) oluşturun ve aktif edin:

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

3. Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

4. Uygulamayı çalıştırın:

```bash
python car_price_guess_model[MODEL_NUMARASI].py
```

---


# 🐧 linux alternatif çalıştırma yöntemi

start dosyasına tıklayın. veya bir sonraki komudu terminalde çalıştırın.

```bash
./start
```

---