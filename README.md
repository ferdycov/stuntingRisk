# Pemodelan Risiko Stunting Anak dengan Machine Learning

Project ini merupakan implementasi **Machine Learning untuk memprediksi risiko kejadian stunting pada anak** berdasarkan data antropometri, riwayat pengukuran, serta faktor risiko yang berkaitan dengan kondisi anak dan lingkungan keluarga.

Project dikembangkan sebagai bagian dari analisis **prediksi insidensi stunting** dengan pendekatan data longitudinal, sehingga informasi dari pengukuran sebelumnya dapat digunakan untuk membantu memprediksi kondisi pada pengukuran berikutnya.

---

## 🎯 Tujuan Project

Tujuan utama project ini adalah:

1. Mengembangkan model Machine Learning untuk memprediksi status/risiko stunting anak.
2. Memanfaatkan data pengukuran sebelumnya sebagai **historical/lag features**.
3. Membandingkan performa beberapa algoritma Machine Learning.
4. Mengidentifikasi variabel yang paling berpengaruh terhadap prediksi stunting.
5. Mengembangkan model yang dapat digunakan sebagai dasar **early risk identification**.
6. Mengintegrasikan model ke dalam aplikasi berbasis **Streamlit** untuk kebutuhan demonstrasi/penerapan.

---

## 📊 Dataset

Dataset berisi informasi mengenai anak, hasil pengukuran antropometri, riwayat kesehatan, karakteristik ibu, serta faktor sosial dan lingkungan.

Beberapa variabel yang digunakan antara lain:

### Identitas dan pengukuran

* `child_id` — ID anak
* `measurement_age_months` — usia anak saat pengukuran
* `measurement_height_cm` — tinggi badan anak
* `measurement_weight_g` — berat badan anak
* `measurement_bmi` — BMI anak
* `sex` — jenis kelamin

### Riwayat pengukuran

Untuk menangkap perubahan kondisi anak dari waktu ke waktu digunakan beberapa **lag features**, antara lain:

* `height_lag_1`
* `weight_lag_1`
* `bmi_lag_1`
* `stunting_lag_1`

`lag_1` menunjukkan nilai dari pengukuran sebelumnya.

Contoh:

```text
Pengukuran sebelumnya
        ↓
height_lag_1
weight_lag_1
stunting_lag_1
        ↓
Pengukuran saat ini
        ↓
Prediksi kondisi berikutnya
```

### Faktor risiko

Beberapa variabel faktor risiko yang digunakan meliputi:

* Riwayat ISPA
* Riwayat diare
* Imunisasi lengkap
* Pemberian vitamin A
* ASI eksklusif
* Pemberian MPASI
* Sumber air minum
* Ketersediaan air bersih
* Fasilitas sanitasi
* Pendidikan ibu
* Tinggi badan ibu
* Kunjungan ANC
* Wealth index

---

## 🧠 Target Prediksi

Target utama model adalah **status stunting**.

Pada tahap multiclass, status dikategorikan menjadi:

* `Normal`
* `Stunted`
* `Severely stunted`
* `Tall`

Model kemudian mempelajari hubungan antara karakteristik anak, kondisi pengukuran, riwayat pengukuran, dan faktor risiko terhadap status tersebut.

---

## 🔄 Workflow Machine Learning

Workflow utama project:

```text
Raw Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Data Preparation
     │
     ▼
Perhitungan HAZ
     │
     ▼
Feature Engineering
     │
     ├── Current Anthropometry
     ├── Historical / Lag Features
     └── Risk Factors
     │
     ▼
Feature Selection
     │
     ▼
Temporal Data Splitting
     │
     ├── Training Data
     └── Testing Data
     │
     ▼
Preprocessing
     │
     ▼
Model Training
     │
     ├── XGBoost
     ├── AdaBoost
     └── KNN
     │
     ▼
Model Evaluation
     │
     ├── Accuracy
     ├── Macro F1
     ├── Weighted F1
     ├── ROC-AUC
     └── Confusion Matrix
     │
     ▼
Model Selection
     │
     ▼
Model Interpretation
     │
     └── SHAP
     │
     ▼
Deployment
     │
     └── Streamlit
```

---

## 🧹 Data Preprocessing

Tahap preprocessing dilakukan untuk memastikan data dapat digunakan oleh algoritma Machine Learning.

Tahapan utama meliputi:

* Pemeriksaan missing values
* Pemeriksaan tipe data
* Konversi variabel numerik
* Penanganan nilai yang tidak valid
* Encoding variabel kategorikal
* Standardisasi variabel numerik
* Feature engineering
* Feature selection

Untuk variabel kategorikal digunakan pendekatan seperti:

* `OneHotEncoder`
* `OrdinalEncoder`

Sedangkan variabel numerik dapat diproses menggunakan:

```python
StandardScaler()
```

---

## 📐 Perhitungan HAZ

Status stunting tidak hanya ditentukan berdasarkan tinggi badan absolut, tetapi mempertimbangkan **usia dan jenis kelamin anak**.

Oleh karena itu digunakan **Height-for-Age Z-score (HAZ)** berdasarkan referensi pertumbuhan WHO.

Secara konseptual:

```text
Tinggi badan
      +
Usia
      +
Jenis kelamin
      ↓
WHO Growth Reference
      ↓
HAZ
      ↓
Klasifikasi status pertumbuhan
```

Pendekatan ini digunakan agar klasifikasi status pertumbuhan tidak hanya bergantung pada tinggi badan absolut.

---

## ⏱️ Temporal Splitting

Karena data memiliki dimensi waktu, pembagian data dilakukan dengan mempertimbangkan urutan pengukuran.

Contoh pendekatan:

```python
train_idx = df_final_ts[df_final_ts["bulan_ke"] < 5].index
test_idx  = df_final_ts[df_final_ts["bulan_ke"] == 5].index
```

Dengan demikian:

```text
Data periode sebelumnya
        ↓
     TRAINING
        ↓
Model Machine Learning
        ↓
Data periode berikutnya
        ↓
      TESTING
```

Pendekatan ini lebih sesuai untuk skenario prediksi karena model tidak menggunakan informasi dari masa depan untuk memprediksi masa sebelumnya.

---

## 🧬 Feature Engineering

Salah satu komponen penting dalam project ini adalah penggunaan **historical/lag features**.

Contohnya:

```python
height_lag_1
weight_lag_1
bmi_lag_1
stunting_lag_1
```

Contoh interpretasi:

Jika seorang anak memiliki:

```text
Pengukuran sebelumnya:
Height = 79 cm
Stunting = 1

Pengukuran saat ini:
Height = 82.5 cm
Age = 24 bulan
```

maka informasi pengukuran sebelumnya dapat digunakan sebagai input model untuk memprediksi kondisi berikutnya.

---

## 🔎 Feature Selection

Feature selection dilakukan untuk mengidentifikasi variabel yang paling informatif terhadap model.

Salah satu pendekatan yang digunakan adalah **Permutation Importance**.

Variabel dengan importance tinggi menunjukkan bahwa perubahan/permutasi pada variabel tersebut memberikan pengaruh besar terhadap performa model.

Contoh feature importance yang diperoleh pada eksperimen:

| Feature                  | Importance |
| ------------------------ | ---------: |
| `measurement_height_cm`  |     0.6525 |
| `measurement_age_months` |     0.3026 |
| `stunting_lag_1`         |     0.2858 |
| `height_lag_1`           |     0.2227 |
| `weight_lag_1`           |     0.0756 |
| `bmi_lag_1`              |     0.0668 |

Hasil tersebut menunjukkan bahwa informasi antropometri dan riwayat status sebelumnya memiliki kontribusi penting terhadap prediksi.

---

# 🤖 Machine Learning Models

Beberapa algoritma dibandingkan dalam project ini.

## 1. XGBoost

```python
XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
```

XGBoost digunakan karena mampu menangani hubungan non-linear dan interaksi antarvariabel dengan baik.

---

## 2. AdaBoost

```python
AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)
```

AdaBoost digunakan sebagai algoritma pembanding berbasis ensemble learning.

---

## 3. K-Nearest Neighbors

```python
KNeighborsClassifier(
    n_neighbors=5
)
```

KNN digunakan sebagai algoritma pembanding berbasis kedekatan antar-observasi.

---

# 📈 Model Evaluation

Performa model dievaluasi menggunakan beberapa metrik.

### Accuracy

Mengukur proporsi prediksi yang benar terhadap seluruh observasi.

### Macro F1

Menghitung rata-rata F1-score dari seluruh kelas tanpa memperhatikan jumlah sampel masing-masing kelas.

Metric ini penting ketika distribusi kelas tidak seimbang.

### Weighted F1

Menghitung F1-score dengan memberikan bobot berdasarkan jumlah sampel pada masing-masing kelas.

### ROC-AUC

Untuk skenario multiclass digunakan pendekatan One-vs-One (`ovo`).

Contoh:

```python
auc_score = roc_auc_score(
    y_test_ts,
    y_prob_ts,
    multi_class="ovo",
    average="macro"
)
```

ROC-AUC dihitung menggunakan probabilitas prediksi:

```python
y_prob_ts = model.predict_proba(X_test_ts)
```

bukan menggunakan hasil kelas:

```python
y_pred_ts = model.predict(X_test_ts)
```

---

# 📊 Confusion Matrix

Confusion matrix digunakan untuk melihat distribusi prediksi benar dan salah pada masing-masing kelas.

Contoh:

```python
cm = confusion_matrix(
    y_test_ts,
    y_pred_ts
)

print(cm)
```

Untuk multiclass, confusion matrix berbentuk matriks:

```text
              Predicted
             C1   C2   C3   C4

Actual C1    TN? ...
Actual C2    ...
Actual C3    ...
Actual C4    ...
```

Dalam multiclass, konsep TP, TN, FP, dan FN perlu dihitung **per kelas**, bukan melakukan:

```python
tp, tn, fp, fn = cm
```

---

# 🧪 Perbandingan Feature Groups

Eksperimen dilakukan dengan beberapa kelompok fitur.

### Current Anthropometry

Berisi variabel pengukuran saat ini.

```text
Current Anthropometry
        ↓
Model
        ↓
Prediction
```

### History / Lag

Berisi informasi dari pengukuran sebelumnya.

```text
Historical Data
      ↓
Lag Features
      ↓
Model
```

### Risk Factors

Berisi faktor risiko sosial, kesehatan, dan lingkungan.

### Current + History

Menggabungkan kondisi pengukuran saat ini dengan riwayat sebelumnya.

### Current + Risk Factors

Menggabungkan kondisi pengukuran saat ini dengan faktor risiko.

Hasil eksperimen menunjukkan bahwa fitur antropometri saat ini dan kombinasi dengan informasi historis memberikan performa yang lebih baik dibandingkan penggunaan faktor risiko saja pada beberapa model.

---

# 🔬 Model Interpretation dengan SHAP

Setelah model terbaik diperoleh, interpretasi model dilakukan menggunakan **SHAP (SHapley Additive exPlanations)**.

Tujuannya bukan hanya mengetahui:

> "Apakah anak diprediksi berisiko?"

tetapi juga:

> "Variabel apa yang paling berkontribusi terhadap prediksi tersebut?"

Contoh interpretasi:

```text
Prediction
    │
    ├── measurement_height_cm
    │        ↓
    │     Contribution
    │
    ├── stunting_lag_1
    │        ↓
    │     Contribution
    │
    ├── measurement_age_months
    │        ↓
    │     Contribution
    │
    └── height_lag_1
             ↓
          Contribution
```

SHAP digunakan untuk meningkatkan **interpretability** model sehingga hasil prediksi dapat dijelaskan kepada pengguna.

---

# 🖥️ Deployment

Model yang telah dipilih dapat disimpan menggunakan `joblib`.

Contoh:

```python
joblib.dump(
    {
        "model": xgb_model,
        "label_encoder": le_ts,
        "feature_columns": feature_columns
    },
    "stunting_xgb.pkl"
)
```

Model kemudian dapat digunakan pada aplikasi **Streamlit**.

Contoh alur aplikasi:

```text
Input Data Anak
      ↓
Preprocessing
      ↓
Machine Learning Model
      ↓
Prediksi Risiko
      ↓
Risk Percentage
      ↓
SHAP Explanation
      ↓
Faktor yang Mempengaruhi Prediksi
```

---

# 📦 Dependencies

Beberapa library utama yang digunakan:

```text
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
shap
joblib
streamlit
plotly
```

---

# ▶️ Running Application

Untuk menjalankan aplikasi Streamlit:

```bash
streamlit run stunting/app.py
```

Aplikasi kemudian dapat digunakan untuk melakukan simulasi prediksi berdasarkan data anak.

---

# ⚠️ Catatan Penting

Project ini merupakan **model prediksi dan risk identification**, bukan alat diagnosis klinis.

Output model sebaiknya digunakan sebagai **decision-support / early warning tool** dan tidak menggantikan:

* pemeriksaan antropometri,
* penilaian pertumbuhan berdasarkan standar yang berlaku,
* pemeriksaan klinis,
* atau keputusan tenaga kesehatan.

Validasi eksternal dan evaluasi lebih lanjut diperlukan sebelum model digunakan pada lingkungan klinis atau operasional secara nyata.

---

# 🚀 Future Development

Pengembangan selanjutnya dapat mencakup:

* [ ] Hyperparameter tuning
* [ ] Cross-validation yang mempertimbangkan struktur temporal/group
* [ ] External validation
* [ ] Calibration model
* [ ] Threshold optimization
* [ ] SHAP dashboard
* [ ] Monitoring model performance
* [ ] Model drift detection
* [ ] Integrasi database
* [ ] REST API untuk model inference
* [ ] Deployment aplikasi
* [ ] Validasi pada dataset eksternal

---

# 👨‍💻 Project Focus

Project ini berfokus pada penerapan **Machine Learning untuk early identification of stunting risk** dengan menggabungkan:

**Current Anthropometry + Historical Measurements + Risk Factors**

Pendekatan tersebut diharapkan dapat memberikan informasi prediktif yang lebih berguna dibandingkan hanya menggunakan kondisi antropometri pada satu titik pengukuran.

---

## 📌 Kesimpulan

Project ini mengembangkan pipeline Machine Learning untuk memprediksi kondisi stunting anak dengan mempertimbangkan **data antropometri, riwayat pengukuran, dan faktor risiko**.

Beberapa algoritma dibandingkan untuk mendapatkan model dengan performa terbaik. Selain performa prediktif, interpretasi model menggunakan SHAP digunakan untuk memahami faktor yang berkontribusi terhadap hasil prediksi.

Pipeline akhir diarahkan untuk dapat digunakan dalam aplikasi berbasis Streamlit sebagai **sistem pendukung identifikasi risiko stunting secara dini**.

## ⚠️ Disclaimer Data

**Data yang digunakan dalam project ini merupakan data sintetis (synthetic data) yang dibuat untuk keperluan pengembangan, pengujian, demonstrasi, dan pembelajaran Machine Learning.**

Data tidak merepresentasikan data pasien atau individu nyata dan **tidak mengandung informasi kesehatan maupun identitas pribadi pasien yang sebenarnya**.

Oleh karena itu:

* Hasil analisis dan performa model pada project ini **tidak dapat dianggap sebagai representasi performa pada populasi nyata**.
* Model yang dihasilkan **tidak ditujukan untuk diagnosis, prognosis, atau pengambilan keputusan klinis**.
* Hasil prediksi tidak boleh digunakan sebagai dasar tunggal dalam menentukan intervensi atau tindakan terhadap pasien.
* Validasi menggunakan **data nyata dan independen** diperlukan sebelum model dapat dipertimbangkan untuk penggunaan operasional atau klinis.
* Struktur variabel dan karakteristik data dibuat menyerupai skenario data stunting untuk tujuan demonstrasi, namun **tidak dimaksudkan untuk menggambarkan kondisi epidemiologis populasi tertentu**.

Project ini ditujukan sebagai **proof of concept (PoC)** penerapan Machine Learning untuk pemodelan risiko/insidensi stunting.
