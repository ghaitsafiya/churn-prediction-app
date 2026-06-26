# Proyek Prediksi Churn Pelanggan

Proyek ini bertujuan untuk membangun model *Machine Learning* yang dapat memprediksi pelanggan yang berpotensi melakukan **churn**, yaitu berhenti menggunakan layanan atau produk perusahaan. Model ini diharapkan dapat membantu perusahaan mengidentifikasi pelanggan yang memiliki risiko tinggi untuk churn sehingga dapat dilakukan strategi *retention* lebih awal.

## Dataset

Dataset yang digunakan merupakan dataset pelanggan pada domain **Sales & Marketing** yang terdiri dari **15.000 data pelanggan** dengan **30 fitur**. Target yang diprediksi adalah kolom **`churn`**, yang memiliki dua kelas:

* **0** : Pelanggan tidak melakukan churn.
* **1** : Pelanggan melakukan churn.

## Metodologi

Proses pengembangan model mengikuti tahapan standar dalam *Machine Learning* sebagai berikut.

### 1. Exploratory Data Analysis (EDA)

Tahap pertama dilakukan eksplorasi data untuk memahami karakteristik dataset. Dari hasil analisis ditemukan beberapa hal berikut:

* Terdapat **missing value** pada kolom `coupon_code`, `age`, `total_spent`, `gender`, dan `satisfaction_score`.
* Distribusi target **tidak seimbang (imbalanced)**, dengan pelanggan yang tidak churn sebesar **84,7%**, sedangkan pelanggan churn hanya **15,3%**.

Temuan tersebut menjadi dasar dalam menentukan proses *preprocessing* yang akan dilakukan.

### 2. Direct Modeling (Baseline Model)

Sebelum melakukan *preprocessing* secara lengkap, dilakukan pembuatan model awal sebagai **baseline** menggunakan tiga algoritma:

* Logistic Regression
* Random Forest
* Voting Classifier

Pada tahap ini hanya dilakukan penanganan sederhana terhadap data, seperti imputasi *missing value* dan *encoding* data kategorikal. Hasil dari model awal digunakan sebagai pembanding terhadap model setelah proses *preprocessing*.

### 3. Data Preprocessing

Tahap *preprocessing* dilakukan agar kualitas data menjadi lebih baik sebelum digunakan dalam pelatihan model. Langkah-langkah yang dilakukan meliputi:

* Menghapus kolom `coupon_code` karena memiliki jumlah *missing value* yang sangat tinggi.
* Mengisi *missing value* pada data numerik menggunakan **median**, sedangkan data kategorikal menggunakan **modus**.
* Menangani *outlier* pada fitur numerik menggunakan metode **Winsorizing**.
* Menghapus fitur yang tidak relevan, seperti `customer_id`, `signup_date`, dan `last_purchase_date`.
* Mengubah data kategorikal menjadi numerik menggunakan **Label Encoding**.
* Membagi dataset menjadi **80% data latih** dan **20% data uji**.
* Melakukan normalisasi data menggunakan **StandardScaler**.
* Mengatasi ketidakseimbangan kelas pada data latih menggunakan **SMOTE (Synthetic Minority Over-sampling Technique)**.

### 4. Modeling Setelah Preprocessing

Setelah seluruh proses *preprocessing* selesai, ketiga model sebelumnya kembali dilatih menggunakan data yang telah diproses dan diseimbangkan dengan SMOTE. Tahap ini bertujuan untuk melihat peningkatan performa model dibandingkan dengan model baseline.

### 5. Seleksi Fitur dan Hyperparameter Tuning

Untuk meningkatkan performa model, dilakukan beberapa langkah optimasi sebagai berikut:

* Menentukan tingkat kepentingan fitur menggunakan **Random Forest Feature Importance**.
* Memilih **15 fitur terbaik** sebagai masukan model.
* Melakukan kembali proses *train-test split*, *scaling*, dan SMOTE menggunakan fitur terpilih.
* Melakukan **hyperparameter tuning** menggunakan **RandomizedSearchCV** dengan fokus optimasi pada metrik **F1-Score**.

## Model Terbaik

Berdasarkan hasil evaluasi, model dengan performa terbaik adalah **Random Forest** yang telah melalui proses **feature selection** dan **hyperparameter tuning**.

| Metrik    |      Nilai |
| --------- | ---------: |
| Accuracy  | **0.8547** |
| Precision | **0.5141** |
| Recall    | **0.9500** |
| F1-Score  | **0.6672** |

Model ini memiliki nilai **Recall** yang sangat tinggi, sehingga mampu mengidentifikasi hampir seluruh pelanggan yang berpotensi melakukan churn. Hal ini penting dalam kasus bisnis karena perusahaan dapat melakukan tindakan pencegahan lebih awal terhadap pelanggan yang berisiko tinggi.

## Output Proyek

Seluruh komponen yang dibutuhkan untuk proses *deployment* telah disimpan sehingga model dapat digunakan kembali tanpa perlu melakukan pelatihan ulang.

* `model_churn.pkl` → Model Random Forest terbaik.
* `scaler.pkl` → Objek StandardScaler untuk normalisasi data.
* `label_encoders.pkl` → Objek LabelEncoder untuk setiap fitur kategorikal.
* `top_features.json` → Daftar 15 fitur terbaik yang digunakan oleh model.

## Implementasi

Model prediksi churn ini dapat diintegrasikan ke dalam sistem perusahaan untuk membantu mengidentifikasi pelanggan yang memiliki potensi tinggi melakukan churn. Informasi tersebut dapat dimanfaatkan sebagai dasar dalam menyusun strategi *customer retention*, seperti pemberian promo, penawaran khusus, atau pendekatan yang lebih personal kepada pelanggan sehingga peluang mereka untuk tetap menggunakan layanan menjadi lebih besar.



Link GitHub   : https://github.com/ghaitsafiya/churn-prediction-app

Link Streamlit: https://churn-prediction-app-4ypdyh6o23ny45i2ebmars.streamlit.app/
