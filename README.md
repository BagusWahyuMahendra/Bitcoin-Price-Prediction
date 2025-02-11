# Laporan Proyek Machine Learning - Bagus Wahyu

## Domain Proyek

Bitcoin sebagai mata uang digital terdesentralisasi telah mengubah sistem keuangan, menawarkan keamanan, transparansi, dan efisiensi. Namun, volatilitas nilai Bitcoin menjadi tantangan bagi investor dan pelaku pasar. Untuk mengatasi hal ini, riset tentang prediksi harga Bitcoin menggunakan kecerdasan buatan menjadi penting.

Prediksi harga Bitcoin berguna bagi investor untuk memperkirakan harga di masa depan dan mencegah potensi kerugian, serta melihat perkembangan cryptocurrency dari tahun ke tahun.

Fluktuasi harga Bitcoin yang tinggi dapat memberikan peluang investasi, tetapi juga berisiko. Oleh karena itu, diperlukan pendekatan berbasis machine learning untuk memahami pola harga dan membuat prediksi berdasarkan data historis. Model prediksi dibuat dengan algoritma deep learning LSTM dan GRU untuk membandingkan model yang mana melakukan prediksi lebih baik. 

Berdasarkan hasil studi literatur yang telah dilakukan, Model Long Short Term Memory (LSTM) digunakan dalam skripsi yang berjudul "Prediksi Harga Bitcoin Menggunakan Metode Long Short Term Memory" untuk memprediksi harga Bitcoin karena kemampuannya mengingat informasi dalam periode waktu yang lebih panjang. LSTM adalah jenis jaringan saraf tiruan (Recurrent Neural Network/RNN) yang dirancang untuk memproses data berurutan dan mengatasi masalah dependensi jangka panjang yang tidak dapat ditangani oleh model RNN tradisional. Pada penelitian ini, model dievaluasi menggunakan metrik RMSE sebesar 478.237, MAE 330.322, dan MAPE 99.01%, yang menunjukkan bahwa akurasi prediksi masih perlu ditingkatkan. Meskipun model dan website yang dikembangkan dapat melakukan prediksi, tingkat error yang cukup tinggi mengindikasikan perlunya optimasi lebih lanjut, seperti tuning parameter atau kombinasi dengan metode lain agar prediksi harga Bitcoin lebih akurat dan andal.
  
Maka dari itu, pembuatan proyek ini bertujuan untuk melakukan perbandingan terhadap model LSTM dan GRU dalam memprediksi harga bitcoin.

Referensi: [Prediksi Harga Bitcoin Menggunakan Metode Long Short Term Memory](https://repository.unja.ac.id/62231/6/FULL%20SKRIPSI.pdf) 

## Business Understanding

### Problem Statements
- Bagaimana cara memprediksi harga Bitcoin berdasarkan data historis menggunakan machine learning?
- Algoritma mana yang memberikan prediksi terbaik untuk harga Bitcoin antara LSTM dan GRU?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Mengembangkan model machine learning yang dapat memprediksi harga Bitcoin berdasarkan data historis.
- Mengevaluasi performa model LSTM dan GRU untuk menentukan model yang lebih akurat.

### Solution statements
- Menggunakan data historis harga Bitcoin dari Yahoo Finance.
- Menerapkan dua metode deep learning, yaitu LSTM dan GRU, untuk prediksi harga Bitcoin.
- Melakukan evaluasi model berdasarkan metrik RMSE, MAE, dan R-squared.

## Data Understanding
Dataset yang digunakan diperoleh dari Yahoo Finance dengan simbol BTC-USD. Data ini mencakup harga penutupan harian Bitcoin selama 10 tahun terakhir yang berjumlah 3654 baris.

Data di load menggunakan kode berikut:
```sh
end = datetime.now()
start = datetime(end.year-10, end.month, end.day)
stock = "BTC-USD"
df = yf.download(stock, start, end)
```
Sebelum menggunakan kode di atas, perlu untuk menginstall library yfinance dan mengimportnya dengan kode berikut:
```sh
pip install yfinance
import yfinance as yf
```

Setelah dataset diunduh menggunakan kode di atas, informasi yang kita dapat antara lain:
- Terdapat 3654 baris yang berisi informasi mengenai data riwayat harga bitcoin saat ini, sampai dengan 10 tahun terakhir.
- Terdapat 5 kolom, yaitu Close, High, Low, Open, dan Volume.
- 4 kolom, yaitu Close, High, Low, dan Open memiliki tipe data float64 dan terdapat 1 kolom, yaitu Volume yang memiliki tipe data int64.
- Tidak terdapat missing value pada dataset.

### Variabel-variabel dataset Bitcoin
- Date: Tanggal pencatatan harga Bitcoin.
- Open: Harga pembukaan Bitcoin.
- High: Harga tertinggi dalam satu hari.
- Low: Harga terendah dalam satu hari.
- Close: Harga penutupan Bitcoin (digunakan sebagai target prediksi).
- Volume: Volume perdagangan Bitcoin dalam satu hari.

### Exploratory Data Analysis
- Mengidentifikasi Missing Value dan Outlier

- Univariate Analysis for Numerical Features

- Multivariate Analysis for Numerical Features

- Heatmap untuk melihat korelasi antar fitur. (gambar)

- Plot tren harga Bitcoin sepanjang waktu. (gambar)

- Moving Average (MA) 100 dan 250 hari untuk melihat pola pergerakan harga jangka panjang. (gambar)

## Data Preparation
Tahapan persiapan data meliputi:
- Normalisasi data: Menggunakan MinMaxScaler untuk mengubah harga Bitcoin dalam rentang 0-1 agar model lebih stabil.
- Pembuatan dataset untuk model LSTM dan GRU: Menggunakan sliding window dengan window size 100 hari untuk membuat input data.
- Pembagian dataset: 80% data digunakan untuk pelatihan dan 20% untuk pengujian.

## Modeling
Dua model deep learning yang digunakan:
### LSTM Model
- Layer pertama: LSTM dengan 128 unit dan return_sequences=True.
- Layer kedua: LSTM dengan 64 unit.
- Fully connected layer dengan 25 unit.
- Output layer dengan 1 unit.
(gambar model summary)

### GRU Model
- Layer pertama: GRU dengan 128 unit dan return_sequences=True.
- Layer kedua: GRU dengan 64 unit.
- Fully connected layer dengan 25 unit.
- Output layer dengan 1 unit.
(gambar model summary)

(buat tabel perbedaan keduanya)
(kelebihan dan kekurangan)
**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Metrik evaluasi yang digunakan:
- Root Mean Squared Error (RMSE): Mengukur seberapa jauh prediksi dari nilai sebenarnya.
- Mean Absolute Error (MAE): Mengukur rata-rata selisih absolut antara nilai prediksi dan aktual.
- R-squared (R2 Score): Mengukur seberapa baik model menjelaskan variabilitas data.

Hasil Evaluasi:
- LSTM Model:
RMSE: X

MAE: X

R-squared: X

- GRU Model:
RMSE: X

MAE: X

R-squared: X

