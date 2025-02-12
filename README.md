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
- **Mengidentifikasi Missing Value dan Outlier**
    <br>
    <image src='https://raw.githubusercontent.com/BagusWahyuMahendra/Bitcoin-Price-Prediction/main/images/boxplot-outlier.png' width= 500/>
    <br> Karena Bitcoin dikenal memiliki volatilitas harga yang tinggi. Lonjakan harga atau volume yang ekstrem sering kali mencerminkan sentimen pasar, pengaruh berita besar, atau aktivitas institusional. Menghapus outlier dapat menghilangkan informasi penting yang bisa membantu model dalam memprediksi tren harga di masa depan, maka dari itu outlier pada dataset tidak dihapus.

- **Univariate Analysis for Numerical Features**
    <br>
    <image src='https://raw.githubusercontent.com/BagusWahyuMahendra/Bitcoin-Price-Prediction/main/images/univariate-analysis.png' width= 500>
    <br> Berdasarkan histogram di atas, kita mendapatkan beberapa informasi, antara lain:
    - Semua fitur harga Bitcoin (Open, High, Low, Close) dan Volume menunjukkan distribusi yang condong ke kiri (positively skewed).
    - Sebagian besar data harga Bitcoin berada di bawah 20.000 USD, dengan frekuensi tinggi pada nilai yang lebih rendah. Ini menunjukkan bahwa harga Bitcoin lebih sering berada di kisaran rendah dibandingkan harga tertinggi yang jarang terjadi.
    - Histogram volume menunjukkan rentang nilai yang luas, dengan sebagian besar transaksi terjadi pada volume yang relatif kecil, tetapi ada beberapa lonjakan volume yang sangat tinggi.

- **Multivariate Analysis for Numerical Features**
    <br>
    <image src='https://raw.githubusercontent.com/BagusWahyuMahendra/Bitcoin-Price-Prediction/main/images/multivariate-analysis.png' width= 500/>
    <br> Berdasarkan hasil pair plot di atas, informasi yang kita dapat antara lain:
    - Hubungan Linier Kuat antara Open, High, Low, dan Close
    - Volume memiliki hubungan yang lebih variatif
    - Histogram diagonal menunjukkan bahwa distribusi data cenderung skewed ke kiri (positively skewed), dengan sebagian besar nilai berada pada kisaran rendah.

- **Heatmap untuk melihat korelasi antar fitur**
    <br>
    <image src='https://raw.githubusercontent.com/BagusWahyuMahendra/Bitcoin-Price-Prediction/main/images/multivariate-analysis.png' width= 500/>
    <br> Heatmap di atas menunjukkan korelasi antara berbagai fitur dalam dataset harga Bitcoin. Terlihat bahwa Close, High, Low, dan Open BTC-USD memiliki korelasi sempurna (1.00), menandakan bahwa harga-harga ini bergerak bersamaan. Volume BTC-USD memiliki korelasi sekitar 0.65-0.67 dengan harga, menunjukkan hubungan yang cukup kuat tetapi tidak sebesar fitur harga lainnya.

- **Plot tren harga Bitcoin sepanjang waktu**
    <br>
    <image src='https://raw.githubusercontent.com/BagusWahyuMahendra/Bitcoin-Price-Prediction/main/images/btc-trend.png' width= 500/>
    <br> Grafik di atas menunjukkan tren harga Bitcoin dari tahun 2015 hingga sekarang, dengan sumbu horizontal merepresentasikan waktu dan sumbu vertikal menunjukkan harga dalam satuan USD. Dari grafik, terlihat bahwa harga Bitcoin mengalami volatilitas tinggi dengan beberapa lonjakan signifikan, terutama pada tahun 2017, 2021, dan 2024. Peningkatan harga yang tajam diikuti oleh koreksi yang cukup besar mencerminkan sifat spekulatif pasar kripto. Tren kenaikan jangka panjang menunjukkan bahwa meskipun mengalami fluktuasi besar, Bitcoin cenderung mengalami apresiasi nilai seiring waktu.

- **Moving Average (MA) 100 dan 250 hari untuk melihat pola pergerakan harga jangka panjang**
    <br>
    <image src='https://raw.githubusercontent.com/BagusWahyuMahendra/Bitcoin-Price-Prediction/main/images/moving-average.png' width= 500/>
    <br> Grafik di atas menampilkan pergerakan harga Bitcoin dengan tambahan indikator Moving Average (MA) 100-hari dan 250-hari, yang ditampilkan sebagai garis putus-putus berwarna oranye dan hijau. Indikator MA membantu mengidentifikasi tren pasar dengan meratakan fluktuasi harga jangka pendek, di mana MA 100-hari lebih responsif terhadap perubahan dibandingkan MA 250-hari yang lebih stabil. Ketika harga Bitcoin berada di atas MA, ini sering mengindikasikan tren bullish, sementara jika berada di bawah MA, ini dapat menjadi sinyal bearish. Analisis MA penting dalam memprediksi harga Bitcoin karena membantu mengurangi noise pasar, memberikan gambaran tren yang lebih jelas, serta digunakan untuk mendeteksi potensi titik masuk dan keluar yang lebih strategis dalam perdagangan kripto.

## Data Preparation
Tahapan persiapan data meliputi:
- **Normalisasi data**
Normalisasi merupakan proses skala ulang data agar berada dalam rentang tertentu. Pada proyek ini, digunakan MinMaxScaler dari library `sklearn.preprocessing`, dengan rentang 0 hingga 1. Langkah ini dilakukan untuk meningkatkan stabilitas model dalam menangani berbagai skala nilai harga Bitcoin. Normalisasi juga membantu jaringan LSTM dalam melakukan komputasi secara lebih efisien dengan menghindari nilai ekstrem yang dapat menyebabkan eksplositas gradien atau vanishing gradient problem.
- **Pembuatan Dataset untuk LSTM (Windowing)**
Setelah data dinormalisasi, langkah selanjutnya adalah membentuk dataset yang sesuai untuk model LSTM. LSTM memanfaatkan sekuens data sebagai input, sehingga dilakukan proses windowing dengan panjang 100 hari. Pada proses ini, setiap 100 hari sebelumnya digunakan sebagai fitur `x_data`. Nilai harga penutupan hari ke-101 digunakan sebagai target `y_data`. Proses ini penting karena LSTM bekerja dengan memori jangka panjang, di mana nilai masa lalu berkontribusi dalam memprediksi nilai masa depan. Dengan window size 100, model dapat memahami pola pergerakan harga berdasarkan data historis dalam 100 hari terakhir.
- **Pembagian dataset**
Membagi dataset menjadi data latih (training set) dan data uji (testing set). Pembagian dilakukan dengan proporsi 80% untuk training dan 20% untuk testing. Pembagian ini penting untuk menghindari overfitting, di mana model terlalu menyesuaikan diri dengan data latih tetapi kurang mampu melakukan generalisasi terhadap data baru

## Modeling
Dua model deep learning yang digunakan:
### LSTM Model
LSTM (Long Short-Term Memory) adalah jenis jaringan saraf berulang (RNN) yang dirancang untuk menangani permasalahan long-term dependencies dalam data deret waktu. LSTM memiliki sel memori yang dapat mempertahankan informasi untuk jangka waktu yang lebih lama dibandingkan RNN standar, sehingga cocok untuk prediksi harga Bitcoin yang bersifat time series.
- Layer pertama: LSTM dengan 128 unit dan `return_sequences=True`
- Layer kedua: LSTM dengan 64 unit.
- Fully connected layer dengan 25 unit.
- Output layer dengan 1 unit. <br>
Model summary:

| Layer (type) | Output Shape | Param # |
|-------------|-------------|---------|
| **lstm (LSTM)** | (None, 100, 128) | 66,560 |
| **lstm_1 (LSTM)** | (None, 64) | 49,408 |
| **dense (Dense)** | (None, 25) | 1,625 |
| **dense_1 (Dense)** | (None, 1) | 26 |

 Total params: 117,619 (459.45 KB)
 Trainable params: 117,619 (459.45 KB)
 Non-trainable params: 0 (0.00 B)

### GRU Model
GRU (Gated Recurrent Unit) adalah varian dari LSTM yang lebih sederhana dan efisien karena memiliki lebih sedikit parameter dibandingkan LSTM. GRU menggabungkan forget gate dan input gate menjadi satu update gate, sehingga dapat mempercepat proses training sambil tetap mempertahankan performa yang baik dalam menangani data sekuensial seperti harga Bitcoin.
- Layer pertama: GRU dengan 128 unit dan `return_sequences=True`.
- Layer kedua: GRU dengan 64 unit.
- Fully connected layer dengan 25 unit.
- Output layer dengan 1 unit. <br>
Model summary:

| Layer (type) | Output Shape | Param # |
|-------------|-------------|---------|
| **gru (GRU)** | (None, 100, 128) | 50,304 |
| **gru_1 (GRU)** | (None, 64) | 37,248 |
| **dense_2 (Dense)** | (None, 25) | 1,625 |
| **dense_3 (Dense)** | (None, 1) | 26 |

 Total params: 89,203 (348.45 KB)
 Trainable params: 89,203 (348.45 KB)
 Non-trainable params: 0 (0.00 B)

### Kelebihan dan Kekurangan LSTM dan GRU
| Faktor                | LSTM (Long Short-Term Memory)                              | GRU (Gated Recurrent Unit)                              |
|-----------------------|----------------------------------------------------------|----------------------------------------------------------|
| **Struktur Arsitektur** | Memiliki tiga gerbang utama: Forget, Input, dan Output.  | Memiliki dua gerbang utama: Update dan Reset.           |
| **Jumlah Parameter**   | Lebih banyak parameter karena struktur yang lebih kompleks. | Lebih sedikit parameter, lebih sederhana dibandingkan LSTM. |
| **Kemampuan Menangani Long-Term Dependencies** | Sangat baik dalam menangani long-term dependencies. | Baik, tetapi terkadang tidak seefektif LSTM untuk data dengan pola sangat panjang. |
| **Kecepatan Training** | Lebih lambat karena kompleksitas arsitektur yang lebih tinggi. | Lebih cepat dibandingkan LSTM karena memiliki struktur lebih sederhana. |
| **Efisiensi Komputasi** | Memerlukan lebih banyak daya komputasi. | Lebih efisien secara komputasi dibandingkan LSTM. |
| **Akurasi Prediksi** | Biasanya lebih akurat untuk data time series yang kompleks. | Akurat, tetapi bisa sedikit lebih rendah dari LSTM dalam beberapa kasus. |
| **Kapasitas Generalisasi** | Dapat menangkap pola yang lebih kompleks dengan lebih baik. | Dapat bekerja dengan baik dalam berbagai skenario dengan parameter yang lebih sedikit. |
| **Overfitting** | Cenderung lebih rentan terhadap overfitting jika tidak dikontrol dengan baik. | Lebih sedikit kemungkinan overfitting karena struktur yang lebih ringkas. |

## Evaluation
Dalam tahap evaluasi, model yang telah dilatih diuji menggunakan metrik evaluasi yang sesuai dengan konteks data, problem statement, dan solusi yang diinginkan. Pada proyek ini, tiga metrik evaluasi utama yang digunakan adalah Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), dan R-squared (R²).

### **Root Mean Squared Error (RMSE)**
RMSE mengukur seberapa jauh prediksi model dari nilai sebenarnya dalam satuan yang sama dengan data asli. RMSE dihitung menggunakan formula berikut:

$$
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

- Semakin kecil nilai RMSE, semakin baik model dalam melakukan prediksi.
- RMSE lebih sensitif terhadap outlier, karena menggunakan kuadrat dari selisih antara nilai prediksi dan nilai aktual.

---

### **Mean Absolute Error (MAE)**
MAE mengukur rata-rata selisih absolut antara nilai prediksi dan aktual. Formula MAE adalah:

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

- MAE tidak mengkuadratkan error sehingga tidak terlalu sensitif terhadap outlier.
- Semakin kecil nilai MAE, semakin baik model dalam memprediksi nilai harga Bitcoin.

---

### **R-squared (R² Score)**
R² atau Koefisien Determinasi mengukur seberapa baik model dapat menjelaskan variasi data aktual. Formula R² adalah:

$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y_i})^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

- Nilai R² mendekati 1 menunjukkan bahwa model dapat menjelaskan variasi data dengan baik.
- Jika R² bernilai 0 atau negatif, berarti model tidak lebih baik dibandingkan model sederhana (seperti rata-rata nilai).

---

### **Hasil Evaluasi**
| Model  | RMSE ↓    | MAE ↓    | R² ↑     |
|--------|----------|----------|---------|
| **LSTM** | 2328.39  | 1905.02  | 0.9903  |
| **GRU**  | 1752.94  | 1177.40  | 0.9945  |

## Kesimpulan
Model prediksi harga Bitcoin menggunakan Deep Learning telah dievaluasi dengan dua jenis arsitektur jaringan saraf, yaitu LSTM dan GRU. Dari hasil tersebut, GRU menunjukkan performa yang lebih baik dibandingkan LSTM dalam hal:

- Error lebih rendah → RMSE dan MAE pada GRU lebih kecil dibandingkan LSTM, yang berarti prediksi harga Bitcoin menggunakan GRU lebih akurat.
- R-squared lebih tinggi → Nilai R² pada GRU lebih tinggi dibandingkan LSTM (0.9945 vs 0.9903), yang menunjukkan bahwa model GRU lebih mampu menjelaskan variabilitas data harga Bitcoin.

Berdasarkan hasil evaluasi, model GRU lebih direkomendasikan untuk digunakan dalam prediksi harga Bitcoin karena:

✅ Memiliki akurasi lebih tinggi (error lebih kecil dan R² lebih besar).

✅ Lebih ringan secara komputasi dibandingkan LSTM.

✅ Cocok untuk dataset dengan urutan panjang seperti time series.
