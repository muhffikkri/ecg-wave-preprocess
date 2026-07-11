# 🎛️ Wearable ECG Live Clinical Workbench & Hardware DSP Analyzer

Aplikasi workbench klinis berbasis web berkecepatan tinggi yang dirancang khusus untuk memproses, memvisualisasikan, dan menguji sinyal Elektrokardiogram (EKG) 3-Lead secara real-time. Proyek ini mengintegrasikan rekayasa pemrosesan sinyal digital (DSP), instrumentasi perangkat keras medis (ADS1293 + Raspberry Pi), serta inferensi Kecerdasan Buatan (AI) berskema Multi-label untuk deteksi dini aritmia.

Aplikasi ini mengimplementasikan **Clean Layered Architecture** yang memisahkan data access layer, core DSP logic, dan lapisan server web/fastapi secara ketat untuk menjamin modularitas sistem.

---

## 🚀 Fitur Utama Sistem

### 1. Tab 1: Live Dataset & Hardware Streaming Workbench

- **Multi-Source Dataset Support:** Mampu memuat data dari dataset gold standard publik seperti PTB-XL Database (100Hz/500Hz), Chapman Dataset (500Hz), hingga data tangkapan fisik mentah perangkat keras (ProSim Simulator via ADS1293).
- **Interactive Preprocessing Pipeline:** Pengguna dapat mengubah parameter penyaringan sinyal secara langsung melalui slider/input interaktif di antarmuka web, meliputi:
  - _Adaptive Wavelet Denoising_ (Daubechies `db4` / Symlets `sym4`) dengan tingkatan dekomposisi dinamis (Level 1–6).
  - _Median Filter Baseline Wander Removal_ dengan lebar jendela kernel adaptif.
  - _Butterworth Bandpass Filter_ dengan pengaturan batas frekuensi atas (_highcut_) dan batas bawah (_lowcut_) interaktif.
  - _Poly-Resampling Engine_ otomatis menuju target spesifikasi model (250Hz).
- **Clinical ECG Grid View:** Visualisasi sinyal mentah (_raw_) berdampingan dengan sinyal bersih (_cleaned_) di atas kertas milimeter blok klinis standar asli (25 mm/s | 10 mm/mV) menggunakan kustom plugin Chart.js.
- **Real-Time Clinical Holter Dashboard:** Menghitung metrik elektrofisiologi jantung secara sekuensial:
  - Heart Rate (HR) dalam satuan BPM.
  - Jarak interval puncak-ke-puncak rata-rata (Mean R-R Interval dalam milidetik).
  - Heart Rate Variability (HRV via RMSSD).
  - Deviasi Elevasi/Depresi Segmen-ST (ST Segment Deviation dalam mV).
  - Corrected QT Interval (QTc).
- **Edge-Computing Profiler:** Memantau metrik efisiensi komputasi perangkat lunak saat dipasang di mikrokontroler tepi (Pi Edge Latency dalam milidetik dan Peak Memory Allocation dalam Megabytes).

### 2. Tab 2: ProSim Perangkat Keras DSP Study (Studi Distorsi)

- **Gold Standard Simulator Validation:** Menghubungkan berkas biner/CSV tangkapan sensor ADS1eran fisik asli (`raw_ecg.csv`) dengan sinyal referensi murni instrumen medis (`latest_prosim_calibrated_mv.csv`).
- **Inherent Hardware Noise Mapping:** Mengotomatisasi Transformasi Fourier Riil (FFT via `rfftfreq`) untuk melacak tumpukan energi derau laten perangkat keras (seperti Power Line Interference 50Hz/60Hz atau fluktuasi catu daya).
- **Interactive Phase Shift & Attenuation Evaluation:** Mensimulasikan perbandingan dampak filter digital interaktif. Menunjukkan secara visual perbedaan fatal filter kausal konvensional (`lfilter`) yang merusak pergeseran fase segmen ST, dibandingkan filter optimasi non-kausal (`filtfilt`) yang memberikan perlindungan _zero-phase distortion_.
- **Kuantifikasi Redaman Puncak R:** Menghitung persentase deviasi akurat pada titik puncak kelistrikan jantung sejati ($R\text{-peak}$) untuk menguji hipotesis agresivitas filter.

### 3. Skema Multi-Label AI Inference Interception

Aplikasi backend dilengkapi mesin inferensi model deep learning `.keras` berskema **Multi-Label Classification** (Sigmoid Activation Layer) yang mengenali karakteristik penyakit tunggal maupun komorbiditas ganda secara bersamaan menggunakan batas keputusan (_decision thresholds_) optimal hasil pencarian kisi (_grid search_):

- **Normal Threshold:** `0.34`
- **AF (Atrial Fibrillation) Threshold:** `0.37`
- **Takikardia Threshold:** `0.57`
- **Bradikardia Threshold:** `0.57`

- **Open-Set Reject Option:** Jika probabilitas keempat neuron sigmoid berada di bawah ambang batas di atas, sistem pasca-pemrosesan otomatis mengalihkan diagnosis ke kategori **"Others / Ragu-ragu (Unsure)"** demi menjaga sterilitas keputusan klinis di dunia nyata.

---

## 📁 Struktur Repositori Proyek

```text
ecg-wave-preproccess/
├── dataset/                  # Lokasi penyimpanan basis data EKG
│   ├── chapman/
│   ├── ptbxl/
│   └── Kalibrasi Prosim/     # Folder rekaman fisik mentah sensor ADS1293
├── output/
│   └── research_experiments/
│       └── best_model.keras  # File model deep learning hasil eksperimen
├── src/
│   ├── app/
│   │   ├── __init__.py
│   │   └── app_layer.py      # FastAPI Server, Routing API, & Logika Inferensi AI
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_layer.py     # Data Access Layer (Parsing .mat, .hea, & .csv ProSim)
│   ├── logic/
│   │   ├── __init__.py
│   │   ├── logic_layer.py    # Eksekusi Pipeline DSP & Hitung Holter Metrics
│   │   └── dsp_simulation_workbench.py  # FFT Noise Mapping & Analisis Distorsi Parameter
│   ├── presentation/
│   │   └── index.html        # Frontend Dashboard (Millimeter Chart & Control Panel)
│   └── config.py             # Global Path & Hardware Konstanta System Configuration
├── venv/                     # Python Virtual Environment
└── README.md                 # Dokumentasi Proyek
```

---

## 🛠️ Cara Menjalankan Proyek

### 1. Prasyarat Sistem

Pastikan perangkat komputer Anda telah terpasang Python versi 3.10 ke atas, serta disarankan menggunakan sistem operasi Windows/Linux/macOS dengan alokasi terminal terminal aktif.

### 2. Aktivasi Virtual Environment & Instalasi Dependensi

Buka terminal/command prompt pada direktori utama proyek (`D:\Project\ecg-wave-preproccess`), lalu jalankan perintah berikut:

```bash
# Buat Virtual Environment (jika belum ada)
python -m venv venv

# Aktivasi Virtual Environment (Windows)
.\venv\Scripts\activate

# Jalankan Instalasi Pustaka Inti (jika belum terpasang lengkap)
pip install -r requirements.txt
```

### 3. Konfigurasi Awal Path Jalur Berkas

Buka file `src/config.py` dan sesuaikan variabel variabel global `BASE_DIR` menuju lokasi folder absolut penyimpanan data Anda, contoh:

```python
BASE_DIR = r"D:\Project\ecg-wave-preproccess\dataset"
```

### 4. Menyalakan Server Workbench EKG

Eksekusi server ASGI lokal menggunakan modul Python dari _root project_ direktori dengan perintah:

```bash
python -m src.app.app_layer
```

Jika berhasil, terminal akan menampilkan log pesan uvicorn sukses berjalan:

```text
INFO:     Started server process [43852]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### 5. Mengakses Dashboard Web Clinis

Buka peramban browser Anda (Google Chrome / Mozilla Firefox / Safari), lalu akses alamat berikut:

```text
http://127.0.0.1:8000/
```

Sistem workbench klinis interaktif Anda kini siap digunakan sepenuhnya untuk pembuktian riset rekayasa biomedis!
