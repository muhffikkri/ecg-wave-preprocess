# Changelog

Semua perubahan penting pada proyek ini akan dicatat di berkas ini.

Format mengikuti gaya [Keep a Changelog](https://keepachangelog.com/id-ID/1.0.0/),
dan versi mengikuti [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-10-08

### Added
- **Per-lead gapless charts** di Tab 1: tiga kanvas terpisah (Lead 1/2/3) tanpa celah antar-grafik.
- **Toggle filter per-lead** (show raw / show cleaned) dengan flag dikirim ke backend.
- **Kalibrasi gain klinis** 5/10/20 mm/mV dan grid EKG standar (1mm/5mm) dijangkar ke nol data.
- **Vertical pan & zoom** pada chart Tab 1; chart dibangun ulang saat gain berubah.
- **Spectral analysis (Welch PSD)** di Tab 2: frekuensi dominan ECG + rasio daya pita 0.5–5 / 5–15 / 15–40 Hz.
- **Grafik PSD** sebagai panel ketiga pada laporan analisis Tab 2.
- **Filter Distortion Comparison interaktif** (Chart.js): klik label legenda untuk menampilkan/menyembunyikan sinyal; semua aktif secara default.
- **Dataset registry lookup** di Tab 2 (`/api/simulator/folders` membaca registry yang sama dengan `/api/records`; format sumber `dataset::record`).
- **Lead selector** (Lead 1–3) dan label hasil `LEAD n` pada kartu metrik Tab 2.
- **Opsi "Sudah calibrated (lewati rekalkulasi)"** — melewati perbandingan ADC↔mV untuk data dataset/sensor yang sudah dalam satuan mV.
- **Grid 5 metrik dalam satu baris** Tab 2 (BPM, RR, noise FFT, attenuation, spectral).

### Changed
- Endpoint `/api/simulator/folders` kini mengembalikan grup terstruktur `{dataset, records}` (bukan daftar string), mendukung optgroup pada dropdown.
- Mode analisis dataset di backend selalu memaksa jalur *already calibrated* (data dataset sudah mV).
- README diperbarui untuk struktur repositori dan perintah menjalankan terkini.

### Fixed
- Crash `loadSimulatorFolders` saat backend lama (format response array) dipakai — kini respons dibentuk dari registry dataset.
- Kesalahan 500 pada analisis dataset tanpa opsi kalibrasi; mode dataset memaksa `already_calibrated=True`.

### Notes
- Versi statis aset frontend: `?v=1.6.0`.