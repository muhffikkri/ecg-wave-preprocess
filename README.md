ecg-wave-preprocessing/
│
├── dataset/ # Tempat penyimpanan berkas medis mentah
│ ├── chapman/sample/_.mat
│ └── ptbxl/sample_100hz/_.hea, \*.dat
│
├── src/
│ ├── **init**.py
│ ├── config.py # Layer Data: Konfigurasi jalur mutlak & konstanta hardware
│ │
│ ├── app/ # APPLICATION LAYER
│ │ ├── **init**.py
│ │ └── app_layer.py # FastAPI Endpoints & Uvicorn Router Engine
│ │
│ ├── data/ # DATA LAYER
│ │ ├── **init**.py
│ │ └── data_layer.py # Dynamic I/O Scanner (.mat & .hea) + LRU Cache
│ │
│ ├── logic/ # LOGIC LAYER (DSP ENGINE)
│ │ ├── **init**.py
│ │ ├── logic_layer.py # Live Dynamic Execution Wrapper
│ │ └── preprocessing.py # Advanced DSP Engine v5.0 (Wavelet, Median, Butter)
│ │
│ └── presentation/ # PRESENTATION LAYER
│ └── index.html # UI Controls Dashboard + Chart.js Engine
│
└── requirements.txt # Minimal dependency list murni untuk ARM / Raspberry Pi

python -m src.app.app_layer
