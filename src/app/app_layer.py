# src/app/app_layer.py
import os
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import src.config as cfg
import src.data.data_layer as dl
import src.logic.logic_layer as ll

app = FastAPI(title="ECG Live Preprocessing Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/records")
def list_records():
    return dl.get_available_records()

@app.get("/api/process")
def process_signal(
    dataset: str, record_id: str, target_fs: float = 250.0,
    wavelet: str = "db4", w_level: int = 4, median_kernel: int = 51,
    lowcut: float = 0.5, highcut: float = 45.0
):
    # 1. Ambil data mentah & fs asal
    raw_signal, src_fs = dl.load_raw_signal(dataset, record_id)
    
    # 2. Ambil kelas target dari manifes database (Dinamis)
    try:
        if dataset == "chapman":
            import pandas as pd
            manifest = pd.read_csv(os.path.join(cfg.BASE_DIR, "chapman", "manifest_chapman_ablation.csv"))
            sample_row = manifest[manifest['filename_npy'] == f"{record_id}.npy"]
            target_class = str(sample_row['target_class'].values[0]) if not sample_row.empty else "Unknown"
        else:
            import wfdb
            sub_folder = "sample_100hz" if dataset == "ptbxl_100hz" else "sample_500hz"
            _, meta = wfdb.rdsamp(os.path.join(cfg.BASE_DIR, "ptbxl", sub_folder, record_id))
            target_class = str(meta.get('comments', ['Unknown'])[0])
    except Exception:
        target_class = "Aritmia Terdeteksi (Sampel)"

    # 3. Jalankan DSP Pipeline
    clean_signal, metrics = ll.execute_live_pipeline(
        raw_signal, src_fs, target_fs, 
        wavelet, w_level, median_kernel, lowcut, highcut
    )
    
    return {
        "src_fs": src_fs,
        "target_fs": target_fs,
        "target_class": target_class,
        "metrics": metrics,
        "raw_signals": {f"lead_{i}": raw_signal[:, i].tolist() for i in range(3)},
        "clean_signals": {f"lead_{i}": clean_signal[:, i].tolist() for i in range(3)}
    }

@app.get("/")
def serve_workbench():
    """
    Melayani file index.html langsung dari server FastAPI.
    Menghilangkan masalah CORS lokal akibat membuka file secara offline (file:/// style).
    """
    # Mencari lokasi index.html secara presisi berbasis letak berkas operasional
    root_project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    html_path = os.path.join(root_project, "src", "presentation", "index.html")
    
    if not os.path.exists(html_path):
        # Fallback cadangan jika struktur folder berjalan langsung di root lokasi kerja
        html_path = os.path.join(os.getcwd(), "src", "presentation", "index.html")
        
    return FileResponse(html_path)


# Tambahkan ini di bagian paling bawah berkas src/app/app_layer.py Anda

if __name__ == "__main__":
    import uvicorn
    # Menyalakan mesin server lokal pada port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)