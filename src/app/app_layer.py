# app_layer.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import src.data.data_layer as dl
import logic_layer as ll

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
    # 1. Fetch Data
    raw_signal, src_fs = dl.load_raw_signal(dataset, record_id)
    
    # 2. Process Data
    clean_signal, metrics = ll.execute_live_pipeline(
        raw_signal, src_fs, target_fs, 
        wavelet, w_level, median_kernel, lowcut, highcut
    )
    
    # 3. Format Response JSON secara ringkas untuk transmisi cepat
    return {
        "src_fs": src_fs,
        "target_fs": target_fs,
        "metrics": metrics,
        "raw_signals": {f"lead_{i}": raw_signal[:, i].tolist() for i in range(3)},
        "clean_signals": {f"lead_{i}": clean_signal[:, i].tolist() for i in range(3)}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)