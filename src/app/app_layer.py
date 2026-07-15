# =====================================================================
# FILE: src/app/app_layer.py
# PURPOSE: ECG LIVE WORKBENCH WITH DUAL INFERENCE (KERAS VS TFLITE)
# =====================================================================

import os
import sys
import numpy as np
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.append(os.getcwd())
from app import config as cfg
from data import data_layer as dl
from logic.logic_layer import execute_live_pipeline
from logic.ai_model_manager import load_models
from logic.inference_manager import resolve_target_class, run_dual_model_inference
from logic.dsp_simulation_workbench import run_dsp_distortion_analysis

app = FastAPI(title="ECG Live Engine: Keras vs TFLite Benchmarking")


@app.on_event("startup")
def startup_load_models():
    load_models()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

presentation_dir = str(cfg.PRESENTATION_DIR)

# =====================================================================
# SOLUSI UTAMA: Mount folder presentation agar asset .css dan .js terbaca
# =====================================================================
if os.path.exists(presentation_dir):
    app.mount("/presentation", StaticFiles(directory=presentation_dir), name="presentation")


# ---------------------------------------------------------------------
# ROUTING CONTROLLER
# ---------------------------------------------------------------------

@app.get("/api/records")
def list_records():
    return dl.get_available_records()

@app.get("/api/process")
def process_signal(
    dataset: str, record_id: str, target_fs: float = 250.0,
    wavelet: str = "db4", w_level: int = 4, median_kernel: int = 51,
    lowcut: float = 0.5, highcut: float = 45.0
):
    raw_signal, src_fs = dl.load_raw_signal(dataset, record_id)
    target_class = resolve_target_class(dataset, record_id)

    # Jalankan Auto-Calibration + Preprocessing
    clean_signal, metrics = execute_live_pipeline(
        raw_signal, src_fs, target_fs, 
        wavelet, w_level, median_kernel, lowcut, highcut
    )
    inference = run_dual_model_inference(clean_signal)

    return {
        "src_fs": src_fs,
        "target_fs": target_fs,
        "target_class": target_class,       
        "keras_prediction": inference["keras_prediction"], 
        "keras_confidence": inference["keras_confidence"],
        "tflite_prediction": inference["tflite_prediction"],
        "tflite_confidence": inference["tflite_confidence"],
        "metrics": metrics,
        "raw_signals": {f"lead_{i}": raw_signal[:, i].tolist() for i in range(3)},
        "clean_signals": {f"lead_{i}": clean_signal[:, i].tolist() for i in range(3)}
    }

# @app.get("/")
# def serve_workbench():
#     html_path = os.path.join(presentation_dir, "index.html")
#     return FileResponse(html_path)

@app.get("/")
def serve_workbench():
    return RedirectResponse(url="/presentation/index.html")

# (Endpoint /api/simulator/folders dan analyze di bawah tetap sama seperti sebelumnya)
from src.logic.dsp_simulation_workbench import run_dsp_distortion_analysis
@app.get("/api/simulator/folders")
def list_simulator_folders():
    try:
        calibrated_record_path = cfg.PROSIM_SIMULATOR_DIR
        if not os.path.exists(calibrated_record_path): return []
        return sorted([d for d in os.listdir(calibrated_record_path) if os.path.isdir(os.path.join(calibrated_record_path, d))])
    except Exception: return []

@app.get("/api/simulator/analyze")
def analyze_simulator_data(folder_name: str, target_fs: float = 250.0, wavelet: str = "db4", w_level: int = 4, median_kernel: int = 51, lowcut: float = 0.5, highcut: float = 45.0):
    full_path = os.path.join(cfg.PROSIM_SIMULATOR_DIR, folder_name)
    return run_dsp_distortion_analysis(full_path, fs=target_fs, p_wavelet=wavelet, p_w_level=w_level, p_median_kernel=median_kernel, p_lowcut=lowcut, p_highcut=highcut)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)