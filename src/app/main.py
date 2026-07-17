# =====================================================================
# FILE : src/app/main.py
# PURPOSE : FastAPI Backend Entry Point
# =====================================================================

import os
import sys

# Ensure the 'src' directory is in the python path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import time
import logging
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import config as cfg
from app import model_registry as reg
from data.data_layer import get_available_records, load_raw_signal
from logic.logic_layer import execute_live_pipeline
from logic.inference_manager import resolve_target_class, run_dual_model_inference
from logic.ai_model_manager import load_models, set_active_model, get_active_model_id
from logic.dsp_simulation_workbench import run_dsp_distortion_analysis

logger = logging.getLogger("ecg_workbench.main")


app = FastAPI(
    title="Wearable ECG Edge Computing Workbench",
    version="2.0.0",
)


@app.on_event("startup")
def startup_load_models():
    # Memuat default model pada startup
    load_models(cfg.DEFAULT_MODEL_ID)


# =====================================================================
# CORS
# =====================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# STATIC FRONTEND
# =====================================================================
FRONTEND_DIR = cfg.PRESENTATION_DIR

if os.path.exists(FRONTEND_DIR):
    app.mount(
        "/static",
        StaticFiles(directory=FRONTEND_DIR),
        name="static",
    )


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


# =====================================================================
# AVAILABLE RECORDS
# =====================================================================
@app.get("/api/records")
def api_records():
    return get_available_records()


# =====================================================================
# LIVE PIPELINE
# =====================================================================
@app.get("/api/process")
def api_process(
    dataset: str,
    record_id: str,
    target_fs: float = cfg.TARGET_FS,
    wavelet: str = cfg.WAVELET_DEFAULT,
    w_level: int = cfg.WAVELET_LEVEL_DEFAULT,
    median_kernel: int = cfg.MEDIAN_KERNEL_DEFAULT,
    lowcut: float = cfg.BUTTERWORTH_LOWCUT,
    highcut: float = cfg.BUTTERWORTH_HIGHCUT_DEFAULT,
    model_id: str = cfg.DEFAULT_MODEL_ID,
):
    t_start = time.perf_counter()

    # Load raw signal
    raw_signal, src_fs = load_raw_signal(dataset, record_id)

    # Set active model di manager
    set_active_model(model_id)
    model_info = reg.get_model_info(model_id)

    # LOG ===== REQUEST =====
    logger.info("===== REQUEST =====")
    logger.info(f"dataset: {dataset}")
    logger.info(f"record: {record_id}")
    logger.info(f"sampling rate: {src_fs} Hz")
    logger.info(f"pipeline yang dipilih: {'upsampling' if src_fs < target_fs else 'offline'}")
    logger.info(f"parameter preprocessing: wavelet={wavelet}, level={w_level}, median_kernel={median_kernel}, lowcut={lowcut}, highcut={highcut}, model_id={model_id}")

    # Run clean pipeline (steps 1-7)
    clean_signal, metrics = execute_live_pipeline(
        raw_signal=raw_signal,
        src_fs=src_fs,
        target_fs=target_fs,
        p_wavelet=wavelet,
        p_w_level=w_level,
        p_median_kernel=median_kernel,
        p_lowcut=lowcut,
        p_highcut=highcut,
    )

    # LOG ===== PREPROCESS =====
    logger.info("===== PREPROCESS =====")
    logger.info(f"shape awal: {raw_signal.shape}")
    logger.info(f"shape setelah resampling: {clean_signal.shape}")
    logger.info(f"shape akhir (sebelum crop): {clean_signal.shape}")
    logger.info(f"mean: {np.mean(clean_signal):.4f}")
    logger.info(f"std: {np.std(clean_signal):.4f}")
    logger.info(f"min: {np.min(clean_signal):.4f}")
    logger.info(f"max: {np.max(clean_signal):.4f}")
    logger.info(f"latency preprocessing: {metrics['latency_ms']:.2f} ms")

    # LOG ===== MODEL =====
    logger.info("===== MODEL =====")
    logger.info(f"model aktif: {model_id} ({model_info['model_name']})")
    logger.info(f"task: {model_info['task_type']}")
    logger.info(f"input tensor shape: (1, {cfg.MODEL_INPUT_LENGTH}, {clean_signal.shape[1]})")

    # Resolve target class (ground truth)
    target_class = resolve_target_class(dataset, record_id)

    # Run AI Model inference (steps 8-10)
    inference = run_dual_model_inference(
        clean_signal,
        model_id=model_id,
    )

    # Convert signals to list for JSON response
    raw_dict = {}
    clean_dict = {}
    n_leads = min(3, raw_signal.shape[1])
    for i in range(n_leads):
        raw_dict[f"lead_{i}"] = raw_signal[:, i].astype(float).tolist()
        clean_dict[f"lead_{i}"] = clean_signal[:, i].astype(float).tolist()

    t_end = time.perf_counter()
    processing_time_ms = (t_end - t_start) * 1000.0

    # LOG ===== RESOURCE =====
    logger.info("===== RESOURCE =====")
    logger.info(f"peak memory: {metrics['peak_memory_mb']:.4f} MB")
    logger.info(f"processing time: {processing_time_ms:.2f} ms")
    logger.info("=========================\n")

    return {
        "target_class": target_class,
        "raw_signals": raw_dict,
        "clean_signals": clean_dict,
        "keras_prediction": inference["keras_prediction"],
        "keras_confidence": inference["keras_confidence"],
        "tflite_prediction": inference["tflite_prediction"],
        "tflite_confidence": inference["tflite_confidence"],
        "metrics": {
            "latency_ms": metrics["latency_ms"],
            "peak_memory_mb": metrics["peak_memory_mb"],
            "total_processing_time_ms": processing_time_ms,
        },
        "holter": metrics["holter"],
    }


# =====================================================================
# SIMULATOR FOLDER
# =====================================================================
@app.get("/api/simulator/folders")
def api_simulator_folders():
    simulator_dir = cfg.PROSIM_SIMULATOR_DIR
    if not os.path.exists(simulator_dir):
        return []
    folders = [
        f
        for f in os.listdir(simulator_dir)
        if os.path.isdir(os.path.join(simulator_dir, f))
    ]
    folders.sort()
    return folders


# =====================================================================
# SIMULATOR ANALYSIS
# =====================================================================
@app.get("/api/simulator/analyze")
def api_simulator_analysis(
    folder_name: str,
    target_fs: float = cfg.TARGET_FS,
    wavelet: str = cfg.WAVELET_DEFAULT,
    w_level: int = cfg.WAVELET_LEVEL_DEFAULT,
    median_kernel: int = cfg.MEDIAN_KERNEL_DEFAULT,
    lowcut: float = cfg.BUTTERWORTH_LOWCUT,
    highcut: float = cfg.BUTTERWORTH_HIGHCUT_DEFAULT,
):
    folder = os.path.join(
        cfg.PROSIM_SIMULATOR_DIR,
        folder_name,
    )
    return run_dsp_distortion_analysis(
        folder_path=folder,
        fs=float(target_fs),
        p_wavelet=wavelet,
        p_w_level=w_level,
        p_median_kernel=median_kernel,
        p_lowcut=lowcut,
        p_highcut=highcut,
    )


# =====================================================================
# HEALTH CHECK
# =====================================================================
@app.get("/health")
def health():
    return {
        "status": "running",
        "backend": "FastAPI",
        "version": "2.0.0",
        "active_model_id": get_active_model_id(),
    }