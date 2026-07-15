# =====================================================================
# FILE : src/app/main.py
# PURPOSE : FastAPI Backend Entry Point
# =====================================================================

import os

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import config as cfg

from data.data_layer import (
    get_available_records,
    load_raw_signal,
)

from logic.logic_layer import (
    execute_live_pipeline,
)

from logic.inference_manager import (
    resolve_target_class,
    run_dual_model_inference,
)
from logic.ai_model_manager import load_models

from logic.dsp_simulation_workbench import (
    run_dsp_distortion_analysis,
)

# =====================================================================
# FASTAPI
# =====================================================================

app = FastAPI(
    title="Wearable ECG Edge Computing Workbench",
    version="2.0.0",
)


@app.on_event("startup")
def startup_load_models():
    load_models()

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
    target_fs: int = 250,
    wavelet: str = "db4",
    w_level: int = 4,
    median_kernel: int = 51,
    lowcut: float = 0.5,
    highcut: float = 45.0,
):

    raw_signal, src_fs = load_raw_signal(dataset, record_id)

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

    target_class = resolve_target_class(
        dataset,
        record_id,
    )

    inference = run_dual_model_inference(
        clean_signal,
    )

    raw_dict = {}
    clean_dict = {}

    n_leads = min(3, raw_signal.shape[1])

    for i in range(n_leads):

        raw_dict[f"lead_{i}"] = (
            raw_signal[:, i]
            .astype(float)
            .tolist()
        )

        clean_dict[f"lead_{i}"] = (
            clean_signal[:, i]
            .astype(float)
            .tolist()
        )

    return {

        "target_class": target_class,

        "raw_signals": raw_dict,

        "clean_signals": clean_dict,

        "keras_prediction":
            inference["keras_prediction"],

        "keras_confidence":
            inference["keras_confidence"],

        "tflite_prediction":
            inference["tflite_prediction"],

        "tflite_confidence":
            inference["tflite_confidence"],

        "metrics": metrics,

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
        if os.path.isdir(
            os.path.join(simulator_dir, f)
        )
    ]

    folders.sort()

    return folders


# =====================================================================
# SIMULATOR ANALYSIS
# =====================================================================

@app.get("/api/simulator/analyze")
def api_simulator_analysis(

    folder_name: str,

    target_fs: int = 250,

    wavelet: str = "db4",

    w_level: int = 4,

    median_kernel: int = 51,

    lowcut: float = 0.5,

    highcut: float = 45.0,

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
    }