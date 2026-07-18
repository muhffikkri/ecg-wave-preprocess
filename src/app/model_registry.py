# =====================================================================
# FILE: src/app/model_registry.py
# PURPOSE: AI Model Registry
# =====================================================================

import os
from pathlib import Path
from app import config as cfg

# Root directory of models

MODELS = {
    "multiclass_100_to_250": {
        "model_name": "Pure CNN Multi Class 100Hz to 250Hz",
        "task_type": "multiclass",
        "keras_model_path": cfg.MODEL_DIR / "Pure CNN Multi Class" / "multiclass_100to250" / "best_model_patched.keras",
        "tflite_model_path": cfg.MODEL_DIR / "Pure CNN Multi Class" / "multiclass_100to250" / "best_model.tflite",
        # "class_list": ["Normal", "AF", "Takikardia", "Bradikardia"],
        # "class_list": ["Takikardia", "Normal", "Bradikardia", "AF"],
        "class_list": ["AF", "Bradikardia", "Normal", "Takikardia"],
        "thresholds": [0.5, 0.5, 0.5, 0.5],
        "source_fs": 100.0,
        "target_fs": 250.0,
    },
    "multiclass_500_to_250": {
        "model_name": "Pure CNN Multi Class 500Hz to 250Hz",
        "task_type": "multiclass",
        "keras_model_path": cfg.MODEL_DIR / "Pure CNN Multi Class" / "multiclass_500to250" / "best_model_patched.keras",
        "tflite_model_path": cfg.MODEL_DIR / "Pure CNN Multi Class" / "multiclass_500to250" / "best_model.tflite",
        # "class_list": ["Normal", "AF", "Takikardia", "Bradikardia"],
        # "class_list": ["Takikardia", "Normal", "Bradikardia", "AF"],
        "class_list": ["AF", "Bradikardia", "Normal", "Takikardia"],
        "thresholds": [0.5, 0.5, 0.5, 0.5],
        "source_fs": 500.0,
        "target_fs": 250.0,
    },
    "multilabel_100_to_250": {
        "model_name": "Pure CNN Multi Label 100Hz to 250Hz",
        "task_type": "multilabel",
        "keras_model_path": cfg.MODEL_DIR / "Pure CNN Multi Label" / "multilabel_100to250" / "best_model_patched.keras",
        "tflite_model_path": cfg.MODEL_DIR / "Pure CNN Multi Label" / "multilabel_100to250" / "best_model.tflite",
        "class_list": ["Normal", "AF", "Takikardia", "Bradikardia"],
        "thresholds": [0.34, 0.37, 0.57, 0.57],
        "source_fs": 100.0,
        "target_fs": 250.0,
    },
    "multilabel_500_to_250": {
        "model_name": "Pure CNN Multi Label 500Hz to 250Hz",
        "task_type": "multilabel",
        "keras_model_path": cfg.MODEL_DIR / "Pure CNN Multi Label" / "multilabel_500to250" / "best_model_patched.keras",
        "tflite_model_path": cfg.MODEL_DIR / "Pure CNN Multi Label" / "multilabel_500to250" / "best_model.tflite",
        "class_list": ["Normal", "AF", "Takikardia", "Bradikardia"],
        "thresholds": [0.42, 0.52, 0.2, 0.37],
        "source_fs": 500.0,
        "target_fs": 250.0,
    },
}

DEFAULT_MODEL_ID = cfg.DEFAULT_MODEL_ID

def get_model_info(model_id: str = None):
    """
    Mengambil metadata model berdasarkan ID model.
    """
    if model_id is None:
        model_id = DEFAULT_MODEL_ID
    if model_id not in MODELS:
        raise ValueError(f"Model ID '{model_id}' tidak ditemukan di registry.")
    return MODELS[model_id]