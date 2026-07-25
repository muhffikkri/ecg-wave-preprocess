# =====================================================================
# FILE: src/app/model_registry.py
# PURPOSE: AI Model Registry
# =====================================================================

import os
from pathlib import Path
from app import config as cfg

# Root directory of models

MODELS = {
    "softmax_filtered_100to250_cnn": {
        "model_name": "Softmax Filtered 100Hz to 250Hz CNN",
        "task_type": "multiclass",
        "keras_model_path": cfg.MODEL_DIR / "Pure CNN Multi Class" / "softmax_filtered_100to250_cnn" / "best_model_patched.keras",
        "tflite_model_path": cfg.MODEL_DIR / "Pure CNN Multi Class" / "softmax_filtered_100to250_cnn" / "best_model.tflite",
        "class_list": ["AF", "Bradikardia", "Normal", "Takikardia"],
        "thresholds": [0.5, 0.5, 0.5, 0.5],
        "source_fs": 100.0,
        "target_fs": 250.0,
    },
    "softmax_filtered_100to250_cnnattention": {
        "model_name": "Softmax Filtered 100Hz to 250Hz CNN-Attention",
        "task_type": "multiclass",
        "keras_model_path": cfg.MODEL_DIR / "Pure CNN Multi Class" / "softmax_filtered_100to250_cnnattention" / "best_model_patched.keras",
        "tflite_model_path": cfg.MODEL_DIR / "Pure CNN Multi Class" / "softmax_filtered_100to250_cnnattention" / "best_model.tflite",
        "class_list": ["AF", "Bradikardia", "Normal", "Takikardia"],
        "thresholds": [0.5, 0.5, 0.5, 0.5],
        "source_fs": 100.0,
        "target_fs": 250.0,
    },
    "softmax_filtered_100to250_lstm": {
        "model_name": "Softmax Filtered 100Hz to 250Hz LSTM",
        "task_type": "multiclass",
        "keras_model_path": cfg.MODEL_DIR / "Pure CNN Multi Class" / "softmax_filtered_100to250_lstm" / "best_model_patched.keras",
        "tflite_model_path": cfg.MODEL_DIR / "Pure CNN Multi Class" / "softmax_filtered_100to250_lstm" / "best_model.tflite",
        "class_list": ["AF", "Bradikardia", "Normal", "Takikardia"],
        "thresholds": [0.5, 0.5, 0.5, 0.5],
        "source_fs": 100.0,
        "target_fs": 250.0,
    },
    "softmax_filtered_500to250_cnn": {
        "model_name": "Softmax Filtered 500Hz to 250Hz CNN",
        "task_type": "multiclass",
        "keras_model_path": cfg.MODEL_DIR / "Pure CNN Multi Class" / "softmax_filtered_500to250_cnn" / "best_model_patched.keras",
        "tflite_model_path": cfg.MODEL_DIR / "Pure CNN Multi Class" / "softmax_filtered_500to250_cnn" / "best_model.tflite",
        "class_list": ["AF", "Bradikardia", "Normal", "Takikardia"],
        "thresholds": [0.5, 0.5, 0.5, 0.5],
        "source_fs": 500.0,
        "target_fs": 250.0,
    },
    "softmax_filtered_500to250_cnnattention": {
        "model_name": "Softmax Filtered 500Hz to 250Hz CNN-Attention",
        "task_type": "multiclass",
        "keras_model_path": cfg.MODEL_DIR / "Pure CNN Multi Class" / "softmax_filtered_500to250_cnnattention" / "best_model_patched.keras",
        "tflite_model_path": cfg.MODEL_DIR / "Pure CNN Multi Class" / "softmax_filtered_500to250_cnnattention" / "best_model.tflite",
        "class_list": ["AF", "Bradikardia", "Normal", "Takikardia"],
        "thresholds": [0.5, 0.5, 0.5, 0.5],
        "source_fs": 500.0,
        "target_fs": 250.0,
    },
    "softmax_filtered_500to250_lstm": {
        "model_name": "Softmax Filtered 500Hz to 250Hz LSTM",
        "task_type": "multiclass",
        "keras_model_path": cfg.MODEL_DIR / "Pure CNN Multi Class" / "softmax_filtered_500to250_lstm" / "best_model_patched.keras",
        "tflite_model_path": cfg.MODEL_DIR / "Pure CNN Multi Class" / "softmax_filtered_500to250_lstm" / "best_model.tflite",
        "class_list": ["AF", "Bradikardia", "Normal", "Takikardia"],
        "thresholds": [0.5, 0.5, 0.5, 0.5],
        "source_fs": 500.0,
        "target_fs": 250.0,
    },
    "softmax_raw_100to250_cnn": {
        "model_name": "Softmax Raw 100Hz to 250Hz CNN",
        "task_type": "multiclass",
        "keras_model_path": cfg.MODEL_DIR / "Pure CNN Multi Class" / "softmax_raw_100to250_cnn" / "best_model_patched.keras",
        "tflite_model_path": cfg.MODEL_DIR / "Pure CNN Multi Class" / "softmax_raw_100to250_cnn" / "best_model.tflite",
        "class_list": ["AF", "Bradikardia", "Normal", "Takikardia"],
        "thresholds": [0.5, 0.5, 0.5, 0.5],
        "source_fs": 100.0,
        "target_fs": 250.0,
    },
    "softmax_raw_500to250_cnn": {
        "model_name": "Softmax Raw 500Hz to 250Hz CNN",
        "task_type": "multiclass",
        "keras_model_path": cfg.MODEL_DIR / "Pure CNN Multi Class" / "softmax_raw_500to250_cnn" / "best_model_patched.keras",
        "tflite_model_path": cfg.MODEL_DIR / "Pure CNN Multi Class" / "softmax_raw_500to250_cnn" / "best_model.tflite",
        "class_list": ["AF", "Bradikardia", "Normal", "Takikardia"],
        "thresholds": [0.5, 0.5, 0.5, 0.5],
        "source_fs": 500.0,
        "target_fs": 250.0,
    },
    "sigmoid_filtered_500to250_cnn": {
        "model_name": "Sigmoid Filtered 500Hz to 250Hz CNN",
        "task_type": "multilabel",
        "keras_model_path": cfg.MODEL_DIR / "Pure CNN Multi Label" / "sigmoid_filtered_500to250_cnn" / "best_model_patched.keras",
        "tflite_model_path": cfg.MODEL_DIR / "Pure CNN Multi Label" / "sigmoid_filtered_500to250_cnn" / "best_model.tflite",
        "class_list": ["Normal", "AF", "Takikardia", "Bradikardia"],
        "thresholds": [0.36, 0.37, 0.57, 0.46],
        "source_fs": 500.0,
        "target_fs": 250.0,
    },
    "sigmoid_filtered_100to250_cnn": {
        "model_name": "Sigmoid Filtered 100Hz to 250Hz CNN",
        "task_type": "multilabel",
        "keras_model_path": cfg.MODEL_DIR / "Pure CNN Multi Label" / "sigmoid_filtered_100to250_cnn" / "best_model_patched.keras",
        "tflite_model_path": cfg.MODEL_DIR / "Pure CNN Multi Label" / "sigmoid_filtered_100to250_cnn" / "best_model.tflite",
        "class_list": ["Normal", "AF", "Takikardia", "Bradikardia"],
        "thresholds": [0.36, 0.37, 0.57, 0.46],
        "source_fs": 100.0,
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