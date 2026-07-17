# =====================================================================
# FILE: src/logic/ai_model_manager.py
# PURPOSE: AI MODEL LOADER & CACHE (KERAS + TFLITE) WITH REGISTRY SUPPORT
# =====================================================================

import os
import threading
import logging
import tensorflow as tf
from tensorflow.keras import layers

from app import config as cfg
from app import model_registry as reg

logger = logging.getLogger("ecg_workbench.ai_model_manager")

# =====================================================================
# CUSTOM LAYERS
# =====================================================================
class StochasticDepth(layers.Layer):
    def __init__(self, survival_probability=1.0, **kwargs):
        super().__init__(**kwargs)
        self.survival_probability = survival_probability

    def call(self, x, residual, training=None):
        if training:
            binary_tensor = tf.cast(
                tf.random.uniform([]) < self.survival_probability,
                tf.float32,
            )
            x = (binary_tensor * x) / self.survival_probability

        return x + residual


class PatchedBatchNorm(layers.BatchNormalization):
    """
    Compatibility wrapper untuk model lama yang masih memakai
    parameter renorm.
    """

    def __init__(self, **kwargs):
        kwargs.pop("renorm", None)
        kwargs.pop("renorm_clipping", None)
        kwargs.pop("renorm_momentum", None)
        super().__init__(**kwargs)


# =====================================================================
# THREAD-SAFE MODEL CACHE
# =====================================================================
_model_lock = threading.Lock()
_keras_models = {}
_tflite_interpreters = {}
_active_model_id = cfg.DEFAULT_MODEL_ID


# =====================================================================
# MODEL LOADER
# =====================================================================
def load_models(model_id: str = None):
    """
    Memuat model AI (Keras + TFLite) untuk model_id tertentu.
    Aman dipanggil saat startup atau reload.
    """
    global _active_model_id

    if model_id is None:
        model_id = _active_model_id

    model_info = reg.get_model_info(model_id)
    keras_path = model_info["keras_model_path"]
    tflite_path = model_info["tflite_model_path"]

    with _model_lock:
        # Load Keras Model
        if model_id not in _keras_models:
            if os.path.exists(keras_path):
                try:
                    model = tf.keras.models.load_model(
                        keras_path,
                        compile=False,
                        custom_objects={
                            "StochasticDepth": StochasticDepth,
                            "PatchedBatchNorm": PatchedBatchNorm,
                        },
                    )
                    _keras_models[model_id] = model
                    logger.info(f"✅ Loaded Keras model: {model_info['model_name']} ({keras_path})")
                except Exception as e:
                    logger.error(f"⚠ Failed loading Keras model for {model_id} from {keras_path}: {e}")
            else:
                logger.warning(f"⚠ Keras model file not found for {model_id} at {keras_path}")

        # Load TFLite Model
        if model_id not in _tflite_interpreters:
            if os.path.exists(tflite_path):
                try:
                    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
                    interpreter.allocate_tensors()
                    _tflite_interpreters[model_id] = interpreter
                    logger.info(f"✅ Loaded TFLite Interpreter: {model_info['model_name']} ({tflite_path})")
                except Exception as e:
                    logger.error(f"⚠ Failed loading TFLite model for {model_id} from {tflite_path}: {e}")
            else:
                logger.warning(f"⚠ TFLite model file not found for {model_id} at {tflite_path}")

    # Set as active model ID
    _active_model_id = model_id


# =====================================================================
# GETTERS & ACTIVE MODIFIERS
# =====================================================================
def set_active_model(model_id: str):
    """
    Mengubah model aktif dan memuatnya ke memori jika belum ada.
    """
    global _active_model_id
    if model_id not in reg.MODELS:
        raise ValueError(f"Model ID '{model_id}' tidak valid.")
    
    if model_id not in _keras_models or model_id not in _tflite_interpreters:
        load_models(model_id)
    else:
        _active_model_id = model_id
        logger.info(f"🔄 Active model switched to: {model_id}")


def get_active_model_id() -> str:
    return _active_model_id


def get_keras_model(model_id: str = None):
    if model_id is None:
        model_id = _active_model_id
    
    if model_id not in _keras_models:
        load_models(model_id)
        
    return _keras_models.get(model_id)


def get_tflite_interpreter(model_id: str = None):
    if model_id is None:
        model_id = _active_model_id
        
    if model_id not in _tflite_interpreters:
        load_models(model_id)
        
    return _tflite_interpreters.get(model_id)


# =====================================================================
# RELOAD
# =====================================================================
def reload_model(model_id: str = None):
    """
    Menghapus cache model tertentu dan memuat ulang dari piringan keras.
    """
    global _active_model_id
    if model_id is None:
        model_id = _active_model_id

    with _model_lock:
        _keras_models.pop(model_id, None)
        _tflite_interpreters.pop(model_id, None)

    logger.info(f"♻ Reloading model {model_id}...")
    load_models(model_id)


# =====================================================================
# STATUS
# =====================================================================
def models_loaded(model_id: str = None):
    if model_id is None:
        model_id = _active_model_id
    return {
        "keras": model_id in _keras_models,
        "tflite": model_id in _tflite_interpreters,
    }