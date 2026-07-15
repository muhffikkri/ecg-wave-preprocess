# =====================================================================
# FILE: src/logic/ai_model_manager.py
# PURPOSE: AI MODEL LOADER (KERAS + TFLITE)
# =====================================================================

import os
import tensorflow as tf
from tensorflow.keras import layers

from app import config as cfg


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
# GLOBAL MODEL OBJECT
# =====================================================================
ai_model_keras = None
tflite_interpreter = None


# =====================================================================
# MODEL LOADER
# =====================================================================
def load_models():
    """
    Memuat seluruh model AI (Keras + TFLite).
    Aman dipanggil saat startup.
    """

    global ai_model_keras
    global tflite_interpreter

    # ----------------------------------------------------------
    # LOAD PATCHED MODEL
    # ----------------------------------------------------------
    if os.path.exists(cfg.PATCHED_MODEL_PATH):
        try:
            ai_model_keras = tf.keras.models.load_model(
                cfg.PATCHED_MODEL_PATH,
                compile=False,
                custom_objects={
                    "StochasticDepth": StochasticDepth,
                    "PatchedBatchNorm": PatchedBatchNorm,
                },
            )
            print("✅ Loaded patched Keras model.")

        except Exception as e:
            print(f"⚠ Failed loading patched model : {e}")

    # ----------------------------------------------------------
    # FALLBACK ORIGINAL MODEL
    # ----------------------------------------------------------
    if ai_model_keras is None and os.path.exists(cfg.ORIGINAL_MODEL_PATH):
        try:
            ai_model_keras = tf.keras.models.load_model(
                cfg.ORIGINAL_MODEL_PATH,
                compile=False,
                custom_objects={
                    "StochasticDepth": StochasticDepth,
                    "PatchedBatchNorm": PatchedBatchNorm,
                },
            )
            print("✅ Loaded original Keras model.")

        except Exception as e:
            print(f"⚠ Failed loading original model : {e}")

    # ----------------------------------------------------------
    # LOAD TFLITE
    # ----------------------------------------------------------
    if os.path.exists(cfg.TFLITE_MODEL_PATH):
        try:
            tflite_interpreter = tf.lite.Interpreter(
                model_path=cfg.TFLITE_MODEL_PATH
            )
            tflite_interpreter.allocate_tensors()
            print("✅ Loaded TFLite Interpreter.")

        except Exception as e:
            print(f"⚠ Failed loading TFLite model : {e}")


# =====================================================================
# GETTERS
# =====================================================================
def get_keras_model():
    return ai_model_keras

def get_tflite_interpreter():
    return tflite_interpreter


# =====================================================================
# STATUS
# =====================================================================
def models_loaded():
    return {
        "keras": ai_model_keras is not None,
        "tflite": tflite_interpreter is not None,
    }