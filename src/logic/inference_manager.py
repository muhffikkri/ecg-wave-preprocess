# =====================================================================
# FILE : logic/inference_manager.py
# PURPOSE : AI Inference & Ground Truth高度 DSP Resolver (v5.0 Sync)
# =====================================================================

import os
import numpy as np
import wfdb

from app import config as cfg
from logic.manifest_manager import lookup_target_class
from logic.ai_model_manager import get_keras_model, get_tflite_interpreter

# Mengimpor modul DSP v5.0 yang sudah diselaraskan dari logic_layer
from logic.logic_layer import ensure_length, apply_zscore_clip, sanitize_signal


# =====================================================================
# GROUND TRUTH RESOLVER
# =====================================================================
def resolve_target_class(dataset, record_id):
    if dataset == "chapman":
        manifest_path = cfg.MANIFEST_CHAPMAN
        candidates = [
            f"chap_{record_id}.npy",
            f"{record_id}.npy",
            record_id,
        ]
        found = lookup_target_class(manifest_path, candidates)
        return found if found else "Unknown"

    elif dataset in ("ptbxl_100hz", "ptbxl_500hz"):
        manifest_path = cfg.MANIFEST_PTBXL
        candidates = [
            record_id,
            f"{record_id}.npy",
            f"{record_id}.hea",
            record_id,
        ]
        found = lookup_target_class(manifest_path, candidates)
        if found:
            return found

        try:
            folder = (
                cfg.PTBXL_100HZ_DIR
                if dataset == "ptbxl_100hz"
                else cfg.PTBXL_500HZ_DIR
            )
            _, meta = wfdb.rdsamp(os.path.join(folder, record_id))
            return str(meta.get("comments", ["Unknown"])[0])
        except Exception:
            return "Unknown"

    elif dataset == "prosim_simulator":
        return f"ProSim Simulator ({record_id})"
    return "Unknown"


# =====================================================================
# MULTILABEL DECODER
# =====================================================================
def interpret_multilabel_probabilities(preds_prob):
    detected = []
    for idx, cls in enumerate(cfg.TARGET_CLASSES):
        print(f"{cls} ({preds_prob[idx]*100:.1f}%)")
        if preds_prob[idx] >= cfg.OPTIMIZED_THRESHOLDS[idx]:
            detected.append(f"{cls} ({preds_prob[idx]*100:.1f}%)")

    if len(detected) == 0:
        return ("Others / Unsure", float(np.max(preds_prob)))

    return (" + ".join(detected), float(np.max(preds_prob)))


# =====================================================================
# PREPROCESS INPUT MODEL (100% IDENTIK DENGAN PROSES TRAINING v5.0)
# =====================================================================
def prepare_model_input(signal):
    """
    Mengubah Clean Signal (mV) -> Tensor Normalisasi AI (1, 2500, 3)
    Proses diisolasi penuh agar tidak merusak visualisasi grafik di frontend.
    """
    # 1. Duplikasi sinyal dan lakukan sanitasi tipe data awal
    x = np.copy(signal)
    x = sanitize_signal(x)

    if x.ndim != 2:
        raise ValueError(f"Signal harus berbentuk [timesteps, channels], didapat ndim={x.ndim}")

    # 2. Kondisikan panjang gelombang via Center Crop / Zero Padding kaku ke 2500 sampel
    target_length = getattr(cfg, "MODEL_INPUT_LENGTH", 2500)
    x = ensure_length(x, target_len=target_length)

    # 3. Jalankan Z-score Normalization per-channel & clipping outlier ekstrem [-5.0, 5.0]
    x = apply_zscore_clip(x, epsilon=1e-8, clip_min=-5.0, clip_max=5.0)

    # 4. Ekspansi dimensi tensor menjadi bentuk batch (1, 2500, 3)
    x = np.expand_dims(x, axis=0)
    return x.astype(np.float32)


# =====================================================================
# DUAL MODEL INFERENCE
# =====================================================================
def run_dual_model_inference(clean_signal):
    result = {
        "keras_prediction": "Offline",
        "keras_confidence": None,
        "tflite_prediction": "Offline",
        "tflite_confidence": None,
    }

    # x sekarang dijamin berupa matriks ternormalisasi berdimensi kaku (1, 2500, 3)
    x = prepare_model_input(clean_signal)

    # ==============================================================
    # KERAS INFERENCE
    # ==============================================================
    ai_model_keras = get_keras_model()
    if ai_model_keras is not None:
        try:
            pred = ai_model_keras.predict(x, verbose=0)[0]
            cls, conf = interpret_multilabel_probabilities(pred)

            result["keras_prediction"] = cls
            result["keras_confidence"] = round(conf * 100, 2)
        except Exception as e:
            result["keras_prediction"] = f"Error: {e}"

    # ==============================================================
    # TFLITE INFERENCE
    # ==============================================================
    tflite_interpreter = get_tflite_interpreter()
    if tflite_interpreter is not None:
        try:
            input_details = tflite_interpreter.get_input_details()
            output_details = tflite_interpreter.get_output_details()

            tflite_interpreter.set_tensor(
                input_details[0]["index"],
                x.astype(input_details[0]["dtype"])
            )
            tflite_interpreter.invoke()

            pred = tflite_interpreter.get_tensor(output_details[0]["index"])[0]
            cls, conf = interpret_multilabel_probabilities(pred)

            result["tflite_prediction"] = cls
            result["tflite_confidence"] = round(conf * 100, 2)
        except Exception as e:
            result["tflite_prediction"] = f"Error: {e}"

    return result