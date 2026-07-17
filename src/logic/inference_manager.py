# =====================================================================
# FILE : logic/inference_manager.py
# PURPOSE : AI Inference & Ground Truth Resolver (Dynamic Registry Integration)
# =====================================================================

import os
import time
import logging
import numpy as np
import wfdb

from app import config as cfg
from app import model_registry as reg
from logic.manifest_manager import lookup_target_class
from logic.ai_model_manager import get_keras_model, get_tflite_interpreter, get_active_model_id

# Mengimpor modul DSP dari preprocessing
from logic.preprocessing import ensure_length, apply_zscore_clip, sanitize_signal

logger = logging.getLogger("ecg_workbench.inference_manager")


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
        # Extract digits to match ptb_xxxxx.npy format in manifest
        digits = "".join([c for c in record_id if c.isdigit()])
        candidates = [
            record_id,
            f"{record_id}.npy",
            f"{record_id}.hea",
            f"ptb_{record_id}.npy",
            f"ptb_{record_id}",
        ]
        if digits:
            candidates.append(f"ptb_{digits}.npy")
            candidates.append(f"ptb_{digits}")
            try:
                candidates.append(f"ptb_{int(digits):05d}.npy")
                candidates.append(f"ptb_{int(digits):05d}")
            except ValueError:
                pass

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
            comments = meta.get("comments", [])
            if comments:
                return str(comments[0])
            return "Unknown"
        except Exception:
            return "Unknown"

    elif dataset == "prosim_simulator":
        return f"ProSim Simulator ({record_id})"
    return "Unknown"



# =====================================================================
# DECODERS (MULTILABEL & MULTICLASS)
# =====================================================================
def interpret_multiclass_probabilities(preds_prob, class_list, threshold=0.5):
    """
    Menafsirkan output probabilitas untuk skema Multi-Class.
    Memilih probabilitas tertinggi (argmax), tolak jika di bawah threshold.
    """
    max_idx = np.argmax(preds_prob)
    max_prob = float(preds_prob[max_idx])

    if max_prob < threshold:
        return ("Others / Unsure", max_prob)

    class_name = class_list[max_idx]
    return (f"{class_name} ({max_prob * 100:.1f}%)", max_prob)


def interpret_multilabel_probabilities(preds_prob, class_list, thresholds):
    """
    Menafsirkan output probabilitas untuk skema Multi-Label.
    """
    detected = []
    for idx, cls in enumerate(class_list):
        prob = preds_prob[idx]
        thresh = thresholds[idx]
        if prob >= thresh:
            detected.append(f"{cls} ({prob * 100:.1f}%)")

    if len(detected) == 0:
        return ("Others / Unsure", float(np.max(preds_prob)))

    return (" + ".join(detected), float(np.max(preds_prob)))


# =====================================================================
# PREPROCESS INPUT MODEL
# =====================================================================
def prepare_model_input(signal):
    """
    Mengubah Clean Signal (mV) -> Tensor Normalisasi AI (1, 2500, 3)
    """
    x = np.copy(signal)
    x = sanitize_signal(x)

    if x.ndim != 2:
        raise ValueError(f"Signal harus berbentuk [timesteps, channels], didapat ndim={x.ndim}")

    # 1. Kondisikan panjang gelombang via Center Crop / Zero Padding kaku ke 2500 sampel
    target_length = getattr(cfg, "MODEL_INPUT_LENGTH", 2500)
    x = ensure_length(x, target_len=target_length)

    # 2. Jalankan Z-score Normalization per-channel & clipping outlier ekstrem [-5.0, 5.0]
    x = apply_zscore_clip(
        x, 
        epsilon=1e-8, 
        clip_min=cfg.DEFAULT_CLIP_MIN, 
        clip_max=cfg.DEFAULT_CLIP_MAX
    )

    # 3. Ekspansi dimensi tensor menjadi bentuk batch (1, 2500, 3)
    x = np.expand_dims(x, axis=0)
    return x.astype(np.float32)


# =====================================================================
# INFERENCE EXECUTION
# =====================================================================
def run_dual_model_inference(clean_signal, model_id=None):
    """
    Eksekusi inferensi model ganda (Keras vs TFLite) menggunakan konfigurasi dari model registry.
    """
    if model_id is None:
        model_id = get_active_model_id()

    model_info = reg.get_model_info(model_id)
    class_list = model_info["class_list"]
    thresholds = model_info["thresholds"]
    task_type = model_info["task_type"]

    result = {
        "keras_prediction": "Offline",
        "keras_confidence": None,
        "keras_latency_ms": None,
        "tflite_prediction": "Offline",
        "tflite_confidence": None,
        "tflite_latency_ms": None,
    }

    # x sekarang berupa matriks ternormalisasi berdimensi kaku (1, 2500, 3)
    x = prepare_model_input(clean_signal)

    # ==============================================================
    # KERAS INFERENCE
    # ==============================================================
    ai_model_keras = get_keras_model(model_id)
    if ai_model_keras is not None:
        try:
            t0 = time.perf_counter()
            pred = ai_model_keras.predict(x, verbose=0)[0]
            t1 = time.perf_counter()
            latency = (t1 - t0) * 1000.0
            result["keras_latency_ms"] = latency

            if task_type == "multilabel":
                cls, conf = interpret_multilabel_probabilities(pred, class_list, thresholds)
            else:
                cls, conf = interpret_multiclass_probabilities(pred, class_list, thresholds[0])

            result["keras_prediction"] = cls
            result["keras_confidence"] = round(conf * 100, 2)

            logger.info(f"===== KERAS PREDICTION ({model_id}) =====")
            logger.info(f"Probabilitas: {pred}")
            logger.info(f"Thresholds: {thresholds}")
            logger.info(f"Prediksi Akhir: {cls}")
            logger.info(f"Confidence: {result['keras_confidence']}%")
            logger.info(f"Latency Inference: {latency:.2f} ms")

        except Exception as e:
            result["keras_prediction"] = f"Error: {e}"
            logger.error(f"Error during Keras inference: {e}", exc_info=True)

    # ==============================================================
    # TFLITE INFERENCE
    # ==============================================================
    tflite_interpreter = get_tflite_interpreter(model_id)
    if tflite_interpreter is not None:
        try:
            t0 = time.perf_counter()
            input_details = tflite_interpreter.get_input_details()
            output_details = tflite_interpreter.get_output_details()

            tflite_interpreter.set_tensor(
                input_details[0]["index"],
                x.astype(input_details[0]["dtype"])
            )
            tflite_interpreter.invoke()

            pred = tflite_interpreter.get_tensor(output_details[0]["index"])[0]
            t1 = time.perf_counter()
            latency = (t1 - t0) * 1000.0
            result["tflite_latency_ms"] = latency

            if task_type == "multilabel":
                cls, conf = interpret_multilabel_probabilities(pred, class_list, thresholds)
            else:
                cls, conf = interpret_multiclass_probabilities(pred, class_list, thresholds[0])

            result["tflite_prediction"] = cls
            result["tflite_confidence"] = round(conf * 100, 2)

            logger.info(f"===== TFLITE PREDICTION ({model_id}) =====")
            logger.info(f"Probabilitas: {pred}")
            logger.info(f"Thresholds: {thresholds}")
            logger.info(f"Prediksi Akhir: {cls}")
            logger.info(f"Confidence: {result['tflite_confidence']}%")
            logger.info(f"Latency Inference: {latency:.2f} ms")

        except Exception as e:
            result["tflite_prediction"] = f"Error: {e}"
            logger.error(f"Error during TFLite inference: {e}", exc_info=True)

    return result