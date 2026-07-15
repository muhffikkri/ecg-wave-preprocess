# =====================================================================
# FILE: src/logic/patch_model.py
# PURPOSE: DUAL PRODUCTION CONVERTER (PATCHED .KERAS & NATIVE .TFLITE)
# =====================================================================
import zipfile
import json
import shutil
import tensorflow as tf
from tensorflow.keras import layers

import src.app.config as cfg

# Ambil kelas StochasticDepth asli bawaan arsitektur Anda
# from logic.ai_model_manager import StochasticDepth
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

# --- Jalur Berkas ---
original_model_path = cfg.ORIGINAL_MODEL_PATH
patched_model_path = cfg.PATCHED_MODEL_PATH
tflite_model_path = cfg.TFLITE_MODEL_PATH
tmp_dir = cfg.MODEL_TMP_EXTRACT_DIR

if not original_model_path.exists():
    print(f"❌ File model tidak ditemukan di {original_model_path}")
    raise SystemExit(1)

# =====================================================================
# FASE 1: STRIP PARAMETER RENORM DENGAN AMAN VIA ZIP MANIFEST PARSING
# =====================================================================
print("⏳ 1. Membongkar kontainer arsip berkas .keras...")
if tmp_dir.exists():
    shutil.rmtree(tmp_dir)
    
with zipfile.ZipFile(original_model_path, 'r') as zip_ref:
    zip_ref.extractall(tmp_dir)

config_json_path = tmp_dir / "config.json"
if not config_json_path.exists():
    print("❌ Format berkas .keras tidak valid (config.json absen).")
    shutil.rmtree(tmp_dir)
    raise SystemExit(1)

print("🧹 2. Membersihkan parameter renorm usang dari berkas arsitektur JSON...")
with open(config_json_path, 'r') as f:
    model_config = json.load(f)

def clean_renorm_keys(obj):
    if isinstance(obj, dict):
        if obj.get("class_name") == "BatchNormalization" or "renorm" in obj:
            if "config" in obj:
                obj["config"].pop("renorm", None)
                obj["config"].pop("renorm_clipping", None)
                obj["config"].pop("renorm_momentum", None)
            obj.pop("renorm", None)
            obj.pop("renorm_clipping", None)
            obj.pop("renorm_momentum", None)
        for k, v in obj.items():
            clean_renorm_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            clean_renorm_keys(item)

clean_renorm_keys(model_config)

with open(config_json_path, 'w') as f:
    json.dump(model_config, f)

print("💾 3. Mengemas kembali menjadi model .keras produksi terkalibrasi...")
with zipfile.ZipFile(patched_model_path, 'w', zipfile.ZIP_DEFLATED) as zip_out:
    for full_path in tmp_dir.rglob("*"):
        if full_path.is_file():
            rel_path = full_path.relative_to(tmp_dir)
            zip_out.write(full_path, rel_path)

shutil.rmtree(tmp_dir)
print(f"   -> Saved Patched Keras: {patched_model_path}")

# =====================================================================
# FASE 2: VERIFIKASI NATIVE LOAD & EXPORT KE TFLITE INTERPRETER
# =====================================================================
print("⏳ 4. Memuat model steril ke memori untuk verifikasi pipa komparatif...")
try:
    final_model = tf.keras.models.load_model(
        patched_model_path, 
        compile=False, 
        custom_objects={"StochasticDepth": StochasticDepth}
    )
    print("   -> Validasi Keras Produksi Native Load: SUKSES!")
except Exception as e:
    print(f"   -> Validasi Keras Load Gagal: {str(e)}")
    raise SystemExit(1)

print("📦 5. Mengonversi arsitektur terkalibrasi ke format TFLite Edge...")
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

print("💾 6. Menyimpan berkas biner TFLite...")
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)
print(f"   -> Saved TFLite Biner: {tflite_model_path}")
print("✅ Sinkronisasi dual-model untuk pengujian benchmarking selesai total!")