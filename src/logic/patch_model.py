# =====================================================================
# FILE: src/logic/patch_model.py
# PURPOSE: DUAL PRODUCTION CONVERTER (PATCHED .KERAS & NATIVE .TFLITE) FOR REGISTRY
# =====================================================================
import os
import sys
from pathlib import Path

# Ensure the 'src' directory is in the python path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import zipfile
import json
import shutil
import tensorflow as tf
from tensorflow.keras import layers

import app.config as cfg
import app.model_registry as reg


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


def patch_and_convert_model(model_id, model_info):
    print(f"\n========================================================")
    print(f"Processing Model: {model_id} ({model_info['model_name']})")
    print(f"========================================================")

    keras_model_path = Path(model_info["keras_model_path"])
    tflite_model_path = Path(model_info["tflite_model_path"])
    parent_dir = keras_model_path.parent
    original_model_path = parent_dir / "best_model.keras"
    tmp_dir = parent_dir / "tmp_extracted"

    # Create directories if they do not exist
    parent_dir.mkdir(parents=True, exist_ok=True)

    # Robust fallback: copy original best_model.keras from parent directory or sibling directories if missing
    if not original_model_path.exists():
        fallback_source = parent_dir.parent / "best_model.keras"
        if fallback_source.exists():
            print(f"   -> Copying raw model from parent fallback: {fallback_source} -> {original_model_path}")
            shutil.copy2(fallback_source, original_model_path)
        else:
            # Check sibling directories
            siblings = [d for d in parent_dir.parent.iterdir() if d.is_dir() and d != parent_dir]
            found_sibling = False
            for sib in siblings:
                sib_source = sib / "best_model.keras"
                if sib_source.exists():
                    print(f"   -> Copying raw model from sibling fallback: {sib_source} -> {original_model_path}")
                    shutil.copy2(sib_source, original_model_path)
                    found_sibling = True
                    break

            if not found_sibling:
                print(f"[-] Original model file 'best_model.keras' not found in {parent_dir}, parent, or siblings.")
                return False


    # =====================================================================
    # FASE 1: STRIP PARAMETER RENORM DENGAN AMAN VIA ZIP MANIFEST PARSING
    # =====================================================================
    print("... 1. Membongkar kontainer arsip berkas .keras...")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    try:
        with zipfile.ZipFile(original_model_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
    except (zipfile.BadZipFile, PermissionError) as e:
        print(f"   -> [Warning] File {original_model_path} tidak valid atau rusak ({e}). Menghapus dan mencoba fallback...")
        if original_model_path.exists():
            os.remove(original_model_path)

        # Jalankan logika fallback
        fallback_source = parent_dir.parent / "best_model.keras"
        found_source = False
        if fallback_source.exists():
            try:
                with zipfile.ZipFile(fallback_source, 'r') as check_zip:
                    check_zip.testzip()
                print(f"   -> Copying raw model from parent fallback: {fallback_source} -> {original_model_path}")
                shutil.copy2(fallback_source, original_model_path)
                found_source = True
            except Exception:
                pass

        if not found_source:
            # Check sibling directories
            siblings = [d for d in parent_dir.parent.iterdir() if d.is_dir() and d != parent_dir]
            for sib in siblings:
                sib_source = sib / "best_model.keras"
                if sib_source.exists():
                    try:
                        with zipfile.ZipFile(sib_source, 'r') as check_zip:
                            check_zip.testzip()
                        print(f"   -> Copying raw model from sibling fallback: {sib_source} -> {original_model_path}")
                        shutil.copy2(sib_source, original_model_path)
                        found_source = True
                        break
                    except Exception:
                        continue

        if not found_source:
            print(f"[-] Gagal menemukan file master best_model.keras yang valid di parent maupun sibling.")
            return False

        # Coba ekstrak kembali
        try:
            with zipfile.ZipFile(original_model_path, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)
        except Exception as retry_err:
            print(f"[-] Gagal mengekstrak berkas setelah menyalin ulang: {retry_err}")
            return False

    config_json_path = tmp_dir / "config.json"
    if not config_json_path.exists():
        print("[-] Format berkas .keras tidak valid (config.json absen).")
        shutil.rmtree(tmp_dir)
        return False


    print("... 2. Membersihkan parameter renorm usang dari berkas arsitektur JSON...")
    with open(config_json_path, 'r') as f:
        model_config = json.load(f)

    clean_renorm_keys(model_config)

    with open(config_json_path, 'w') as f:
        json.dump(model_config, f)

    print("... 3. Mengemas kembali menjadi model .keras produksi terkalibrasi...")
    with zipfile.ZipFile(keras_model_path, 'w', zipfile.ZIP_DEFLATED) as zip_out:
        for full_path in tmp_dir.rglob("*"):
            if full_path.is_file():
                rel_path = full_path.relative_to(tmp_dir)
                zip_out.write(full_path, rel_path)

    shutil.rmtree(tmp_dir)
    print(f"   -> Saved Patched Keras: {keras_model_path}")

    # =====================================================================
    # FASE 2: VERIFIKASI NATIVE LOAD & EXPORT KE TFLITE INTERPRETER
    # =====================================================================
    print("... 4. Memuat model steril ke memori untuk verifikasi pipa komparatif...")
    try:
        final_model = tf.keras.models.load_model(
            keras_model_path, 
            compile=False, 
            custom_objects={"StochasticDepth": StochasticDepth}
        )
        print("   -> Validasi Keras Produksi Native Load: SUKSES!")
    except Exception as e:
        print(f"   -> Validasi Keras Load Gagal: {str(e)}")
        return False

    print("... 5. Mengonversi arsitektur terkalibrasi ke format TFLite Edge...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        print("... 6. Menyimpan berkas biner TFLite...")
        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)
        print(f"   -> Saved TFLite Biner: {tflite_model_path}")
        print(f"[+] Conversion successful for: {model_id}!")
        return True
    except Exception as e:
        print(f"[-] Failed converting model {model_id} to TFLite: {e}")
        return False


def main():
    print("Starting Batch Model Patcher and Converter...")
    success_count = 0
    total_count = 0

    for model_id, model_info in reg.MODELS.items():
        total_count += 1
        success = patch_and_convert_model(model_id, model_info)
        if success:
            success_count += 1

    print(f"\n========================================================")
    print(f"Batch Processing Complete: {success_count}/{total_count} models converted successfully.")
    print(f"========================================================")



if __name__ == "__main__":
    main()