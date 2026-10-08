# =====================================================================
# FILE: src/logic/logic_layer.py
# PURPOSE: Pure Research-Grade v5.0 DSP Engine (No Backend Overrides)
# =====================================================================

import logging
import math
import os
from pathlib import Path
import time
import tracemalloc
from fractions import Fraction

import numpy as np
import pandas as pd
import pywt
from scipy import signal

from app import config as cfg


# =====================================================================
# 2. HOLTER METRIC COMPUTATION (Murni Satuan mV Klinis Tanpa Normalisasi)
# =====================================================================

def extract_holter_metrics(signal_1d, fs):
    """Estimasi metrik Holter secara klinis murni (Bebas dari Error 500 JSON)."""
    try:
        diff_sig = np.diff(signal_1d) ** 2
        threshold = np.percentile(diff_sig, 95)
        peaks = []
        min_spacing = int(0.30 * fs)
        prev_peak = -min_spacing

        for idx, val in enumerate(diff_sig):
            if val > threshold and (idx - prev_peak) > min_spacing:
                s = max(0, idx - 10)
                e = min(len(signal_1d), idx + 10)
                true_peak = s + np.argmax(signal_1d[s:e])
                peaks.append(true_peak)
                prev_peak = true_peak

        peaks = np.asarray(peaks)

        if len(peaks) < 2:
            return {
                "hr": "--", "rr_avg_ms": "--", "rmssd_ms": "--",
                "st_dev_mv": 0.0, "qtc_ms": "--", "events": ["🟢 Loading..."],
            }

        rr_samples = np.diff(peaks)
        rr_ms = (rr_samples / fs) * 1000.0
        rr_avg = np.mean(rr_ms)
        hr = 60000.0 / rr_avg if rr_avg > 0 else 0

        rr_diff = np.diff(rr_ms)
        rmssd = np.sqrt(np.mean(rr_diff ** 2)) if len(rr_diff) else 0

        st_offset = int(0.060 * fs)
        st_values = [
            signal_1d[p + st_offset]
            for p in peaks
            if (p + st_offset) < len(signal_1d)
        ]
        st_dev = np.mean(st_values) if len(st_values) else 0

        qt_ms = 360.0
        qtc = qt_ms / np.sqrt(rr_avg / 1000.0) if rr_avg > 0 else 0

        events = []
        if hr > 100: events.append("⚠️ Tachycardia")
        elif hr < 60: events.append("⚠️ Bradycardia")

        if np.std(rr_ms) > 100: events.append("⚠️ Irregular Rhythm")
        if len(events) == 0: events.append("🟢 Normal Rhythm")

        # Solusi Konversi Eksplisit float() untuk Menghilangkan Error 500 ASGI Jsonable Encoder
        return {
            "hr": float(round(hr, 1)) if isinstance(hr, (int, float, np.number)) else hr,
            "rr_avg_ms": float(round(rr_avg, 1)) if isinstance(rr_avg, (int, float, np.number)) else rr_avg,
            "rmssd_ms": float(round(rmssd, 1)) if isinstance(rmssd, (int, float, np.number)) else rmssd,
            "st_dev_mv": float(round(st_dev, 3)) if isinstance(st_dev, (int, float, np.number)) else st_dev,
            "qtc_ms": float(round(qtc, 1)) if isinstance(qtc, (int, float, np.number)) else qtc,
            "events": events,
        }
    except Exception as e:
        return {
            "hr": "Err", "rr_avg_ms": "Err", "rmssd_ms": "Err",
            "st_dev_mv": 0, "qtc_ms": "Err", "events": [str(e)],
        }


# =====================================================================
# 3. FILTERED FRAME SAVER & STORAGE UTILITIES
# =====================================================================
import os
from pathlib import Path
import pandas as pd

def save_filtered_frames(
    filtered_signal,
    output_dir=None,
    record_id="record",
    frame_size=cfg.MODEL_INPUT_LENGTH,
    stride=None,
    file_format="npy",
    save_full_signal=True,
):
    """
    Menyimpan sinyal EKG yang telah difilter ke dalam bentuk frame / segmen terpotong (.npy / .csv / .npz).

    Parameters
    ----------
    filtered_signal : np.ndarray
        Array sinyal EKG 2D [timesteps, channels] yang sudah melalui pipeline filtering.
    output_dir : str or Path, optional
        Direktori tujuan penyimpanan file frame. Jika None, default ke cfg.FILTERED_FRAMES_DIR.
    record_id : str, optional
        ID atau nama rekaman sebagai prefix file (default: "record").
    frame_size : int, optional
        Ukuran frame (jumlah sampel) per potongan (default: 2500 sampel sesuai input model).
    stride : int, optional
        Pergeseran sampel antar frame (default: sama dengan frame_size / non-overlapping).
    file_format : str, optional
        Format penyimpanan file: 'npy', 'npz', atau 'csv' (default: 'npy').
    save_full_signal : bool, optional
        Jika True, juga menyimpan sinyal utuh filtered_signal sebagai file tersendiri.

    Returns
    -------
    dict
        Metadata penyimpanan frame (jumlah frame, daftar path tersimpan, shape, status).
    """
    try:
        if output_dir is None:
            output_dir = getattr(cfg, "FILTERED_FRAMES_DIR", cfg.PROJECT_ROOT / "output" / "filtered_frames")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filtered_signal = np.asarray(filtered_signal, dtype=np.float32)
        if filtered_signal.ndim != 2:
            raise ValueError(f"filtered_signal harus 2D [timesteps, channels], didapat ndim={filtered_signal.ndim}")

        total_samples, num_channels = filtered_signal.shape
        if stride is None or stride <= 0:
            stride = frame_size

        saved_files = []
        file_format = file_format.lower().strip()

        # 1. Simpan sinyal filtered utuh
        if save_full_signal:
            full_filename = f"{record_id}_full_filtered.{file_format}"
            full_filepath = output_path / full_filename

            if file_format == "npy":
                np.save(full_filepath, filtered_signal)
            elif file_format == "npz":
                np.savez_compressed(full_filepath, signal=filtered_signal)
            elif file_format == "csv":
                cols = [f"lead_{i}" for i in range(num_channels)]
                df = pd.DataFrame(filtered_signal, columns=cols)
                df.to_csv(full_filepath, index=False)
            
            saved_files.append(str(full_filepath))

        # 2. Pemotongan sinyal menjadi frame-frame (windowing)
        frame_count = 0
        if frame_size is not None and frame_size > 0:
            idx = 0
            while idx + frame_size <= total_samples:
                frame = filtered_signal[idx : idx + frame_size, :]
                frame_filename = f"{record_id}_frame_{frame_count:04d}.{file_format}"
                frame_filepath = output_path / frame_filename

                if file_format == "npy":
                    np.save(frame_filepath, frame)
                elif file_format == "npz":
                    np.savez_compressed(frame_filepath, frame=frame)
                elif file_format == "csv":
                    cols = [f"lead_{i}" for i in range(num_channels)]
                    df = pd.DataFrame(frame, columns=cols)
                    df.to_csv(frame_filepath, index=False)

                saved_files.append(str(frame_filepath))
                frame_count += 1
                idx += stride

            # Handing sisa sinyal pendek / residual jika belum ada frame yang dibuat
            if idx < total_samples and frame_count == 0:
                from logic.preprocessing import ensure_length
                padded_frame = ensure_length(filtered_signal[idx:, :], target_len=frame_size)
                frame_filename = f"{record_id}_frame_0000.{file_format}"
                frame_filepath = output_path / frame_filename

                if file_format == "npy":
                    np.save(frame_filepath, padded_frame)
                elif file_format == "npz":
                    np.savez_compressed(frame_filepath, frame=padded_frame)
                elif file_format == "csv":
                    cols = [f"lead_{i}" for i in range(num_channels)]
                    df = pd.DataFrame(padded_frame, columns=cols)
                    df.to_csv(frame_filepath, index=False)

                saved_files.append(str(frame_filepath))
                frame_count = 1

        logger.info(f"Berhasil menyimpan {len(saved_files)} file frame ({frame_count} frame) ke: {output_path}")

        return {
            "status": "success",
            "output_dir": str(output_path),
            "total_files_saved": len(saved_files),
            "frame_count": frame_count,
            "frame_size": frame_size,
            "saved_files": saved_files,
        }
    except Exception as e:
        logger.error(f"Gagal menyimpan frame yang ter-filter: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "total_files_saved": 0,
            "frame_count": 0,
            "saved_files": [],
        }


# =====================================================================
# 4. PURE PIPELINE EXECUTIVE INTERFACE (No Overrides)
# =====================================================================
import logging
from app import config as cfg
from logic.preprocessing import sanitize_signal, validate_signal_shape, advanced_cleaning_pipeline

logger = logging.getLogger("ecg_workbench.logic_layer")

def execute_live_pipeline(
    raw_signal,
    src_fs,
    target_fs,
    p_wavelet,
    p_w_level,
    p_median_kernel,
    p_lowcut,
    p_highcut,
    save_frames: bool = False,
    output_dir: str = None,
    record_id: str = "record",
    frame_size: int = cfg.MODEL_INPUT_LENGTH,
    file_format: str = "csv",
    use_wavelet: bool = True,
    use_median: bool = True,
    use_bandpass: bool = True,
):
    """
    Eksekusi Murni Parameter UI Workbench Tanpa Pemaksaan Logika Alur di Backend.
    Mendukung opsi penyimpanan frame yang telah ter-filter jika save_frames=True.
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    x = sanitize_signal(raw_signal)
    x = validate_signal_shape(x)

    # 1. Komparasi Kalibrasi Tegangan Hardware ADS1293
    # if np.abs(np.mean(x)) > 10000:
    #     v_ref = cfg.ADS1293_VREF
    #     gain = cfg.ADS1293_GAIN
    #     mid = cfg.ADS1293_MID
    #     x = ((x - mid) / (mid - 1.0)) * (v_ref / gain) * 1000.0

    # 2. Jalankan Filter Sesuai Urutan Eksperimen Riset v5.0 Anda
    x = advanced_cleaning_pipeline(
        raw_signal=x,
        src_fs=src_fs,
        target_fs=target_fs,
        p_wavelet=p_wavelet,
        p_w_level=p_w_level,
        p_median_kernel=p_median_kernel,
        p_lowcut=p_lowcut,
        p_highcut=p_highcut,
        p_use_wavelet=use_wavelet,
        p_use_median=use_median,
        p_use_bandpass=use_bandpass,
    )

    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Ekstraksi metrik klinis Holter murni (Menggunakan Lead II / Indeks 1 jika multisaluran)
    ch_idx = 1 if x.shape[1] > 1 else 0
    holter = extract_holter_metrics(x[:, ch_idx], target_fs)

    metrics = {
        "latency_ms": (t1 - t0) * 1000,
        "peak_memory_mb": peak / (1024 * 1024),
        "holter": holter,
    }

    # 3. Simpan frame yang sudah ter-filter jika opsi diaktifkan
    if save_frames:
        save_info = save_filtered_frames(
            filtered_signal=x,
            output_dir=output_dir,
            record_id=record_id,
            frame_size=frame_size,
            file_format=file_format,
        )
        metrics["saved_frames"] = save_info

    return x, metrics

