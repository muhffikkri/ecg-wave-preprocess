import math
from fractions import Fraction

import numpy as np
import pywt
from scipy import signal


def sanitize_signal(raw_signal):
    x = np.asarray(raw_signal, dtype=float)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def validate_signal_shape(signal_array):
    x = np.asarray(signal_array, dtype=float)
    if x.ndim != 2:
        raise ValueError("Signal must be a 2D array with shape [timesteps, channels].")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("Signal cannot be empty.")
    return x


def apply_wavelet_denoising(signal_array, wavelet="db4", level=4):
    x = validate_signal_shape(signal_array)
    denoised = np.empty_like(x)

    for channel_index in range(x.shape[1]):
        channel = x[:, channel_index]
        coeffs = pywt.wavedec(channel, wavelet=wavelet, level=level)
        if len(coeffs) > 1:
            detail = coeffs[-1]
            sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745 if detail.size else 0.0
            threshold = sigma * math.sqrt(2.0 * math.log(max(channel.size, 2)))
            coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]
        denoised[:, channel_index] = pywt.waverec(coeffs, wavelet=wavelet)[: channel.size]

    return denoised


def apply_median_baseline(signal_array, kernel_size=51):
    x = validate_signal_shape(signal_array)
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    corrected = np.empty_like(x)
    for channel_index in range(x.shape[1]):
        baseline = signal.medfilt(x[:, channel_index], kernel_size=kernel_size)
        corrected[:, channel_index] = x[:, channel_index] - baseline
    return corrected


def apply_butter_bandpass(signal_array, fs, lowcut=0.5, highcut=45.0, order=4):
    x = validate_signal_shape(signal_array)
    nyquist = 0.5 * float(fs)
    low = max(float(lowcut) / nyquist, 1e-6)
    high = min(float(highcut) / nyquist, 0.999999)
    if not low < high:
        raise ValueError("lowcut must be lower than highcut and both must be within (0, Nyquist).")

    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def apply_poly_resample(signal_array, src_fs, target_fs):
    x = validate_signal_shape(signal_array)
    src_fs = float(src_fs)
    target_fs = float(target_fs)
    if src_fs <= 0 or target_fs <= 0:
        raise ValueError("Sampling rates must be positive.")
    if np.isclose(src_fs, target_fs):
        return x

    ratio = Fraction(target_fs / src_fs).limit_denominator(1000)
    return signal.resample_poly(x, ratio.numerator, ratio.denominator, axis=0)


def extract_signal_statistics(signal_array):
    x = np.asarray(signal_array, dtype=float)
    if x.ndim == 1:
        x = x[:, np.newaxis]

    return {
        "shape": [int(x.shape[0]), int(x.shape[1])],
        "mean": np.mean(x, axis=0).tolist(),
        "std": np.std(x, axis=0).tolist(),
        "min": np.min(x, axis=0).tolist(),
        "max": np.max(x, axis=0).tolist(),
        "median": np.median(x, axis=0).tolist(),
    }