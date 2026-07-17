"""
src/app/config.py

Global configuration for ECG Wave Preprocessing Workbench.

This module centralizes all configurable parameters including:
- Project paths
- Dataset locations
- AI model locations
- DSP parameters
- Inference configuration
"""

from pathlib import Path
import logging

# =============================================================================
# PROJECT ROOT
# ========================================= ====================================

# src/app/config.py -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# =============================================================================
# DATASET PATHS
# =============================================================================

DATASET_DIR = PROJECT_ROOT / "dataset"
CHAPMAN_DIR = DATASET_DIR / "chapman" / "sample"
PTBXL_100HZ_DIR = DATASET_DIR / "ptbxl" / "sample_100hz"
PTBXL_500HZ_DIR = DATASET_DIR / "ptbxl" / "sample_500hz"
PROSIM_SIMULATOR_DIR = DATASET_DIR / "Kalibrasi Prosim"
SENSOR_RECORD_DIR = DATASET_DIR / "sensor record"
CLEANED_SAMPLE_DIR = DATASET_DIR / "cleaned_sample"
MANIFEST_CHAPMAN = CLEANED_SAMPLE_DIR / "manifest_chapman.csv"
MANIFEST_PTBXL = CLEANED_SAMPLE_DIR / "manifest_ptbxl.csv"

# =============================================================================
# MODEL PATHS
# =============================================================================

MODEL_DIR = PROJECT_ROOT / "models" 

# =============================================================================
# FRONTEND PATHS
# =============================================================================

PRESENTATION_DIR = PROJECT_ROOT / "src" / "presentation"

# =============================================================================
# DSP CONFIGURATION
# =============================================================================

TARGET_FS = 250.0
DEFAULT_LEADS = (0, 1, 2)

# Panjang sinyal yang diberikan ke CNN
MODEL_INPUT_LENGTH = 2500

# Preprocessing Defaults
WAVELET_DEFAULT = "db4"
WAVELET_LEVEL_DEFAULT = 4
MEDIAN_KERNEL_DEFAULT = 51
MEDIAN_KERNEL_500 = 101
BUTTERWORTH_LOWCUT = 0.5
BUTTERWORTH_HIGHCUT_DEFAULT = 45.0
BUTTERWORTH_HIGHCUT_250 = 100.0  # Used if src_fs >= 250 in offline BP

DEFAULT_CLIP_MIN = -5.0
DEFAULT_CLIP_MAX = 5.0

# ADS1293 Calibration Defaults
ADS1293_VREF = 2.4
ADS1293_GAIN = 3.5
ADS1293_MID = 8388608.0

# =============================================================================
# AI MODEL CONFIGURATION
# =============================================================================

TARGET_CLASSES = (
    "Normal",
    "AF",
    "Takikardia",
    "Bradikardia",
)

DEFAULT_MODEL_ID = "multilabel_500_to_250"

# =============================================================================
# APPLICATION METADATA
# =============================================================================

PIPELINE_VERSION = "v5.0_research_grade"
APP_NAME = "ECG Live Engine"
APP_DESCRIPTION = "ECG Filtering and AI Benchmark Workbench"

# =============================================================================
# LOGGER CONFIGURATION
# =============================================================================

LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(message)s"  # Clean format as requested by output guidelines

# Configure root logging options
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    force=True
)

# =============================================================================
# OPTIONAL DEBUG FLAGS
# =============================================================================

DEBUG = True
