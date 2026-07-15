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

# =============================================================================
# PROJECT ROOT
# =============================================================================

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
CLEANED_SAMPLE_DIR = DATASET_DIR / "cleaned_sample"
MANIFEST_CHAPMAN = CLEANED_SAMPLE_DIR / "manifest_chapman.csv"
MANIFEST_PTBXL = CLEANED_SAMPLE_DIR / "manifest_ptbxl.csv"

# =============================================================================
# MODEL PATHS
# =============================================================================

MODEL_DIR = PROJECT_ROOT / "model" / "Pure CNN Multi Label"
ORIGINAL_MODEL_PATH = MODEL_DIR / "best_model.keras"
PATCHED_MODEL_PATH = MODEL_DIR / "best_model_patched.keras"
TFLITE_MODEL_PATH = MODEL_DIR / "best_model.tflite"
MODEL_TMP_EXTRACT_DIR = MODEL_DIR / "tmp_extracted"

# =============================================================================
# FRONTEND PATHS
# =============================================================================

PRESENTATION_DIR = PROJECT_ROOT / "src" / "presentation"

# =============================================================================
# DSP CONFIGURATION
# =============================================================================

TARGET_FS = 250.0
DEFAULT_LEADS = (0, 1, 2)

# panjang sinyal yang diberikan ke CNN
MODEL_INPUT_LENGTH = 2500

# =============================================================================
# AI MODEL CONFIGURATION
# =============================================================================

TARGET_CLASSES = (
    "Normal",
    "AF",
    "Takikardia",
    "Bradikardia",
)

OPTIMIZED_THRESHOLDS = (
    0.34,
    0.37,
    0.57,
    0.57,
)

# =============================================================================
# APPLICATION METADATA
# =============================================================================

PIPELINE_VERSION = "v5.0_research_grade"
APP_NAME = "ECG Live Engine"
APP_DESCRIPTION = "ECG Filtering and AI Benchmark Workbench"

# =============================================================================
# OPTIONAL DEBUG FLAGS
# =============================================================================

DEBUG = True