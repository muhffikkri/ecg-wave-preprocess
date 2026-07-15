# =====================================================================
# FILE: src/logic/manifest_manager.py
# PURPOSE: Dataset Manifest Manager
# =====================================================================

from functools import lru_cache
from pathlib import Path

import pandas as pd


# =====================================================================
# LOAD MANIFEST
# =====================================================================

@lru_cache(maxsize=8)
def load_manifest_lookup(manifest_path):
    """
    Membaca file manifest CSV dan mengembalikan dictionary
    {filename -> target_class}.

    Hasil akan di-cache agar tidak membaca CSV berulang kali.
    """

    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        return {}

    try:
        df = pd.read_csv(manifest_path)

        return {
            str(row["filename_npy"]).strip():
            str(row.get("target_class", "Unknown")).strip()
            for _, row in df.iterrows()
            if pd.notna(row["filename_npy"])
        }

    except Exception as e:
        print(f"[Manifest] Failed reading {manifest_path.name}: {e}")
        return {}


# =====================================================================
# LOOKUP
# =====================================================================

def lookup_target_class(manifest_path, key_candidates):
    """
    Mencari target_class berdasarkan beberapa kemungkinan nama file.

    Parameters
    ----------
    manifest_path : str

    key_candidates : list[str]

    Returns
    -------
    str
        target_class atau "Unknown"
    """

    lookup = load_manifest_lookup(manifest_path)

    for key in key_candidates:
        key = str(key).strip()

        if key in lookup:
            return lookup[key]

    return "Unknown"