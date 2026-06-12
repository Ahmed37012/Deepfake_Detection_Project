"""
Module rPPG (remote Photoplethysmography) - Extraction du signal physiologique.
"""

import numpy as np
from scipy.signal import butter, filtfilt, periodogram


def bandpass_filter(sig: np.ndarray, fs: float = 30.0,
                    low: float = 0.7, high: float = 4.0, order: int = 3) -> np.ndarray:
    """Applique un filtre passe-bande Butterworth."""
    if len(sig) < (order * 6 + 3):
        return sig
    nyq   = 0.5 * fs
    low_n = max(1e-5, low  / nyq)
    high_n = min(0.999, high / nyq)
    b, a  = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, sig)


def estimate_rppg_from_faces(face_seq_rgb: np.ndarray, fs: float = 30.0) -> np.ndarray:
    """
    Estime le signal rPPG à partir d'une séquence de visages RGB.
    face_seq_rgb : (T, H, W, 3) float32 [0..1]
    Retourne : (T,) signal rPPG filtré
    """
    # Extrait le canal vert (sensible aux variations de pouls)
    green = face_seq_rgb[:, :, :, 1].mean(axis=(1, 2))
    
    # Normalise
    green = green - green.mean()
    if green.std() > 1e-8:
        green = green / green.std()
    
    return bandpass_filter(green, fs=fs).astype(np.float32)


def rppg_features(sig: np.ndarray, fs: float = 30.0) -> np.ndarray:
    """
    Extrait 6 features du signal rPPG.
    Retourne : (6,) array de features
    """
    if len(sig) < 4:
        return np.zeros(6, dtype=np.float32)
    
    f, pxx = periodogram(sig, fs=fs)
    mask   = (f >= 0.7) & (f <= 4.0)
    
    if not mask.any():
        return np.zeros(6, dtype=np.float32)
    
    f_band = f[mask]
    p_band = pxx[mask]
    peak_idx = int(np.argmax(p_band))
    hr_bpm = float(f_band[peak_idx]) * 60.0
    snr = float((p_band[peak_idx] + 1e-8) / (np.mean(p_band) + 1e-8))
    
    return np.array([
        float(sig.mean()),
        float(sig.std()),
        float(np.max(sig) - np.min(sig)),
        float(np.percentile(sig, 75) - np.percentile(sig, 25)),
        hr_bpm,
        snr,
    ], dtype=np.float32)
