"""
Module de traitement vidéo pour la détection de deepfakes.
Extrait les frames, détecte les visages et calcule les features de fréquence.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from scipy.fft import dctn
from PIL import Image


# ════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def normalize_map(x: np.ndarray) -> np.ndarray:
    """Normalise une carte entre 0 et 1."""
    x = x.astype(np.float32)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def compute_fft_log_magnitude(gray_img):
    """Calcule la magnitude log FFT normalisée."""
    fft_shift = np.fft.fftshift(np.fft.fft2(gray_img))
    return normalize_map(np.log1p(np.abs(fft_shift)))


def compute_dct_log_magnitude(gray_img):
    """Calcule la magnitude log DCT normalisée."""
    dct = dctn(gray_img.astype(np.float32), type=2, norm="ortho")
    return normalize_map(np.log1p(np.abs(dct)))


def compute_frequency_metrics(fft_map, dct_map) -> Dict[str, float]:
    """Calcule les métriques de fréquence (FFT et DCT)."""
    h, w = fft_map.shape
    yy, xx = np.ogrid[:h, :w]
    fft_r = np.sqrt((yy-(h-1)/2)**2 + (xx-(w-1)/2)**2) / max(1e-8, np.sqrt(((h-1)/2)**2+((w-1)/2)**2))
    dct_r = np.sqrt(yy**2 + xx**2) / max(1e-8, np.sqrt((h-1)**2+(w-1)**2))

    def smean(arr, mask):
        return float(np.mean(arr[mask])) if mask.any() else 0.0

    fc = smean(fft_map, fft_r <= 0.25)
    fo = smean(fft_map, fft_r >= 0.65)
    dc = smean(dct_map, dct_r <= 0.25)
    do_ = smean(dct_map, dct_r >= 0.65)
    
    return {
        "fft_mean": float(np.mean(fft_map)),
        "fft_std": float(np.std(fft_map)),
        "fft_center_mean": fc,
        "fft_outer_mean": fo,
        "fft_outer_to_center": fo / max(1e-8, fc),
        "dct_mean": float(np.mean(dct_map)),
        "dct_std": float(np.std(dct_map)),
        "dct_center_mean": dc,
        "dct_outer_mean": do_,
        "dct_outer_to_center": do_ / max(1e-8, dc),
    }


def compute_freq_from_bgr(face_bgr: np.ndarray, include_maps: bool = False) -> Optional[Dict]:
    """Calcule les features de fréquence depuis une image BGR."""
    if face_bgr is None or face_bgr.size == 0:
        return None
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fft_map = compute_fft_log_magnitude(gray)
    dct_map = compute_dct_log_magnitude(gray)
    metrics = compute_frequency_metrics(fft_map, dct_map)
    if include_maps:
        return {
            "metrics": metrics,
            "fft_map": fft_map,
            "dct_map": dct_map,
        }
    return metrics


# ════════════════════════════════════════════════════════════════════════════
# FACE DETECTION
# ════════════════════════════════════════════════════════════════════════════

class DeepfakeFaceDetector:
    """Détecteur de visages avec support RetinaFace/MTCNN."""
    
    def __init__(self, backend_preference: str = "retinaface", allow_fallback: bool = True):
        self.backend = None
        self.detector = None
        backend_preference = backend_preference.lower().strip()
        
        if backend_preference in {"retinaface", "auto"}:
            try:
                from retinaface import RetinaFace
                self.backend = "retinaface"
                self.detector = RetinaFace
                print("[INFO] Backend: RetinaFace")
                return
            except Exception as exc:
                print(f"[WARN] RetinaFace unavailable: {exc}")
                if backend_preference == "retinaface" and not allow_fallback:
                    raise RuntimeError("RetinaFace required but unavailable.")
        
        try:
            from mtcnn import MTCNN
            self.backend = "mtcnn"
            self.detector = MTCNN()
            print("[INFO] Backend: MTCNN (fallback)")
            return
        except Exception as exc:
            print(f"[WARN] MTCNN unavailable: {exc}")

        # Final fallback: OpenCV Haar Cascade (no extra deps)
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                raise RuntimeError("OpenCV Haar cascade not found")
            self.backend = "haar"
            self.detector = cascade
            print("[INFO] Backend: OpenCV Haar Cascade (final fallback)")
        except Exception as exc:
            raise RuntimeError(f"No face detector available: {exc}")

    def detect(self, frame_bgr: np.ndarray, confidence_threshold: float) -> List[Dict]:
        """Détecte les visages dans une frame."""
        detections: List[Dict] = []
        
        if self.backend == "retinaface":
            faces = self.detector.detect_faces(frame_bgr)
            if isinstance(faces, dict):
                for _, fd in faces.items():
                    score = float(fd.get("score", 0.0))
                    if score < confidence_threshold:
                        continue
                    x1, y1, x2, y2 = fd.get("facial_area", [0, 0, 0, 0])
                    detections.append({"bbox": [int(x1), int(y1), int(x2), int(y2)], "confidence": score})
        
        elif self.backend == "mtcnn":
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            faces = self.detector.detect_faces(frame_rgb)
            if isinstance(faces, list):
                for fd in faces:
                    score = float(fd.get("confidence", 0.0))
                    if score < confidence_threshold:
                        continue
                    x, y, w, h = fd.get("box", [0, 0, 0, 0])
                    detections.append({"bbox": [int(x), int(y), int(x+w), int(y+h)], "confidence": score})
        
        elif self.backend == "haar":
            # OpenCV Haar Cascade with optimized parameters
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)  # Improve contrast
            
            # Try multiple scale factors for better detection
            detected_faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.05,  # Sensitive to scale changes
                minNeighbors=4,    # Lower threshold for more detections
                minSize=(50, 50),  # Minimum face size
                maxSize=(500, 500) # Maximum face size
            )
            
            # Convert to (0-1) confidence scores based on detection quality
            for (x, y, w, h) in detected_faces:
                # Normalize confidence (0.7 baseline for Haar, boosted if face is in good size range)
                confidence = 0.7
                face_area = w * h
                img_area = gray.shape[0] * gray.shape[1]
                if 0.05 <= face_area / img_area <= 0.4:  # Face is reasonable size
                    confidence = 0.8
                
                if confidence >= confidence_threshold:
                    detections.append({"bbox": [int(x), int(y), int(x+w), int(y+h)], "confidence": confidence})
        
        detections.sort(
            key=lambda d: (d["bbox"][2]-d["bbox"][0]) * (d["bbox"][3]-d["bbox"][1]) * d["confidence"],
            reverse=True
        )
        return detections


def _clip_bbox(bbox, w, h):
    """Clip une bounding box dans les limites de l'image."""
    x1, y1, x2, y2 = bbox
    return [max(0, min(x1, w-1)), max(0, min(y1, h-1)),
            max(0, min(x2, w-1)), max(0, min(y2, h-1))]


def crop_face_with_padding(frame_bgr, bbox, padding_ratio):
    """Crop un visage avec padding."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(bbox, w, h)
    padx = int(max(1, x2-x1) * padding_ratio)
    pady = int(max(1, y2-y1) * padding_ratio)
    cx1, cy1 = max(0, x1-padx), max(0, y1-pady)
    cx2, cy2 = min(w, x2+padx), min(h, y2+pady)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = frame_bgr[cy1:cy2, cx1:cx2]
    return crop if crop.size > 0 else None


def resize_face_to_standard(face_bgr, output_size):
    """Redimensionne un visage à une taille standard."""
    pil = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
    pil = pil.resize((output_size, output_size), Image.BILINEAR)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _select_detections(detections, frame_shape, mode="largest"):
    """Sélectionne les détections selon le mode."""
    if mode == "all" or not detections:
        return detections
    if mode == "largest":
        return [detections[0]]
    h, w = frame_shape[:2]
    cx_img, cy_img = w / 2.0, h / 2.0
    def center_dist(d):
        x1, y1, x2, y2 = d["bbox"]
        return (((x1+x2)/2 - cx_img)**2 + ((y1+y2)/2 - cy_img)**2)
    return [min(detections, key=center_dist)]


# ════════════════════════════════════════════════════════════════════════════
# VIDEO PROCESSING
# ════════════════════════════════════════════════════════════════════════════

class VideoProcessor:
    """Traite une vidéo pour extraire les séquences de visages."""
    
    def __init__(self, 
                 face_size: int = 224,
                 confidence_threshold: float = 0.70,
                 padding_ratio: float = 0.35,
                 seq_len: int = 16,
                 detector_backend: str = "retinaface"):
        self.face_size = face_size
        self.confidence_threshold = confidence_threshold
        self.padding_ratio = padding_ratio
        self.seq_len = seq_len
        self.detector = DeepfakeFaceDetector(backend_preference=detector_backend, allow_fallback=True)

    def extract_face_sequence(self, video_path: str, verbose: bool = True) -> Optional[Dict]:
        """
        Extrait une séquence de visages d'une vidéo.
        Retourne :
        {
            'frames': list of (H, W, 3) uint8 RGB arrays,
            'freq_features': (seq_len, 10) numpy array,
            'success': bool
        }
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if verbose:
            print(f"[INFO] Video has {total_frames} frames")

        # Try with normal confidence threshold first, then with lower threshold
        for attempt, confidence_threshold_override in enumerate([self.confidence_threshold, 0.4, 0.0]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
            
            frames_rgb = []
            source_frames_rgb = []
            fft_maps = []
            dct_maps = []
            freq_features = []
            frame_idx = 0
            faces_found = 0

            if verbose and attempt > 0:
                print(f"[INFO] Retry {attempt}: Using lower confidence threshold ({confidence_threshold_override})")

            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                # Détecte les visages
                detections = self.detector.detect(frame_bgr, confidence_threshold_override)
                if not detections:
                    frame_idx += 1
                    continue

                faces_found += 1
                
                # Sélectionne le visage
                selected = _select_detections(detections, frame_bgr.shape, mode="largest")
                if not selected:
                    frame_idx += 1
                    continue

                det = selected[0]
                crop = crop_face_with_padding(frame_bgr, det["bbox"], self.padding_ratio)
                if crop is None:
                    frame_idx += 1
                    continue

                # Redimensionne
                face_resized = resize_face_to_standard(crop, self.face_size)
                frame_resized = cv2.resize(frame_bgr, (self.face_size, self.face_size))
                
                # Calcule les features de fréquence
                freq_output = compute_freq_from_bgr(face_resized, include_maps=True)
                if freq_output is None:
                    frame_idx += 1
                    continue
                freq_metrics = freq_output["metrics"]

                # Ajoute le visage (en RGB)
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                source_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frames_rgb.append(face_rgb)
                source_frames_rgb.append(source_rgb)

                fft_map_uint8 = (np.clip(freq_output["fft_map"], 0.0, 1.0) * 255).astype(np.uint8)
                dct_map_uint8 = (np.clip(freq_output["dct_map"], 0.0, 1.0) * 255).astype(np.uint8)
                fft_maps.append(fft_map_uint8)
                dct_maps.append(dct_map_uint8)

                # Ajoute les features
                freq_vals = [
                    freq_metrics["fft_mean"],
                    freq_metrics["fft_std"],
                    freq_metrics["fft_center_mean"],
                    freq_metrics["fft_outer_mean"],
                    freq_metrics["fft_outer_to_center"],
                    freq_metrics["dct_mean"],
                    freq_metrics["dct_std"],
                    freq_metrics["dct_center_mean"],
                    freq_metrics["dct_outer_mean"],
                    freq_metrics["dct_outer_to_center"],
                ]
                freq_features.append(freq_vals)

                frame_idx += 1
                
                # Limite à seq_len frames
                if len(frames_rgb) >= self.seq_len:
                    break

            if verbose:
                print(f"[INFO] Found {faces_found} faces in {frame_idx} frames checked, extracted {len(frames_rgb)} frames")

            # If we got enough frames, use them
            if len(frames_rgb) >= self.seq_len:
                frames_rgb = frames_rgb[:self.seq_len]
                source_frames_rgb = source_frames_rgb[:self.seq_len]
                fft_maps = fft_maps[:self.seq_len]
                dct_maps = dct_maps[:self.seq_len]
                freq_features = freq_features[:self.seq_len]
                cap.release()
                if verbose:
                    print(f"[INFO] Extracted {len(frames_rgb)} faces with frequency features")
                return {
                    "source_frames": source_frames_rgb,
                    "frames": frames_rgb,
                    "fft_maps": fft_maps,
                    "dct_maps": dct_maps,
                    "freq_features": np.array(freq_features, dtype=np.float32),
                    "extraction_mode": "face_detection",
                    "success": True,
                }
            
            # If we got some frames (at least 50% of needed), pad with last frame
            if len(frames_rgb) >= self.seq_len // 2:
                if frames_rgb:
                    last_frame = frames_rgb[-1]
                    last_source = source_frames_rgb[-1]
                    last_fft = fft_maps[-1]
                    last_dct = dct_maps[-1]
                    last_freq = freq_features[-1]
                    while len(frames_rgb) < self.seq_len:
                        frames_rgb.append(last_frame.copy())
                        source_frames_rgb.append(last_source.copy())
                        fft_maps.append(last_fft.copy())
                        dct_maps.append(last_dct.copy())
                        freq_features.append(last_freq.copy())
                    
                    frames_rgb = frames_rgb[:self.seq_len]
                    source_frames_rgb = source_frames_rgb[:self.seq_len]
                    fft_maps = fft_maps[:self.seq_len]
                    dct_maps = dct_maps[:self.seq_len]
                    freq_features = freq_features[:self.seq_len]
                    cap.release()
                    if verbose:
                        print(f"[INFO] Extracted {len(frames_rgb)} faces (padded) with frequency features")
                    return {
                        "source_frames": source_frames_rgb,
                        "frames": frames_rgb,
                        "fft_maps": fft_maps,
                        "dct_maps": dct_maps,
                        "freq_features": np.array(freq_features, dtype=np.float32),
                        "extraction_mode": "face_detection_padded",
                        "success": True,
                    }

        cap.release()
        
        # Final fallback: use full frames if no faces can be extracted
        if verbose:
            print("[WARN] Could not extract faces reliably. Using full video frames as fallback.")
        
        return self._extract_full_frames_fallback(video_path, verbose)

    def _extract_full_frames_fallback(self, video_path: str, verbose: bool = True) -> Optional[Dict]:
        """Fallback: use full video frames when face detection fails."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        frames_rgb = []
        source_frames_rgb = []
        fft_maps = []
        dct_maps = []
        freq_features = []
        frame_count = 0
        
        while len(frames_rgb) < self.seq_len:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            
            # Resize full frame to face_size
            frame_resized = cv2.resize(frame_bgr, (self.face_size, self.face_size))
            
            # Calculate frequency features
            freq_output = compute_freq_from_bgr(frame_resized, include_maps=True)
            if freq_output is None:
                continue
            freq_metrics = freq_output["metrics"]
            
            # Add frame (in RGB)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames_rgb.append(frame_rgb)
            source_frames_rgb.append(frame_rgb.copy())

            fft_map_uint8 = (np.clip(freq_output["fft_map"], 0.0, 1.0) * 255).astype(np.uint8)
            dct_map_uint8 = (np.clip(freq_output["dct_map"], 0.0, 1.0) * 255).astype(np.uint8)
            fft_maps.append(fft_map_uint8)
            dct_maps.append(dct_map_uint8)
            
            # Add features
            freq_vals = [
                freq_metrics["fft_mean"],
                freq_metrics["fft_std"],
                freq_metrics["fft_center_mean"],
                freq_metrics["fft_outer_mean"],
                freq_metrics["fft_outer_to_center"],
                freq_metrics["dct_mean"],
                freq_metrics["dct_std"],
                freq_metrics["dct_center_mean"],
                freq_metrics["dct_outer_mean"],
                freq_metrics["dct_outer_to_center"],
            ]
            freq_features.append(freq_vals)
            frame_count += 1
        
        cap.release()
        
        if not frames_rgb:
            if verbose:
                print("[ERROR] Could not extract any frames from video")
            return None
        
        # Pad if needed
        if len(frames_rgb) < self.seq_len:
            last_frame = frames_rgb[-1]
            last_source = source_frames_rgb[-1]
            last_fft = fft_maps[-1]
            last_dct = dct_maps[-1]
            last_freq = freq_features[-1]
            while len(frames_rgb) < self.seq_len:
                frames_rgb.append(last_frame.copy())
                source_frames_rgb.append(last_source.copy())
                fft_maps.append(last_fft.copy())
                dct_maps.append(last_dct.copy())
                freq_features.append(last_freq.copy())
        
        frames_rgb = frames_rgb[:self.seq_len]
        source_frames_rgb = source_frames_rgb[:self.seq_len]
        fft_maps = fft_maps[:self.seq_len]
        dct_maps = dct_maps[:self.seq_len]
        freq_features = freq_features[:self.seq_len]
        
        if verbose:
            print(f"[INFO] Extracted {len(frames_rgb)} full frames as fallback")
        
        return {
            "source_frames": source_frames_rgb,
            "frames": frames_rgb,
            "fft_maps": fft_maps,
            "dct_maps": dct_maps,
            "freq_features": np.array(freq_features, dtype=np.float32),
            "extraction_mode": "full_frame_fallback",
            "success": True,
        }

