"""
Module d'inférence pour la détection de deepfakes.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

import torchvision.transforms as T
from PIL import Image

from video_processor import VideoProcessor
from rppg_extractor import estimate_rppg_from_faces, rppg_features
from model import load_model


class DeepfakeDetector:
    """Prédiction de deepfakes à partir d'une vidéo."""
    
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        """
        Initialise le détecteur.
        
        Args:
            checkpoint_path : chemin vers le checkpoint du modèle
            device : "cpu" ou "cuda"
        """
        self.device = device
        self.model = load_model(checkpoint_path, device=device)
        self.video_processor = VideoProcessor(
            face_size=224,
            confidence_threshold=0.70,
            padding_ratio=0.35,
            seq_len=16,
            detector_backend="mtcnn",
        )
        
        # Transforms pour normaliser les images
        self.tf = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])
        
        print(f"[INFO] Modèle chargé sur {device}")

    def predict(self, video_path: str, verbose: bool = True) -> Dict:
        """
        Prédit si une vidéo est réelle ou fake.
        
        Args:
            video_path : chemin vers la vidéo
            verbose : affiche les messages
            
        Returns:
            {
                'success': bool,
                'is_fake': bool,  # True = fake, False = real
                'confidence': float,  # 0.0 à 1.0
                'probabilities': {'real': float, 'fake': float},
                'message': str,
            }
        """
        try:
            # ── 1. Extract faces from video ──────────────────────────
            if verbose:
                print(f"\n[STEP 1] Extracting faces from {Path(video_path).name}...")
            
            result = self.video_processor.extract_face_sequence(video_path, verbose=verbose)
            if result is None or not result['success']:
                return {
                    'success': False,
                    'is_fake': None,
                    'confidence': 0.0,
                    'probabilities': {'real': 0.0, 'fake': 0.0},
                    'message': 'Impossible d\'extraire les visages de la vidéo',
                }
            
            frames_rgb = result['frames']  # list of (H, W, 3) uint8
            freq_features = result['freq_features']  # (T, 10)
            source_frames = result.get('source_frames', frames_rgb)
            fft_maps = result.get('fft_maps', [])
            dct_maps = result.get('dct_maps', [])
            extraction_mode = result.get('extraction_mode', 'unknown')

            feature_names = [
                'fft_mean',
                'fft_std',
                'fft_center_mean',
                'fft_outer_mean',
                'fft_outer_to_center',
                'dct_mean',
                'dct_std',
                'dct_center_mean',
                'dct_outer_mean',
                'dct_outer_to_center',
            ]
            
            # ── 2. Extract rPPG signal ──────────────────────────────
            if verbose:
                print("[STEP 2] Extracting rPPG signal...")
            
            frames_rgb_normalized = np.stack(frames_rgb, axis=0).astype(np.float32) / 255.0
            rppg_sig = estimate_rppg_from_faces(frames_rgb_normalized, fs=30.0)
            rppg_feat = rppg_features(rppg_sig, fs=30.0)
            
            # ── 3. Prepare tensors for model ─────────────────────────
            if verbose:
                print("[STEP 3] Preparing tensors...")
            
            # Frames -> tensors with normalization
            frame_tensors = []
            for frame_rgb in frames_rgb:
                pil_img = Image.fromarray(frame_rgb)
                tensor = self.tf(pil_img)  # (3, H, W) normalized
                frame_tensors.append(tensor)
            
            frames_tensor = torch.stack(frame_tensors, dim=0)  # (T, 3, H, W)
            freq_tensor = torch.tensor(freq_features, dtype=torch.float32)  # (T, 10)
            rppg_tensor = torch.tensor(rppg_feat, dtype=torch.float32)  # (6,)
            
            # Add batch dimension
            frames_tensor = frames_tensor.unsqueeze(0).to(self.device)  # (1, T, 3, H, W)
            freq_tensor = freq_tensor.unsqueeze(0).to(self.device)     # (1, T, 10)
            rppg_tensor = rppg_tensor.unsqueeze(0).to(self.device)     # (1, 6)
            
            # ── 4. Inference ────────────────────────────────────────
            if verbose:
                print("[STEP 4] Running inference...")
            
            with torch.no_grad():
                logits = self.model(frames_tensor, freq_tensor, rppg_tensor)  # (1, 2)
                probs = torch.softmax(logits, dim=-1)  # (1, 2)
                
                prob_real = float(probs[0, 0].cpu().numpy())
                prob_fake = float(probs[0, 1].cpu().numpy())
                
                is_fake = prob_fake > prob_real
                confidence = max(prob_real, prob_fake)
            
            if verbose:
                result_label = "FAKE ⚠️" if is_fake else "REAL ✓"
                print(f"\n[RESULT] {result_label}")
                print(f"  Real: {prob_real:.2%}")
                print(f"  Fake: {prob_fake:.2%}")
                print(f"  Confidence: {confidence:.2%}")
            
            return {
                'success': True,
                'is_fake': is_fake,
                'confidence': float(confidence),
                'probabilities': {
                    'real': float(prob_real),
                    'fake': float(prob_fake),
                },
                'artifacts': {
                    'source_frames': source_frames,
                    'face_frames': frames_rgb,
                    'fft_maps': fft_maps,
                    'dct_maps': dct_maps,
                    'freq_feature_names': feature_names,
                    'freq_features_mean': np.mean(freq_features, axis=0).astype(np.float32).tolist(),
                    'rppg_features': np.asarray(rppg_feat, dtype=np.float32).tolist(),
                    'extraction_mode': extraction_mode,
                },
                'message': f"{'Détecté comme FAKE' if is_fake else 'Détecté comme REAL'} (confiance: {confidence:.1%})",
            }
        
        except Exception as e:
            print(f"[ERROR] Erreur lors de la prédiction: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'is_fake': None,
                'confidence': 0.0,
                'probabilities': {'real': 0.0, 'fake': 0.0},
                'message': f'Erreur: {str(e)}',
            }
