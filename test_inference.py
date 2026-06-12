"""
Script de démonstration pour tester l'inférence sur une vidéo.
Utilise directement le module inference.py sans passer par Streamlit.
"""

import sys
import argparse
from pathlib import Path
from inference import DeepfakeDetector


def main():
    parser = argparse.ArgumentParser(
        description="Test la détection de deepfakes sur une vidéo"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Chemin vers la vidéo à analyser"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="best_model_fold4.pt",
        help="Chemin vers le checkpoint du modèle (défaut: best_model_fold4.pt)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device à utiliser (défaut: cpu)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Réduit les messages d'affichage"
    )
    
    args = parser.parse_args()
    
    # Vérifie que le fichier vidéo existe
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"❌ Erreur: Fichier vidéo non trouvé: {args.video_path}")
        return 1
    
    # Vérifie que le modèle existe
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Erreur: Modèle non trouvé: {args.model}")
        return 1
    
    # Charge le détecteur
    try:
        if not args.quiet:
            print(f"[INIT] Chargement du modèle depuis {args.model}...")
        detector = DeepfakeDetector(
            checkpoint_path=str(model_path),
            device=args.device
        )
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return 1
    
    # Lance la prédiction
    if not args.quiet:
        print(f"[INPUT] Analyse de: {args.video_path}")
    
    try:
        result = detector.predict(str(video_path), verbose=not args.quiet)
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction: {e}")
        return 1
    
    # Affiche les résultats
    print("\n" + "="*60)
    print("  RÉSULTATS")
    print("="*60)
    
    if result['success']:
        print(f"\n✓ Analyse réussie")
        print(f"\n  Résultat: {'🚨 DEEPFAKE DÉTECTÉ' if result['is_fake'] else '✅ VIDÉO AUTHENTIQUE'}")
        print(f"  Confiance: {result['confidence']:.1%}")
        print(f"\n  Probabilités:")
        print(f"    - Réelle: {result['probabilities']['real']:.2%}")
        print(f"    - Fake:   {result['probabilities']['fake']:.2%}")
    else:
        print(f"\n✗ Analyse échouée")
        print(f"  Erreur: {result['message']}")
        return 1
    
    print("\n" + "="*60 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
