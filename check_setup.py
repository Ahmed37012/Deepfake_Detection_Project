"""
Script de test des dépendances et du setup.
Vérifie que tout est installé correctement et que le modèle existe.
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """Vérifie la version de Python."""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ⚠️  Python 3.8+ recommandé")
        return False
    return True


def check_package(package_name, import_name=None):
    """Vérifie qu'un paquet est installé."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name}")
        return True
    except ImportError:
        print(f"✗ {package_name} - NON INSTALLÉ")
        return False


def check_model_file():
    """Vérifie que le fichier du modèle existe."""
    model_path = Path("best_model_fold4.pt")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ Modèle trouvé ({size_mb:.1f} MB)")
        return True
    else:
        print(f"✗ Modèle NON TROUVÉ (best_model_fold4.pt)")
        return False


def check_cuda():
    """Vérifie la disponibilité de CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA disponible ({torch.cuda.get_device_name(0)})")
            return True
        else:
            print("ℹ️  CUDA non disponible (utilisation de CPU)")
            return True
    except Exception as e:
        print(f"⚠️  Impossible de vérifier CUDA: {e}")
        return True


def main():
    print("\n" + "="*50)
    print("  Vérification du Setup")
    print("="*50 + "\n")
    
    checks = {
        "Python": check_python_version(),
        "torch": check_package("PyTorch", "torch"),
        "torchvision": check_package("TorchVision", "torchvision"),
        "cv2": check_package("OpenCV", "cv2"),
        "numpy": check_package("NumPy", "numpy"),
        "scipy": check_package("SciPy", "scipy"),
        "sklearn": check_package("Scikit-learn", "sklearn"),
        "streamlit": check_package("Streamlit", "streamlit"),
        "retinaface": check_package("RetinaFace", "retinaface"),
        "mtcnn": check_package("MTCNN", "mtcnn"),
        "PIL": check_package("Pillow", "PIL"),
    }
    
    print("\n" + "-"*50)
    print("  Fichiers et Hardware")
    print("-"*50 + "\n")
    
    files_ok = check_model_file()
    cuda_ok = check_cuda()
    
    print("\n" + "-"*50)
    print("  Résumé")
    print("-"*50 + "\n")
    
    all_ok = all(checks.values()) and files_ok
    
    packages_ok = sum(checks.values())
    total_packages = len(checks)
    
    print(f"Paquets: {packages_ok}/{total_packages}")
    print(f"Modèle: {'✓' if files_ok else '✗'}")
    print(f"CUDA: {'✓' if cuda_ok else 'ℹ️  (CPU will be used)'}")
    
    print("\n" + "="*50)
    
    if all_ok:
        print("✓ Tout est OK! Vous pouvez lancer l'interface.")
        print("\n  Commande: streamlit run app.py")
    else:
        if files_ok is False:
            print("✗ Le modèle est manquant!")
            print("  Assurez-vous que 'best_model_fold4.pt' est présent.")
        
        if packages_ok < total_packages:
            print("✗ Des paquets sont manquants!")
            print("  Exécutez: pip install -r requirements.txt")
    
    print("="*50 + "\n")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
