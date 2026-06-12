# 📁 Structure du Projet

Vue d'ensemble complète des fichiers et leur rôle.

## 🎯 Fichiers Principaux

### Interface & Inférence

| Fichier | Rôle | Langage |
|---------|------|---------|
| **app.py** | Interface web Streamlit | Python |
| **inference.py** | Moteur de prédiction | Python |
| **model.py** | Architecture du modèle | Python |
| **video_processor.py** | Traitement vidéo | Python |
| **rppg_extractor.py** | Signal physiologique | Python |

### Configuration & Lancement

| Fichier | Rôle | Type |
|---------|------|------|
| **requirements.txt** | Dépendances Python | Requirements |
| **run.bat** | Lancer l'interface | Batch (Windows) |
| **install.bat** | Installer l'environnement | Batch (Windows) |
| **check_setup.py** | Vérifier le setup | Python |
| **test_inference.py** | Tester inférence CLI | Python |

### Documentation

| Fichier | Contenu | Format |
|---------|---------|--------|
| **README.md** | Documentation complète | Markdown |
| **QUICKSTART.md** | Démarrage rapide | Markdown |
| **ARCHITECTURE.md** | Extraction des notebooks | Markdown |
| **FILES.md** | Ce fichier | Markdown |

### Configuration Streamlit

| Fichier | Rôle | Format |
|---------|------|--------|
| **.streamlit/config.toml** | Configuration Streamlit | TOML |

### Données & Modèle

| Fichier | Rôle | Note |
|---------|------|------|
| **best_model_fold4.pt** | Modèle entraîné | À télécharger |
| **.gitignore** | Fichiers à ignorer | Git |

---

## 📊 Dépendances du Code

```
app.py (Interface)
  ├─ streamlit
  ├─ inference.py
  │  ├─ video_processor.py
  │  │  ├─ cv2 (OpenCV)
  │  │  ├─ numpy
  │  │  ├─ scipy.fft
  │  │  └─ PIL
  │  ├─ rppg_extractor.py
  │  │  ├─ numpy
  │  │  ├─ scipy.signal
  │  ├─ model.py
  │  │  ├─ torch
  │  │  ├─ torchvision
  │  │  └─ best_model_fold4.pt
  │  └─ torchvision.transforms
  └─ PIL
```

---

## 🔄 Flux des Données

```
Utilisateur (web)
    ↓
app.py (Streamlit interface)
    ├─ Affiche formulaire
    └─ Appelle inference.py
         ↓
    DeepfakeDetector.predict()
         ├─ video_processor.extract_face_sequence()
         │  ├─ Lit vidéo (cv2)
         │  ├─ Détecte visages (RetinaFace/MTCNN)
         │  ├─ Extrait features fréquence
         │  └─ Retourne: frames + freq_features
         │
         ├─ rppg_extractor.estimate_rppg_from_faces()
         │  ├─ Extrait signal vert
         │  └─ Filtre signal
         │
         ├─ model.CNNFreqRPPGTransformer()
         │  ├─ Charge checkpoint
         │  ├─ Forward pass
         │  └─ Retourne logits
         │
         └─ Post-traite & retourne résultat
              ↓
           Résultat (Real/Fake + Confidence)
              ↓
           app.py affiche les résultats
              ↓
           Utilisateur voit réponse
```

---

## 🎓 Qu'est-ce qui a été extrait?

### Des Notebooks Kaggle

**pcdlastwork (1).ipynb:**
- ✅ Extraction de frames
- ✅ Détection de visages (RetinaFace)
- ✅ Cropping & Redimensionnement
- ✅ Calcul FFT/DCT

**notebooke5ca2fbf26.ipynb:**
- ✅ Architecture CNN + Transformer
- ✅ Extraction rPPG
- ✅ Cross-Attention layers
- ✅ Classifier head

### Refactorisations

- ✨ Code organisé en modules indépendants
- ✨ Élimination des dépendances Kaggle
- ✨ Gestion d'erreurs robuste
- ✨ Caching pour performance
- ✨ Interface utilisateur

Voir [ARCHITECTURE.md](ARCHITECTURE.md) pour détails complets.

---

## 🚀 Utilisation des Fichiers

### Installation
```bash
1. double-cliquez install.bat
2. Attendez (1-2 min)
3. Prêt!
```

### Lancement
```bash
double-cliquez run.bat
# ou
streamlit run app.py
```

### Tests
```bash
python check_setup.py                    # Vérifie setup
python test_inference.py mon_video.mp4   # Teste inférence
```

### Personnalisation
```python
# Changer le modèle:
detector = DeepfakeDetector("mon_modele.pt", device="cuda")

# Ou via Streamlit sidebar: Configuration
```

---

## 📦 Taille des Fichiers

| Fichier | Taille | Note |
|---------|--------|------|
| app.py | ~8 KB | Interface |
| model.py | ~12 KB | Architecture |
| video_processor.py | ~15 KB | Traitement vidéo |
| inference.py | ~8 KB | Inférence |
| rppg_extractor.py | ~3 KB | rPPG |
| requirements.txt | ~1 KB | Dépendances |
| best_model_fold4.pt | ~50-200 MB | Modèle (à télécharger) |
| **Total (sans modèle)** | ~47 KB | Code source |

---

## ✅ Checklist Installation

- [ ] Télécharger tous les fichiers Python
- [ ] Exécuter `install.bat`
- [ ] Télécharger `best_model_fold4.pt`
- [ ] Placer le modèle dans le répertoire
- [ ] Exécuter `run.bat`
- [ ] Ouvrir http://localhost:8501
- [ ] Télécharger une vidéo
- [ ] Cliquer "Analyser"
- [ ] Voir les résultats

---

## 🔗 Références Rapides

### Démarrer
- [QUICKSTART.md](QUICKSTART.md) - 5 minutes pour commencer

### Apprendre
- [ARCHITECTURE.md](ARCHITECTURE.md) - Extraction des notebooks
- [README.md](README.md) - Documentation complète

### Développer
- [model.py](model.py) - Architecture du modèle
- [inference.py](inference.py) - Pipeline inférence

### Dépanner
- [README.md#troubleshooting](README.md#troubleshooting) - Solutions
- Exécutez `python check_setup.py`

---

## 💾 Flux de Sauvegarde

Aucun fichier de résultat n'est sauvegardé par défaut.

Pour sauvegarder les résultats, modifiez `app.py`:

```python
# Ajouter après les résultats
if result['success']:
    with open("results.txt", "a") as f:
        f.write(f"{uploaded_file.name}: {result['is_fake']}\n")
```

---

## 🎨 Personnalisation

### Changer la couleur du thème
Modifier `.streamlit/config.toml`:
```toml
primaryColor = "#FF6B6B"  # Changez-moi!
```

### Changer la taille max upload
```toml
maxUploadSize = 1000  # MB
```

### Ajouter un modèle différent
```python
# Dans app.py:
detector = get_detector("mon_modele.pt", device)
```

---

## 🆘 Aide Rapide

| Question | Solution |
|----------|----------|
| Où mettre le modèle? | Dans le dossier `interface/` |
| Comment utiliser un autre modèle? | Changez le chemin dans Streamlit sidebar |
| Peut-on utiliser GPU? | Oui, changez device en "cuda" |
| Comment sauvegarder les résultats? | Modifiez app.py (voir section flux) |
| Où trouver le fichier du modèle? | C'est votre `best_model_fold4.pt` |

---

**Version**: 1.0  
**Mise à jour**: 2024  
**Status**: ✅ Production-ready

