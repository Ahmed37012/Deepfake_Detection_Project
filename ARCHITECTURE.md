# 🏗️ Architecture et Sources

Ce document explique comment l'interface a été construite à partir des notebooks Kaggle.

## 📚 Source: Les Deux Notebooks

### Notebook 1: `pcdlastwork (1).ipynb`
**Rôle**: Traitement des données brutes (vidéos → visages)

**Composants extraits:**
- Configuration des chemins et paramètres
- Extraction de frames depuis vidéos
- Détection et cropping de visages
- Calcul de métriques de fréquence (FFT, DCT)
- Sauvegardes de checkpoints

**Code réutilisé dans:**
- ✅ `video_processor.py` - Classe `VideoProcessor`

### Notebook 2: `notebooke5ca2fbf26.ipynb`
**Rôle**: Entraînement et validation du modèle multimodal

**Composants extraits:**
- Architecture complète du modèle CNN + Transformer
- Extraction du signal rPPG (physiologique)
- Stratégie de cross-validation
- Loss function (Focal Loss)
- Pipeline d'inférence

**Code réutilisé dans:**
- ✅ `model.py` - Classe `CNNFreqRPPGTransformer`
- ✅ `rppg_extractor.py` - Fonctions d'extraction rPPG
- ✅ `inference.py` - Classe `DeepfakeDetector`

---

## 🔄 Pipeline Complet

```
VIDEO INPUT
    ↓
┌─────────────────────────────────────────┐
│ video_processor.py                      │
│ ├─ VideoProcessor class                 │
│ │  ├─ extract_face_sequence()           │
│ │  │  ├─ Lit chaque frame               │
│ │  │  ├─ Détecte les visages (RetinaFace/MTCNN)
│ │  │  ├─ Crop + Resize (224×224)        │
│ │  │  └─ Calcule freq features (FFT/DCT)│
│ │  └─ Retourne: frames + freq_features  │
│ └─ Face detection helpers               │
│    └─ DeepfakeFaceDetector class        │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ rppg_extractor.py                       │
│ ├─ estimate_rppg_from_faces()           │
│ │  └─ Extrait signal physiologique      │
│ └─ rppg_features()                      │
│    └─ 6 features du signal rPPG         │
└─────────────────────────────────────────┘
    ↓
   [Tensors préparés]
    ├─ frames: (1, 16, 3, 224, 224)
    ├─ freq: (1, 16, 10)
    └─ rppg: (1, 6)
    ↓
┌─────────────────────────────────────────┐
│ model.py                                │
│ └─ CNNFreqRPPGTransformer               │
│    ├─ CNN (ResNet-18)                   │
│    ├─ Cross-Attention A: freq → CNN     │
│    ├─ Temporal Encoder (Transformer)    │
│    ├─ Cross-Attention B: rPPG → temporal│
│    └─ Classifier Head (2 classes)       │
└─────────────────────────────────────────┘
    ↓
   [Logits]
    ↓
   [Softmax]
    ↓
REAL/FAKE + Confidence
```

---

## 📋 Extraction Détaillée

### 1. `video_processor.py`

**Source**: Notebook 1, Cells 3-4, 5

**Fonctions clés:**
```python
# Fréquence (de Notebook 1 Cell 3)
compute_fft_log_magnitude()      # FFT
compute_dct_log_magnitude()      # DCT
compute_frequency_metrics()      # 10 features

# Face detection (de Notebook 1 Cell 4)
class DeepfakeFaceDetector       # RetinaFace/MTCNN
crop_face_with_padding()         # Cropping intelligent
resize_face_to_standard()        # Redimensionnement

# Traitement vidéo
class VideoProcessor             # Pipeline complet
```

**Différences:**
- ✨ Refactorisé pour accepter `video_path` en entrée
- ✨ Retourne directement les résultats (pas de sauvegarde disque)
- ✨ Cache simple du détecteur pour performance

### 2. `rppg_extractor.py`

**Source**: Notebook 2, Cell 3

**Extraits:**
```python
bandpass_filter()           # Butterworth [0.7-4.0 Hz]
estimate_rppg_from_faces()  # Signal vert canal
rppg_features()             # 6 features statistiques
```

**Changements:**
- ✨ Module indépendant (plus facile à tester)
- ✨ Pas de dépendance au dataset

### 3. `model.py`

**Source**: Notebook 2, Cell 4

**Architecture exacte extraite:**
```
CNNFreqRPPGTransformer
├─ CNN backbone: ResNet-18
├─ freq_proj: MLP
├─ cross_attn_A: Freq → CNN
├─ temporal_encoder: Transformer
├─ cross_attn_B: rPPG → temporal
├─ head: Classifier
```

**Identique au notebook:**
- ✅ Pre-LN Transformer Encoder
- ✅ Gated fusion: g * attended + (1-g) * pooled
- ✅ FocalLoss compatible (mais retourné en weights)

### 4. `inference.py`

**Source**: Notebook 2, Cells 2, 9

**Pipeline extrait:**
```python
class DeepfakeDetector:
    ├─ __init__()      # Charge le modèle + setup
    └─ predict()       # Pipeline complet inférence
        ├─ extract_face_sequence()   (video_processor)
        ├─ estimate_rppg_from_faces() (rppg_extractor)
        ├─ Prépare les tensors
        ├─ Infère
        └─ Post-traite résultats
```

---

## 🎛️ Paramètres Defaults

Tous les paramètres ont été extraits des notebooks:

| Paramètre | Valeur | Source |
|-----------|--------|--------|
| `FACE_SIZE` | 224 | Notebook 1, Cell 3 |
| `SEQ_LEN` | 16 | Notebook 2, Cell 3 |
| `CONFIDENCE_THRESHOLD` | 0.70 | Notebook 1, Cell 3 |
| `PADDING_RATIO` | 0.35 | Notebook 1, Cell 3 |
| `d_model` | 256 | Notebook 2, Cell 4 |
| `nhead` | 8 | Notebook 2, Cell 4 |
| `num_layers` | 3 | Notebook 2, Cell 4 |

---

## 🔍 Extraction du Checkpoint

Le modèle entraîné est sauvegardé via:

```python
# Notebook 2, Cell 9
torch.save(model.state_dict(), f"best_model_fold{fold+1}.pt")
```

Pour l'interface, on charge:
```python
# model.py
model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
```

**5-fold cross-validation**: 
- Chaque fold a son checkpoint
- L'interface utilise `best_model_fold4.pt` (défaut)
- Changeable via sidebar

---

## 📊 Mapping des Features

### Features de Fréquence (10 éléments)

Calculés dans `video_processor.compute_frequency_metrics()`:

```python
freq_features = [
    "fft_mean",              # 1
    "fft_std",               # 2
    "fft_center_mean",       # 3  (r <= 0.25)
    "fft_outer_mean",        # 4  (r >= 0.65)
    "fft_outer_to_center",   # 5  Ratio
    "dct_mean",              # 6
    "dct_std",               # 7
    "dct_center_mean",       # 8
    "dct_outer_mean",        # 9
    "dct_outer_to_center",   # 10 Ratio
]
```

### Features rPPG (6 éléments)

Calculés dans `rppg_extractor.rppg_features()`:

```python
rppg_features = [
    mean(signal),            # 1
    std(signal),             # 2
    max - min,               # 3
    Q75 - Q25,               # 4
    heart_rate_bpm,          # 5  (peak freq * 60)
    snr,                     # 6  (peak power / mean)
]
```

---

## 🎓 Choix de Design

### Pourquoi Streamlit?

✅ Rapide à développer
✅ Web interface prête à l'emploi
✅ Cache des ressources
✅ Pas d'API back-end requise

### Pourquoi structure modulaire?

✅ `video_processor.py` peut être réutilisé
✅ `rppg_extractor.py` peut être testé seul
✅ `model.py` peut charger n'importe quel checkpoint
✅ `inference.py` peut être utilisé sans GUI

### Exemple: Utilisation sans Streamlit

```python
from inference import DeepfakeDetector

detector = DeepfakeDetector("best_model_fold4.pt", device="cuda")
result = detector.predict("mon_video.mp4")
print(result)
```

---

## 🧪 Validations

### Tailles de tensors

À travers le pipeline (Batch=1, T=16):

```
Input video: Variable (H, W)
    ↓
Frames RGB: (16, 224, 224, 3)  [uint8]
Freq features: (16, 10)         [float32]
rPPG signal: (1,)               [float32]
    ↓
Model input:
  - frames: (1, 16, 3, 224, 224)
  - freq: (1, 16, 10)
  - rppg: (1, 6)
    ↓
Model output: (1, 2)            [logits]
    ↓
Probas: (1, 2)                  [0-1]
```

---

## 🔐 Compatibilité Backwards

Le modèle charge correctement:

```python
# Notebook 2, Cell 9 (training)
torch.save(model.state_dict(), "best_model_fold4.pt")

# model.py (interface)
model = CNNFreqRPPGTransformer(...)
model.load_state_dict(torch.load("best_model_fold4.pt", weights_only=True))
```

✅ Aucune conversion nécessaire
✅ Checkpoints directs réutilisables

---

## 📝 Résumé Extraction

| Composant | Notebook | Cells | Modification |
|-----------|----------|-------|-------------|
| Frequency | 1 | 3 | Refactorisé en classe |
| Face Detection | 1 | 4 | Ajout gestion d'erreurs |
| Video Processing | 1 | 3-4 | Nouvelle classe wrapper |
| rPPG | 2 | 3 | Module indépendant |
| Model | 2 | 4 | Extraction directe |
| Inference Loop | 2 | 9 | Classe `DeepfakeDetector` |
| Validation | 2 | 9 | Pipeline simplifié |

---

**Total**: ~2000 lignes de code extrait et restructuré ✨

