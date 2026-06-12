# 🎬 Deepfake Detection Interface

Interface web pour détecter si une vidéo est un deepfake ou authentique.

## 🎯 Caractéristiques

- **Détection multimodale**: Combine RGB, fréquences (FFT/DCT) et signal rPPG
- **Architecture avancée**: CNN ResNet-18 + Transformer avec cross-attention
- **Interface web intuitive**: Utilise Streamlit pour l'expérience utilisateur
- **Support vidéo**: MP4, AVI, MOV, MKV
- **Détection de visages**: RetinaFace ou MTCNN

## 📋 Prérequis

- Python 3.8+
- CUDA 11.0+ (optionnel pour GPU)
- 4GB RAM minimum
- 2GB espace disque

## 🚀 Installation rapide

### 1. Créer un environnement virtuel

```bash
python -m venv .venv
.venv\Scripts\activate.bat  # Windows
source .venv/bin/activate   # Linux/Mac
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

**Note**: Si vous avez une GPU NVIDIA CUDA, installez plutôt:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt --no-deps
```

### 3. Placer le modèle

Assurez-vous que le fichier `best_model_fold4.pt` se trouve dans le même répertoire que `app.py`.

```
interface/
├── app.py
├── best_model_fold4.pt  ← Ici!
├── video_processor.py
├── inference.py
├── model.py
├── rppg_extractor.py
└── requirements.txt
```

## 🏃 Lancer l'application

### Option 1: Avec le script batch (Windows)

Double-cliquez sur `run.bat`:

```bash
.\run.bat
```

### Option 2: Ligne de commande

```bash
# Activez l'environnement
.venv\Scripts\activate.bat  # Windows
source .venv/bin/activate   # Linux/Mac

# Lancez Streamlit
streamlit run app.py
```

L'interface s'ouvre automatiquement dans votre navigateur sur `http://localhost:8501`

## 📖 Utilisation

1. **Téléchargez une vidéo** en utilisant le formulaire dans l'interface
2. **Cliquez sur "Analyser la vidéo"**
3. **Attendez les résultats** (30-60 secondes selon la qualité vidéo)
4. **Consultez les résultats**:
   - Statut: DEEPFAKE ou AUTHENTIQUE
   - Probabilité totale
   - Détails des probabilités (réelle/fake)

## 📊 Architecture du modèle

```
Input: Vidéo
    ↓
[Frame Extraction] → 16 frames de 224×224 pixels
    ↓
┌───────────────────────────────────────────────┐
│ Multimodal Processing                         │
├───────────────────────────────────────────────┤
│ • CNN (EfficientNet 4B):     (B, T, 512) → (B, T, 256)
│ • Frequency Features:  (B, T, 10)  → (B, T, 256)
│ • rPPG Signal:         (B, 6)      → (B, 256)
└───────────────────────────────────────────────┘
    ↓
[Cross-Attention A]: Freq → CNN
    ↓
[Temporal Encoder]: Cohérence temporelle (Transformer)
    ↓
[Cross-Attention B]: rPPG → Temporal Features
    ↓
[Classifier Head]: (B, 2) → probas [Real, Fake]
    ↓
Output: Real/Fake + Confidence
```

## 🔧 Configuration

Dans la sidebar de l'interface:

- **Chemin du modèle**: Spécifiez le chemin vers le checkpoint
- **Device**: Choisissez CPU ou CUDA (GPU)





## 📁 Structure des fichiers

```
interface/
├── app.py                   # Application Streamlit
├── inference.py             # Module d'inférence
├── model.py                 # Architecture du modèle
├── video_processor.py       # Traitement vidéo et détection de visages
├── rppg_extractor.py        # Extraction du signal physiologique
├── requirements.txt         # Dépendances Python
├── run.bat                  # Script de lancement (Windows)
└── README.md               # Ce fichier
```

## 🎓 Sources des notebooks

Le code a été extrait et refactorisé à partir de:
- `pcdlastwork (1).ipynb`: Extraction de frames, détection de visages, analyse de fréquences
- `notebooke5ca2fbf26.ipynb`: Entraînement du modèle multimodal

## 📝 Licence

Projet éducatif pour la détection de deepfakes.

## 💡 Conseils de performance

1. **GPU**: Si disponible, augmente la vitesse 5-10x
2. **Qualité vidéo**: Meilleures détections avec vidéos HD
3. **Durée**: 5-30 secondes optimal
4. **Résolution**: Minimum 240p, idéalement 720p+

## 📞 Support

Pour les problèmes d'installation ou d'utilisation:
1. Vérifiez les prérequis
2. Consultez le section Troubleshooting
3. Vérifiez que tous les fichiers sont présents

---

**Version**: 1.0  
**Créé**: 2024  
**Modèle**: CNN + Transformer multimodal (5-fold CV)
