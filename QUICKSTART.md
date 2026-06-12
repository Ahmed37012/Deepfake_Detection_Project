# 🚀 Démarrage Rapide

Suivez ces étapes pour commencer à utiliser l'interface de détection de deepfakes.

## 1️⃣ Installation (5 min)

### Première fois seulement:

```bash
# Double-cliquez sur install.bat (Windows)
# Ou tapez dans le terminal:
python install_dependencies.bat
```

**Ou manuellement:**

```bash
# Créer l'environnement virtuel
python -m venv .venv

# L'activer
.venv\Scripts\activate.bat  # Windows
source .venv/bin/activate   # Linux/Mac

# Installer les paquets
pip install -r requirements.txt
```

## 2️⃣ Placer le modèle

Téléchargez le fichier `best_model_fold4.pt` et placez-le dans le répertoire `interface/`:

```
interface/
├── app.py
├── best_model_fold4.pt  ← Placez-le ici!
├── video_processor.py
└── ...
```

## 3️⃣ Lancer l'interface

### Option A: Double-cliquez sur `run.bat` (Windows)

### Option B: Terminal

```bash
# Activez l'environnement
.venv\Scripts\activate.bat  # Windows

# Lancez l'interface
streamlit run app.py
```

L'interface s'ouvrira automatiquement sur `http://localhost:8501`

## 4️⃣ Utiliser l'interface

1. **Cliquez sur "Browse files"** et sélectionnez une vidéo
2. **Cliquez sur "Analyser la vidéo"**
3. **Attendez les résultats** (30-60 sec)

## 📊 Résultats

L'interface affiche:
- ✅ **AUTHENTIQUE** ou 🚨 **DEEPFAKE**
- **Confiance**: 0-100%
- **Probabilités détaillées**:
  - Réelle: X%
  - Fake: Y%

## 🧪 Tester sans l'interface web

```bash
# Activez l'environnement
.venv\Scripts\activate.bat

# Testez une vidéo
python test_inference.py mon_video.mp4 --device cpu
```

## ✓ Vérifier le setup

```bash
# Vérifie que tout est correct
python check_setup.py
```

## 🎥 Formats vidéo supportés

- MP4 ✓
- AVI ✓
- MOV ✓
- MKV ✓
- WebM ✓

## ⚠️ Conseils pour les meilleurs résultats

✅ **BON:**
- Vidéo 720p+
- Durée 5-30 secondes
- Visage visible et net
- Bon éclairage

❌ **MAUVAIS:**
- Vidéo basse résolution
- Durée < 2 sec ou > 2 min
- Visage trop petit ou flou
- Éclairage très faible

## 🆘 Problèmes courants

### "best_model_fold4.pt not found"
→ Placez le fichier du modèle dans le répertoire

### "ModuleNotFoundError: No module named..."
→ Exécutez: `pip install -r requirements.txt`

### "No faces detected"
→ La vidéo ne contient pas de visages visibles

### "CUDA out of memory"
→ Utilisez CPU: lancez avec `--device cpu`

### La prédiction prend trop longtemps
→ C'est normal (30-60 sec). Soyez patient!

## 📂 Fichiers principaux

- `app.py` - Interface Streamlit
- `inference.py` - Moteur de prédiction
- `model.py` - Architecture du modèle
- `video_processor.py` - Traitement vidéo
- `rppg_extractor.py` - Extraction du signal physiologique

## 🔧 Configuration

### Via Streamlit (dans l'interface):

Sidebar ⚙️ **Configuration**:
- **Chemin du modèle** - Changez si vous utilisez un autre modèle
- **Device** - CPU ou CUDA (GPU)

## 💡 Pour aller plus loin

Voir `README.md` pour:
- Architecture complète du modèle
- Troubleshooting détaillé
- Limitations et précision
- Sources des données

## ✨ Résumé

```
install.bat → Configure l'environnement
↓
Placer best_model_fold4.pt
↓
run.bat → Lance l'interface
↓
Téléchargez vidéo → Cliquez "Analyser" → Voyez les résultats
```

---

**✓ Vous êtes prêt!** 🎉

Lancez maintenant: `.\run.bat`

