"""
Interface de démonstration Streamlit pour la détection de deepfakes.
Cette version fonctionne sans dépendre du fichier du modèle.
"""

import streamlit as st

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎬 Détecteur de Deepfakes")
st.markdown("""
Cette application utilise un modèle de deep learning pour déterminer si une vidéo 
contient un deepfake ou si elle est authentique.

**Modèle**: CNN multimodal + Transformer (combines spatial, frequency, et physiological signals)
""")

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR - Configuration
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Chemin du checkpoint
    checkpoint_path = st.text_input(
        "Chemin du modèle:",
        value="best_model_fold4.pt",
        help="Chemin vers le fichier .pt du modèle entraîné"
    )
    
    # Device
    device = st.radio(
        "Device:",
        options=["cpu", "cuda"],
        help="cpu pour tous, cuda si GPU disponible"
    )
    
    # Information
    st.markdown("---")
    st.subheader("ℹ️ Information")
    st.markdown("""
    - **Formats supportés**: MP4, AVI, MOV, MKV
    - **Durée recommandée**: 5-30 secondes
    - **Résolution**: Hauteur minimale 240px
    - **Temps de traitement**: ~30-60 sec par vidéo
    """)


# ════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════════════════════════════════════════

st.subheader("📹 Téléchargez une vidéo")

uploaded_file = st.file_uploader(
    "Choisissez un fichier vidéo:",
    type=["mp4", "avi", "mov", "mkv", "webm"],
    help="Formats: MP4, AVI, MOV, MKV, WebM"
)

if uploaded_file is not None:
    st.markdown("---")
    
    # Show video
    st.video(uploaded_file, start_time=0)
    
    # Demo results
    if st.button("🔍 Analyser la vidéo (DÉMO)", type="primary"):
        st.info("⏳ Analyse simulée... (en démo)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("✅ VIDÉO AUTHENTIQUE")
        
        with col2:
            st.metric("Probabilité", "87.3%", delta=None)
        
        with col3:
            st.write("**Status**: :green[Positif]")
        
        st.subheader("Détails des probabilités")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("🎬 Probabilité RÉELLE", "87.30%")
        
        with col2:
            st.metric("🚨 Probabilité FAKE", "12.70%")
        
        st.subheader("Visualisation des probabilités")
        st.progress(0.1273, text="12.73% - Probabilité FAKE")
        
        st.info("💡 Détecté comme REAL (confiance: 87.3%)")
        
        st.success("""
        **Note**: Ceci est une démonstration. 
        Pour utiliser le vrai modèle:
        
        1. Téléchargez le fichier `best_model_fold4.pt` 
        2. Placez-le dans le dossier `interface/`
        3. Relancez l'application avec `streamlit run app.py`
        """)

# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
### 📝 Notes importantes
- Ce modèle n'est pas 100% précis. Utilisez comme outil de détection préalable.
- Les résultats dépendent de la qualité de la vidéo.
- Pour les cas sensibles, veuillez faire valider par un expert.

### 🔧 Technologie
- **Framework**: PyTorch
- **Architecture**: CNN ResNet-18 + Transformer
- **Modèles**: Multimodal (RGB + Fréquence + rPPG)
- **Détection de visages**: RetinaFace / MTCNN

---
*Créé à partir des notebooks de détection de deepfakes*
""")
