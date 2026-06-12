"""Interface Streamlit moderne pour l'analyse de deepfakes."""

from pathlib import Path
import os
import tempfile

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw

from inference import DeepfakeDetector


st.set_page_config(
    page_title="Deepfake Detector Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        :root {
            --bg-primary: #0f1419;
            --bg-secondary: #1a1f2e;
            --panel: #1f2937;
            --panel-hover: #2d3748;
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
            --text-tertiary: #94a3b8;
            --accent: #3b82f6;
            --accent-alt: #60a5fa;
            --accent-dark: #1e40af;
            --accent-light: #0ea5e9;
            --success: #10b981;
            --danger: #dc2626;
            --warning: #f59e0b;
        }

        * {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .stApp {
            background:
                radial-gradient(ellipse at 20% 50%, rgba(59, 130, 246, 0.15), transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(139, 92, 246, 0.1), transparent 50%),
                linear-gradient(135deg, #0f1419 0%, #1a1f2e 50%, #141829 100%);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif;
            letter-spacing: 0.5px;
            line-height: 1.65;
        }
        [data-testid="stHeader"] { background: rgba(0,0,0,0); }
        [data-testid="stToolbar"] { right: 0.75rem; }
        [data-testid="stSidebar"] { display: none; }
        .block-container {
            max-width: 1440px;
            padding-top: 3.5rem;
            padding-bottom: 4rem;
        }

        /* Hero Section */
        .hero {
            padding: 4rem 4rem 3.5rem 4rem;
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: 40px;
            background: linear-gradient(135deg, rgba(31, 41, 55, 0.8) 0%, rgba(37, 45, 61, 0.6) 50%, rgba(25, 35, 50, 0.8) 100%);
            box-shadow:
                0 0 0 1px rgba(59, 130, 246, 0.1),
                0 8px 16px rgba(15, 20, 25, 0.3),
                0 20px 60px rgba(59, 130, 246, 0.08),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(20px);
            animation: fadeInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1) forwards;
        }

        .hero::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 4px;
            background: linear-gradient(180deg, var(--accent) 0%, var(--accent-light) 50%, rgba(59, 130, 246, 0) 100%);
            box-shadow: 0 0 24px rgba(59, 130, 246, 0.5), 0 0 48px rgba(59, 130, 246, 0.3);
        }

        .hero::after {
            content: "";
            position: absolute;
            top: -30%;
            right: -5%;
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, rgba(59, 130, 246, 0.08), transparent);
            border-radius: 50%;
            pointer-events: none;
            filter: blur(40px);
        }

        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.28em;
            font-size: 0.8rem;
            background: linear-gradient(90deg, var(--accent) 0%, var(--accent-light) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 1.2rem;
            font-weight: 950;
            opacity: 0.95;
        }

        .hero h1 {
            margin: 0;
            font-size: 4rem;
            line-height: 1.15;
            color: var(--text-primary);
            font-weight: 975;
            letter-spacing: -1.5px;
            background: linear-gradient(135deg, #f8fafc 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .hero p {
            margin-top: 1.5rem;
            max-width: 950px;
            color: var(--text-secondary);
            font-size: 1.15rem;
            line-height: 1.75;
            font-weight: 400;
            letter-spacing: 0.2px;
        }

        .pill-row {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            margin-top: 2.5rem;
        }

        .pill {
            padding: 0.8rem 1.6rem;
            border-radius: 9999px;
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(6, 182, 212, 0.1) 100%);
            border: 1.5px solid rgba(59, 130, 246, 0.3);
            color: var(--accent-alt);
            font-size: 0.9rem;
            font-weight: 700;
            cursor: default;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.1);
            letter-spacing: 0.3px;
            backdrop-filter: blur(10px);
        }

        .pill:hover {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.25) 0%, rgba(6, 182, 212, 0.15) 100%);
            box-shadow: 0 8px 20px rgba(59, 130, 246, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.15);
            transform: translateY(-4px);
            border-color: rgba(59, 130, 246, 0.5);
        }

        /* Panel & Cards */
        .panel {
            padding: 2rem 2.2rem;
            border-radius: 28px;
            border: 1px solid rgba(59, 130, 246, 0.15);
            background: linear-gradient(135deg, rgba(31, 41, 55, 0.6) 0%, rgba(37, 45, 61, 0.4) 100%);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2), 0 8px 24px rgba(59, 130, 246, 0.05), inset 0 1px 0 rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(15px);
            animation: fadeInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1) 0.1s forwards;
            opacity: 0;
        }

        .panel:hover {
            border-color: rgba(59, 130, 246, 0.3);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3), 0 12px 32px rgba(59, 130, 246, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.08);
            background: linear-gradient(135deg, rgba(31, 41, 55, 0.8) 0%, rgba(37, 45, 61, 0.6) 100%);
        }

        .section-title {
            font-size: 1.5rem;
            font-weight: 900;
            color: var(--text-primary);
            margin: 0 0 1.5rem 0;
            letter-spacing: -0.7px;
            background: linear-gradient(90deg, #f8fafc 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .muted {
            color: var(--text-secondary);
            font-size: 1rem;
            line-height: 1.7;
            letter-spacing: 0.2px;
        }

        /* Decision Card */
        .decision {
            padding: 3rem 3.5rem;
            border-radius: 40px;
            border: 1px solid rgba(59, 130, 246, 0.2);
            background: linear-gradient(135deg, rgba(31, 41, 55, 0.8) 0%, rgba(37, 45, 61, 0.6) 50%, rgba(25, 35, 50, 0.8) 100%);
            box-shadow: 0 0 0 1px rgba(59, 130, 246, 0.1), 0 12px 32px rgba(15, 20, 25, 0.4), 0 32px 80px rgba(59, 130, 246, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.05);
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(20px);
            animation: fadeInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1) 0.2s forwards;
            opacity: 0;
        }

        .decision::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 5px;
            background: linear-gradient(180deg, var(--success) 0%, #10b981 50%, rgba(16, 185, 129, 0) 100%);
            box-shadow: 0 0 24px rgba(16, 185, 129, 0.4), 0 0 48px rgba(16, 185, 129, 0.2);
        }

        .decision::after {
            content: "";
            position: absolute;
            top: -100%;
            right: -15%;
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(16, 185, 129, 0.08), transparent);
            border-radius: 50%;
            pointer-events: none;
            filter: blur(40px);
        }

        .decision.fake {
            border-color: rgba(220, 38, 38, 0.2);
        }

        .decision.fake::before {
            background: linear-gradient(180deg, var(--danger) 0%, #f87171 50%, rgba(220, 38, 38, 0) 100%);
            box-shadow: 0 0 24px rgba(220, 38, 38, 0.4), 0 0 48px rgba(220, 38, 38, 0.2);
        }

        .decision.fake::after {
            background: radial-gradient(circle, rgba(220, 38, 38, 0.08), transparent);
        }

        .decision-kicker {
            text-transform: uppercase;
            letter-spacing: 0.24em;
            font-size: 0.8rem;
            color: var(--text-tertiary);
            margin-bottom: 1rem;
            font-weight: 900;
            opacity: 0.85;
        }

        .decision-title {
            font-size: 3rem;
            font-weight: 975;
            color: var(--text-primary);
            margin-bottom: 0.8rem;
            letter-spacing: -1.5px;
        }

        .decision.fake .decision-title {
            color: var(--danger);
        }

        .decision-subtitle {
            color: var(--text-secondary);
            margin-bottom: 2rem;
            font-size: 1.1rem;
            line-height: 1.7;
            letter-spacing: 0.2px;
        }

        .decision-score {
            font-size: 4rem;
            font-weight: 975;
            line-height: 1;
            color: var(--success);
            letter-spacing: -1.5px;
            background: linear-gradient(90deg, var(--success) 0%, #34d399 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .decision.fake .decision-score {
            background: linear-gradient(90deg, var(--danger) 0%, #f87171 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .mini-card {
            padding: 1.8rem 2rem;
            border-radius: 24px;
            border: 1px solid rgba(59, 130, 246, 0.15);
            background: linear-gradient(135deg, rgba(37, 45, 61, 0.5) 0%, rgba(31, 41, 55, 0.3) 100%);
            height: 100%;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15), 0 8px 16px rgba(59, 130, 246, 0.03), inset 0 1px 0 rgba(255, 255, 255, 0.05);
            cursor: default;
            backdrop-filter: blur(10px);
            position: relative;
        }

        .mini-card:hover {
            background: linear-gradient(135deg, rgba(37, 45, 61, 0.7) 0%, rgba(31, 41, 55, 0.5) 100%);
            border-color: rgba(59, 130, 246, 0.3);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25), 0 12px 28px rgba(59, 130, 246, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.08);
            transform: translateY(-4px);
        }

        .mini-card .label {
            font-size: 0.8rem;
            color: var(--text-tertiary);
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-weight: 900;
            opacity: 0.8;
        }

        .mini-card .value {
            font-size: 1.85rem;
            font-weight: 950;
            color: var(--text-primary);
            margin-top: 0.8rem;
            letter-spacing: -0.8px;
            background: linear-gradient(90deg, #f8fafc 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .mini-card .help {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-top: 0.8rem;
            line-height: 1.6;
            letter-spacing: 0.2px;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(180deg, rgba(59, 130, 246, 1) 0%, rgba(37, 99, 235, 1) 100%);
            color: #f8fafc;
            border: 1px solid rgba(59, 130, 246, 0.4);
            border-radius: 20px;
            font-weight: 950;
            box-shadow: 0 6px 16px rgba(59, 130, 246, 0.35), 0 12px 32px rgba(59, 130, 246, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.2);
            padding: 1.1rem 2rem;
            letter-spacing: 0.4px;
            cursor: pointer;
            position: relative;
            overflow: hidden;
            font-size: 1rem;
            text-transform: uppercase;
        }

        .stButton > button::before {
            content: "";
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.15), transparent);
            transition: left 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .stButton > button:hover {
            background: linear-gradient(180deg, rgba(96, 165, 250, 1) 0%, rgba(59, 130, 246, 1) 100%);
            box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4), 0 16px 40px rgba(59, 130, 246, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.3);
            transform: translateY(-4px);
            border-color: rgba(96, 165, 250, 0.6);
        }

        .stButton > button:hover::before {
            left: 100%;
        }

        .stButton > button:active {
            transform: translateY(-2px);
        }

        .stButton > button:disabled {
            background: linear-gradient(180deg, rgba(71, 85, 105, 1) 0%, rgba(55, 65, 81, 1) 100%);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            opacity: 0.6;
            cursor: not-allowed;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.8rem;
            background: rgba(37, 45, 61, 0.4);
            padding: 0.7rem;
            border-radius: 20px;
            border: 1px solid rgba(59, 130, 246, 0.15);
            backdrop-filter: blur(10px);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 16px;
            padding: 0.9rem 1.6rem;
            font-weight: 900;
            color: var(--text-secondary);
            letter-spacing: 0.3px;
            cursor: pointer;
            font-size: 0.98rem;
            text-transform: uppercase;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: var(--text-primary);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.3) 0%, rgba(6, 182, 212, 0.2) 100%);
            color: var(--accent-alt);
            font-weight: 950;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2), 0 8px 20px rgba(59, 130, 246, 0.08);
            border: 1px solid rgba(59, 130, 246, 0.3);
        }

        /* File Uploader */
        .stFileUploader {
            border: 2px dashed rgba(59, 130, 246, 0.3);
            border-radius: 24px;
            background: rgba(37, 45, 61, 0.3);
            padding: 1.5rem;
            backdrop-filter: blur(10px);
        }

        .stFileUploader:hover {
            background: rgba(59, 130, 246, 0.08);
            border-color: rgba(59, 130, 246, 0.5);
        }

        /* Progress Bar */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, var(--danger) 0%, #f87171 100%);
            border-radius: 10px;
            height: 8px;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }

        ::-webkit-scrollbar-track {
            background: transparent;
        }

        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, var(--accent) 0%, var(--accent-light) 100%);
            border-radius: 6px;
            border: 2px solid var(--bg-primary);
        }

        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, var(--accent-alt) 0%, var(--accent) 100%);
        }

        /* Animations */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.4); }
            50% { box-shadow: 0 0 40px rgba(59, 130, 246, 0.6); }
        }

        /* Alert Styling */
        .stAlert { border-radius: 20px !important; border: 1px solid rgba(59, 130, 246, 0.2) !important; background: rgba(37, 45, 61, 0.4) !important; backdrop-filter: blur(10px) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_CHECKPOINT = "best_model_fold4.pt"
UPLOAD_TYPES = ["mp4", "avi", "mov", "mkv", "webm"]


@st.cache_resource
def get_detector(checkpoint_path: str, device: str):
    return DeepfakeDetector(checkpoint_path=checkpoint_path, device=device)


def get_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def feature_label(name: str) -> str:
    mapping = {
        "fft_mean": "FFT mean",
        "fft_std": "FFT std",
        "fft_center_mean": "FFT center",
        "fft_outer_mean": "FFT outer",
        "fft_outer_to_center": "FFT ratio",
        "dct_mean": "DCT mean",
        "dct_std": "DCT std",
        "dct_center_mean": "DCT center",
        "dct_outer_mean": "DCT outer",
        "dct_outer_to_center": "DCT ratio",
    }
    return mapping.get(name, name.replace("_", " ").title())


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Deepfake analysis dashboard</div>
            <h1>Détection multimodale, claire et professionnelle.</h1>
            <p>
                Importez une vidéo et obtenez un diagnostic structuré avec les frames extraites,
                les visages détectés, l'analyse fréquentielle et la décision finale.
            </p>
            <div class="pill-row">
                <span class="pill">Analyse vidéo</span>
                <span class="pill">Visages extraits</span>
                <span class="pill">FFT / DCT</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, help_text: str) -> None:
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_image_grid(title: str, images, captions, columns: int = 4) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if not images:
        st.info("Aucune image disponible pour cette section.")
        return

    for start in range(0, len(images), columns):
        row_images = images[start : start + columns]
        row_captions = captions[start : start + columns]
        cols = st.columns(len(row_images))
        for idx, col in enumerate(cols):
            with col:
                st.image(row_images[idx], caption=row_captions[idx], use_container_width=True)


def render_frequency_section(artifacts: dict) -> None:
    st.markdown('<div class="section-title">Analyse fréquentielle</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="muted">Cartes FFT et DCT générées à partir des visages extraits. '
        'Les extraits affichés ci-dessous sont des échantillons de la séquence complète.</div>',
        unsafe_allow_html=True,
    )

    fft_maps = artifacts.get("fft_maps", [])
    dct_maps = artifacts.get("dct_maps", [])
    count = min(len(fft_maps), len(dct_maps))
    if count == 0:
        st.info("Aucune carte fréquentielle disponible.")
        return

    sample_indices = sorted({0, max(0, count // 2), count - 1}) if count > 2 else list(range(count))
    cols = st.columns(len(sample_indices))
    for col, index in zip(cols, sample_indices):
        with col:
            st.caption(f"Frame {index + 1}")
            st.image(fft_maps[index], caption="FFT", use_container_width=True)
            st.image(dct_maps[index], caption="DCT", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    names = artifacts.get("freq_feature_names", [])
    values = artifacts.get("freq_features_mean", [])
    if names and values:
        metric_cols = st.columns(5)
        for idx, (name, value) in enumerate(zip(names, values)):
            with metric_cols[idx % 5]:
                render_metric_card(feature_label(name), f"{float(value):.4f}", "Moyenne sur la séquence")


def render_decision(result: dict) -> None:
    is_fake = bool(result.get("is_fake"))
    confidence = float(result.get("confidence", 0.0))
    probabilities = result.get("probabilities", {})
    real_prob = float(probabilities.get("real", 0.0))
    fake_prob = float(probabilities.get("fake", 0.0))
    message = result.get("message", "")
    artifacts = result.get("artifacts", {})
    extraction_mode = artifacts.get("extraction_mode", "unknown")

    decision_class = "decision fake" if is_fake else "decision"
    decision_label = "Deepfake détecté" if is_fake else "Vidéo authentique"
    subtitle = "Le modèle privilégie la classe FAKE." if is_fake else "Le modèle privilégie la classe REAL."

    st.markdown(
        f"""
        <div class="{decision_class}">
            <div class="decision-kicker">Décision finale</div>
            <div class="decision-title">{decision_label}</div>
            <div class="decision-subtitle">{subtitle}</div>
            <div class="decision-score">{confidence:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("Probabilité réelle", f"{real_prob:.1%}", "Classe authentique")
    with col2:
        render_metric_card("Probabilité fake", f"{fake_prob:.1%}", "Classe manipulée")
    with col3:
        render_metric_card("Mode d'extraction", extraction_mode, "Pipeline utilisé pour la séquence")

    st.progress(float(fake_prob))
    st.info(message)


def main() -> None:
    render_hero()

    try:
        device = get_default_device()
        detector = get_detector(DEFAULT_CHECKPOINT, device)
    except Exception as exc:
        st.error(f"Impossible de charger le modèle: {exc}")
        st.stop()

    st.markdown('<div class="section-title">Analyse vidéo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="muted">Chargez une vidéo, puis lancez l’analyse. '
        'Aucune configuration manuelle n’est nécessaire.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choisir une vidéo",
        type=UPLOAD_TYPES,
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown(
            '<div class="panel"><div class="muted">'
            'Importez une vidéo pour afficher automatiquement les frames extraites, les visages détectés, '
            'les cartes FFT/DCT et la décision finale.'
            '</div></div>',
            unsafe_allow_html=True,
        )
        return

    # Create tabs for analysis
    tabs = st.tabs(["📹 Analyse vidéo", "👤 Visages extraits", "📊 FFT/DCT", "🎯 Décision"])

    with tabs[0]:
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Aperçu vidéo</div>', unsafe_allow_html=True)
            st.video(uploaded_file)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("<div class=\"section-title\">Lancer l'analyse</div>", unsafe_allow_html=True)
            st.markdown(
                '<div class="muted">Cliquez sur le bouton pour démarrer l\'analyse multimodale complète de la vidéo.</div>',
                unsafe_allow_html=True,
            )

            analyze_clicked = st.button(
                "🚀 Analyser la vidéo",
                type="primary",
                use_container_width=True,
            )

            if not analyze_clicked:
                st.markdown('</div>', unsafe_allow_html=True)
                st.stop()

            st.markdown('</div>', unsafe_allow_html=True)

            suffix = Path(uploaded_file.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_video_path = tmp_file.name

            try:
                with st.spinner("Analyse multimodale en cours..."):
                    result = detector.predict(tmp_video_path, verbose=True)

                if not result.get("success"):
                    st.error(result.get("message", "Erreur inconnue"))
                    st.stop()

                artifacts = result.get("artifacts", {})
                source_frames = artifacts.get("source_frames", [])
                face_frames = artifacts.get("face_frames", [])

                # Display results in tabs
                with tabs[0]:
                    st.markdown('<div class="section-title">Frames extraites</div>', unsafe_allow_html=True)
                    render_image_grid(
                        "",
                        source_frames,
                        [f"Frame {idx + 1}" for idx in range(len(source_frames))],
                        columns=4,
                    )

                with tabs[1]:
                    st.markdown('<div class="section-title">Visages extraits</div>', unsafe_allow_html=True)
                    render_image_grid(
                        "",
                        face_frames,
                        [f"Visage {idx + 1}" for idx in range(len(face_frames))],
                        columns=4,
                    )

                with tabs[2]:
                    st.markdown('<div class="section-title">Analyse fréquentielle</div>', unsafe_allow_html=True)
                    render_frequency_section(artifacts)

                with tabs[3]:
                    st.markdown('<div class="section-title">Résultat final</div>', unsafe_allow_html=True)
                    render_decision(result)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="panel"><div class="section-title">Résumé technique</div>'
                    '<div class="muted">'
                    'Le pipeline combine extraction de visages et analyse fréquentielle '
                    'avant une décision finale du réseau.'
                    '</div></div>',
                    unsafe_allow_html=True,
                )
            finally:
                if os.path.exists(tmp_video_path):
                    os.remove(tmp_video_path)


if __name__ == "__main__":
    main()
