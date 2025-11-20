# app.py
"""
Premium Streamlit GUI for NeuroScan AI (Alzheimer MRI) — SHAP only (no Captum)
Place at project root. Assumes:
 - models/resnet18_model.pth
 - utils/gradcam.py
 - utils/difficulty.py
 - utils/mri_explainer.py
 - utils/shap_manager.py
 - model_definition.py (SafeResNet18)
"""

import os
# pragmatic workaround for OpenMP duplicate runtime on Windows (see console warnings)
# NOTE: This is a workaround — if you can remove duplicate OpenMP installs, do that for production.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import json
import time
from pathlib import Path

# -------------------------
# App configuration & paths
# -------------------------
st.set_page_config(page_title="NeuroScan AI", layout="wide", page_icon="🧠")

PROJECT_ROOT = Path(".")
MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_model.pth"
SAMPLE_DIR = PROJECT_ROOT / "sample_data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SHAP_OUTPUT_DIR = OUTPUT_DIR / "shap_analysis"
GRADCAM_OUTPUT_DIR = OUTPUT_DIR / "gradcam_analysis"
MRI_OUTPUT_DIR = OUTPUT_DIR / "mri_explanations"

for p in (OUTPUT_DIR, SHAP_OUTPUT_DIR, GRADCAM_OUTPUT_DIR, MRI_OUTPUT_DIR):
    os.makedirs(p, exist_ok=True)

CLASS_NAMES = ['Non Demented', 'Very mild Dementia', 'Mild Dementia', 'Moderate Dementia']

# -------------------------
# Premium CSS
# -------------------------
def load_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

        :root{
            --bg-1: #071126;
            --bg-2: #0b1a28; /* slightly closer to bg-1 for a gentler transition */
            --glass-weak: rgba(255,255,255,0.02);
            --glass-strong: rgba(255,255,255,0.03);
            --muted: #9bb0c7;
            --text: #dfeaf6;
            --accent-a: rgba(140,170,200,0.04);
            --accent-b: rgba(190,170,220,0.03);
            --accent-edge: rgba(160,190,220,0.05);
        }

        .stApp {
            /* Subtle layered background: soft linear gradient + faint radial highlights */
            background:
                radial-gradient(700px 300px at 50% 8%, rgba(140,170,200,0.02), transparent 45%),
                linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
            font-family: 'Inter', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
            color: var(--text);
            min-height: 100vh;
        }

        .block-container { padding: 1.6rem 1.2rem; }

        /* Glass card */
        .glass-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.014), rgba(255,255,255,0.008));
            border: 1px solid var(--glass-weak);
            backdrop-filter: blur(6px) saturate(105%);
            -webkit-backdrop-filter: blur(6px) saturate(105%);
            padding: 18px;
            border-radius: 12px;
            box-shadow: 0 6px 18px rgba(4,10,20,0.36);
            color: var(--text);
            transition: transform .14s ease, box-shadow .14s ease;
        }
        .glass-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 28px rgba(4,10,20,0.44);
        }

        /* Title & subtitle */
        .title {
            font-size: 30px;
            font-weight: 800;
            letter-spacing: -0.4px;
            margin: 0;
            background: linear-gradient(90deg, rgba(140,170,200,0.10), rgba(190,170,220,0.08));
            -webkit-background-clip: text;
            color: transparent;
            display:inline-block;
        }
        .subtitle {
            color: var(--muted);
            margin-top: 6px;
            margin-bottom: 0;
            font-weight: 500;
            font-size: 0.95rem;
        }

        .badge {
            padding: 7px 12px;
            border-radius: 999px;
            font-weight: 700;
            color: #071025;
            background: rgba(255,255,255,0.05);
            box-shadow: inset 0 -3px 8px rgba(0,0,0,0.04);
            font-size: 0.95rem;
        }

        .small-muted { color: var(--muted); font-size: 0.9rem; }

        img {
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.02);
        }

        /* Tabs */
        .stTabs [role="tab"] { color: var(--muted); }
        .stTabs [role="tab"][aria-selected="true"] {
            color: var(--text);
            font-weight:700;
            box-shadow: inset 0 -2px 0 var(--accent-edge);
        }

        /* Buttons & download controls */
        .stButton>button, .stDownloadButton>button, .st-ae {
            background: transparent;
            color: var(--text) !important;
            border: 1px solid var(--glass-weak);
            padding: 8px 12px;
            border-radius: 10px;
            font-weight: 600;
            box-shadow: 0 6px 12px rgba(3,9,23,0.18);
            transition: background .12s ease, transform .08s ease;
        }
        .stButton>button:hover, .stDownloadButton>button:hover {
            background: rgba(255,255,255,0.015);
            transform: translateY(-2px);
        }

        .brand-accent {
            height: 3px;
            border-radius: 3px;
            background: linear-gradient(90deg, var(--accent-a), var(--accent-b));
            opacity: 0.10;
            margin-top: 10px;
            margin-bottom: 14px;
        }

        @media (max-width: 640px) {
            .title { font-size: 20px; }
            .block-container { padding: 0.9rem; }
            .glass-card { padding: 12px; border-radius: 10px; }
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(255,255,255,0.012), rgba(255,255,255,0.006));
            border: 1px solid rgba(255,255,255,0.02);
            padding: 14px;
            border-radius: 10px;
            box-shadow: 0 8px 22px rgba(2,6,23,0.38);
            margin: 10px;
        }

        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3,
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] label {
            color: var(--text);
        }
        section[data-testid="stSidebar"] .stMarkdown p { color: var(--muted); margin-top: 6px; }

        /* Inputs look compact and glassy */
        section[data-testid="stSidebar"] .stFileUploader,
        section[data-testid="stSidebar"] .stSelectbox,
        section[data-testid="stSidebar"] .stMultiSelect,
        section[data-testid="stSidebar"] .stRadio,
        section[data-testid="stSidebar"] .stSlider {
            background: rgba(255,255,255,0.01);
            border: 1px solid rgba(255,255,255,0.015);
            padding: 8px;
            border-radius: 8px;
            margin-bottom: 10px;
        }

        section[data-testid="stSidebar"] .stButton>button,
        section[data-testid="stSidebar"] .stDownloadButton>button {
            width: 100%;
            background: transparent;
            color: var(--text) !important;
            border-radius: 10px;
            padding: 8px 12px;
            font-weight: 600;
            border: 1px solid rgba(255,255,255,0.02);
            box-shadow: 0 6px 12px rgba(3,9,23,0.12);
        }

        section[data-testid="stSidebar"] .stCheckbox>div,
        section[data-testid="stSidebar"] .stSwitch>div {
            padding: 6px 4px;
            border-radius: 8px;
        }
        section[data-testid="stSidebar"] .stCheckbox>div label,
        section[data-testid="stSidebar"] .stSwitch>div label {
            color: var(--text);
            font-weight: 600;
        }

        section[data-testid="stSidebar"] input[type="file"] {
            color: var(--muted);
        }

        section[data-testid="stSidebar"] .small-muted {
            color: var(--muted);
            font-size: 0.85rem;
        }

        /* Fallback selectors for nested sidebar divs used by Streamlit */
        [data-testid="stSidebar"] .css-1d391kg, [data-testid="stSidebar"] .css-1oe6wy0 {
            background: transparent;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

load_css()

# -------------------------
# Transforms (same as training)
# -------------------------
transform = transforms.Compose([
    transforms.Resize((248, 496)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------
# Load model and utils (cached)
# -------------------------
@st.cache_resource
def load_model_and_utils():
    info = {"model": None, "gradcam": None, "difficulty": None, "mri_explainer": None, "shap_manager": None, "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}
    device = info["device"]
    # load model_definition
    try:
        from model_definition import SafeResNet18
        model = SafeResNet18(num_classes=4).to(device)
        if MODEL_PATH.exists():
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
        else:
            # still provide model object (random weights) but warn in UI
            model.eval()
        info["model"] = model
    except Exception as e:
        st.error(f"Error loading model_definition or weights: {e}")
        return info

    # load utilities if available (fail gracefully)
    try:
        from utils.gradcam import GradCAMAnalyzer
        info["gradcam"] = GradCAMAnalyzer(model, device, CLASS_NAMES)
    except Exception as e:
        st.warning(f"GradCAM not available: {e}")

    try:
        from utils.difficulty import DifficultyAnalyzer
        info["difficulty"] = DifficultyAnalyzer(model, device, CLASS_NAMES)
    except Exception as e:
        st.warning(f"Difficulty analyzer not available: {e}")

    try:
        from utils.mri_explainer import MRIAlzheimerExplainer
        info["mri_explainer"] = MRIAlzheimerExplainer(model, device, CLASS_NAMES)
    except Exception as e:
        st.warning(f"MRI explainer not available: {e}")

    try:
        from utils.shap_manager import SHAPInteractiveManager
        info["shap_manager"] = SHAPInteractiveManager(model, device, CLASS_NAMES, str(SHAP_OUTPUT_DIR))
    except Exception as e:
        # shap_manager should handle shap import internally and fallback if shap missing
        st.warning(f"SHAP manager unavailable: {e}")

    return info

helpers = load_model_and_utils()
model = helpers.get("model")
gradcam = helpers.get("gradcam")
difficulty = helpers.get("difficulty")
mri_explainer = helpers.get("mri_explainer")
shap_manager = helpers.get("shap_manager")
DEVICE = helpers.get("device")

# -------------------------
# Top header
# -------------------------
st.markdown("<div style='display:flex;align-items:center;gap:18px'>", unsafe_allow_html=True)
st.markdown("<div><h1 class='title'>NeuroScan AI</h1><div class='subtitle'>Alzheimer MRI analysis — ResNet18 · Explainable AI (Grad-CAM & SHAP)</div></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("Configuration")
st.sidebar.caption("Toggle explainability and input options")

show_gradcam = st.sidebar.checkbox("Grad-CAM heatmap", True)
show_shap = st.sidebar.checkbox("SHAP explainability", False)
show_mri = st.sidebar.checkbox("Clinical explanation", True)
show_difficulty = st.sidebar.checkbox("Difficulty hint", True)

uploaded = st.sidebar.file_uploader("Upload MRI scan (jpg/png)", type=["jpg", "png", "jpeg"])
sample_files = sorted([p for p in (SAMPLE_DIR.exists() and SAMPLE_DIR or []).glob("*.*")] ) if SAMPLE_DIR.exists() else []
if sample_files:
    sel = st.sidebar.selectbox("Or choose sample", ["-- none --"] + [p.name for p in sample_files])
    if sel != "-- none --" and uploaded is None:
        uploaded = open(SAMPLE_DIR / sel, "rb")

# helper to preprocess
def load_and_preprocess(fileobj):
    img = Image.open(fileobj).convert("RGB")
    tensor = transform(img)
    return img, tensor

# Main
if not uploaded:
    st.markdown("<div class='glass-card' style='text-align:center'><h3>Ready to analyze</h3><p class='small-muted'>Upload an MRI on the left to start — the app will show prediction, Grad-CAM and SHAP results.</p></div>", unsafe_allow_html=True)
else:
    try:
        img, tensor = load_and_preprocess(uploaded)
        filename = getattr(uploaded, "name", f"uploaded_{int(time.time())}.png")

        # Top row: image + summary
        col1, col2 = st.columns([1.2, 2])
        with col1:
            st.image(img, caption="Input MRI", use_container_width=True)

        # Prediction
        if model is None:
            st.error("Model not loaded — place model weights at 'models/resnet18_model.pth'.")
            pred_idx = None
            conf_val = None
        else:
            model.eval()
            with torch.no_grad():
                batch = tensor.unsqueeze(0).to(DEVICE)
                outputs = model(batch)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
                pred_idx = int(pred.item())
                conf_val = float(conf.item())

        with col2:
            if pred_idx is not None:
                badge_color = "#4ade80" if pred_idx == 0 else ("#facc15" if pred_idx == 1 else "#fb7185")
                st.markdown(f"<div class='glass-card'><h3>Result</h3><div><span class='badge' style='background:{badge_color};color:black'>{CLASS_NAMES[pred_idx]}</span> <span class='small-muted' style='margin-left:12px'>Confidence: {conf_val:.2%}</span></div><p class='small-muted' style='margin-top:10px'>Model: ResNet18</p></div>", unsafe_allow_html=True)
            else:
                st.info("No prediction available")

        # Tabs: Grad-CAM, SHAP, MRI report, Difficulty
        tab1, tab2, tab3, tab4 = st.tabs(["Grad-CAM", "SHAP", "Clinical report", "Difficulty"])

        # Grad-CAM
        with tab1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Grad-CAM")
            if gradcam is None:
                st.warning("Grad-CAM utility not loaded.")
            elif not show_gradcam:
                st.info("Enable Grad-CAM in the sidebar.")
            else:
                try:
                    heatmap, _ = gradcam.generate_heatmap(tensor, pred_idx)
                    fig = gradcam.visualize_gradcam(tensor, filename, target_class=pred_idx, save_path=str(GRADCAM_OUTPUT_DIR / f"gradcam_{Path(filename).stem}.png"))
                    st.pyplot(fig)
                    # download
                    saved = GRADCAM_OUTPUT_DIR / f"gradcam_{Path(filename).stem}.png"
                    if saved.exists():
                        with open(saved, "rb") as fh:
                            st.download_button("Download Grad-CAM PNG", fh.read(), file_name=saved.name, mime="image/png")
                except Exception as e:
                    st.error(f"Grad-CAM failed: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        # SHAP
        with tab2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("SHAP Explainability")
            if shap_manager is None:
                st.warning("SHAP manager not loaded.")
            elif not show_shap:
                st.info("Enable SHAP in the sidebar (may be slow).")
            else:
                with st.spinner("Running SHAP (or fallback) — this may take a few seconds..."):
                    try:
                        # Prepare background/test if needed
                        if hasattr(shap_manager, "prepare_shap_data_from_single_image"):
                            shap_manager.prepare_shap_data_from_single_image(tensor.unsqueeze(0))
                        result = shap_manager.analyze_single_image(tensor, filename, save_prefix=f"shap_{Path(filename).stem}")
                        # show image
                        if "shap_file" in result and Path(result["shap_file"]).exists():
                            st.image(result["shap_file"], caption="SHAP result", use_container_width=True)
                            with open(result["shap_file"], "rb") as fh:
                                st.download_button("Download SHAP PNG", fh.read(), file_name=Path(result["shap_file"]).name, mime="image/png")
                        else:
                            st.info("SHAP produced no PNG output — check logs.")
                    except Exception as e:
                        st.error(f"SHAP analysis failed: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        # Clinical report
        with tab3:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Clinical-style Explanation")
            if mri_explainer is None:
                st.warning("MRI explainer not available.")
            elif not show_mri:
                st.info("Enable clinical explanation in the sidebar.")
            else:
                try:
                    # explain and save figure into MRI_OUTPUT_DIR
                    mri_explainer.explain_mri_findings(tensor, filename, pred_idx)
                    expl = MRI_OUTPUT_DIR / f"mri_explanation_{CLASS_NAMES[pred_idx].replace(' ', '_')}.png"
                    if expl.exists():
                        st.image(str(expl), caption="Clinical Explanation", use_container_width=True)
                        with open(expl, "rb") as fh:
                            st.download_button("Download Clinical PNG", fh.read(), file_name=expl.name, mime="image/png")
                    else:
                        st.info("No clinical figure generated.")
                except Exception as e:
                    st.error(f"MRI explainer error: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        # Difficulty
        with tab4:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Model Certainty & Difficulty")
            if difficulty is None:
                st.warning("Difficulty analyzer not loaded.")
            else:
                try:
                    if conf_val is not None:
                        if conf_val > 0.8:
                            st.success("Easy: High confidence")
                        elif conf_val < 0.6:
                            st.error("Hard: Low confidence — recommend expert review")
                        else:
                            st.warning("Medium difficulty")
                        st.metric("Confidence", f"{conf_val:.4f}")
                except Exception as e:
                    st.error(f"Difficulty failed: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        # Footer JSON
        with st.expander("Download JSON report"):
            report = {"image": filename, "prediction": CLASS_NAMES[pred_idx] if pred_idx is not None else None,
                      "confidence": conf_val, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
            st.json(report)
            st.download_button("Download report JSON", data=json.dumps(report, indent=2), file_name=f"report_{Path(filename).stem}.json")

    except Exception as e:
        st.error(f"Unhandled error: {e}")
