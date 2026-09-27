import streamlit as st
import cv2
import numpy as np
from PIL import Image

from detection_pipeline import run_pipeline
from severity_index import compute_severity

# Configure page to wide mode for a more desktop-like feel
st.set_page_config(page_title="Carrot Defect Detector", page_icon="🥕", layout="wide")

# Inject Custom CSS for a modern, CustomTkinter-style dark aesthetic
st.markdown("""
<style>
    /* Main header styling */
    .main-title {
        font-family: 'Segoe UI', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF8C00; /* Carrot Orange */
        margin-bottom: 0px;
    }
    .sub-title {
        font-family: 'Segoe UI', sans-serif;
        color: #A0A0A0;
        margin-bottom: 30px;
    }
    /* Style the tabs to look more like desktop buttons */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        background-color: rgba(255, 140, 0, 0.1);
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 140, 0, 0.2);
        border-bottom: 2px solid #FF8C00;
    }
    /* Metric Card Styling */
    div[data-testid="metric-container"] {
        background-color: #1E1E1E;
        border: 1px solid #333333;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    /* Button styling */
    div.stButton > button {
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<div class='main-title'>🥕 Carrot Defect Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-Powered Agricultural Quality Inspection System</div>", unsafe_allow_html=True)
st.markdown("---")

# Layout: Left column for Input, Right column for Output
col1, col2 = st.columns([1, 1.2], gap="large")

img_buffer = None
run_btn = False

with col1:
    st.subheader("📷 Image Input")
    tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Use Camera"])

    with tab1:
        uploaded_photo = st.file_uploader("Choose an image from your device", type=["jpg", "jpeg", "png"])
        if uploaded_photo:
            img_buffer = uploaded_photo

    with tab2:
        camera_photo = st.camera_input("Capture using device camera")
        if camera_photo:
            img_buffer = camera_photo

    if img_buffer is not None:
        image = Image.open(img_buffer)
        st.image(image, caption="Image Ready for Analysis", use_container_width=True)
        # Use full width button for modern UI feel
        run_btn = st.button("🔍 Run Inspection Pipeline", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Analysis Results")
    
    if img_buffer is None:
        st.info("Awaiting image input... Upload or capture a photo to initialize the pipeline.")
    else:
        if run_btn:
            with st.spinner("Analyzing image array..."):
                # Convert PIL image to OpenCV format (BGR) for your pipeline
                img_array = np.array(image)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                try:
                    # Execute your backend logic
                    results = run_pipeline(img_bgr)
                    grade, composite_score, precautions = compute_severity(results)

                    st.success("✅ Inspection Complete!")
                    
                    # Display metrics in styled grid cards
                    mcol1, mcol2, mcol3 = st.columns(3)
                    mcol1.metric("Defect Area", f"{results.get('defect_area', 0)}%")
                    mcol2.metric("Model Confidence", f"{results.get('confidence', 0)}%")
                    mcol3.metric("Severity Score", f"{composite_score}/100")
                    
                    st.markdown("---")
                    
                    # Highlighted Grade and Disease Box
                    st.markdown(f"### 🏆 Market Grade: **{grade}**")
                    st.error(f"**Detected Condition:** {results.get('disease_name', 'None Detected')}")
                    
                    # Treatment Section
                    st.markdown("#### 💊 Treatment & Precautions")
                    st.info(precautions)

                except Exception as e:
                    st.error(f"System Error during analysis: {e}")
