import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Import your existing backend logic
from detection_pipeline import run_pipeline
from severity_index import compute_severity

# Configure page to wide mode for a more desktop-like feel
st.set_page_config(page_title="Carrot Defect Detector", page_icon="🥕", layout="wide")

# Inject Custom CSS for a modern, dark aesthetic
st.markdown("""
<style>
    .main-title {
        font-family: 'Segoe UI', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF8C00;
        margin-bottom: 0px;
    }
    .sub-title {
        font-family: 'Segoe UI', sans-serif;
        color: #A0A0A0;
        margin-bottom: 30px;
    }
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
    div[data-testid="metric-container"] {
        background-color: #1E1E1E;
        border: 1px solid #333333;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div.stButton > button {
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
        font-size: 18px;
    }
    .report-box {
        background-color: #2b2b2b;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF8C00;
        margin-top: 15px;
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
        run_btn = st.button("🔍 Run Inspection Pipeline", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Analysis Results")
    
    if img_buffer is None:
        st.info("Awaiting image input... Upload or capture a photo to initialize the pipeline.")
    else:
        if run_btn:
            with st.spinner("Analyzing image array..."):
                img_array = np.array(image)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                try:
                    # Execute backend logic
                    results = run_pipeline(img_bgr)
                    
                    # Fix: Use getattr() to safely pull attributes from the PipelineResult object
                    blemish_ratio = getattr(results, 'blemish_pixel_ratio', 0.0) 
                    grade, composite_score, precautions = compute_severity(results, blemish_ratio)

                    # Determine if healthy based on the disease name
                    disease_name = getattr(results, 'disease_name', 'None Detected')
                    is_healthy = str(disease_name).lower() in ['none', 'none detected', 'healthy']

                    st.success("✅ Inspection Complete!")
                    
                    # Display metrics
                    mcol1, mcol2, mcol3 = st.columns(3)
                    defect_area = getattr(results, 'defect_area', 0)
                    confidence = getattr(results, 'confidence', 0)
                    
                    mcol1.metric("Defect Area", f"{defect_area}%")
                    mcol2.metric("Model Confidence", f"{confidence}%")
                    mcol3.metric("Severity Score", f"{composite_score}/100")
                    
                    # Comprehensive Report
                    st.markdown("### 📋 Final Inspection Report")
                    
                    # Status Box
                    if is_healthy:
                        st.success(f"**Health Status:** ✅ Healthy Carrot\n\n**Market Grade:** {grade}")
                    else:
                        st.error(f"**Health Status:** ⚠️ Disease Detected - {disease_name}\n\n**Market Grade:** {grade}")
                    
                    # Treatment and Precautions Box
                    st.markdown("""
                    <div class="report-box">
                        <h4 style='margin-top:0px;'>💊 Treatment & Precautions</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.info(precautions)

                except Exception as e:
                    st.error(f"System Error during analysis: {e}")
