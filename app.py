import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Import your existing backend logic
from detection_pipeline import run_pipeline
from severity_index import compute_severity

# Configure page to wide mode for a more desktop-like feel
st.set_page_config(page_title="Carrot Defect Detector", page_icon="🥕", layout="wide")

# Inject Custom CSS mapped directly to your CustomTkinter color palette
st.markdown("""
<style>
    /* Force Light Mode / Peach Background */
    .stApp {
        background-color: #FFF0E0;
    }
    
    /* General Typography */
    h1, h2, h3, p, span, label, .stMarkdown {
        color: #3D2B1F !important;
    }
    
    /* Header Styling */
    .main-title {
        font-family: 'Segoe UI', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0px;
        color: #3D2B1F;
    }
    .main-title span {
        color: #FF8C42; /* Carrot Orange Accent */
    }
    .sub-title {
        font-family: 'Segoe UI', sans-serif;
        color: #7A6050 !important;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    
    /* Sleek Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #FDE8D0;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF8C42 !important;
    }
    .stTabs [aria-selected="true"] div {
        color: #FFFFFF !important;
        font-weight: bold;
    }
    
    /* Soft Metric Tiles */
    div[data-testid="metric-container"] {
        background-color: #FDE8D0 !important;
        border: 1.5px solid #F0E0D0 !important;
        padding: 15px;
        border-radius: 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    /* Modern Orange Primary Button */
    div.stButton > button {
        border-radius: 14px;
        height: 50px;
        font-weight: bold;
        font-size: 16px;
        background-color: #FF8C42 !important;
        color: white !important;
        border: none;
        box-shadow: 0 4px 10px rgba(255, 140, 66, 0.3);
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        background-color: #E07030 !important;
        box-shadow: 0 4px 15px rgba(224, 112, 48, 0.4);
    }
    
    /* File Uploader styling */
    [data-testid="stFileUploadDropzone"] {
        background-color: #FFFFFF;
        border: 2px dashed #FF8C42;
        border-radius: 14px;
    }
    
    /* White Card for Report Box */
    .report-box {
        background-color: #FFFFFF;
        padding: 25px;
        border-radius: 14px;
        margin-top: 15px;
        border: 1.5px solid #F0E0D0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<div class='main-title'><span>🥕</span> Carrot Defect Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-powered quality inspection & grading</div>", unsafe_allow_html=True)
st.markdown("---")

# Layout: Left column for Input, Right column for Output
col1, col2 = st.columns([1, 1.2], gap="large")

img_buffer = None
run_btn = False

with col1:
    st.subheader("📷 Image Preview")
    tab1, tab2 = st.tabs(["📂 Upload Image", "📸 Use Camera"])

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
        # Wrap image in a white "card" style container visually
        st.markdown("<div style='background: white; padding: 15px; border-radius: 14px; border: 1.5px solid #F0E0D0;'>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown("</div><br>", unsafe_allow_html=True)
        
        run_btn = st.button("🔍 Run Inspection Pipeline", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Statistics & Grading")
    
    if img_buffer is None:
        st.info("Awaiting image input... Upload or capture a photo to initialize the pipeline.")
    else:
        if run_btn:
            with st.spinner("Analyzing image array..."):
                img_array = np.array(image)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                try:
                    # 1. Execute backend vision pipeline
                    results = run_pipeline(img_bgr)
                    
                    # 2. Extract EXACT attribute names mapped to your PipelineResult object
                    blemish_ratio = getattr(results, 'blemish_ratio', 0.0) 
                    classifier_conf = getattr(results, 'classifier_confidence', 0.0)
                    raw_defect_name = getattr(results, 'defect_name', 'Unknown')
                    model_used = getattr(results, 'model_used', False)
                    
                    # 3. Calculate severity mapping to your compute_severity parameters
                    severity_report = compute_severity(
                        classifier_confidence=classifier_conf,
                        blemish_pixel_ratio=blemish_ratio,
                        defect_name=raw_defect_name,
                        model_used=model_used
                    )

                    # 4. Extract final answers from your SeverityReport object
                    grade = severity_report.grade
                    composite_score = severity_report.composite_score
                    treatment = severity_report.treatment
                    defect_pct = severity_report.defect_pct
                    conf_pct = severity_report.classifier_conf * 100
                    final_disease = severity_report.defect_name
                    grade_colour = severity_report.grade_colour

                    # Determine if healthy
                    is_healthy = str(final_disease).lower() in ['none', 'none detected', 'healthy', 'unknown'] and defect_pct < 5.0

                    # Dynamic Grade Badge mimicking the desktop app
                    st.markdown(f"""
                    <div style="background-color: {grade_colour}; color: white; padding: 12px; border-radius: 12px; text-align: center; font-size: 1.2rem; font-weight: bold; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        {grade}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display metrics in warm peach tiles
                    mcol1, mcol2 = st.columns(2)
                    mcol1.metric("Defect Area", f"{defect_pct:.1f} %")
                    mcol2.metric("Model Confidence", f"{conf_pct:.1f} %")
                    
                    mcol3, mcol4 = st.columns(2)
                    mcol3.metric("Composite Score", f"{composite_score:.1f}")
                    mcol4.metric("Pipeline", "MobileNetV2" if model_used else "HSV only")
                    
                    # Treatment Box (White Card)
                    st.markdown(f"""
                    <div class="report-box" style="border-top: 5px solid {grade_colour};">
                        <h4 style='margin-top:0px; color: #3D2B1F;'>💊 Treatment Guidance</h4>
                        <div style='color: #7A6050; font-size: 0.95rem; line-height: 1.6; font-weight: 500;'>
                            {treatment.replace(chr(10), '<br>')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"System Error during analysis: {e}")
