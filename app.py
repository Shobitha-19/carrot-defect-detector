import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Import your existing backend logic
from detection_pipeline import run_pipeline
from severity_index import compute_severity

# Configure page to wide mode for a more desktop-like feel
st.set_page_config(page_title="Carrot Defect Detector", page_icon="🥕", layout="wide")

# Inject Custom CSS for a premium dark aesthetic with a glowing background
st.markdown("""
<style>
    /* Premium Mesh Gradient Background */
    .stApp {
        background-color: #0f1015;
        background-image: 
            radial-gradient(at 0% 0%, rgba(255, 140, 0, 0.12) 0px, transparent 40%),
            radial-gradient(at 100% 100%, rgba(255, 140, 0, 0.08) 0px, transparent 50%);
        background-attachment: fixed;
    }
    
    /* Typography */
    .main-title {
        font-family: 'Segoe UI', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF8C00, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        font-family: 'Segoe UI', sans-serif;
        color: #A0A0A0;
        font-size: 1.1rem;
        margin-bottom: 30px;
        letter-spacing: 0.5px;
    }
    
    /* Sleek Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0px 0px;
        padding: 10px 20px;
        background-color: rgba(255, 140, 0, 0.05);
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 140, 0, 0.15);
        border-bottom: 2px solid #FF8C00;
    }
    
    /* Glassmorphism Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(30, 30, 35, 0.6);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
    }
    
    /* Modern Button */
    div.stButton > button {
        border-radius: 8px;
        height: 55px;
        font-weight: 600;
        font-size: 16px;
        background-color: #FF8C00;
        border: none;
        box-shadow: 0 4px 15px rgba(255, 140, 0, 0.3);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #e67e00;
        box-shadow: 0 6px 20px rgba(255, 140, 0, 0.4);
    }
    
    /* Glassmorphism Report Box */
    .report-box {
        background: rgba(43, 43, 50, 0.5);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 12px;
        margin-top: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.05);
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
                    # Execute backend vision pipeline
                    results = run_pipeline(img_bgr)
                    
                    # Extract EXACT attribute names mapped to your PipelineResult object
                    blemish_ratio = getattr(results, 'blemish_ratio', 0.0) 
                    classifier_conf = getattr(results, 'classifier_confidence', 0.0)
                    raw_defect_name = getattr(results, 'defect_name', 'Unknown')
                    model_used = getattr(results, 'model_used', False)
                    
                    # Calculate severity mapping to your compute_severity parameters
                    severity_report = compute_severity(
                        classifier_confidence=classifier_conf,
                        blemish_pixel_ratio=blemish_ratio,
                        defect_name=raw_defect_name,
                        model_used=model_used
                    )

                    # Extract final answers from your SeverityReport object
                    grade = severity_report.grade
                    composite_score = severity_report.composite_score
                    treatment = severity_report.treatment
                    defect_pct = severity_report.defect_pct
                    conf_pct = severity_report.classifier_conf * 100
                    final_disease = severity_report.defect_name
                    grade_colour = severity_report.grade_colour

                    # Determine if healthy
                    is_healthy = str(final_disease).lower() in ['none', 'none detected', 'healthy', 'unknown'] and defect_pct < 5.0

                    st.success("✅ Inspection Complete!")
                    
                    # Display metrics
                    mcol1, mcol2, mcol3 = st.columns(3)
                    mcol1.metric("Defect Area", f"{defect_pct:.2f}%")
                    mcol2.metric("Model Confidence", f"{conf_pct:.2f}%")
                    mcol3.metric("Severity Score", f"{composite_score:.2f}/100")
                    
                    # Comprehensive Report
                    st.markdown("### 📋 Final Inspection Report")
                    
                    # Status Box
                    if is_healthy:
                        st.success(f"**Health Status:** ✅ Healthy Carrot\n\n**Market Grade:** {grade}")
                    else:
                        st.error(f"**Health Status:** ⚠️ {final_disease}\n\n**Market Grade:** {grade}")
                    
                    # Treatment Box (dynamically uses your custom grade colors)
                    st.markdown(f"""
                    <div class="report-box" style="border-left: 5px solid {grade_colour};">
                        <h4 style='margin-top:0px; margin-bottom:15px;'>💊 Treatment & Precautions</h4>
                        <div style='color: #e0e0e0; font-size: 0.95rem; line-height: 1.6;'>
                            {treatment.replace(chr(10), '<br>')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"System Error during analysis: {e}")
