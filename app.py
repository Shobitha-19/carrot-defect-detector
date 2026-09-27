import streamlit as st
from PIL import Image
import numpy as np
import cv2

from detection_pipeline import run_pipeline
from severity_index import compute_severity

st.set_page_config(page_title="Carrot Defect Detector", page_icon="🥕", layout="centered")
st.title("🥕 Carrot Defect Detector")
st.write("Take a picture or upload an image to inspect carrot quality.")

camera_photo = st.camera_input("Take a picture of the carrot")
uploaded_photo = st.file_uploader("Or upload an image from gallery", type=["jpg", "jpeg", "png"])
img_buffer = camera_photo or uploaded_photo

if img_buffer is not None:
    image = Image.open(img_buffer)
    st.image(image, caption="Image for Analysis", use_column_width=True)
    
    if st.button("Run Inspection", type="primary"):
        with st.spinner("Analyzing image..."):
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            try:
                results = run_pipeline(img_bgr)
                grade, composite_score, precautions = compute_severity(results)
                
                st.success("Inspection Complete!")
                col1, col2 = st.columns(2)
                col1.metric("Defect Area", f"{results.get('defect_area', 0)}%")
                col2.metric("Confidence", f"{results.get('confidence', 0)}%")
                
                st.subheader(f"Grade: {grade}")
                st.info(f"Disease/Defect: {results.get('disease_name', 'None Detected')}")
                st.write("**Treatment & Precautions:**")
                st.write(precautions)
                
            except Exception as e:
                st.error(f"Error during analysis: {e}")