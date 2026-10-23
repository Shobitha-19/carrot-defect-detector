import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO

# --- Page Config ---
st.set_page_config(page_title="Carrot Defect Detection", layout="centered")
st.title("🥕 Automated Carrot Defect Detection")
st.write("Upload an image. The system will first verify a carrot is present before analyzing its health.")

# --- Model Loading (Cached for Performance) ---
@st.cache_resource
def load_models():
    # Load YOLOv8 for object detection (auto-downloads yolov8n.pt if missing)
    yolo_model = YOLO("yolov8n.pt")
    
    # Load your custom trained MobileNetV2 model
    # Update this path if your model is named differently in the Data folder
    try:
        mobilenet_model = tf.keras.models.load_model("Data/mobilenetv2_carrot.h5")
    except Exception as e:
        st.warning("MobileNetV2 model not found. Using a dummy model for demonstration. Ensure 'Data/mobilenetv2_carrot.h5' exists.")
        mobilenet_model = None 
        
    return yolo_model, mobilenet_model

yolo_model, mobilenet_model = load_models()

# --- Defect Detection Logic (MobileNetV2 + HSV) ---
def analyze_carrot_health(cropped_img_bgr):
    # 1. Deep Learning Layer (MobileNetV2)
    img_resized = cv2.resize(cropped_img_bgr, (224, 224))
    img_tensor = np.expand_dims(img_resized, axis=0) / 255.0
    
    dl_confidence = 0.0
    dl_prediction = "Healthy"
    
    if mobilenet_model is not None:
        prediction = mobilenet_model.predict(img_tensor)[0][0]
        # Assuming 1 is Defective and 0 is Healthy
        if prediction > 0.5:
            dl_prediction = "Defective"
            dl_confidence = float(prediction)
        else:
            dl_prediction = "Healthy"
            dl_confidence = float(1.0 - prediction)
            
    # 2. Classical CV Layer (HSV Segmentation)
    hsv_img = cv2.cvtColor(cropped_img_bgr, cv2.COLOR_BGR2HSV)
    
    # Thresholds for Dark Rot (Dark/Black pixels)
    lower_rot = np.array([0, 0, 0])
    upper_rot = np.array([180, 255, 50])
    mask_rot = cv2.inRange(hsv_img, lower_rot, upper_rot)
    
    # Thresholds for Green Sprouting
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv_img, lower_green, upper_green)
    
    # Calculate defect ratio
    total_pixels = cropped_img_bgr.shape[0] * cropped_img_bgr.shape[1]
    rot_pixels = cv2.countNonZero(mask_rot)
    green_pixels = cv2.countNonZero(mask_green)
    
    defect_ratio = (rot_pixels + green_pixels) / total_pixels
    hsv_status = "Defective" if defect_ratio > 0.03 else "Healthy"
    
    # Combine logic: Defective if EITHER layer flags it
    final_status = "Defective" if (dl_prediction == "Defective" or hsv_status == "Defective") else "Healthy"
    
    return final_status, dl_confidence, defect_ratio

# --- UI and File Handling ---
uploaded_file = st.file_uploader("Choose a carrot image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    image = Image.open(uploaded_file).convert('RGB')
    img_cv2 = np.array(image)
    img_bgr = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
    
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Scanning for carrots..."):
        # Run YOLOv8 inference
        results = yolo_model(img_bgr)
        
        carrot_detected = False
        best_box = None
        
        # Parse YOLO results (COCO class ID for 'carrot' is 51)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id == 51 and conf > 0.5: # 51 is carrot
                    carrot_detected = True
                    # Get coordinates [x1, y1, x2, y2]
                    best_box = [int(x) for x in box.xyxy[0].tolist()]
                    break
            if carrot_detected:
                break
                
    if not carrot_detected:
        st.warning("⚠️ No carrot detected in the uploaded image. Please upload a valid photo of a carrot.")
    else:
        st.success("✅ Carrot detected! Analyzing health...")
        
        # Crop image to the detected carrot
        x1, y1, x2, y2 = best_box
        carrot_crop = img_bgr[y1:y2, x1:x2]
        
        with st.spinner("Running Dual-Layer Analysis..."):
            status, dl_conf, defect_ratio = analyze_carrot_health(carrot_crop)
            
            # Draw bounding box on original image based on health status
            color = (0, 255, 0) if status == "Healthy" else (0, 0, 255) # BGR
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)
            label = f"{status} (DL: {dl_conf:.2f})"
            cv2.putText(img_bgr, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Convert back to RGB for Streamlit display
            final_display_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Display Results
            st.image(final_display_img, caption=f"Analysis Result: {status}", use_container_width=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Final Status", status)
            col2.metric("MobileNetV2 Confidence", f"{dl_conf * 100:.1f}%")
            col3.metric("HSV Defect Ratio", f"{defect_ratio * 100:.2f}%")
