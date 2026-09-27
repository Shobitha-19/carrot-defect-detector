# 🥕 AI-Powered Carrot Defect Detector

An end-to-end machine learning application designed to automate the quality inspection and grading of carrots. It analyzes images to detect diseases or defects, calculates a severity score, and provides actionable treatment guidance. 

This project features both a **standalone Windows desktop application** and a **mobile-friendly web interface**.

## ✨ Key Features
*   **Dual Interfaces:** Accessible via a modern CustomTkinter Windows desktop app or a responsive Streamlit web application.
*   **Computer Vision Pipeline:** Utilizes OpenCV and a custom-trained dual-layer machine learning pipeline for image processing and defect classification.
*   **Automated Grading System:** Evaluates defect area and confidence to generate a severity index, composite score, and market grade (e.g., Grade A, Grade B).
*   **Actionable Reporting:** Outputs specific disease names (e.g., Cavity Spot/Lesion) and recommended agricultural treatments.

## 🛠️ Tech Stack
*   **Language:** Python
*   **Machine Learning & Vision:** TensorFlow, OpenCV, NumPy, Pillow
*   **User Interfaces:** Streamlit (Web/Mobile), CustomTkinter (Desktop)
*   **Packaging:** PyInstaller

## 🚀 How to Run

### 1. Web Application (Streamlit)
To run the mobile-friendly web version locally:
```bash
pip install -r requirements.txt
streamlit run app.py

**## 2. Standalone Windows Desktop App (.exe)**

pyinstaller --onefile --windowed --collect-all customtkinter main.py

**### 📂 Project Structure**

carrot-defect-detector/
├── app.py                 # Streamlit web application for mobile/browser
├── main.py                # CustomTkinter desktop GUI application
├── detection_pipeline.py  # Image processing & model inference
├── severity_index.py      # Severity scoring & grading algorithms
├── requirements.txt       # Application dependencies
└── README.md              # Project documentation

**### 📬 Contact**

Author: Shobitha
Email: bshobi05@gmail.com
