import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Set page title
st.set_page_config(page_title="Plant Disease Detection", layout="centered")

# Load model
model = load_model("/home/prateep/Documents/CropCare-AI/model.h5")

# Class labels
class_names = {
    0: 'Apple___Apple_scab', 1: 'Apple___Black_rot', 2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy', 4: 'Blueberry___healthy', 5: 'Cherry_(including_sour)___Powdery_mildew',
    6: 'Cherry_(including_sour)___healthy', 7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    8: 'Corn_(maize)___Common_rust_', 9: 'Corn_(maize)___Northern_Leaf_Blight',
    10: 'Corn_(maize)___healthy', 11: 'Grape___Black_rot', 12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)', 16: 'Peach___Bacterial_spot',
    17: 'Peach___healthy', 18: 'Pepper,_bell___Bacterial_spot', 19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight', 21: 'Potato___Late_blight', 22: 'Potato___healthy',
    23: 'Raspberry___healthy', 24: 'Soybean___healthy', 25: 'Squash___Powdery_mildew',
    26: 'Strawberry___Leaf_scorch', 27: 'Strawberry___healthy', 28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight', 30: 'Tomato___Late_blight', 31: 'Tomato___Leaf_Mold',
    32: 'Tomato___Septoria_leaf_spot', 33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot', 35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato___Tomato_mosaic_virus', 37: 'Tomato___healthy'
}

# App title
st.title("🌿 Plant Disease Detection")
st.markdown("Upload a leaf image or use your camera to detect plant diseases.")

# File uploader (Drag and drop multiple images)
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# If files are uploaded
if uploaded_files:
    # Process each file
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file)
        st.image(img, caption=f'Uploaded Image: {uploaded_file.name}', use_container_width=True, width=500)

        # Preprocess image
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        top1_index = prediction[0].argmax()  # Get the index of the top prediction
        top1_conf = prediction[0][top1_index] * 100
        top1_label = class_names[top1_index]

        # Show top-1 prediction
        st.markdown("### 🔍 Prediction:")
        st.write(f"{top1_label}: **{top1_conf:.2f}%**")

# Check if camera is active in session state
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

# Button to toggle camera input
camera_button = st.button("Use Camera")

# Toggle the camera state
if camera_button:
    st.session_state.camera_active = not st.session_state.camera_active

# If the camera is active, show the camera input
if st.session_state.camera_active:
    camera_img = st.camera_input("Capture Image")
    if camera_img:
        img = Image.open(camera_img)
        st.image(img, use_container_width=True, width=500)
        st.markdown("### 🔍 Processing...")

        # Preprocess image
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        top1_index = prediction[0].argmax()  # Get the index of the top prediction
        top1_conf = prediction[0][top1_index] * 100
        top1_label = class_names[top1_index]

        # Show top-1 prediction
        st.markdown("### 🔍 Prediction:")
        st.write(f"{top1_label}: **{top1_conf:.2f}%**")

# Add a sidebar
st.sidebar.title("About")
st.sidebar.info(
    """
    This app uses a deep learning model to detect plant diseases from leaf images.
    It provides the top prediction with confidence score.
    """
)

# Add a contact form
st.sidebar.title("Contact")
st.sidebar.info(
    """
    If you have any questions or feedback, please reach out to us at:
    - Email: bobplant692@gmail.com
    - Phone: +91-8367755747
    """
)

# Add a link to the GitHub repository
st.sidebar.title("GitHub Repository")
st.sidebar.markdown(
    """
    [GitHub Repository](https://github.com/Bobby-111)
    """
)

# Add a disclaimer
st.markdown(
    """
    <style>
    .disclaimer {
        font-size: 12px;
        color: gray;
        text-align: center;
    }
    </style>
    <div class="disclaimer">
    This app is for educational purposes only. Please consult a professional for accurate diagnosis.
    </div>
    """,
    unsafe_allow_html=True
)

# Footer
st.markdown(
    """
    <style>
    .footer {
        font-size: 12px;
        color: gray;
        text-align: center;
    }
    </style>
    <div class="footer">
    &copy; 2025 Plant Disease Detection. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .made-with {
        font-size: 12px;
        color: gray;
        text-align: center;
        margin-top: 50px;
    }
    </style>
    <div class="made-with">
    Made with ❤️ by <a href= "https://github.com/Bobby-111" target = "blank">Bobby</a>
    </div>
    """,
    unsafe_allow_html=True
)

