import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# --- 1. Page Configuration (Must be the first command) ---
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",  # Uses the full width of the screen
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 10px;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Load Model (Cached) ---
@st.cache_resource
def load_my_model():
    # Path to your model file (Make sure this file is in the same folder!)
    model_path = "plant_disease_recog_model_pwp.keras"
    
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        return None

model = load_my_model()

# --- 3. Class Names ---
# (Paste your full list here if you have it. This is a shortened example for display)
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
               'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 
               'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 
               'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
               'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
               'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
               'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
               'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
               'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
               'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
               'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# --- 4. Sidebar (Instructions) ---
with st.sidebar:
    st.title("🌿 Plant Doctor AI")
    st.markdown("---")
    st.subheader("How to use:")
    st.write("1. **Upload** a clear photo of a plant leaf.")
    st.write("2. **Click** the 'Analyze Leaf' button.")
    st.write("3. **Get** the disease diagnosis instantly.")
    st.markdown("---")
    st.info("This AI model was trained on over 70,000 images with 93.7% accuracy.")
    st.write("Created by: ANESHYA DAS,AASHI GARG AND ARSHI ARYA")

# --- 5. Main Interface ---
st.title("🌱 Plant Disease Recognition System")
st.write("upload an image to detect the disease and get a cure recommendation.")

# Create two columns for a better layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Upload Image")
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf", use_column_width=True, output_format="auto")

with col2:
    st.subheader("🔍 Diagnosis Results")
    
    if uploaded_file and model:
        # Add a big analyze button
        if st.button("🧪 Analyze Leaf"):
            with st.spinner("Analyzing leaf patterns..."):
                # Preprocess
                image = ImageOps.fit(image, (160, 160), Image.Resampling.LANCZOS)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                img_array = np.array(image)
                img_array = np.expand_dims(img_array, 0) # Create batch

                # Predict
                predictions = model.predict(img_array)
                score = predictions[0]
                
                # Get Result
                top_class_index = np.argmax(score)
                top_confidence = 100 * np.max(score)
                
                # Handle case where class list might be shorter than prediction index
                if top_class_index < len(class_names):
                    prediction_label = class_names[top_class_index]
                else:
                    prediction_label = f"Class {top_class_index}"

                # --- DISPLAY RESULTS ---
                
                # Dynamic Status Color
                if "healthy" in prediction_label.lower():
                    st.success(f"✅ **Status: HEALTHY**")
                else:
                    st.error(f"⚠️ **Disease Detected**")

                st.markdown(f"### **Diagnosis:** {prediction_label.replace('_', ' ')}")
                
                # Confidence Bar
                st.write("Confidence Score:")
                st.progress(int(top_confidence))
                st.caption(f"The AI is **{top_confidence:.2f}%** sure of this result.")

    elif not uploaded_file:
        st.info("👈 Please upload an image to see the analysis here.")
    
    elif model is None:
        st.error("🚨 Error: Model file not found. Please make sure the .keras file is in the folder.")