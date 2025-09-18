import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore

# --- Page Configuration ---
st.set_page_config(
    page_title="Lesion Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS (same as before) ---
CSS = """
/* General Body and Font Styles */
body {
    color: #FFFFFF;
    background-color: #001f3f; /* A navy blue background */
}
/* ... all other CSS from the previous version ... */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 5rem;
    padding-right: 5rem;
}
#MainMenu, footer, header { visibility: hidden; }
.navbar { display: flex; justify-content: space-between; align-items: center; padding: 1rem 5rem; background-color: transparent; }
.navbar .logo { font-size: 1.5rem; font-weight: bold; color: #FFFFFF; }
.navbar .nav-links a { color: #CCCCCC; margin-left: 2rem; text-decoration: none; transition: color 0.3s; }
.navbar .nav-links a:hover, .navbar .nav-links a.active { color: #FFFFFF; text-decoration: underline; }
.stButton>button { width: 100%; border-radius: 0.5rem; color: #ffffff; background-color: #007BFF; border: none; padding: 0.75rem 1rem; transition: background-color 0.3s; }
.stButton>button:hover { background-color: #0056b3; color: #ffffff; }
.stFileUploader { border: 2px dashed #007BFF; border-radius: 0.5rem; padding: 1rem; background-color: rgba(0, 123, 255, 0.05); }
.stFileUploader label { font-size: 1.1rem; font-weight: bold; color: #FFFFFF; }
.center-text { text-align: center; }
.footer { padding: 2rem 5rem; margin-top: 3rem; border-top: 1px solid #333; }
.footer h4 { color: #FFFFFF; margin-bottom: 1rem; }
.footer a { color: #CCCCCC; text-decoration: none; }
.footer a:hover { color: #FFFFFF; }
.footer .social-icons a { margin-right: 1rem; font-size: 1.5rem; }
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

# --- Configuration & Model Loading ---
MODEL_PATH = 'skin_lesion_classifier.h5'
IMG_HEIGHT = 75
IMG_WIDTH = 100

# Class mapping based on the training script's LabelEncoder
# The order is crucial: ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
LESION_CLASSES_INFO = {
    0: ('Actinic Keratoses (akiec)', 'Often benign, but can be a precursor to skin cancer. A dermatologist visit is recommended.'),
    1: ('Basal Cell Carcinoma (bcc)', 'A common type of skin cancer. Usually not life-threatening but requires medical treatment.'),
    2: ('Benign Keratosis-like Lesions (bkl)', 'Non-cancerous skin growths, like "age spots" or seborrheic keratoses.'),
    3: ('Dermatofibroma (df)', 'A common benign skin nodule. Typically harmless.'),
    4: ('Melanoma (mel)', 'The most serious type of skin cancer. Early detection is crucial. See a doctor immediately.'),
    5: ('Melanocytic Nevi (nv)', 'Common moles. Mostly benign, but changes should be monitored.'),
    6: ('Vascular Lesions (vasc)', 'Benign lesions like cherry angiomas or spider veins. Usually not a cause for concern.')
}

@st.cache_resource
def load_sk_model(path):
    """Loads the trained Keras model."""
    try:
        model = load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_sk_model(MODEL_PATH)

# --- Real Prediction Logic ---
def predict(image: Image.Image):
    """
    Preprocesses the image and returns the predicted class, description, and confidence score.
    """
    # 1. Convert PIL image to NumPy array
    img_array = np.array(image)
    
    # 2. Resize the image to the model's expected input size
    img_resized = tf.image.resize(img_array, [IMG_HEIGHT, IMG_WIDTH])
    
    # 3. Normalize pixel values to be between 0 and 1
    img_normalized = img_resized / 255.0
    
    # 4. Add a batch dimension
    img_expanded = np.expand_dims(img_normalized, axis=0)
    
    # 5. Make a prediction
    prediction_array = model.predict(img_expanded)
    
    # 6. Get the results
    confidence_score = np.max(prediction_array)
    class_index = np.argmax(prediction_array)
    
    class_name, description = LESION_CLASSES_INFO[class_index]
    
    return class_name, description, confidence_score

# --- UI Layout (same as before) ---

# Custom "Navbar"
st.markdown("""
<div class="navbar">
    <div class="logo">Lesion Detection</div>
    <div class="nav-links">
        <a href="#" class="active">Home</a> <a href="#">Project Details</a> <a href="#">Approach</a> <a href="#">About Us</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Hero Section
st.markdown('<h1 class="center-text">Skin Lesion Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="center-text">Upload an image of a skin lesion to get AI-based classification with expert-backed information.</p>', unsafe_allow_html=True)
st.write("")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Drag and drop or click to upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if st.button("Predict", type="primary"):
        if uploaded_file is not None and model is not None:
            with st.spinner('Analyzing...'):
                image = Image.open(uploaded_file)
                # Call the REAL prediction function
                class_name, description, score = predict(image)
                # Store results in session state
                st.session_state.prediction_result = (class_name, description, score)
                st.session_state.uploaded_image = image
        elif model is None:
            st.error("Model not loaded. Please ensure 'skin_lesion_classifier.h5' is in the folder.")
        else:
            st.warning("Please upload an image first.")

with col2:
    st.subheader("Analysis Results")
    if 'prediction_result' in st.session_state:
        class_name, description, score = st.session_state.prediction_result
        
        st.success(f"**Predicted Class:** {class_name}")
        st.metric(label="Confidence Score", value=f"{score:.2%}")
        st.info(f"**Information:** {description}")

        with st.expander("Show Uploaded Image"):
            st.image(st.session_state.uploaded_image, use_container_width=True)
        
        st.warning("Disclaimer: This is a student project and not for medical diagnosis. Consult a professional for any health concerns.")
    else:
        st.info("Upload an image and click 'Predict' to see results.")

# --- Custom Footer ---
st.markdown('<div style="margin-top: 5rem;"></div>', unsafe_allow_html=True) # Spacer
st.divider()
footer_cols = st.columns([2, 1, 2])
with footer_cols[0]:
    st.markdown("<h4>Lesion Detection</h4>", unsafe_allow_html=True)
    st.write("AI-powered skin lesion classification system for early detection and improved healthcare outcomes.")
with footer_cols[1]:
    st.markdown("<h4>Quick Links</h4>", unsafe_allow_html=True)
    st.markdown("""
        <a href="#">Home</a><br>
        <a href="#">Project Details</a><br>
        <a href="#">About Us</a>
    """, unsafe_allow_html=True)
with footer_cols[2]:
    st.markdown("<h4>Our Team</h4>", unsafe_allow_html=True)
    st.write("Kartik Patel")
    st.write("*Second Year at VIT Chennai*")
    st.markdown("""
        <div class="social-icons">
            <a href="#" target="_blank">🔗</a>
            <a href="#" target="_blank">💼</a>
            <a href="#" target="_blank">👾</a>
        </div>
    """, unsafe_allow_html=True)