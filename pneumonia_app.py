import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained binary model
@st.cache_resource
def load_cnn_model():
    model_path = r"C:\Users\User\Desktop\projects\pneumonia\best_pneumonia_model (1).h5"
    model = load_model(model_path)
    return model

model = load_cnn_model()

# App UI
st.title("Pneumonia Detection App ")
st.write("Upload a chest X-ray image to check for pneumonia.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    
    # Show prediction
    st.subheader("Prediction:")
    if prediction >= 0.5:
        st.error(f"Pneumonia Detected (Confidence: {prediction:.2f})")
    else:
        st.success(f"Normal (Confidence: {1 - prediction:.2f})")
