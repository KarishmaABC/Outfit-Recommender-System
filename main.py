import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Set page config
st.set_page_config(page_title="Fashion Recommender", layout="wide")

# Load data
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))


# Load model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Set title
st.markdown("<h1 style='text-align: center; color: #C71585;'>👗 Outfit Recommender System👗</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Upload file
def save_uploaded_file(uploaded_file):
    try:
        upload_path = os.path.join('uploads', uploaded_file.name)
        with open(upload_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return upload_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# Feature extraction
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Recommendation logic
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices


# Upload and process
st.markdown("### 📤 Upload a Outfit image to get style recommendations")
uploaded_file = st.file_uploader("Choose an image file (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_path = save_uploaded_file(uploaded_file)
    if img_path:
        display_image = Image.open(uploaded_file)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("#### 🖼️ Your Uploaded Image", unsafe_allow_html=True)

        # Center image using columns
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(display_image, width=400)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("#### 🧠 You may also like 👇", unsafe_allow_html=True)

        # Extract features and recommend
        features = feature_extraction(img_path, model)
        indices = recommend(features, feature_list)

        # Show recommendations
        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                st.image(filenames[indices[0][i]], use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color: gray;'>Made with ❤️ by Karishma</p>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Failed to save the uploaded image.")
