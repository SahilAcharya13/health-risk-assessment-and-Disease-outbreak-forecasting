import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

st.set_page_config(initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }

        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
        footer {visibility: hidden;}
    [data-testid="collapsedControl"] {
        display: none
    }
    </style>
""", unsafe_allow_html=True)

# Load the model
model = load_model('xray_fracture_detection_model.h5')  # Load the model you saved

# Define the classes
classes = ['Fracture', 'Normal']

def predict(image_path):
    print("Loading and preprocessing image...")
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    print("Image loaded and preprocessed. Shape:", img_array.shape)

    print("Making prediction...")
    prediction = model.predict(img_array)
    print("Prediction received. Shape:", prediction.shape)

    predicted_class = classes[int(np.round(prediction)[0][0])]
    return predicted_class


# Streamlit app
def main():
    st.title('X-ray Image Classification')
    st.write('Upload an X-ray image to classify it as "Fracture" or "Normal".')

    uploaded_file = st.file_uploader("Choose an X-ray image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded X-ray Image.', use_column_width=True)

        # Make a prediction
        predicted_class = predict(uploaded_file)
        st.write('Prediction:', predicted_class)


if __name__ == "__main__":
    main()
