import streamlit as st

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

from keras.models import load_model
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model("") #give the brain tumor model path with .h5 formate

# Define a function to preprocess the uploaded image
from keras.preprocessing import image

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

def preprocess_image(img):
    # Resize the image to (224, 224)
    img = img.resize((150, 150))
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    # Expand the dimensions to match the model's expected input shape
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess the image (normalize pixel values)
    img_array = preprocess_input(img_array)
    return img_array

# Define a function to make a prediction
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit app
def main():
    st.title("Brain Tumor Detection")

    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            prediction = predict(image)
            if prediction[0] > 0.5:
                st.write("The model predicts that there is a brain tumor in the image.")
            else:
                st.write("The model predicts that there is no brain tumor in the image.")


if __name__ == "__main__":
    main()


# import numpy as np
# import tensorflow as tf
# from keras.preprocessing import image
#
# # Load the saved model
# model = tf.keras.models.load_model('D:/8th_Sem/brain_tumor/brain_tumor_detection_model2.h5')
#
# # Preprocess the input image
# def preprocess_image(image_path):
#     img = image.load_img(image_path, target_size=(150, 150))
#     img_array = image.array_to_img(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array
#
# # Function to predict whether the image contains a brain tumor or not
# def predict_image(image_path):
#     img_array = preprocess_image(image_path)
#     prediction = model.predict(img_array)
#     if prediction[0] < 0.5:
#         return "No brain tumor"
#     else:
#         return "Brain tumor"
#
# # Path to the image you want to predict
# st.title("Brain Tumor Detection")
# image_path = st.text_input("Enter Image Path")
#
# # Predict whether the image contains a brain tumor or not
# prediction = predict_image(image_path)
# st.write("Prediction:", prediction)