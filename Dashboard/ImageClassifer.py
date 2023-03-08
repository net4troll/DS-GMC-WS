import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model=tf.keras.models.load_model('/Users/alfahwun/GMC/CNN/models/angrysmilingclassifier.h5')

st.write("""
         # Image Classification
         """
         )

file = st.file_uploader("", type=["jpg","jpeg", "png"])
def upload_predict(upload_image, model):
    img = tf.image.resize(upload_image, (64,64))
    image = np.asarray(img)
    return model.predict(np.expand_dims(image/255, 0))

if file is None:
    st.write("Upload an image ")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = upload_predict(image, model)
    st.write(predictions)
    emotion = "smiling" if predictions > 0.5 else "not smiling"
    st.write("The image is classified as",emotion)
