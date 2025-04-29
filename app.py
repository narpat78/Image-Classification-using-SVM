# importing dependencies
import streamlit as st
import numpy as np
from skimage.transform import resize 
import pickle
from PIL import Image

# app title
st.title('Image Classifier')

# loading the image classification model
model = pickle.load(open('img_model.pkl', 'rb'))

# uploading test image
uploaded_file = st.file_uploader("Choose an Image.", type="jpg")

# prediction of the uploaded image
if uploaded_file is not None:
  img = Image.open(uploaded_file)
  st.image(img, caption='Uploaded Image')

  if st.button('PREDICT'):
    categories = ['ball_leather', 'cone', 'sunflower']
    st.write('Result:')
    flat_data = []
    img = np.array(img)
    img_resized = resize(img, (150,150,3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    y_out = model.predict(flat_data)
    y_out = categories[y_out[0]]
    st.title(f'PREDICTED OUTPUT: {y_out}')
    q = model.predict_proba(flat_data)
    for index, item in enumerate(categories):
      st.write(f'{item} : {q[0][index]*100:.2f}%')