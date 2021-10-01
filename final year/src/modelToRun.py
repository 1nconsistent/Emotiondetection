import streamlit as st
import os
from PIL import Image
import io
import numpy as np
from keras.models import model_from_json
import cv2
from emotions123 import webcam

from keras.preprocessing.image import load_img, img_to_array

st.set_option('deprecation.showfileUploaderEncoding', False)

def get_opened_image(image):
    return Image.open(image)

def get_list_of_images():
    file_list = os.listdir('test_img')
    return [str(filename) for filename in file_list if str(filename).endswith('.jpg')]

def image_pred(filename):
    emo = []
    model = model_from_json(open("fer.json", "r").read())
    model.load_weights('fer.h5')
    emotion_dict = {0: "Not Engaged", 1: "Not Engaged", 2: "Nominally Engaged", 3: "Very Engaged", 4: "Not Engaged",
                    5: "Very Engaged", 6: "Nominally Engaged"}
    facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(filename, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(filename, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        emo.append(emotion_dict[maxindex])
    return emo

st.title("Online Engagement Detection")
st.sidebar.title('Home')

choice = st.sidebar.selectbox("Select Option", ["Choose Existing Image","Webcam"])

if choice == "Choose Existing Image":
    image_file_chosen = st.sidebar.selectbox('Select an existing test image:', get_list_of_images())
    image2 = cv2.imread(str('test_img/' + image_file_chosen))
    with st.beta_expander('Selected Image', expanded=True):
        st.image(str('test_img/' + image_file_chosen), use_column_width=True) #need to add test file path to source
    prediction = image_pred(image2)
    st.subheader("Prediction")
    st.markdown(f'The predicted label is: ')
    print(prediction)
    for i,j in enumerate(prediction,1):
        st.markdown(f'**{str(i)+") "+str(j)}**')
elif choice == "Webcam":
    webcam()

