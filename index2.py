import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np
import PIL
from playsound import playsound
#import pyttsx3



# Load the face detection model
face_detector = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')

# Load the age and gender recognition models
age_net = cv2.dnn.readNet('models/age_net.caffemodel', 'models/deploy_age.prototxt')
gender_net = cv2.dnn.readNet('models/gender_net.caffemodel', 'models/deploy_gender.prototxt')
model=load_model('models/model_file_30epochs.h5')


AGE_LABELS = ['(1-10)','(11-20)', '(21-30)', '(31-40)', '(41-50)', '(51-60)', '(61-70)', '(71-100)']
GENDER_LABELS = ['Male', 'Female']
labels_dict={0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}


video_capture = cv2.VideoCapture(0)
#engine = pyttsx3.init()

directory_emotions="Emotion_teller/"

while True:
  
    ret, frame = video_capture.read() 


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    emotion = ""

    for (x, y, w, h) in faces:
    
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        #emotion detection
        sub_face_img=gray[y:y+h, x:x+w]
        resized=cv2.resize(sub_face_img,(48,48))
        normalize=resized/255.0
        reshaped=np.reshape(normalize, (1, 48, 48, 1))
        result=model.predict(reshaped)
        label=np.argmax(result, axis=1)[0]

      
        face_roi = frame[y:y+h, x:x+w]

        
        face_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        #age detection 
        age_net.setInput(face_blob)
        age_preds = age_net.forward()
        age = AGE_LABELS[age_preds[0].argmax()]

        #gender detection
        gender_net.setInput(face_blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LABELS[gender_preds[0].argmax()]

  
        labels = "{}, {}, {}".format(gender, age, labels_dict[label])

#        engine.say(label)

        emotion = labels_dict[label]

  #      playsound(directory_emotions+emotion+".mp3")
        
        cv2.putText(frame, labels, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#        engine.runAndWait()
  
    cv2.imshow('Video', frame)

 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
