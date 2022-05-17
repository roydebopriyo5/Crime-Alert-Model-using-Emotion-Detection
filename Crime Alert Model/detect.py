from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

import winsound

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email import message

import geocoder
import ipinfo

import schedule

detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = '_mini_XCEPTION.102-0.66.hdf5'

face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "", "sad", "", ""]     #["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

sender = '...sender email id'
password = '...password of sender email id'
receivers = '...receiver email id'

message = MIMEMultipart()
message['From'] = sender
message['To'] = receivers
message['Subject'] = 'Help me! come fast beer belly(s)'

s = smtplib.SMTP('smtp.gmail.com', 587)   # creates SMTP session
s.starttls()                              # start TLS for security
s.login(sender, password)                 # Authentication

access_token = '...token'   # alternative of API
g = geocoder.ipinfo('me')         # finding IP Address and Location
ip = g.ip                         # storing IP Address   

def knocking(count):
    img_name = "img_{}.png".format(count)
    cv2.imwrite(img_name, frame)

    handler = ipinfo.getHandler(access_token)   # find ip information
    details = handler.getDetails(ip)            # find coordinates

    body = """
    Emergency...!! Emergency...!!

        Location coordinates - {}
    """.format(details.loc)

    message.attach(MIMEText(body, 'plain'))
    attachment = open("C:/Users/Debopriyo Roy/Downloads/Crime Alert Model/{}".format(img_name), "rb")
    p = MIMEBase('application', 'octet-stream')   # instance of MIMEBase
    p.set_payload((attachment).read())            # to change the payload into encoded form
    encoders.encode_base64(p)                     # encode into base64
    p.add_header('Content-Disposition', "attachment; filename= %s" % img_name)
    message.attach(p)
    e_mail = message.as_string()   # Converts the Multipart message into a string

    s.sendmail(sender, receivers, e_mail)

count = 0
cv2.namedWindow('face')
camera = cv2.VideoCapture(0)
while True:
    frame = camera.read()[1]
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame,width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        
        if label == "disgust" or label == "scared" or label == "sad":
            count += 1
            winsound.Beep(500, 500)   # sound effect
            knocking(count)
            #.....add Arduino Pass Code or any relevant

    else: continue
 
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                text = "{}: {:.2f}%".format(emotion, prob * 100)

                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5),
                (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 2)
                cv2.putText(frameClone, label, (fX, fY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                              (0, 0, 255), 2)

    cv2.imshow('face', frameClone)
    #cv2.imshow("probabilities", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
