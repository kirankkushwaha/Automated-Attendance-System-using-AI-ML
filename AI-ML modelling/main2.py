import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import json
import requests
# from PIL import ImageGrab

# path = 'Images'
# images = []
# classNames = []
# for folder in os.listdir(path):
#     folder_path = os.path.join(path, folder)
#     if os.path.isdir(folder_path):
#         for i, file in enumerate(os.listdir(folder_path)):
#             if file.endswith('.png') or file.endswith('.jpg'):
#                 image_path = os.path.join(folder_path, file)
#                 image = cv2.imread(image_path)
#                 images.append(image)
#                 classNames.append(f"{folder}/{i}")
# print(classNames)

path = 'Images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
 
 
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(img)
        if len(face_locations) > 0:
            encode = face_recognition.face_encodings(img, face_locations)[0]
            encodeList.append(encode)
    return encodeList
 



def markAttendance(name):
    # Send the attendance data to your API
    API_URL="https://attendance-api-rdn8.vercel.app/attendance"
    data = {"name": name, "time": datetime.now().strftime('%H:%M:%S')}
    response = requests.post(API_URL, json=data)
    if response.status_code == 200:
        print(f"Attendance marked for {name}")
    else:
        print(f"Attendance marked for {name}")
    # with open('Attendance.csv','r+') as f:
    #     myDataList = f.readlines()
    #     nameList = []
    #     for line in myDataList:
    #         entry = line.split(',')
    #         nameList.append(entry[0])
    #     if name not in nameList:
    #         now = datetime.now()
    #         dtString = now.strftime('%H:%M:%S')
    #         f.writelines(f'\n{name},{dtString}')
 
#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr
 
encodeListKnown = findEncodings(images)
print('Encoding Complete')
 
cap = cv2.VideoCapture(0)
 
while True:
    success, img = cap.read()
    #img = captureScreen()
    # imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
    facesCurFrame = face_recognition.face_locations(img)
    encodesCurFrame = face_recognition.face_encodings(img,facesCurFrame)
 
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)
 
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            # y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
            
 
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)

# Write a code to crop the face from the image and save it in the folder

