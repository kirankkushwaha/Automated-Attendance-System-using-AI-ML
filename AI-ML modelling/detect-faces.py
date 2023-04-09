import cv2
import os

# Load the cascade classifier

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# Set the path to the image directory
image_dir = 'C:/Users/Kiran/Downloads/Face Dataset-20230407T134716Z-001'
image_path='mummy.jpg'

# Loop over the images in the directory
for filename in os.listdir(image_dir):
    # Set the path to the image file
    image_path = os.path.join(image_dir, filename)

    # Load the image
    img = cv2.imread('1.jpg')
    #print(img)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    print(faces)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the output image
    cv2.imshow('output', img)
    cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()
