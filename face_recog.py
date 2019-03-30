import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition

cap = cv2.VideoCapture(0)

def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Live Camera Code
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    haar_cascade_face = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')

    # Face detection
    faces_rects = haar_cascade_face.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # Let us print the no. of faces found
    print('Faces found: ', len(faces_rects))

    # Our next step is to loop over all the co-ordinates it returned and draw rectangles around them using Open CV.We will be drawing a green rectangle with thicknessof 2
    for (x, y, w, h) in faces_rects:
        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
