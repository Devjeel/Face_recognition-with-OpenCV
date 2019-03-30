import numpy as np
import cv2, os
import matplotlib.pyplot as plt
import face_recognition

cap = cv2.VideoCapture(0)

def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

known_face_encodings = []

known_face_names = []

user_appeared = []
root = "images/"
for filename in os.listdir(root):
    if filename.endswith('.jpg' or '.png'):
        try:
            print(filename)
            path = os.path.join(root, filename)
            filter_image = face_recognition.load_image_file(path)
            filter_face_encoding = face_recognition.face_encodings(filter_image)
            known_face_encodings.append(filter_face_encoding[0])
            known_face_names.append(filename)

        except:
            print("An exception occurred : " + filename)

# print(known_face_encodings)
print(known_face_names)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

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

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    k = cv2.waitKey(1)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                print(name)
            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4


        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('frame', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
