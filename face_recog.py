import numpy as np
import cv2
import matplotlib.pyplot as plt
cap = cv2.VideoCapture(0)
''' Live Camera Code
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''
#----------------------------------------------
#Image Tranform 

def convertToRGB(image):
    # Loading the image to be tested
    test_image = cv2.imread('data/jeel.jpg')

    # Converting to grayscale
    test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    # Displaying the grayscale image
    plt.imshow(test_image_gray, cmap='gray')

    return cv2.cvtColor(test_image_gray, cv2.COLOR_BGR2RGB)

# When everything done, release the capture
#cap.release()
#cv2.destroyAllWindows()
