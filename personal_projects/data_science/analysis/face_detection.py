import random
import numpy as np
import sys
import os
import cv2

path = os.getcwd()
data_direct = os.path.abspath(os.path.join(path, os.pardir)) + '/data/'

# Load trained data on fronts of faces from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(data_direct + 'haarcascade_frontalface_default.xml')

#Ask user what media type they want analyzed
media_type = input('Analyze picture, video, or webcam?')
media_type = media_type.lower().strip()

# Perform face detection for a single image/frame
if media_type == 'picture':

    # Choose a default image to detect faces in
    img = cv2.imread(data_direct + 'star_wars.jpeg')


    # Make selected image grayscale
    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangles around all faces with random colors
    for (x, y, w, h) in face_coordinates:
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        cv2.rectangle(img, (x,y), (x+w, y+h), (blue, green, red), 2)

    # Shows selected image
    cv2.imshow('Face Detector', img)

    # Show image until any key is pressed
    cv2.waitKey()

else:
    #Check either 'video' or 'webcam' input by user or raise exception
    if media_type == 'video':
        video = cv2.VideoCapture(data_direct + 'mo_code_mo_problems.mp4')
    elif media_type == 'webcam':
        video = cv2.VideoCapture(0)
    else:
        raise Exception("Sorry! Your response is limited to either `picture`, `video`, or `webcam`.")

    # Color arrays for rectangles
    red_arr = np.array([])
    green_arr = np.array([])
    blue_arr = np.array([])

    #Iterate forever over all frames
    while True:

        #Read each frame of video starting at beginning
        successful_frame_read, frame = video.read()

        # Make selected image grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame, scaleFactor=1.1, minNeighbors=3)

        # Current face coordinate in loop
        ind=0

        # Draw rectangles around all faces with random colors that are saved and re-used each iteration
        for (x, y, w, h) in face_coordinates:
            ind+=1
            if ind > len(red_arr):
                red_arr = np.append(red_arr, random.randint(0, 255))
                green_arr = np.append(green_arr, random.randint(0, 255))
                blue_arr = np.append(blue_arr, random.randint(0, 255))

            cv2.rectangle(frame, (x,y), (x+w, y+h), (blue_arr[ind-1], green_arr[ind-1], red_arr[ind-1]), 8)

        # Show each frame for 1 ms.
        cv2.imshow('Face Detector', frame)
        key = cv2.waitKey(1)

        #If 'esc' pressed, terminate program
        if key == 27:
            break

#trainCascadeObjectDetector

print('Code Completed')
