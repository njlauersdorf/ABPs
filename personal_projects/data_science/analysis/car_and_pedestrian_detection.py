import random
import numpy as np
import sys
import os
import cv2

path = os.getcwd()
data_direct = os.path.abspath(os.path.join(path, os.pardir)) + '/data/'

# Load trained data on cars from opencv (haar cascade algorithm)
trained_car_data = cv2.CascadeClassifier(data_direct + 'car_detector.xml')
trained_pedestrian_data = cv2.CascadeClassifier(data_direct + 'pedestrian_detector.xml')

#Ask user what media type they want analyzed
media_type = input('Analyze picture, video, or webcam?')
media_type = media_type.lower().strip()

# Perform face detection for a single image/frame
if media_type == 'picture':

    # Choose a default image to detect faces in
    img = cv2.imread(data_direct + 'car_image.jpeg')

    # Make selected image grayscale
    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    car_coordinates = trained_car_data.detectMultiScale(grayscaled_img, scaleFactor=1.1, minNeighbors=3)
    pedestrian_coordinates = trained_pedestrian_data.detectMultiScale(grayscaled_img, scaleFactor=1.1, minNeighbors=3)

    # Draw rectangles around all faces with random colors that are saved and re-used each iteration
    for (x, y, w, h) in car_coordinates:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 6)

    for (x, y, w, h) in pedestrian_coordinates:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 4)

    # Shows selected image
    cv2.imshow('Car Detector', img)

    # Show image until any key is pressed
    cv2.waitKey()

elif media_type == 'video':
    video = cv2.VideoCapture(data_direct + 'dashcam_footage.mp4')
    # Color arrays for rectangles
    red_arr = np.array([])
    green_arr = np.array([])
    blue_arr = np.array([])

    #Iterate forever over all frames
    while True:

        #Read each frame of video starting at beginning
        (successful_frame_read, frame) = video.read()

        if successful_frame_read:
            # Make selected image grayscale
            grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            break

        # Detect faces
        car_coordinates = trained_car_data.detectMultiScale(grayscaled_frame, scaleFactor=1.1, minNeighbors=3)
        pedestrian_coordinates = trained_pedestrian_data.detectMultiScale(grayscaled_frame, scaleFactor=1.1, minNeighbors=3)
        # Current face coordinate in loop

        # Draw rectangles around all faces with random colors that are saved and re-used each iteration
        for (x, y, w, h) in car_coordinates:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 6)

        for (x, y, w, h) in pedestrian_coordinates:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 4)

        # Show each frame for 1 ms.
        cv2.imshow('Car Detector', frame)
        key = cv2.waitKey(1)

        #If 'esc' pressed, terminate program
        if key == 27:
            break
else:
    raise Exception("Sorry! Your response is limited to either `picture` or `video`.")
