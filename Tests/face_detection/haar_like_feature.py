# Script imported by openCv official doc https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

import cv2 as cv
import argparse
import face_recognition
import numpy as np


def detect_and_display(edited_frame, face_only=True):
    frame_gray = cv.cvtColor(edited_frame, cv.COLOR_BGR2GRAY)

    # -- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    all_rois = []

    # This for statement extracts the 4 vertices of the rectangle area that contains the face.
    # If a face is detected the for loop will be executed one time for each frame
    for (x, y, w, h) in faces:
        # Design an ellipse to frame the face
        center = (x + w // 2, y + h // 2)
        edited_frame = cv.rectangle(edited_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Obtain a range of interest in the frame identified by the ellipse.
        # This range represents the area where the face is detected, and then it will
        # be used to detect the eyes in it
        face_roi = frame_gray[y:y + h, x:x + w]
        all_rois.append(face_roi)

        if not face_only:
            # -- In each face, detect eyes
            eyes = eyes_cascade.detectMultiScale(face_roi)

            for (x2, y2, w2, h2) in eyes:
                eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
                radius = int(round((w2 + h2) * 0.25))
                edited_frame = cv.circle(edited_frame, eye_center, radius, (255, 0, 0), 4)

    cv.imshow('Capture - Face detection', edited_frame)

    return all_rois


# Path to the data folder that contains the trained cascade classifier
data_path = cv.data.haarcascades

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.',
                    default=data_path + 'haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.',
                    default=data_path + 'haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)

if __name__ == "__main__":
    args = parser.parse_args()
    face_cascade_name = args.face_cascade
    eyes_cascade_name = args.eyes_cascade

    face_cascade = cv.CascadeClassifier()
    eyes_cascade = cv.CascadeClassifier()

    # -- 1. Load the cascades
    if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)

    if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
        print('--(!)Error loading eyes cascade')
        exit(0)

    camera_device = args.camera
    # -- 2. Read the video stream
    cap = cv.VideoCapture(camera_device, cv.CAP_DSHOW)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)

    while True:
        ret, frame = cap.read()

        if frame is None:
            print('--(!) No captured frame -- Break!')
            break

        print(detect_and_display(frame, face_only=True))

        if cv.waitKey(10) == 27:  # the esc key
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
