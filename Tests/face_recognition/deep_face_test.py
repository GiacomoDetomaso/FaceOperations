from deepface import DeepFace as df
import matplotlib.pyplot as plt
import numpy
import os
import pandas
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

detectors_backends = [
    'opencv',
    'ssd',
    'dlib',
    'mtcnn',
    'retinaface',
    'mediapipe'
]

# face detection and alignment
img = cv2.imread("assets/trump.jpg")
face_objs = df.extract_faces(img, target_size=(224, 224), detector_backend=detectors_backends[5], enforce_detection=False)
print(face_objs)
plt.imshow(face_objs[0]['face'])
plt.show()
