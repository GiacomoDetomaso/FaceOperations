import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
import os

path = "C:/Users/Jak/Desktop/faces/31.jfif"
img = cv2.imread(path)

face = DeepFace.extract_faces(img, target_size=(300, 300), detector_backend="ssd")
