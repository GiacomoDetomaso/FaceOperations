from deepface import DeepFace as df
import matplotlib.pyplot as plt
import numpy
import os
import pandas

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
face_objs = df.extract_faces(img_path="assets/biden.jpg", target_size=(224, 224), detector_backend=detectors_backends[4])
print(face_objs)
plt.imshow(face_objs[0]['face'])
plt.show()

print(pandas.DataFrame(df.analyze(img_path="assets/biden.jpg", detector_backend=detectors_backends[4])).to_string())
