from PIL import Image, ImageDraw  # image manipulation library
import face_recognition

# Load the jpg file into a numpy array
image = face_recognition.load_image_file("../face_recognition/assets/biden.jpg")

# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image)
pil_image = Image.fromarray(image)

print("Detected features: ", face_landmarks_list[0].keys())

# Manipulate the face by adding some makeup
for face_landmarks in face_landmarks_list:
    d = ImageDraw.Draw(pil_image, 'RGBA')

    # Make the eyebrows into a nightmare
    d.line(face_landmarks['left_eyebrow'], fill=(255, 255, 255), width=2)
    d.line(face_landmarks['right_eyebrow'], fill=(255, 255, 255), width=2)
    d.line(face_landmarks['left_eyebrow'], fill=(255, 255, 255), width=2)
    d.line(face_landmarks['right_eyebrow'], fill=(255, 255, 255), width=2)

    # Gloss the lips
    d.line(face_landmarks['top_lip'], fill=(255, 255, 255), width=2)
    d.line(face_landmarks['bottom_lip'], fill=(255, 255, 255), width=2)
    d.line(face_landmarks['top_lip'], fill=(255, 255, 255), width=2)
    d.line(face_landmarks['bottom_lip'], fill=(255, 255, 255), width=2)

    # Sparkle the eyes
    d.line(face_landmarks['left_eye'], fill=(255, 255, 255, 30), width=2)
    d.line(face_landmarks['right_eye'], fill=(255, 255, 255, 30), width=2)

    # Apply some eyeliner
    d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(255, 255, 255), width=2)
    d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(255, 255, 255), width=2)

    print(face_landmarks['left_eye'])

    d.line(face_landmarks['chin'], fill=(255, 255, 255), width=2)
    d.line(face_landmarks['nose_bridge'], fill=(255, 255, 255), width=2)
    d.line(face_landmarks['nose_tip'], fill=(255, 255, 255), width=2)

    pil_image.show()


