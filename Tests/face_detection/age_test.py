import cv2
import numpy


def extract_face_box(face_net, video_frame, confidence_min_level):
    # Extract the height and width of the frame from its shape (first and second dimension)
    video_frame_height = video_frame.shape[0]
    video_frame_width = video_frame.shape[1]

    # Create a blob from image defining the mean (to perform mean subtraction
    # of the RGB channels of the input image) of the data set and
    # the scale factor that will be used to normalize the RGB channels
    blob = cv2.dnn.blobFromImage(video_frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)

    face_net.setInput(blob)

    # Returns the detected blob: a 4 dimensional array with the following shape (1, 1, 200, 7).
    # The first two dimension both store a single list.
    # The third dimension stores 200 list with 7 values each. These values are important property of the blob
    detection = face_net.forward()

    boxes = []

    for i in range(detection.shape[2]):
        # The value needed in this matrix is the confidence, that is stored in the element
        # with index [0, 0, i, 2], as it is possible to see in the detection_example.txt file.
        # This value is useful because tells us how likely the image is a face.
        confidence = detection[0, 0, i, 2]

        # Coordinates of the rectangle extraction. The coordinates of the blob are stored in the
        # element with index [0, 0, i, j] where 3 <= j <= 6. These 4 values will be extracted and
        # multiplied by the frame height and weight. The rectangle will be then added to the frame
        if confidence > confidence_min_level:
            left = int(detection[0, 0, i, 3] * video_frame_width)
            top = int(detection[0, 0, i, 4] * video_frame_height)
            right = int(detection[0, 0, i, 5] * video_frame_width)
            bottom = int(detection[0, 0, i, 6] * video_frame_height)

            # Opposite corners of the rectangle that will be added to the frame
            corner_left_top = (left, top)
            corner_right_bottom = (right, bottom)

            boxes.append([left, top, right, bottom])

            cv2.rectangle(video_frame, corner_left_top, corner_right_bottom, (255, 255, 0), 2)

    return boxes


def predict_gender_and_age(video_frame, boxes, mean, all_genders, all_ages_classes, show_gender=False):
    for box in boxes:
        # Slice the frame to extract the part containing the detected face
        face_box = video_frame[max(0, box[1] - padding):min(box[3] + padding, video_frame.shape[0] - 1),
                   max(0, box[0] - padding):min(box[2] + padding, video_frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face_box, 1.0, (227, 227), mean, swapRB=False)

        cv_net_gender.setInput(blob)
        predicted_gender = cv_net_gender.forward()[0].argmax()

        cv_net_age.setInput(blob)
        predicted_age_class = cv_net_age.forward()[0].argmax()

        label = "{}".format(all_ages_classes[predicted_age_class])

        if show_gender:
            label = "{},{}".format(all_genders[predicted_gender], all_ages_classes[predicted_age_class])
            print(label)

        cv2.putText(video_frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)

        #return cv_net_age.forward()[0][predicted_age_class] * 100, label


face_proto = "assets/opencv_face_detector.pbtxt"  # configuration of the dnn for face detection
face_model = "assets/opencv_face_detector_uint8.pb"  # the model of the dnn for face detection

age_proto = "assets/age_deploy.prototxt"  # configuration of the dnn for age classification
age_model = "assets/age_net.caffemodel"  # model of the dnn for age classification

gender_proto = "assets/gender_deploy.prototxt"  # configuration of the dnn for gender prediction
gender_model = "assets/gender_net.caffemodel"  # model of the dnn for gender prediction

# Read the three dnns
cv_net_face = cv2.dnn.readNet(face_model, face_proto)
cv_net_age = cv2.dnn.readNet(age_model, age_proto)
cv_net_gender = cv2.dnn.readNet(gender_model, gender_proto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
age_list_named = ['Newborn', 'Baby', 'Boy', 'Teenager', 'Young adult', 'Adult', 'Middle age', 'Elder']
gender_list = ['Male', 'Female']

padding = 20

print("Age detection mod\n1) Video stream\n2) Test image")
mode = int(input())

if mode == 1:
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = video.read()
        boxes_list = extract_face_box(cv_net_face, frame, 0.7)
        predict_gender_and_age(frame, boxes_list, MODEL_MEAN_VALUES, gender_list, age_list_named)

        cv2.imshow("Age prediction", frame)

        key_pressed = cv2.waitKey(1)

        if key_pressed == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
elif mode == 2:
    frame = cv2.imread("assets/elder.jpg")
    boxes_list = extract_face_box(cv_net_face, frame, 0.7)
    print("Faces' detected: ", len(boxes_list))

    conf, result = predict_gender_and_age(frame, boxes_list, MODEL_MEAN_VALUES, gender_list, age_list_named)

    print("Confidence: {}\nResult: {}".format(conf, result))
    cv2.imshow("ciao", frame)
    cv2.waitKey(0)
