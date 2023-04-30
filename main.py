from nn_utils import *
from facial_detection import *
import tensorflow as tf


def main():
    face_detector = cv2.CascadeClassifier(r'haar_tools/haarcascade_frontalface_default.xml')
    mouth_detector = cv2.CascadeClassifier(r'haar_tools/mouth.xml')
    file_path = 0

    loaded_model = tf.keras.models.load_model('nn_model')
    facial_capture(loaded_model, face_detector, mouth_detector, file_path)

    return 0


def facial_capture(loaded_model, face_detector, mouth_detector, file_path=0):
    capture = cv2.VideoCapture(file_path)
    while capture.isOpened() and cv2.waitKey(1) not in (ord('s'), ord('S')):  # While 's' or 'S' not pressed
        read_successfully, main_frame = capture.read()
        if read_successfully:
            face_frame = get_face_frame(main_frame, face_detector)
            if face_frame is not None:
                print("face found")
                mouth_frame = get_mouth_frame(face_frame, mouth_detector)
                if mouth_frame is not None:
                    print("mouth found")
                    target = to_nn_input(mouth_frame)
                    prediction = loaded_model.predict(target)
                    print("Mi predicción es:", prediction)
                else:
                    print("No mouths found")
            else:
                print("No faces found")
            cv2.imshow('Captura', main_frame)
        else:
            print("Read unsuccessful")
    return 0


if __name__ == '__main__':
    main()
