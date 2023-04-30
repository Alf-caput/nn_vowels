import cv2
import numpy as np
import os


def main():
    face_detector = cv2.CascadeClassifier(r'haar_tools/haarcascade_frontalface_default.xml')
    mouth_detector = cv2.CascadeClassifier(r'haar_tools/mouth.xml')
    file_path = 0
    capture = cv2.VideoCapture(file_path)
    while capture.isOpened() and cv2.waitKey(1) not in (ord('s'), ord('S')):  # While 's' or 'S' not pressed
        read_successfully, main_frame = capture.read()
        if read_successfully:
            face_frame = get_face_frame(main_frame, face_detector)
            print(face_frame)
            if face_frame is not None:
                print("face found")
                mouth_frame = get_mouth_frame(face_frame, mouth_detector)
                if mouth_frame is not None:
                    print("mouth found")
                else:
                    print("No mouths found")
            else:
                print("No faces found")
            cv2.imshow('Captura', main_frame)
        else:
            print("Read unsuccessful")
    return 0


def get_face_frame(frame, face_detector):
    # For fewer false positives we convert to grayscale and blur the frame
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed_frame = cv2.blur(processed_frame, (10, 10))

    # Faces inside processed_frame
    faces = face_detector.detectMultiScale(processed_frame, minNeighbors=5)

    if np.any(faces):
        # faces is a np.array of np.arrays with x, y, w, h of each face found inside processed_frame
        x1, y1, w1, h1 = max(faces, key=lambda face: face[2] * face[3])

        # Slices for readability
        face_ypx, face_xpx = slice(y1, y1 + h1), slice(x1, x1 + w1)

        # Output frame
        face_frame = frame[face_ypx, face_xpx]

        # Face rectangle
        cv2.rectangle(frame,
                      (x1, y1), (x1 + w1, y1 + h1),
                      (255, 0, 0), 4)

        # Line 2 // 3 face (start point to check for mouth)
        cv2.line(face_frame,
                 (0, 2 * h1 // 3), (w1, 2 * h1 // 3),
                 (0, 255, 0), 4)
    else:
        face_frame = None
    return face_frame


def get_mouth_frame(face_frame, mouth_detector):
    # Mouths found in last 1 // 3 of face
    height, width = face_frame.shape[0:2]  # .shape atribute is a tuple with (height, width, channels)

    # Slices for readability
    y_third_ypx, x_xpx = slice(2 * height // 3, height), slice(0, width)

    # Mouths found in last 1 // 3 of face
    mouths = mouth_detector.detectMultiScale(face_frame[y_third_ypx, x_xpx])

    if np.any(mouths):
        # Biggest mouth found inside current face
        x2, y2, w2, h2 = max(mouths, key=lambda mouth: mouth[2] * mouth[3])

        # Slices for readability
        mouth_ypx, mouth_xpx = slice(y2, y2 + h2), slice(x2, x2 + w2)

        # Output frame
        mouth_frame = face_frame[y_third_ypx, x_xpx][mouth_ypx, mouth_xpx]

        # Mouth rectangle
        cv2.rectangle(face_frame[y_third_ypx, x_xpx],
                      (x2, y2), (x2 + w2, y2 + h2),
                      (0, 0, 255), 4)
    else:
        mouth_frame = None
    return mouth_frame


if __name__ == '__main__':
    main()
