import cv2


def to_nn_input(frame):
    # Preprocess the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))

    # Flatten the image
    flat = resized.reshape((1, 784))

    # Normalize the pixel values
    normalized = flat / 255.0

    return normalized
