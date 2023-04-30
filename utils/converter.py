import cv2
import numpy as np


def to_mnist(frame, file_path):
    # Preprocess the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))

    # Flatten the image
    flat = resized.reshape((1, 784))

    # Normalize the pixel values
    normalized = flat / 255.0

    # Save the image as a CSV file in MNIST format
    np.savetxt(file_path, normalized, delimiter=',')
    return
