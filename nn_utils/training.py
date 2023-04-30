import numpy as np
import tensorflow as tf

batch_size = 32

# Load CSV data into NumPy matrices
mnist_images = np.loadtxt('imagenes_mnist.csv', delimiter=',')
mnist_labels = np.loadtxt('etiquetas_mnist.csv', delimiter=',')

# Convert data to tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((mnist_images, mnist_labels))

# Shuffle and divide dataset in batches for training
dataset = dataset.shuffle(buffer_size=len(mnist_labels)).batch(batch_size)

# Define neuronal network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Model compiling
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model training
model.fit(dataset, epochs=200)

model.save('../nn_model')
