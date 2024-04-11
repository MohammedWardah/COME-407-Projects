import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
import collections 
import tensorflow as tf
from sklearn import datasets 
from sklearn.model_selection import train_test_split

print('Running 1.1 ...')

# ---------- project implementation start:

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data
# Rescale the pixel values to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Flatten the images from 28x28 to 784
train_images = train_images.reshape((train_images.shape[0], -1))
test_images = test_images.reshape((test_images.shape[0], -1))

# Convert labels to float32
train_labels = train_labels.astype('float32')
test_labels = test_labels.astype('float32')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_images, test_images, train_labels, test_labels

# Define totalItems (number of features in each feature vector)
totalItems = X_train.shape[1]

# Define and compile the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(totalItems,)),  # No need for an extra dimension (1) here
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
epochs = 100
for epoch in range(1, epochs + 1):
    model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=0)
    print(f'Epoch {epoch}/{epochs}')
    print(f'Test loss: {test_loss:.4f} - Test accuracy: {test_acc:.4f}')

print('Training completed.')

# ---------- project implementation end:


# df_data = pd.DataFrame(data)
# df_data=df_data.astype('float32')    

# X_train, X_test, y_train, y_test = train_test_split(
#     df_data, digits.target, test_size=0.2, shuffle=False
# )

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(totalItems, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)
