import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
import collections 
import tensorflow as tf
from sklearn import datasets 
from sklearn.model_selection import train_test_split

print('Running 1.2 ...')

# ---------- project implementation start:

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Compute moments for each image
moments_train = []
for img in train_images:
    moments = cv2.moments(img)
    moments_train.append([moments[key] for key in moments.keys()])

moments_test = []
for img in test_images:
    moments = cv2.moments(img)
    moments_test.append([moments[key] for key in moments.keys()])

# Convert moments to DataFrames
df_train = pd.DataFrame(moments_train)
df_test = pd.DataFrame(moments_test)

# Reshape the input images
X_train = df_train.values.reshape((df_train.shape[0], -1)).astype('float32')
X_test = df_test.values.reshape((df_test.shape[0], -1)).astype('float32')

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(train_labels, num_classes=10)
y_test = tf.keras.utils.to_categorical(test_labels, num_classes=10)

# Define totalItems (number of features in each feature vector)
totalItems = X_train.shape[1]
data = X_train
digits_target = y_train


# ---------- project implementation end


# Fixed given code for classification:
df_data = pd.DataFrame(data)
df_data=df_data.astype('float32')    

X_train, X_test, y_train, y_test = train_test_split(
    df_data, digits_target, test_size=0.2, shuffle=False
)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(totalItems, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# edited
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# old
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#             metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)

