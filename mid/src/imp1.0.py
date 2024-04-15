import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
import collections 
import tensorflow as tf
from sklearn import datasets 
from sklearn.model_selection import train_test_split

print('Running...')

# ---------- project implementation start:






# ---------- project implementation end:


df_data = pd.DataFrame(data)
df_data=df_data.astype('float32')    

X_train, X_test, y_train, y_test = train_test_split(
    df_data, digits.target, test_size=0.2, shuffle=False
)

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
