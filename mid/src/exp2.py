from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import collections
import numpy as np
import pandas as pd
import tensorflow as tf

print('Running Version xp w ...')

# -------------------- project implementation start:

digits = load_digits()

processed_images = []
centroids = []

for image in digits.images:

    resized_image = cv2.resize(image, (8, 8))
    ret, thresh = cv2.threshold(resized_image, 7, 255, cv2.THRESH_BINARY)
    thresh = np.uint8(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = contours[0]
        M = cv2.moments(cnt)
        if M['m00'] != 0:  # Check if the denominator is not zero
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids.append((cx, cy))
        else:
            # Handle the case where the denominator is zero
            centroids.append((-1, -1))
    else:
        # Handle the case where no contours are detected
        centroids.append((-1, -1))

    processed_images.append(resized_image)

# Digit occurrence count and data visualization part:
digit_counts = collections.Counter(digits.target)
print(digit_counts)

num_images_to_display = 1
for i in range(num_images_to_display):
    plt.figure(figsize=(2, 2))
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f"Label: {digits.images[i]}")
    plt.axis('off')
    plt.show()

num_images_to_display = 1
for i in range(num_images_to_display):
    plt.figure(figsize=(2, 2))
    plt.imshow(processed_images[i], cmap='gray')
    plt.title(f"Label: {processed_images[i]}")
    plt.axis('off')
    plt.show()

data = np.array(processed_images).reshape(len(processed_images), -1)
print(data)

# -------------------- project implementation end


# Given code for classification (unedited):
df_data = pd.DataFrame(data)
df_data = df_data.astype('float32')

X_train, X_test, y_train, y_test = train_test_split(
    df_data, digits.target, test_size=0.2, shuffle=False
)

totalItems = df_data.shape[1]
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(totalItems, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=100)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)
