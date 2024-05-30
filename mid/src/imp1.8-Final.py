from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import collections
import numpy as np
import pandas as pd
import tensorflow as tf

print("Running Version 1.8 ...")

# --------------------------------------------- project implementation start

drawn_digits = load_digits()

processed_images = []
centroids = []

for image in drawn_digits.images:
    resized_image = cv2.resize(image, (8, 8))

    _, thresh = cv2.threshold(resized_image, 7, 255, cv2.THRESH_BINARY)
    thresh = np.uint8(thresh)

    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])
            centroids.append((centroid_x, centroid_y))
            cv2.circle(thresh_color, (centroid_x, centroid_y), 0, (255, 0, 0), -1)
        else:
            centroids.append((-1, -1))
            
    if not contours:
        centroids.append((-1, -1))

    processed_images.append(thresh_color)

# Visualization: Digit occurrence count and data visualization part
digit_counts = collections.Counter(drawn_digits.target)
print("Digit Occurrences: ", digit_counts)

# Original and Processed Images
plt.figure(figsize=(12, 5))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(drawn_digits.images[i], cmap='gray')
    plt.title(f"Original Image ({drawn_digits.target[i]})")
    plt.axis('off')

for i in range(5):
    plt.subplot(2, 5, i + 6)
    plt.imshow(processed_images[i], cmap='gray')
    plt.title(f"Processed Image ({drawn_digits.target[i]})")
    plt.axis('off')

plt.tight_layout()
plt.show()

# Processed Data
data = np.array(processed_images).reshape(len(processed_images), -1)

# --------------------------------------------- project implementation end

# Given code for classification (Unedited):
df_data = pd.DataFrame(data)
df_data = df_data.astype('float32')

X_train, X_test, y_train, y_test = train_test_split(
    df_data, drawn_digits.target, test_size=0.2, shuffle=False
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