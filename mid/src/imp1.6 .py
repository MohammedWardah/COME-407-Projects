from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import collections
import numpy as np
import pandas as pd
import tensorflow as tf

print("Running 1.6 Version...")

# ------------------------------ project implementation start

# Load the digits dataset
digits = load_digits()

# Initialize lists to store processed images and centroids
processed_images = []
centroids = []

# Process each image
for image in digits.images:
    # Resize the image to 8x8 pixels
    resized_image = cv2.resize(image, (8, 8))

    # Apply binary thresholding
    _, thresh = cv2.threshold(resized_image, 7, 255, cv2.THRESH_BINARY)
    thresh = np.uint8(thresh)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = contours[0]
        M = cv2.moments(cnt)
    else:
        centroids.append((-1, -1))

    processed_images.append(thresh)  # Store the processed image


# Digit occurrence count and data visualization part:
digit_counts = collections.Counter(digits.target)
print(digit_counts)

# Original Images
plt.figure(figsize=(12, 5))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f"Original Image ({digits.target[i]})")
    plt.axis('off')

# Processed Images
for i in range(5):
    plt.subplot(2, 5, i + 6)
    plt.imshow(processed_images[i], cmap='gray')
    plt.title(f"Processed Image ({digits.target[i]})")
    plt.axis('off')

plt.tight_layout()
plt.show()


# Reshape data for further analysis
data = np.array(processed_images).reshape(len(processed_images), -1)
print("Processed data shape:", data.shape)

# ------------------------------ project implementation end

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
