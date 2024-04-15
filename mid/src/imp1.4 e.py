import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

print('Running 1.4 ...')

# -------------------- project implementation start:

# Load the digits dataset
digits = load_digits()

# Preprocess the images and calculate moments
processed_images = []
centroids = []

for image in digits.images:
    # Resize the image to 8x8
    resized_image = cv2.resize(image, (8, 8))

    # Apply thresholding
    ret, thresh = cv2.threshold(resized_image, 127, 255, cv2.THRESH_BINARY)

    # Convert to binary format (8-bit, single-channel)
    thresh = np.uint8(thresh)

    # Find contours - what is _ ?
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if contours are found
    if contours:
        cnt = contours[0]  # Assuming only one contour is found
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        centroids.append((cx, cy))
    else:
        # If no contours are found, set centroid to (-1, -1)
        centroids.append((-1, -1))

    processed_images.append(resized_image)


# Flatten the images
data =  np.array(processed_images).reshape(len(processed_images), -1)

# Visualization - optional
# Display some images from the dataset
num_images_to_display = 10  # Change this number as needed
for i in range(num_images_to_display):
    plt.figure(figsize=(2, 2))
    plt.imshow(digits.images[i], cmap='gray')  # Display grayscale images
    plt.title(f"Label: {digits.images[i]}")
    plt.axis('off')  # Hide axis
    plt.show()

# -------------------- project implementation end

# Create dataframe
df_data = pd.DataFrame(data)
df_data=df_data.astype('float32')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df_data, digits.target, test_size=0.2, shuffle=False
)
totalItems = df_data.shape[1]
# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(totalItems, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model
model.fit(X_train, y_train, epochs=100)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)