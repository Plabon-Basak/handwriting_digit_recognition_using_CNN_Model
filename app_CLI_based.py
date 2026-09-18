import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values (0-255) to (0-1) for better performance
X_train, X_test = X_train / 255.0, X_test / 255.0

# Reshape image for CNN input (28x28x1)
X_train_reshaped = X_train.reshape(-1, 28, 28, 1)
X_test_reshaped = X_test.reshape(-1, 28, 28, 1)


# Build the CNN model
model = Sequential([
	Input(shape=(28, 28, 1)),
	Conv2D(32, (3,3), activation='relu'),
	MaxPooling2D(2,2),
	Conv2D(64, (3,3), activation='relu'),
	MaxPooling2D(2,2),
	Flatten(),
	Dense(128, activation='relu'),
	Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_reshaped, y_train, epochs=1, validation_data=(X_test_reshaped, y_test))

test_loss, test_acc = model.evaluate(X_test_reshaped, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Select a test image
index = 0
test_image = X_test[index].reshape(1, 28, 28, 1)

# Predict digit
prediction = np.argmax(model.predict(test_image))
plt.imshow(X_test[index], cmap="gray")
plt.title(f"Predicted Digit: {prediction}")
plt.show()

# Model Summary
# model.summary()

# Display a sample digit
# plt.imshow(X_train[4], cmap="gray")
# plt.title(f"Label: {y_train[4]}")
# plt.show()


