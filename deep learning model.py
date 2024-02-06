import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Load and preprocess the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channel dimension (for grayscale images)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Create the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)







# In this code:

# We first import the necessary libraries and load the MNIST dataset.
# We preprocess the dataset by normalizing the pixel values to the range [0, 1] and adding a channel dimension to accommodate the grayscale images.
# We define a sequential model using Keras's Sequential API. This model consists of a series of convolutional and pooling layers followed by fully connected layers.
# We compile the model, specifying the optimizer, loss function, and evaluation metrics.
# We train the model on the training data for 5 epochs and validate it using the test data.
# Finally, we evaluate the model's performance on the test data and print the test accuracy.