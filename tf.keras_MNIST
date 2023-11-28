# Importing libraries

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# tensorflow.keras is the cover command that is used to access most of its functions

# models.Sequential is a function that creates and initializes the neural network
from tensorflow.keras.models import Sequential

# Use the layers library to find prebuilt neural net layers such as:
# Dense for ANN's
# Conv2D for CNN's
# BatchNormalization for data control
# MaxPooling2D for pooling operations in CNN's
# Dropout to prevent overfitting
# Masking to manipulate images

# layers.Dense creates a fully-connected ANN layer
from tensorflow.keras.layers import Dense

# Use the optimizers library to find prebuilt optimizers like RMSprop, Adam, SGD, and many more
# optimizers.SGD imports the Stochastic Gradient Descent algorithm as an optimizer
from tensorflow.keras.optimizers import SGD

# Use the datasets library to upload precurated datasets
# In this example, we upload the MNIST dataset, which contains handwritten drawings of numerals from 0-9
from tensorflow.keras.datasets import mnist

# Use the backend library to manage applications running within the program
from tensorflow.keras import backend as K

# The rest of these libraries are utilities to provide basic functions and visualize the output
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Starting Keras Training

((trainX, trainY), (testX, testY)) = mnist.load_data()

trainX = trainX.reshape((trainX.shape[0], 28 * 28 * 1))
testX = testX.reshape((testX.shape[0], 28 * 28 * 1))

trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# Creating and initializing the model
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(64, activation="sigmoid"))
model.add(Dense(32, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
	metrics=["accuracy"])

# This one line, line 66, constitutes the entire training process
# That's how easy it is to implement neural nets with tf.keras
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=500, batch_size=128)

# This one line, line 70, constitutes the entire testing process
predictions = model.predict(testX, batch_size=128)

# Gives us a classification report with all relevant statistics on accuracy
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=[str(x) for x in lb.classes_]))

# Visualizing data
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(keras_mnist.jpg)
