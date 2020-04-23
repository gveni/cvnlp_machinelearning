import tensorflow as tf
import numpy as np
from tensorflow import keras


# model hyper-parameters
num_epochs = 10
LR = 0.005

# load data
fashion_mnist = keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()
print('Size of training set:', len(training_images))
print('Size of test set:', len(test_images))
 
# scale images to [0 <-> 1]
training_images = training_images/255.
test_images = test_images/255.

# define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# build the model by compiling it with optimizer and loss function
model.compile(loss='sparse_categorical_crossentropy',
optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=LR))

# train model
model.fit(training_images, training_labels, epochs = num_epochs)

# evaluate model
model.evaluate(test_images, test_labels)
