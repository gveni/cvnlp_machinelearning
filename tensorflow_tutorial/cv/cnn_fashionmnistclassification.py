import tensorflow as tf
import numpy as np
from tensorflow import keras


# model hyper-parameters
num_epochs = 10
LR = 0.005


# class to handle callback
class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.1:
            print("\nReached accuracy > 60%, so terminating training")
            self.model.stop_training=True


callbacks = myCallback()  # create callback object

# load data
fashion_mnist = keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()
print('Size of training set:', len(training_images))
print('Size of test set:', len(test_images))
 
# reshape and scale images to [0 <-> 1]
training_images = training_images.reshape(len(training_images), len(training_images[0]), len(training_images[0][0]), 1)
training_images = training_images/255.
test_images = test_images.reshape(len(test_images), len(test_images[0]), len(test_images[0][0]), 1)
test_images = test_images/255.
print('Shape of training set:', np.shape(training_images))
print('Shape of test set:', np.shape(test_images))

# define model
model = tf.keras.models.Sequential([
    keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64,(3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# build the model by compiling it with optimizer and loss function
model.compile(loss='sparse_categorical_crossentropy',
optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=LR))
model.summary()

# train model
model.fit(training_images, training_labels, epochs=num_epochs, callbacks=[callbacks])

# evaluate model
model.evaluate(test_images, test_labels)
