import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt


(x_train,y_train), (x_test,y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 250.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test  = x_test.reshape(x_test.shape[0],   28, 28, 1)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    #horizontal_flip=1,
    #shear_range=1,
    #brightness_range=(0,1),
)
datagen.fit(x_train)

initial_learning_rate = 0.01
batch_size = 64
epochs = 10

dropout_rate = 0.2

def lr_schedule(epoch,lr):
    if epoch > 0 and epoch%3 == 0:
        return lr * 0.5
    return lr

model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128,activation="relu"),
    Dropout(dropout_rate),
    Dense(64,activation="relu"),
    Dropout(dropout_rate),
    Dense(10,activation="softmax")
])

optimizer = SGD(learning_rate=initial_learning_rate)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(lr_schedule)

history = model.fit(datagen.flow(x_train,y_train,batch_size=batch_size),
                    epochs=epochs,
                    validation_data=(x_test,y_test),
                    callbacks=[early_stopping,lr_scheduler])
test_loss,test_accuracy = model.evaluate(x_test,y_test)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

plt.plot(history.history["accuracy"],label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.plot(history.history["loss"],label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
