import kagglehub
path = kagglehub.dataset_download("hongweicao/catanddogsmall")
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

print("Path to dataset Files:", path)
#r"C:\Users\flori\.cache\kagglehub\datasets\hongweicao\catanddogsmall\versions\1"

train_dir = path + "/dogvscat_small/train"
val_dir = path + "/dogvscat_small/validation"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode="binary"
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode="binary"
)

base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))

for layer in base_model.layers:
    layer.trainable=False


x = base_model.output
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_generator, epochs=10, validation_data=val_generator)

loss, accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {accuracy*100:.2f}%")

img_path = path + "/dogvscat_small/test/dogs/1500.jpg"
img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)
class_label = "Dog" if prediction[0] > 0.5 else "Cat"
print(f"Predicted Class: {class_label}")

images, labels = next(val_generator)

predictions = model.predict(images)

plt.figure(figsize=(12,12))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(images[i])
    predicted_label = "Dog" if predictions[i] > 0.5 else "Cat"
    true_label = "Dog" if labels[i] == 1 else "Cat"
    plt.title(f"True: {true_label}, Pred: {predicted_label}")
    plt.axis("Off")
plt.show()