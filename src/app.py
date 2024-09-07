import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import image_dataset_from_directory
from keras import Sequential
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization
import tensorflow as tf


class CatDog:
    def __init__(self):
        self.train, self.test = image_dataset_from_directory('../.venv/CatDog/dogs-vs-cats/train', image_size=(224, 224),
                                                             subset='both', validation_split=0.2, seed=42, batch_size=8, label_mode='binary')
        # # self.pre_check()
        self.model = Sequential()
        self.model_init()
        try:
            self.model.load_weights('../.venv/CatDog/checkpoint.model.keras')
        except FileNotFoundError:
            self.model.fit(self.train, batch_size=8, epochs=50, callbacks=[ModelCheckpoint('CatDog/checkpoint.model.keras', monitor='binary_accuracy', mode='max', save_best_only=True), EarlyStopping(monitor='loss', patience=6)])
        finally:
            self.model.predict(self.test)

    def model_init(self):
        self.model.add(
            Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(units=4096, activation="relu"))
        self.model.add(Dense(units=4096, activation="relu"))
        self.model.add(Dense(units=1, activation="sigmoid"))
        self.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3), metrics=['binary_accuracy'])

    def pre_check(self):
        cats = []
        dogs = []
        for i in range(9):
            cats.append(image.img_to_array(image.load_img(f'CatDog/dogs-vs-cats/train/Cat/cat.{i}.jpg')))
            dogs.append(image.img_to_array(image.load_img(f'CatDog/dogs-vs-cats/train/Dog/dog.{i}.jpg')))
        fig, axs = plt.subplots(len(cats), 2)
        for i in range(len(cats)):
            axs[i, 0].imshow(cats[i] / 255)
            axs[i, 1].imshow(dogs[i] / 255)
        plt.show()
        # Fucking adorable


if __name__ == '__main__':
    CatDog()
