import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D,MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.utils import plot_model


base_model=VGG16(input_shape=(224,224,3),include_top=False)
base_model.trainable=False

# Pre_trained model

def define_pre_trained_model(total_class):
    model = tf.keras.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(total_class,activation='softmax')
    ])
    model.summary()
    plot_model(base_model, to_file='models/base_model.png', show_shapes=True, dpi=600)
    plot_model(model, to_file='models/pre_trained_model.png', show_shapes=True, dpi=400)
    return model

# Customize model

def define_customize_model(total_class):
    model=Sequential()
    model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(224,224,3),activation='relu',padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(total_class,activation='softmax'))
    model.summary()
    plot_model(model, to_file='models/my_model.png', show_shapes=True, dpi=600)
    return model

   
   



