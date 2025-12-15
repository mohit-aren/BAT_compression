import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Convolution2D, Conv2D, MaxPool2D, AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.utils import get_file

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg19 import VGG19
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout, Lambda
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import load_img
import random

import numpy as np
from keras import applications
from keras.layers import Input
from keras.models import Model
from keras import optimizers
from keras.utils import get_file

import random
img_width, img_height = 256, 256
inp_1 = Input(shape=(img_height, img_width, 3))

vgg19 = keras.applications.vgg19.VGG19(
  weights='imagenet',
  include_top=False,
  input_tensor = inp_1)

x = vgg19.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)


model1=applications.ResNet50(weights="imagenet", include_top=False, input_tensor = inp_1)

y = model1.output
y = Flatten()(y)
y = Dense(1024, activation="relu")(y)
y = Dropout(0.5)(y)
y = Dense(512, activation="relu")(y)
#x = Dropout(0.5)(x)

combined = Lambda(lambda a: a[0] + a[1])([x, y])

predictions = Dense(2, activation="softmax")(combined) # 4-way softmax classifier at the end

model2 = Model(inp_1, predictions)

model2.summary()
#model.add(Dense(4, activation='softmax'))

#opt = Adam(learning_rate=0.001, beta_1=0.9)
classifier = model2
classifier.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.9), metrics=["accuracy"])

classifier.summary()

from google.colab import drive
drive.mount('/content/drive')

import glob
Covid = glob.glob('/content/drive/MyDrive/Covid/train/covid/*.*')
Control = glob.glob('/content/drive/MyDrive/Covid/train/control/*.*')


data = []
labels = []

idx = 0
for i in Covid:
    print(i)
    image=load_img(i,
    target_size= (256,256))
    image=np.array(image)
    data.append(image)
    labels.append(0)
    if(idx > 200):
      break
    idx += 1
idx = 0
for i in Control:
    print(i)
    image=load_img(i,
    target_size= (256,256))
    image=np.array(image)
    data.append(image)
    labels.append(1)
    if(idx > 200):
      break
    idx += 1

data = np.array(data)
labels = np.array(labels)

from keras.utils import to_categorical

categorical_labels = to_categorical(labels, num_classes=2)


from sklearn.model_selection import train_test_split

#X_train1, X_test0, ytrain1, ytest0 = train_test_split(data, categorical_labels, test_size=0.1,
                                                    #random_state=random.randint(0,100))
X_train1 = data
ytrain1 = categorical_labels


#classifier = Model(input=model.input, output=predictions)


X_train, X_test1, ytrain, ytest1 = train_test_split(X_train1, ytrain1, test_size=0.1,
                                                random_state=random.randint(0,100))

X_val, X_test, yval, ytest = train_test_split(X_test1, ytest1, test_size=0.5,
                                                random_state=random.randint(0,100))

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

val_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow(
        X_train, ytrain,
        batch_size=64)

val_set = val_datagen.flow(
        X_val, yval,
        batch_size=64)

test_set = test_datagen.flow(
        X_test, ytest,
        batch_size=64)

classifier.fit(
        training_set,
        steps_per_epoch=20,
        epochs=5,
        validation_data=val_set,
        validation_steps=20)

w_file = '/content/drive/MyDrive/Covid/keras_ensemble_main1.weights.h5'
classifier.save_weights(w_file)

arr = classifier.evaluate(test_set)
print(arr)
