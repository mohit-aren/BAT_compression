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
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.utils import get_file

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16
import random
img_width, img_height = 256, 256
'''
vgg19 = tf.keras.applications.vgg19.VGG19(
  weights='imagenet',
  include_top=False,
  input_shape=(img_height, img_width, 3))

for layer in vgg19.layers:
  layer.trainable = False

model = Sequential(layers=vgg19.layers)
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='softmax'))
'''
model=VGG16(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

#initialise top model
"""
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(10, activation='softmax'))

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
weights_path = get_file('vgg16_weights.h5', WEIGHTS_PATH_NO_TOP)

vgg_model.load_weights(weights_path)

# add the model on top of the convolutional base

model_final = Model(input= vgg_model.input, output= top_model(vgg_model.output))
"""
# Freeze first 15 layers
for layer in model.layers[:45]:
	layer.trainable = False
for layer in model.layers[45:]:
   layer.trainable = True


x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation="softmax")(x) # 4-way softmax classifier at the end

model_final = Model(inputs=model.input, outputs=predictions)

model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(learning_rate=1e-3, momentum=0.9), metrics=["accuracy"])

model_final.summary()

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
print(np.array(X_train1).shape)
print(np.array(ytrain1).shape)

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

X, y = next(test_set)
model_final.fit(
        training_set,
        steps_per_epoch=20,
        epochs=20,
        validation_data=val_set,
        validation_steps=20)

#model_final.save_weights('/content/drive/MyDrive/Covid/keras_vgg19_main.weights.h5')
model = model_final
model_path = '/content/drive/MyDrive/Covid/keras_vgg19_main.weights.h5'
#model = tf.keras.models.load_model(model_path)
model.save_weights(model_path)

model_final.load_weights('/content/drive/MyDrive/Covid/keras_vgg19_main.weights.h5')
arr = model_final .evaluate(X, y)
print(arr)

layer1_b = 128
layer2_b = 256
layer3_b = 512
layer4_b = 512

layer5_b = 1024
layer6_b = 1024

layer1_a = 128
layer2_a = 256
layer3_a = 512
layer4_a = 512

layer5_a = 1024
layer6_a = 1024

model1 = Sequential()
model1.add(Conv2D(input_shape=(256,256,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
#model1.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model1.add(Conv2D(filters=layer1_a, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model1.add(Conv2D(filters=layer2_a, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model1.add(Conv2D(filters=layer3_a, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model1.add(Conv2D(filters=layer4_a, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model1.add(Flatten())
model1.add(Dense(units=layer5_a,activation="relu"))
model1.add(Dense(units=layer6_a,activation="relu"))
model1.add(Dense(2))
model1.add(Activation('softmax'))

model1.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(learning_rate=1e-3, momentum=0.9), metrics=["accuracy"])

model1.summary()

#y_test = y_test.reshape(len(y_test), 1)


#onehot_encoded = onehot_encoder.fit_transform(y_test)

####################### 1st convolution layer with 128 filters
print('1st convolution layer with 128 filters')
A = []
Acc = []

arr = model_final .evaluate(X, y)
print(arr)

filters, biases = model_final.layers[4].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

def ensure_bounds(par):
    new_par = []
    for index in range(0, len(par)):
        if(par[index] >= 0.5):
            new_par.append(1)
        else:
            new_par.append(0)

    return new_par

# Parameters
num_bats = 10
dim = 128
num_iterations = 10
freq_min = 0
freq_max = 2
A = 0.5
r0 = 0.5
alpha = 0.9
gamma = 0.9
lb = 0
ub = 1

# Initialize bat positions and velocities
positions = np.random.uniform(lb, ub, (num_bats, dim))
velocities = np.zeros((num_bats, dim))
frequencies = np.zeros(num_bats)
loudness = A * np.ones(num_bats)
pulse_rate = r0 * np.ones(num_bats)

fitness = []
for i1 in range(num_bats):
    pop = positions[i1]
    #print(pop)
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    # Evaluate initial fitness
    retain_nodes = 0
    for i in range(0,128):
        f = filters[:, :, :, i]
        #for j in range(3):
            #if(child[i] == 0):
                #filters1[:, :, :, i][:,:,j] = 0
        if(pop[i] < 0.5):
            biases1[i] = 0
            filters1[:, :, :, i] = 0
        else:
            retain_nodes += 1

    model_final.layers[4].set_weights([filters1, biases1])
    arr = model_final.evaluate(X, y)
    fitness.append(arr[1])


#fitness = np.apply_along_axis(sphere_function, 1, positions)
best_position = positions[np.argmax(fitness)]
best_fitness = np.max(fitness)
print(np.array(loudness).shape)
for iteration in range(num_iterations):
    avg_loudness = np.mean(loudness)
    avg_pulse_rate = np.mean(pulse_rate)

    # Update bats
    for i in range(num_bats):
        beta = np.random.uniform(0, 1)
        frequencies[i] = freq_min + (freq_max - freq_min) * beta
        velocities[i] += (best_position - positions[i]) * frequencies[i]
        new_position = positions[i] + velocities[i]

        # Boundary check
        new_position = np.clip(new_position, lb, ub)

        # Local search
        if np.random.uniform(0, 1) > pulse_rate[i]:
            epsilon = np.random.uniform(-1, 1)
            new_position = positions[i] + epsilon * avg_loudness

        # Evaluate new solution
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        # Evaluate initial fitness
        retain_nodes = 0
        for i1 in range(0,128):
            f = filters[:, :, :, i1]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(new_position[i1] < 0.5):
                biases1[i1] = 0
                filters1[:, :, :, i1] = 0
            else:
                retain_nodes += 1
        print('Nodes left', retain_nodes)
        model_final.layers[4].set_weights([filters1, biases1])
        arr = model_final .evaluate(X, y)
        #fitness.append(arr[1])
        new_fitness = arr[1] #sphere_function(new_position)

        # Greedy mechanism to update if new solution is better and random value is less than loudness
        if new_fitness > fitness[i] and np.random.uniform(0, 1) < loudness[i]:
            positions[i] = new_position
            fitness[i] = new_fitness

        # Update global best
        if fitness[i] > best_fitness:
            best_position = positions[i]
            best_fitness = fitness[i]

        loudness[i] *= alpha
        pulse_rate[i] = r0 * (1 - np.exp(-gamma * iteration))

    # Print the best fitness value in each iteration
    print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")

print("\nOptimized Solution:", best_position)
print("Best Fitness Value:", best_fitness)


A1 = np.copy(best_position)
new_num = np.sum(ensure_bounds(best_position))

print(new_num)
layer1_a = new_num

####################### 1st convolution layer with 256 filters
print('1st convolution layer with 256 filters')
A = []
Acc = []
# Evaluate new solution
filters1 = np.copy(filters)
biases1 = np.copy(biases)
# Evaluate initial fitness
retain_nodes = 0
for i1 in range(0,128):
    f = filters[:, :, :, i1]
    #for j in range(3):
        #if(child[i] == 0):
            #filters1[:, :, :, i][:,:,j] = 0
    if(new_position[i1] < 0.5):
        biases1[i1] = 0
        filters1[:, :, :, i1] = 0
    else:
        retain_nodes += 1
print('Nodes left', retain_nodes)
model_final.layers[4].set_weights([filters1, biases1])
arr = model_final .evaluate(X, y)
#arr = model_final .evaluate(X, y)
print(arr)

filters, biases = model_final.layers[7].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

def ensure_bounds(par):
    new_par = []
    for index in range(0, len(par)):
        if(par[index] >= 0.5):
            new_par.append(1)
        else:
            new_par.append(0)

    return new_par

# Parameters
num_bats = 10
dim = 256
num_iterations = 10
freq_min = 0
freq_max = 2
A = 0.5
r0 = 0.5
alpha = 0.9
gamma = 0.9
lb = 0
ub = 1

# Initialize bat positions and velocities
positions = np.random.uniform(lb, ub, (num_bats, dim))
velocities = np.zeros((num_bats, dim))
frequencies = np.zeros(num_bats)
loudness = A * np.ones(num_bats)
pulse_rate = r0 * np.ones(num_bats)

fitness = []
for i1 in range(num_bats):
    pop = positions[i1]
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    # Evaluate initial fitness
    retain_nodes = 0
    for i in range(0,256):
        f = filters[:, :, :, i]
        #for j in range(3):
            #if(child[i] == 0):
                #filters1[:, :, :, i][:,:,j] = 0
        if(pop[i] < 0.5):
            biases1[i] = 0
            filters1[:, :, :, i] = 0
        else:
            retain_nodes += 1

    model_final.layers[7].set_weights([filters1, biases1])
    arr = model_final .evaluate(X, y)
    fitness.append(arr[1])


#fitness = np.apply_along_axis(sphere_function, 1, positions)
best_position = positions[np.argmax(fitness)]
best_fitness = np.max(fitness)

for iteration in range(num_iterations):
    avg_loudness = np.mean(loudness)
    avg_pulse_rate = np.mean(pulse_rate)

    # Update bats
    for i in range(num_bats):
        beta = np.random.uniform(0, 1)
        frequencies[i] = freq_min + (freq_max - freq_min) * beta
        velocities[i] += (best_position - positions[i]) * frequencies[i]
        new_position = positions[i] + velocities[i]

        # Boundary check
        new_position = np.clip(new_position, lb, ub)

        # Local search
        if np.random.uniform(0, 1) > pulse_rate[i]:
            epsilon = np.random.uniform(-1, 1)
            new_position = positions[i] + epsilon * avg_loudness

        # Evaluate new solution
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        # Evaluate initial fitness
        retain_nodes = 0
        for i1 in range(0,256):
            f = filters[:, :, :, i1]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(new_position[i1] < 0.5):
                biases1[i1] = 0
                filters1[:, :, :, i1] = 0
            else:
                retain_nodes += 1

        model_final.layers[7].set_weights([filters1, biases1])
        arr = model_final .evaluate(X, y)
        #fitness.append(arr[1])
        new_fitness = arr[1] #sphere_function(new_position)

        # Greedy mechanism to update if new solution is better and random value is less than loudness
        if new_fitness > fitness[i] and np.random.uniform(0, 1) < loudness[i]:
            positions[i] = new_position
            fitness[i] = new_fitness

        # Update global best
        if fitness[i] > best_fitness:
            best_position = positions[i]
            best_fitness = fitness[i]

        loudness[i] *= alpha
        pulse_rate[i] = r0 * (1 - np.exp(-gamma * iteration))

    # Print the best fitness value in each iteration
    print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")

print("\nOptimized Solution:", best_position)
print("Best Fitness Value:", best_fitness)


A2 = np.copy(best_position)
new_num = np.sum(ensure_bounds(best_position))

print(new_num)
layer2_a = new_num

####################### 1st convolution layer with 512 filters
print('1st convolution layer with 512 filters')
A = []
Acc = []

filters1 = np.copy(filters)
biases1 = np.copy(biases)
# Evaluate initial fitness
retain_nodes = 0
for i1 in range(0,256):
    f = filters[:, :, :, i1]
    #for j in range(3):
        #if(child[i] == 0):
            #filters1[:, :, :, i][:,:,j] = 0
    if(new_position[i1] < 0.5):
        biases1[i1] = 0
        filters1[:, :, :, i1] = 0
    else:
        retain_nodes += 1
print('Nodes left', retain_nodes)
model_final.layers[7].set_weights([filters1, biases1])
arr = model_final .evaluate(X, y)
print(arr)

filters, biases = model_final.layers[11].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

def ensure_bounds(par):
    new_par = []
    for index in range(0, len(par)):
        if(par[index] >= 0.5):
            new_par.append(1)
        else:
            new_par.append(0)

    return new_par

# Parameters
num_bats = 10
dim = 512
num_iterations = 10
freq_min = 0
freq_max = 2
A = 0.5
r0 = 0.5
alpha = 0.9
gamma = 0.9
lb = 0
ub = 1

# Initialize bat positions and velocities
positions = np.random.uniform(lb, ub, (num_bats, dim))
velocities = np.zeros((num_bats, dim))
frequencies = np.zeros(num_bats)
loudness = A * np.ones(num_bats)
pulse_rate = r0 * np.ones(num_bats)

fitness = []
for i1 in range(num_bats):
    pop = positions[i1]
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    # Evaluate initial fitness
    retain_nodes = 0
    for i in range(0,512):
        f = filters[:, :, :, i]
        #for j in range(3):
            #if(child[i] == 0):
                #filters1[:, :, :, i][:,:,j] = 0
        if(pop[i] < 0.5):
            biases1[i] = 0
            filters1[:, :, :, i] = 0
        else:
            retain_nodes += 1

    model_final.layers[11].set_weights([filters1, biases1])
    arr = model_final .evaluate(X, y)
    fitness.append(arr[1])


#fitness = np.apply_along_axis(sphere_function, 1, positions)
best_position = positions[np.argmax(fitness)]
best_fitness = np.max(fitness)

for iteration in range(num_iterations):
    avg_loudness = np.mean(loudness)
    avg_pulse_rate = np.mean(pulse_rate)

    # Update bats
    for i in range(num_bats):
        beta = np.random.uniform(0, 1)
        frequencies[i] = freq_min + (freq_max - freq_min) * beta
        velocities[i] += (best_position - positions[i]) * frequencies[i]
        new_position = positions[i] + velocities[i]

        # Boundary check
        new_position = np.clip(new_position, lb, ub)

        # Local search
        if np.random.uniform(0, 1) > pulse_rate[i]:
            epsilon = np.random.uniform(-1, 1)
            new_position = positions[i] + epsilon * avg_loudness

        # Evaluate new solution
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        # Evaluate initial fitness
        retain_nodes = 0
        for i1 in range(0,512):
            f = filters[:, :, :, i1]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(new_position[i1] < 0.5):
                biases1[i1] = 0
                filters1[:, :, :, i1] = 0
            else:
                retain_nodes += 1

        model_final.layers[11].set_weights([filters1, biases1])
        arr = model_final .evaluate(X, y)
        #fitness.append(arr[1])
        new_fitness = arr[1] #sphere_function(new_position)

        # Greedy mechanism to update if new solution is better and random value is less than loudness
        if new_fitness > fitness[i] and np.random.uniform(0, 1) < loudness[i]:
            positions[i] = new_position
            fitness[i] = new_fitness

        # Update global best
        if fitness[i] > best_fitness:
            best_position = positions[i]
            best_fitness = fitness[i]

        loudness[i] *= alpha
        pulse_rate[i] = r0 * (1 - np.exp(-gamma * iteration))

    # Print the best fitness value in each iteration
    print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")

print("\nOptimized Solution:", best_position)
print("Best Fitness Value:", best_fitness)


A3 = np.copy(best_position)
new_num = np.sum(ensure_bounds(best_position))

print(new_num)
layer3_a = new_num

####################### 2nd convolution layer with 512 filters
print('2nd convolution layer with 512 filters')
A = []
Acc = []

filters1 = np.copy(filters)
biases1 = np.copy(biases)
# Evaluate initial fitness
retain_nodes = 0
for i1 in range(0,512):
    f = filters[:, :, :, i1]
    #for j in range(3):
        #if(child[i] == 0):
            #filters1[:, :, :, i][:,:,j] = 0
    if(new_position[i1] < 0.5):
        biases1[i1] = 0
        filters1[:, :, :, i1] = 0
    else:
        retain_nodes += 1
print('Nodes left', retain_nodes)
model_final.layers[11].set_weights([filters1, biases1])
arr = model_final .evaluate(X, y)
print(arr)

filters, biases = model_final.layers[15].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

def ensure_bounds(par):
    new_par = []
    for index in range(0, len(par)):
        if(par[index] >= 0.5):
            new_par.append(1)
        else:
            new_par.append(0)

    return new_par

# Parameters
num_bats = 10
dim = 512
num_iterations = 10
freq_min = 0
freq_max = 2
A = 0.5
r0 = 0.5
alpha = 0.9
gamma = 0.9
lb = 0
ub = 1

# Initialize bat positions and velocities
positions = np.random.uniform(lb, ub, (num_bats, dim))
velocities = np.zeros((num_bats, dim))
frequencies = np.zeros(num_bats)
loudness = A * np.ones(num_bats)
pulse_rate = r0 * np.ones(num_bats)

fitness = []
for i1 in range(num_bats):
    pop = positions[i1]
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    # Evaluate initial fitness
    retain_nodes = 0
    for i in range(0,512):
        f = filters[:, :, :, i]
        #for j in range(3):
            #if(child[i] == 0):
                #filters1[:, :, :, i][:,:,j] = 0
        if(pop[i] < 0.5):
            biases1[i] = 0
            filters1[:, :, :, i] = 0
        else:
            retain_nodes += 1

    model_final.layers[15].set_weights([filters1, biases1])
    arr = model_final .evaluate(X, y)
    fitness.append(arr[1])


#fitness = np.apply_along_axis(sphere_function, 1, positions)
best_position = positions[np.argmax(fitness)]
best_fitness = np.max(fitness)

for iteration in range(num_iterations):
    avg_loudness = np.mean(loudness)
    avg_pulse_rate = np.mean(pulse_rate)

    # Update bats
    for i in range(num_bats):
        beta = np.random.uniform(0, 1)
        frequencies[i] = freq_min + (freq_max - freq_min) * beta
        velocities[i] += (best_position - positions[i]) * frequencies[i]
        new_position = positions[i] + velocities[i]

        # Boundary check
        new_position = np.clip(new_position, lb, ub)

        # Local search
        if np.random.uniform(0, 1) > pulse_rate[i]:
            epsilon = np.random.uniform(-1, 1)
            new_position = positions[i] + epsilon * avg_loudness

        # Evaluate new solution
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        # Evaluate initial fitness
        retain_nodes = 0
        for i1 in range(0,512):
            f = filters[:, :, :, i1]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(new_position[i1] < 0.5):
                biases1[i1] = 0
                filters1[:, :, :, i1] = 0
            else:
                retain_nodes += 1

        model_final.layers[15].set_weights([filters1, biases1])
        arr = model_final .evaluate(X, y)
        #fitness.append(arr[1])
        new_fitness = arr[1] #sphere_function(new_position)

        # Greedy mechanism to update if new solution is better and random value is less than loudness
        if new_fitness > fitness[i] and np.random.uniform(0, 1) < loudness[i]:
            positions[i] = new_position
            fitness[i] = new_fitness

        # Update global best
        if fitness[i] > best_fitness:
            best_position = positions[i]
            best_fitness = fitness[i]

        loudness[i] *= alpha
        pulse_rate[i] = r0 * (1 - np.exp(-gamma * iteration))

    # Print the best fitness value in each iteration
    print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")

print("\nOptimized Solution:", best_position)
print("Best Fitness Value:", best_fitness)


A4 = np.copy(best_position)
new_num = np.sum(ensure_bounds(best_position))

print(new_num)
layer4_a = new_num

####################### 1st dense layer with 1024 filters
print('1st dense layer with 1024 filters')
A = []
Acc = []

filters1 = np.copy(filters)
biases1 = np.copy(biases)
# Evaluate initial fitness
retain_nodes = 0
for i1 in range(0,512):
    f = filters[:, :, :, i1]
    #for j in range(3):
        #if(child[i] == 0):
            #filters1[:, :, :, i][:,:,j] = 0
    if(new_position[i1] < 0.5):
        biases1[i1] = 0
        filters1[:, :, :, i1] = 0
    else:
        retain_nodes += 1
print('Nodes left', retain_nodes)
model_final.layers[15].set_weights([filters1, biases1])
arr = model_final .evaluate(X, y)
print(arr)

filters, biases = model_final.layers[20].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

def ensure_bounds(par):
    new_par = []
    for index in range(0, len(par)):
        if(par[index] >= 0.5):
            new_par.append(1)
        else:
            new_par.append(0)

    return new_par

# Parameters
num_bats = 10
dim = 1024
num_iterations = 10
freq_min = 0
freq_max = 2
A = 0.5
r0 = 0.5
alpha = 0.9
gamma = 0.9
lb = 0
ub = 1

# Initialize bat positions and velocities
positions = np.random.uniform(lb, ub, (num_bats, dim))
velocities = np.zeros((num_bats, dim))
frequencies = np.zeros(num_bats)
loudness = A * np.ones(num_bats)
pulse_rate = r0 * np.ones(num_bats)

fitness = []
for i1 in range(num_bats):
    pop = positions[i1]
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    # Evaluate initial fitness
    retain_nodes = 0
    for i in range(0,1024):
        f = filters[:, i]
        #for j in range(3):
            #if(child[i] == 0):
                #filters1[:, :, :, i][:,:,j] = 0
        if(pop[i] < 0.5):
            biases1[i] = 0
            filters1[:, i] = 0
        else:
            retain_nodes += 1

    model_final.layers[20].set_weights([filters1, biases1])
    arr = model_final .evaluate(X, y)
    fitness.append(arr[1])


#fitness = np.apply_along_axis(sphere_function, 1, positions)
best_position = positions[np.argmax(fitness)]
best_fitness = np.max(fitness)

for iteration in range(num_iterations):
    avg_loudness = np.mean(loudness)
    avg_pulse_rate = np.mean(pulse_rate)

    # Update bats
    for i in range(num_bats):
        beta = np.random.uniform(0, 1)
        frequencies[i] = freq_min + (freq_max - freq_min) * beta
        velocities[i] += (best_position - positions[i]) * frequencies[i]
        new_position = positions[i] + velocities[i]

        # Boundary check
        new_position = np.clip(new_position, lb, ub)

        # Local search
        if np.random.uniform(0, 1) > pulse_rate[i]:
            epsilon = np.random.uniform(-1, 1)
            new_position = positions[i] + epsilon * avg_loudness

        # Evaluate new solution
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        # Evaluate initial fitness
        retain_nodes = 0
        for i1 in range(0,1024):
            f = filters[:, i1]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(new_position[i1] < 0.5):
                biases1[i1] = 0
                filters1[:, i1] = 0
            else:
                retain_nodes += 1

        model_final.layers[20].set_weights([filters1, biases1])
        arr = model_final .evaluate(X, y)
        #fitness.append(arr[1])
        new_fitness = arr[1] #sphere_function(new_position)

        # Greedy mechanism to update if new solution is better and random value is less than loudness
        if new_fitness > fitness[i] and np.random.uniform(0, 1) < loudness[i]:
            positions[i] = new_position
            fitness[i] = new_fitness

        # Update global best
        if fitness[i] > best_fitness:
            best_position = positions[i]
            best_fitness = fitness[i]

        loudness[i] *= alpha
        pulse_rate[i] = r0 * (1 - np.exp(-gamma * iteration))

    # Print the best fitness value in each iteration
    print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")

print("\nOptimized Solution:", best_position)
print("Best Fitness Value:", best_fitness)


A5 = np.copy(best_position)
new_num = np.sum(ensure_bounds(best_position))

print(new_num)
layer5_a = new_num


####################### 2nd dense layer with 1024 filters
print('2nd dense layer with 1024 filters')
A = []
Acc = []

filters1 = np.copy(filters)
biases1 = np.copy(biases)
# Evaluate initial fitness
retain_nodes = 0
for i1 in range(0,1024):
    f = filters[:, i1]
    #for j in range(3):
        #if(child[i] == 0):
            #filters1[:, :, :, i][:,:,j] = 0
    if(new_position[i1] < 0.5):
        biases1[i1] = 0
        filters1[:, i1] = 0
    else:
        retain_nodes += 1
print('Nodes left', retain_nodes)
model_final.layers[20].set_weights([filters1, biases1])
arr = model_final .evaluate(X, y)
print(arr)

filters, biases = model_final.layers[22].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

def ensure_bounds(par):
    new_par = []
    for index in range(0, len(par)):
        if(par[index] >= 0.5):
            new_par.append(1)
        else:
            new_par.append(0)

    return new_par

# Parameters
num_bats = 10
dim = 1024
num_iterations = 10
freq_min = 0
freq_max = 2
A = 0.5
r0 = 0.5
alpha = 0.9
gamma = 0.9
lb = 0
ub = 1

# Initialize bat positions and velocities
positions = np.random.uniform(lb, ub, (num_bats, dim))
velocities = np.zeros((num_bats, dim))
frequencies = np.zeros(num_bats)
loudness = A * np.ones(num_bats)
pulse_rate = r0 * np.ones(num_bats)

fitness = []
for i1 in range(num_bats):
    pop = positions[i1]
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    # Evaluate initial fitness
    retain_nodes = 0
    for i in range(0,1024):
        f = filters[:, i]
        #for j in range(3):
            #if(child[i] == 0):
                #filters1[:, :, :, i][:,:,j] = 0
        if(pop[i] < 0.5):
            biases1[i] = 0
            filters1[:, i] = 0
        else:
            retain_nodes += 1

    model_final.layers[22].set_weights([filters1, biases1])
    arr = model_final .evaluate(X, y)
    fitness.append(arr[1])


#fitness = np.apply_along_axis(sphere_function, 1, positions)
best_position = positions[np.argmax(fitness)]
best_fitness = np.max(fitness)

for iteration in range(num_iterations):
    avg_loudness = np.mean(loudness)
    avg_pulse_rate = np.mean(pulse_rate)

    # Update bats
    for i in range(num_bats):
        beta = np.random.uniform(0, 1)
        frequencies[i] = freq_min + (freq_max - freq_min) * beta
        velocities[i] += (best_position - positions[i]) * frequencies[i]
        new_position = positions[i] + velocities[i]

        # Boundary check
        new_position = np.clip(new_position, lb, ub)

        # Local search
        if np.random.uniform(0, 1) > pulse_rate[i]:
            epsilon = np.random.uniform(-1, 1)
            new_position = positions[i] + epsilon * avg_loudness

        # Evaluate new solution
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        # Evaluate initial fitness
        retain_nodes = 0
        for i1 in range(0,1024):
            f = filters[:, i1]
            #for j in range(3):
                #if(child[i] == 0):
                    #filters1[:, :, :, i][:,:,j] = 0
            if(new_position[i1] < 0.5):
                biases1[i1] = 0
                filters1[:, i1] = 0
            else:
                retain_nodes += 1

        model_final.layers[22].set_weights([filters1, biases1])
        arr = model_final .evaluate(X, y)
        #fitness.append(arr[1])
        new_fitness = arr[1] #sphere_function(new_position)

        # Greedy mechanism to update if new solution is better and random value is less than loudness
        if new_fitness > fitness[i] and np.random.uniform(0, 1) < loudness[i]:
            positions[i] = new_position
            fitness[i] = new_fitness

        # Update global best
        if fitness[i] > best_fitness:
            best_position = positions[i]
            best_fitness = fitness[i]

        loudness[i] *= alpha
        pulse_rate[i] = r0 * (1 - np.exp(-gamma * iteration))

    # Print the best fitness value in each iteration
    print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")

print("\nOptimized Solution:", best_position)
print("Best Fitness Value:", best_fitness)


A6 = np.copy(best_position)
new_num = np.sum(ensure_bounds(best_position))

print(new_num)
layer6_a = new_num


model1 = Sequential()
model1.add(Conv2D(input_shape=(256,256,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
#model1.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model1.add(Conv2D(filters=layer1_a, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model1.add(Conv2D(filters=layer2_a, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model1.add(Conv2D(filters=layer3_a, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model1.add(Conv2D(filters=layer4_a, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model1.add(Flatten())
model1.add(Dense(units=layer5_a,activation="relu"))
model1.add(Dense(units=layer6_a,activation="relu"))
model1.add(Dense(2))
model1.add(Activation('softmax'))

model1.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(learning_rate=1e-3, momentum=0.9), metrics=["accuracy"])

model1.summary()

layerr = model_final.layers[1].get_weights()
model1.layers[0].set_weights(layerr)

model = model_final
######################## 1st convolution layer with 128 filters
filters, biases = model.layers[4].get_weights()
filters1, biases1 = model1.layers[2].get_weights()
print('Shape', filters.shape)
print('Shape', filters1.shape)
# normalize filter values to 0-1 so we can visualize them
# plot first few filters
n_filters, ix = 128, 1

"""
for i in range(n_filters):
    f = filters[:, :, i]
"""
index1 = 0
# plot each channel separately
for j in range(128):
    if(A1[j] == 1) :
        """
        for i1 in range (0,3):
            for j1 in range(0,3):
                filters1[:, :, index1][:,:,j][i1][j1] = filters[:, :, i][:,:,j][i1][j1]
        """
        filters1[:, :, :, index1] = filters[:, :, :, j]
        #biases1[:, :, :, index1] = biases[:, :, :, i]
        biases1[index1] = biases[j]
        #print(index1,i)
        index1 += 1
#print(biases1, biases)
model1.layers[2].set_weights([filters1, biases1])

######################## 1st convolution layer with 256 filters
filters, biases = model.layers[7].get_weights()
filters1, biases1 = model1.layers[4].get_weights()
print('Shape', filters.shape)
print('Shape', filters1.shape)
# normalize filter values to 0-1 so we can visualize them
# plot first few filters
n_filters, ix = 256, 1

"""
for i in range(n_filters):
    f = filters[:, :, i]
"""
index1 = 0
# plot each channel separately
for j in range(256):
    if(A2[j] == 1) :
        index2 = 0
        for l in range(128):
            if(A1[l] == 1):
                filters1[:, :, index2, index1] = filters[:, :, l, j]
                index2 += 1
        #biases1[:, :, :, index1] = biases[:, :, :, i]
        biases1[index1] = biases[j]
        #print(index1,i)
        index1 += 1
#print(biases1, biases)
model1.layers[4].set_weights([filters1, biases1])

######################## 1st convolution layer with 512 filters
filters, biases = model.layers[11].get_weights()
filters1, biases1 = model1.layers[6].get_weights()
print('Shape', filters.shape)
print('Shape', filters1.shape)
# normalize filter values to 0-1 so we can visualize them
# plot first few filters
n_filters, ix = 512, 1

"""
for i in range(n_filters):
    f = filters[:, :, i]
"""
index1 = 0
# plot each channel separately
for j in range(512):
    if(A3[j] == 1) :
        index2 = 0
        for l in range(256):
            if(A2[l] == 1):
                filters1[:, :, index2, index1] = filters[:, :, l, j]
                index2 += 1
        biases1[index1] = biases[j]
        #print(index1,i)
        index1 += 1
#print(biases1, biases)
model1.layers[6].set_weights([filters1, biases1])

######################## 2nd convolution layer with 512 filters
filters, biases = model.layers[15].get_weights()
filters1, biases1 = model1.layers[8].get_weights()
print('Shape', filters.shape)
print('Shape', filters1.shape)
# normalize filter values to 0-1 so we can visualize them
# plot first few filters
n_filters, ix = 512, 1

"""
for i in range(n_filters):
    f = filters[:, :, i]
"""
index1 = 0
# plot each channel separately
for j in range(512):
    if(A4[j] == 1) :
        index2 = 0
        for l in range(512):
            if(A3[l] == 1):
                filters1[:, :, index2, index1] = filters[:, :, l, j]
                index2 += 1
        #biases1[:, :, :, index1] = biases[:, :, :, i]
        biases1[index1] = biases[j]
        #print(index1,i)
        index1 += 1
#print(biases1, biases)
model1.layers[8].set_weights([filters1, biases1])

######################## 1st dense layer with 1024 filters
filters, biases = model.layers[20].get_weights()
filters1, biases1 = model1.layers[11].get_weights()
print('Shape', filters.shape)
print('Shape', filters1.shape)
# normalize filter values to 0-1 so we can visualize them
# plot first few filters
n_filters, ix = 1024, 1

"""
for i in range(n_filters):
    f = filters[:, :, i]
"""
index1 = 0
# plot each channel separately
for j in range(1024):
    if(A5[j] == 1) :
        index2 = 0
        for l in range(512):
            if(A4[l] == 1):
                filters1[index2, index1] = filters[l, j]
                index2 += 1
        #biases1[:, :, :, index1] = biases[:, :, :, i]
        biases1[index1] = biases[j]
        #print(index1,i)
        index1 += 1
#print(biases1, biases)
model1.layers[11].set_weights([filters1, biases1])

######################## 2nd dense layer with 1024 filters
filters, biases = model.layers[22].get_weights()
filters1, biases1 = model1.layers[12].get_weights()
print('Shape', filters.shape)
print('Shape', filters1.shape)
# normalize filter values to 0-1 so we can visualize them
# plot first few filters
n_filters, ix = 1024, 1

"""
for i in range(n_filters):
    f = filters[:, :, i]
"""
index1 = 0
# plot each channel separately
for j in range(1024):
    if(A6[j] == 1) :
        index2 = 0
        for l in range(1024):
            if(A5[l] == 1):
                filters1[index2, index1] = filters[l, j]
                index2 += 1
        biases1[index1] = biases[j]
        #print(index1,i)
        index1 += 1
#print(biases1, biases)
model1.layers[12].set_weights([filters1, biases1])


arr = model1 .evaluate(X, y)
print(arr)


model1.fit(
    training_set,
    steps_per_epoch=len(X_train)//64,
    epochs=20,
    validation_data=val_set,
    validation_steps=len(X_val)//64)

model1.summary()
model1.save_weights('/content/drive/MyDrive/Covid/VGG16_pruned.weights.h5')

arr = model1 .evaluate(test_set)


print(arr)

