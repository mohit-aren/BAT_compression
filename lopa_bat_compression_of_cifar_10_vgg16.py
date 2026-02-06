

from google.colab import drive
drive.mount('/content/drive')

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
from keras.datasets import cifar10
img_width, img_height = 48, 48

model=VGG16(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

#initialise top model

# Freeze first 45 layers
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
predictions = Dense(10, activation="softmax")(x) # 4-way softmax classifier at the end

model_final = Model(inputs=model.input, outputs=predictions)

model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(learning_rate=1e-3, momentum=0.9), metrics=["accuracy"])

model_final.summary()

from google.colab import drive
drive.mount('/content/drive')

#Read CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = [cv2.resize(i, (48,48)) for i in X_train]
X_train = np.concatenate([arr[np.newaxis] for arr in X_train]).astype('float32')

X_test = [cv2.resize(i, (48,48)) for i in X_test]
X_test = np.concatenate([arr[np.newaxis] for arr in X_test]).astype('float32')


y_train = y_train.reshape(len(y_train), 1)
y_test = y_test.reshape(len(y_test), 1)

from keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

X = X_test
y = y_test

print(np.array(X_train).shape)
print(np.array(y_train).shape)

#Train the base VGG16 model with train set
#model_final.fit(X_train,y_train,batch_size=64,epochs=20)

model = model_final
model_path = '/content/drive/MyDrive/CIFAR-10/keras_vgg16_main.weights.h5'

#Save the model weights
model.save_weights(model_path)

model_final.load_weights('/content/drive/MyDrive/CIFAR-10/keras_vgg16_main.weights.h5')
arr = model_final .evaluate(X, y)
print(arr)

#Design a compressed model with duplicate rows pruned and filter size given as parameters which will be obtained from BAT optimization
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
model1.add(Conv2D(input_shape=(48,48,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
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
model1.add(Dense(10))
model1.add(Activation('softmax'))

model1.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(learning_rate=1e-3, momentum=0.9), metrics=["accuracy"])

model1.summary()

#y_test = y_test.reshape(len(y_test), 1)


#onehot_encoded = onehot_encoder.fit_transform(y_test)

####################### Perform BAT optimization for 1st convolution layer with 128 filters
model_final.load_weights('/content/drive/MyDrive/CIFAR-10/keras_vgg16_main.weights.h5')
print('1st convolution layer with 128 filters')
A = []
Acc = []

arr = model_final .evaluate(X, y)
print(arr)

filters, biases = model_final.layers[4].get_weights()
filters1 = np.copy(filters)
biases1 = np.copy(biases)

#A function to make real values between 0 and 1 in discrete values 0,1
def ensure_bounds(par):
    new_par = []
    for index in range(0, len(par)):
        if(par[index] >= 0.5):
            new_par.append(1)
        else:
            new_par.append(0)

    return new_par

# Initialize BAT optmization Parameters
num_bats = 10
dim = 128
num_iterations = 5
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
    #Obtain the fitness value from fitness function Minimize (F) = 0.5*loss + 0.5*(retained nodes/original nodes)
    fitness.append(0.5*arr[0] + 0.5*retain_nodes/len(pop))


#Find the best position and fitness from initial population
best_position = positions[np.argmin(fitness)]
best_fitness = np.min(fitness)
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
        #fitness.append(0.5*arr[0] + 0.5*retain_nodes/len(pop))
        new_fitness = 0.5*arr[0] + 0.5*retain_nodes/len(new_position) #sphere_function(new_position)

        # Greedy mechanism to update if new solution is better and random value is less than loudness
        if new_fitness < fitness[i] and np.random.uniform(0, 1) < loudness[i]:
            positions[i] = new_position
            fitness[i] = new_fitness

        # Update global best
        if fitness[i] < best_fitness:
            best_position = positions[i]
            best_fitness = fitness[i]

        loudness[i] *= alpha
        pulse_rate[i] = r0 * (1 - np.exp(-gamma * iteration))

    # Print the best fitness value in each iteration
    print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")

print("\nOptimized Solution:", best_position)
print("Best Fitness Value:", best_fitness)

#Create a vector to store dominant filters in this layer
A1 = np.copy(ensure_bounds(best_position))
new_num = np.sum(ensure_bounds(best_position))

print(new_num)
print(A1)
layer1_a = new_num

####################### Perform BAT optimization  1st convolution layer with 256 filters
model_final.load_weights('/content/drive/MyDrive/CIFAR-10/keras_vgg16_main.weights.h5')
print('1st convolution layer with 256 filters')
A = []
Acc = []

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
num_iterations = 5
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
    fitness.append(0.5*arr[0] + 0.5*retain_nodes/len(pop))


#fitness = np.apply_along_axis(sphere_function, 1, positions)
best_position = positions[np.argmin(fitness)]
best_fitness = np.min(fitness)

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
        #fitness.append(0.5*arr[0] + 0.5*retain_nodes/len(pop))
        new_fitness = 0.5*arr[0] + 0.5*retain_nodes/len(new_position) #sphere_function(new_position)

        # Greedy mechanism to update if new solution is better and random value is less than loudness
        if new_fitness < fitness[i] and np.random.uniform(0, 1) < loudness[i]:
            positions[i] = new_position
            fitness[i] = new_fitness

        # Update global best
        if fitness[i] < best_fitness:
            best_position = positions[i]
            best_fitness = fitness[i]

        loudness[i] *= alpha
        pulse_rate[i] = r0 * (1 - np.exp(-gamma * iteration))

    # Print the best fitness value in each iteration
    print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")

print("\nOptimized Solution:", best_position)
print("Best Fitness Value:", best_fitness)


A2 = np.copy(ensure_bounds(best_position))
new_num = np.sum(ensure_bounds(best_position))

print(new_num)
print(A2)
layer2_a = new_num

####################### Perform BAT optimization  1st convolution layer with 512 filters
model_final.load_weights('/content/drive/MyDrive/CIFAR-10/keras_vgg16_main.weights.h5')
print('1st convolution layer with 512 filters')
A = []
Acc = []

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
num_iterations = 5
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
    fitness.append(0.5*arr[0] + 0.5*retain_nodes/len(pop))


#fitness = np.apply_along_axis(sphere_function, 1, positions)
best_position = positions[np.argmin(fitness)]
best_fitness = np.min(fitness)

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
        #fitness.append(0.5*arr[0] + 0.5*retain_nodes/len(pop))
        new_fitness = 0.5*arr[0] + 0.5*retain_nodes/len(new_position) #sphere_function(new_position)

        # Greedy mechanism to update if new solution is better and random value is less than loudness
        if new_fitness < fitness[i] and np.random.uniform(0, 1) < loudness[i]:
            positions[i] = new_position
            fitness[i] = new_fitness

        # Update global best
        if fitness[i] < best_fitness:
            best_position = positions[i]
            best_fitness = fitness[i]

        loudness[i] *= alpha
        pulse_rate[i] = r0 * (1 - np.exp(-gamma * iteration))

    # Print the best fitness value in each iteration
    print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")

print("\nOptimized Solution:", best_position)
print("Best Fitness Value:", best_fitness)


A3 = np.copy(ensure_bounds(best_position))
new_num = np.sum(ensure_bounds(best_position))

print(new_num)
print(A3)
layer3_a = new_num

####################### Perform BAT optimization  2nd convolution layer with 512 filters
model_final.load_weights('/content/drive/MyDrive/CIFAR-10/keras_vgg16_main.weights.h5')
print('2nd convolution layer with 512 filters')
A = []
Acc = []

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
num_iterations = 5
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
    fitness.append(0.5*arr[0] + 0.5*retain_nodes/len(pop))


#fitness = np.apply_along_axis(sphere_function, 1, positions)
best_position = positions[np.argmin(fitness)]
best_fitness = np.min(fitness)

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
        #fitness.append(0.5*arr[0] + 0.5*retain_nodes/len(pop))
        new_fitness = 0.5*arr[0] + 0.5*retain_nodes/len(new_position) #sphere_function(new_position)

        # Greedy mechanism to update if new solution is better and random value is less than loudness
        if new_fitness < fitness[i] and np.random.uniform(0, 1) < loudness[i]:
            positions[i] = new_position
            fitness[i] = new_fitness

        # Update global best
        if fitness[i] < best_fitness:
            best_position = positions[i]
            best_fitness = fitness[i]

        loudness[i] *= alpha
        pulse_rate[i] = r0 * (1 - np.exp(-gamma * iteration))

    # Print the best fitness value in each iteration
    print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")

print("\nOptimized Solution:", best_position)
print("Best Fitness Value:", best_fitness)


A4 = np.copy(ensure_bounds(best_position))
new_num = np.sum(ensure_bounds(best_position))

print(new_num)
print(A4)
layer4_a = new_num

####################### Perform BAT optimization  1st dense layer with 1024 filters
model_final.load_weights('/content/drive/MyDrive/CIFAR-10/keras_vgg16_main.weights.h5')
print('1st dense layer with 1024 filters')
A = []
Acc = []

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
num_iterations = 5
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
    fitness.append(0.5*arr[0] + 0.5*retain_nodes/len(pop))


#fitness = np.apply_along_axis(sphere_function, 1, positions)
best_position = positions[np.argmin(fitness)]
best_fitness = np.min(fitness)

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
        #fitness.append(0.5*arr[0] + 0.5*retain_nodes/len(pop))
        new_fitness = 0.5*arr[0] + 0.5*retain_nodes/len(new_position) #sphere_function(new_position)

        # Greedy mechanism to update if new solution is better and random value is less than loudness
        if new_fitness < fitness[i] and np.random.uniform(0, 1) < loudness[i]:
            positions[i] = new_position
            fitness[i] = new_fitness

        # Update global best
        if fitness[i] < best_fitness:
            best_position = positions[i]
            best_fitness = fitness[i]

        loudness[i] *= alpha
        pulse_rate[i] = r0 * (1 - np.exp(-gamma * iteration))

    # Print the best fitness value in each iteration
    print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")

print("\nOptimized Solution:", best_position)
print("Best Fitness Value:", best_fitness)


A5 = np.copy(ensure_bounds(best_position))
new_num = np.sum(ensure_bounds(best_position))

print(new_num)
print(A5)
layer5_a = new_num


####################### Perform BAT optimization  2nd dense layer with 1024 filters
model_final.load_weights('/content/drive/MyDrive/CIFAR-10/keras_vgg16_main.weights.h5')
print('2nd dense layer with 1024 filters')
A = []
Acc = []

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
num_iterations = 5
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
    fitness.append(0.5*arr[0] + 0.5*retain_nodes/len(pop))


#fitness = np.apply_along_axis(sphere_function, 1, positions)
best_position = positions[np.argmin(fitness)]
best_fitness = np.min(fitness)

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
        #fitness.append(0.5*arr[0] + 0.5*retain_nodes/len(pop))
        new_fitness = 0.5*arr[0] + 0.5*retain_nodes/len(new_position) #sphere_function(new_position)

        # Greedy mechanism to update if new solution is better and random value is less than loudness
        if new_fitness < fitness[i] and np.random.uniform(0, 1) < loudness[i]:
            positions[i] = new_position
            fitness[i] = new_fitness

        # Update global best
        if fitness[i] < best_fitness:
            best_position = positions[i]
            best_fitness = fitness[i]

        loudness[i] *= alpha
        pulse_rate[i] = r0 * (1 - np.exp(-gamma * iteration))

    # Print the best fitness value in each iteration
    print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")

print("\nOptimized Solution:", best_position)
print("Best Fitness Value:", best_fitness)


A6 = np.copy(ensure_bounds(best_position))
new_num = np.sum(ensure_bounds(best_position))

print(new_num)
print(A6)
layer6_a = new_num

#Design the compressed model with reduced number of nodes as obtained from BAT optmization
model1 = Sequential()
model1.add(Conv2D(input_shape=(48,48,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
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
model1.add(Dense(10))
model1.add(Activation('softmax'))

model1.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(learning_rate=1e-3, momentum=0.9), metrics=["accuracy"])

model1.summary()

layerr = model_final.layers[1].get_weights()
model1.layers[0].set_weights(layerr)

model_final.load_weights('/content/drive/MyDrive/CIFAR-10/keras_vgg16_main.weights.h5')
model = model_final
######################## Copy the weights from original trained model in corresponding neurons for 1st convolution layer with 128 filters
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

########################  Copy the weights from original trained model in corresponding neurons for 1st convolution layer with 256 filters
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

########################  Copy the weights from original trained model in corresponding neurons for 1st convolution layer with 512 filters
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

########################  Copy the weights from original trained model in corresponding neurons for 2nd convolution layer with 512 filters
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

########################  Copy the weights from original trained model in corresponding neurons for 1st dense layer with 1024 filters
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

########################  Copy the weights from original trained model in corresponding neurons for 2nd dense layer with 1024 filters
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

#y_train = y_train.reshape(len(y_train), 1)

#onehot_encoded = onehot_encoder.fit_transform(ytrain)

#Train the compressed model after weights copying from original model
model1.fit(X_train,y_train,batch_size=64,epochs=20)

model1.summary()
model1.save_weights('/content/drive/MyDrive/CIFAR-10/VGG16_pruned.weights.h5')

#y_test = y_test.reshape(len(y_test), 1)

#Test the performance of compressed model
arr = model1 .evaluate(X,y)


print(arr)