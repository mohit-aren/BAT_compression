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
base = VGG19(weights="imagenet", include_top=False, input_shape=(256,256,3))
base.trainable = False
for layer in base.layers[-4:]:   # last 4 layers trainable
    layer.trainable = True
x = Flatten()(base.output)
x = Dense(1024, activation="relu")(x)
x = Dense(512, activation="relu")(x)
outputs = Dense(2, activation="softmax")(x)
model = tf.keras.Model(inputs=base.input, outputs=outputs)
#opt = Adam(learning_rate=0.001, beta_1=0.9)
model_final = model

#model_final = model
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

#X, y = test_set.next()
w_file = '/content/drive/MyDrive/Covid/keras_vgg19_main.weights.h5'
model_final.load_weights(w_file)

model_final.fit(
        training_set,
        steps_per_epoch=20,
        epochs=2,
        validation_data=val_set,
        validation_steps=20)

w_file = '/content/drive/MyDrive/Covid/keras_vgg19_main1.weights.h5'
model_final.save_weights(w_file)

model = model_final
model_path = "/content/drive/MyDrive/Covid/keras_vgg19_main1.weights.h5"
model.load_weights(model_path)

class_names = ["control","covid"]

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2

def load_and_preprocess_image(img_path, target_size=(256, 256)):
    """Load image from disk and preprocess for VGG16.

    Returns:
        original_img: original image as numpy array (RGB) for plotting/overlay.
        img_array: preprocessed image ready for model (shape: (1,224,224,3)).
    """
    img = Image.open(img_path).convert('RGB')
    img = img.resize((256,256))
    img_array = image.img_to_array(img)
    original_img = img_array.astype(np.uint8).copy()
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return original_img, img_array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generates a Grad-CAM heatmap for a predicted class.

    Args:
        img_array: preprocessed image tensor with shape (1, H, W, C).
        model: tf.keras Model.
        last_conv_layer_name: name of the convolutional layer to target.
        pred_index: index of the class to explain. If None, uses model's top predicted class.

    Returns:
        heatmap: 2D numpy array (H_conv, W_conv) normalized to [0,1].
    """
    # Create a model that maps the input image to the activations of the last conv layer
    # and the model's predictions
    grad_model = tf.keras.models.Model(
        model.input, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    img_tensor = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)

        #predictions = tf.convert_to_tensor(predictions)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
        print(predictions, pred_index)

    #logit = tf.math.log(predictions[:, pred_index] + 1e-8)
    #grads = tape.gradient(logit, conv_outputs)
    # Compute the gradient of the class output value with respect to the feature map

    grads = tape.gradient(class_channel, conv_outputs)
    print("Gradients min/max:", tf.reduce_min(grads).numpy(), tf.reduce_max(grads).numpy())
    if grads is None:
        raise ValueError(f"Gradients are None. Is `{last_conv_layer_name}` the last Conv2D layer?")

    # Pool the gradients over all the axes leaving out the channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by "how important this channel is"
    conv_outputs = conv_outputs[0]
    '''
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)


    # Apply ReLU to the heatmap (only positive influence)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    '''
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)  # weighted sum
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8  # normalize to [0,1]
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path='cam.jpg', alpha=0.4):
    """Superimpose heatmap on image and save to cam_path.

    Args:
        img_path: original image path (used for loading full-res image)
        heatmap: 2D numpy array with values in [0,1]
        cam_path: where to write the overlay
        alpha: blending factor for overlay
    """
    # Load original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    #heatmap_norm = np.uint8(255 * (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8))


    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    heatmap = heatmap.numpy()

    # Convert to 0–255 uint8 for OpenCV
    heatmap_uint8 = np.uint8(255 * heatmap)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    # Superimpose heatmap onto original image
    heatmap_resized = cv2.resize(heatmap_colored, (img.shape[1], img.shape[0]))

    superimposed_img = cv2.addWeighted(img, 1-alpha, heatmap_resized, alpha, 0)
    #superimposed_img = heatmap_resized * alpha + img
    #superimposed_img = heatmap_resized * alpha + img
    superimposed_img = np.clip(superimposed_img / superimposed_img.max() * 255, 0, 255).astype(np.uint8)

    # Save
    out = Image.fromarray(superimposed_img)
    out.save(cam_path)
    print(f"Grad-CAM saved to: {cam_path}")

    # Display inline (simple matplotlib figure)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.axis('off')
    plt.imshow(img)

    plt.subplot(1, 3, 2)
    plt.title('Grad-CAM')
    plt.axis('off')
    plt.imshow(superimposed_img)
    plt.show()

    plt.subplot(1, 3, 3)
    plt.title('Grad-CAM-heatmap')
    plt.axis('off')
    plt.imshow(heatmap_colored)
    plt.show()

test_images = [
    "/content/drive/MyDrive/Covid/test/covid/segm-Novara 100-1.png",
    "/content/drive/MyDrive/Covid/test/covid/segm-Novara 101-14.png",
    "/content/drive/MyDrive/Covid/test/covid/segm-Novara 102-26.png",
    "/content/drive/MyDrive/Covid/test/covid/segm-Novara 103-45.png",
    "/content/drive/MyDrive/Covid/test/covid/segm-Novara 105-56.png",
    "/content/drive/MyDrive/Covid/test/covid/segm-Novara 106-28.png",
    "/content/drive/MyDrive/Covid/test/covid/segm-Novara 107-30.png",
    "/content/drive/MyDrive/Covid/segm-Novara 113-19.png"
]
idx = 0
for path in test_images:
   original_img, img_array = load_and_preprocess_image(path)

   last_conv_layer = model.get_layer("block4_conv4")
   dummy_input = tf.random.normal([1, 256, 256, 3])
   _ = model(dummy_input)
   print(model.outputs)
   # Build grad-model
   grad_model = tf.keras.models.Model(
       [model.inputs],
       [last_conv_layer.output, model.outputs[0]]
   )

   preds = model.predict(img_array)
   pred_index = np.argmax(preds[0])
   print(preds,pred_index)
   heatmap = make_gradcam_heatmap(img_array, model, 'block4_conv4', pred_index)
   save_and_display_gradcam(path, heatmap, cam_path='gradcam_out' + str(idx) + '.jpg', alpha=0.4)
   idx += 1