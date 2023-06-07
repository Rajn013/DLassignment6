#!/usr/bin/env python
# coding: utf-8

# What are the advantages of a CNN over a fully connected DNN for image classification?
# 

# 1.CNNs excel at image classification due to their ability to exploit spatial locality, parameter sharing, translation invariance, and hierarchical feature learning.
# They are specifically designed for image data, capturing local patterns and structures efficiently.
# CNNs require fewer parameters compared to fully connected DNNs, making them more memory-efficient and computationally efficient.
# They are more robust to variations in input images and can handle spatial transformations better.
# Overall, CNNs provide superior performance and are the preferred choice for image classification tasks.

# Consider a CNN composed of three convolutional layers, each with 3 × 3 kernels, a stride of 2, and "same" padding. The lowest layer outputs 100 feature maps, the middle one outputs 200, and the top one outputs 400. The input images are RGB images of 200 × 300 pixels.
# What is the total number of parameters in the CNN? If we are using 32-bit floats, at least how much RAM will this network require when making a prediction for a single instance? What about when training on a mini-batch of 50 images?
# 

# 2.The total number of parameters in the CNN is the sum of the parameters in each convolutional layer, including the bias terms.
# RAM required for prediction on a single instance is the sum of the input size and the output size of the top layer.
# RAM required for training on a mini-batch of 50 images is the sum of the input size multiplied by the batch size and the output size multiplied by the batch size.
# Please note that these calculations provide an estimate and do not include additional memory requirements for intermediate activations, gradients, optimizer states, or other training-related overhead.
# 

# If your GPU runs out of memory while training a CNN, what are five things you could try to solve the problem?
# 

# 3.Reduce batch size.
# Resize or crop input images.
# Simplify the model architecture.
# Use mixed precision training.
# Enable memory optimization techniques.
# These solutions aim to reduce GPU memory usage while training a CNN. However, the effectiveness of each solution depends on the specific circumstances, and trade-offs must be considered.
# 

# Why would you want to add a max pooling layer rather than a convolutional layer with the same stride?
# 

# 4.Max pooling layer provides dimensionality reduction, translation invariance, feature selection, and reduces spatial information.
# It reduces the number of parameters and computations, making the model more efficient.
# Max pooling helps in capturing salient features and emphasizing discriminative information.
# It operates locally, allowing for parameter sharing and reducing overfitting.
# However, the choice depends on the specific requirements of the task and the nature of the input data.
# 

# When would you want to add a local response normalization layer?
# 

# 5.LRN layers can be added to enhance generalization, address activation scaling issues, and promote local contrast.
# They encourage competition between adjacent neurons and normalize activations across neighboring channels.
# LRN layers are particularly useful for improving performance on large-scale datasets and tasks that require sensitivity to local patterns or texture information.
# However, the use of LRN layers has become less common in favor of techniques like batch normalization.
# LRN layers can still be considered in specific cases to experiment and improve model performance.
# 

# Can you name the main innovations in AlexNet, compared to LeNet-5? What about the main innovations in GoogLeNet, ResNet, SENet, and Xception?
# 

# 6.AlexNet introduced a deeper architecture, ReLU activation, dropout regularization, and extensive data augmentation.
# GoogLeNet introduced the inception module, auxiliary classifiers, and global average pooling.
# ResNet introduced residual connections and skip connections.
# SENet introduced channel-wise attention mechanisms for adaptive feature recalibration.
# Xception employed depthwise separable convolutions and a fully convolutional architecture.
# These innovations in each model contributed to improved performance, model capacity, training efficiency, and interpretability in deep learning.
# 

# What is a fully convolutional network? How can you convert a dense layer into a convolutional layer?
# 

# 7.A fully convolutional network (FCN) is a neural network architecture composed entirely of convolutional layers, without any fully connected layers.
# To convert a dense layer into a convolutional layer:
# Reshape the input or output to have the shape (batch_size, height, width, num_channels).
# Replace the dense layer with a 1x1 convolutional layer.
# Adjust the parameters of the convolutional layer to match the behavior of the original dense layer.
# Converting a dense layer to a convolutional layer allows the model to handle inputs of varying sizes and is useful for tasks that require spatial information.
# 

# What is the main technical difficulty of semantic segmentation?
# 

# 8.the main technical difficulty of semantic segmentation is accurately assigning the correct semantic label to each pixel in an image due to challenges such as pixel-level prediction, modeling spatial context, handling variability in object appearance, dealing with class imbalance, localizing boundaries, and ensuring efficiency and computational complexity. Addressing these challenges requires advanced techniques and continuous research efforts.
# 

# Build your own CNN from scratch and try to achieve the highest possible accuracy on MNIST.
# 

# In[1]:


#9
import numpy as np
import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)


# Use transfer learning for large image classification, going through these steps:
# Create a training set containing at least 100 images per class. For example, you could classify your own pictures based on the location (beach, mountain, city, etc.), or alternatively you can use an existing dataset (e.g., from TensorFlow Datasets).
# 

# 1 Choose or create a dataset with at least 100 images per class.
# Load a pre-trained model without the top classification layers.
# Freeze the pre-trained layers to retain their weights.
# Add new classification layers on top of the pre-trained model.
# Train the model on your dataset, updating only the new classification layers.
# Evaluate the model's performance on a separate test set.
# Optionally, fine-tune the entire model by unfreezing the pre-trained layers and continuing training with a lower learning rate.
# Consider using data augmentation techniques to enhance the diversity of your training data.
# 

# Split it into a training set, a validation set, and a test set.
# 

# 2.when using transfer learning, you should split your dataset into three sets:
# 
# Training set: Used to train the model and update the weights of the new classification layers.
# Validation set: Used to tune hyperparameters, monitor model performance, and make adjustments if needed.
# Test set: Used to evaluate the final performance of the model on unseen data.
# A typical split ratio is around 70% for training, 15% for validation, and 15% for testing. Ensure that each set contains a representative distribution of images from all classes. This separation allows for accurate evaluation and helps make informed decisions about the model's architecture and parameters.
# 

# Build the input pipeline, including the appropriate preprocessing operations, and optionally add data augmentation.
# 

# In[56]:


import tensorflow as tf

file_path = 'C:\\Users\\ACER\\OneDrive\\Desktop\\gallery_15103_969_110712.jpg'

if tf.io.gfile.exists(file_path):
    print("File exists." , file_path)
else:
    print("File does not exist.")


# In[1]:


import tensorflow as tf

directory = ' C:\\Users\\ACER\\OneDrive\\Desktop\\gallery_15103_969_110712.jpg'

file_list = tf.io.gfile.listdir(directory)
print(file_list)


# In[47]:


import tensorflow as tf

def preprocess_image(image, label):
    return preprocessed_image, label

def augment_image(image, label):
    return augmented_image, label

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'C:\Users\ACER\OneDrive\Desktop\gallery_15103_969_110712.jpg',
    labels='inferred',
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(222, 222),
    batch_size=3
)

# Preprocess and augment the train dataset
train_dataset = train_dataset.map(preprocess_image)
if AUGMENT_DATA:
    train_dataset = train_dataset.map(augment_image)

# Configure the train dataset
train_dataset = train_dataset.cache().shuffle(SHUFFLE_BUFFER_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

# Create the validation dataset
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'https://url.to/example/train_dataset',
    labels='inferred',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(222, 222),
    batch_size=3
)

# Preprocess the validation dataset
validation_dataset = validation_dataset.map(preprocess_image)

# Configure the validation dataset
validation_dataset = validation_dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)


# Fine-tune a pretrained model on this dataset.
# 

# In[37]:


#4.
import tensorflow as tf

base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(18, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, validation_data=validation_dataset, epochs=5)


# In[ ]:




