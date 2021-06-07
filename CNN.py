from google.colab import drive
from pathlib import Path
import sys

drive.mount('/content/gdrive', force_remount=True)

base = Path('gdrive/My Drive/TFM/data/')
sys.path.append(str(base))

# unzipper from drive to local colab disk

zip_path = base/'epilep/zips/chb18_30s_cmexican.zip'
!cp "{zip_path}" .
!unzip -q chb18_30s_cmexican.zip
!rm chb18_30s_cmexican.zip

!pip install split-folders &> /dev/null
import splitfolders
import os, os.path
import shutil
import itertools

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

import keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam, SGD, Adadelta
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Convolution2D , Convolution3D, MaxPooling2D , MaxPooling3D, Flatten , Dropout, Activation
from keras import applications 
from keras.utils.np_utils import to_categorical 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, f1_score, precision_score

import tensorflow as tf

import splitfolders
import os, os.path
import cv2

import numpy as np
import pandas as pd

%matplotlib inline
import math 
import datetime
import time

# Organize data into train, valid, test dirs

patient = 'chb18_30s_cmexican/'

#shutil.rmtree('output')

os.makedirs('output')
os.makedirs('output/'+patient+'train')
os.makedirs('output/'+patient+'test')
os.makedirs('output/'+patient+'val')

# We split our data into train/validation/test folders with given ratio
splitfolders.ratio(patient, output= 'output/' + patient, seed=1337, ratio=(0.7, 0.2, 0.1))

# Number of images in each subset (train/test/val)
subsets = ['train','test','val']
total_patient_images = 0

for subset in subsets:
    path = 'output/' + patient + subset
    folders = ([name for name in os.listdir(path)
                if os.path.isdir(os.path.join(path, name))]) # get all directories 
    total_files = 0
    
    for folder in folders:
        contents = os.listdir(os.path.join(path,folder)) # get list of contents
        print(folder,len(contents))
        total_files += len(contents)
    print('Número total de archivos en la carpeta ', path, ': ', total_files)
    total_patient_images = total_patient_images + total_files
print('Número total de archivos para el paciente: ', total_patient_images)

### Defining Dimensions and locating images ###

data_path = 'output/' + patient

# loading up our datasets
train_data_dir = data_path + 'train'
validation_data_dir = data_path + 'val'
test_data_dir = data_path + 'test'
 
# I create several fix input variables 
INIT_LR = 1e-3
epochs = 30
batch_size = 32
num_folds = 5

#img_rows, img_cols = 54, 83    #if we eliminate the alpha layer
img_rows, img_cols = 217, 334  # input size of image
activationFunction='relu'


# Define class_weight for my imbalanced dataset
path = 'output/' + patient + 'train'
folders = ([name for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))]) # get all directories 
total_files = 0

class_weight = {0:0, 1:0, 2:0}

for folder in folders:
  contents = os.listdir(os.path.join(path,folder)) # get list of contents
  print(folder,len(contents))
  if folder=='ictal':
    class_weight[0] = len(contents)
  elif folder =='preictal':
    class_weight[1] = len(contents)
  else:
    class_weight[2] = len(contents)

  total_files += len(contents)
print('Número total de archivos en la carpeta ', path, ': ', total_files)

total = sum(class_weight.values(), 0.0)
class_weight = {k: (1/v) * (total/2) for k, v in class_weight.items()}

print('Weight for class 0: {:.2f}'.format(class_weight[0]))
print('Weight for class 1: {:.2f}'.format(class_weight[1]))
print('Weight for class 2: {:.2f}'.format(class_weight[2]))


# class_weight_ = {0: 14.55, 1: 2.91 ,2: 0.63}


datagen = ImageDataGenerator(rescale=1. / 255)

train_batches = datagen.flow_from_directory(directory=train_data_dir, 
                                            target_size=(img_rows,img_cols), 
                                            classes=['ictal', 'preictal', 'interictal'], 
                                            batch_size=batch_size)

valid_batches = datagen.flow_from_directory(directory=validation_data_dir, 
                                            target_size=(img_rows,img_cols), 
                                            classes=['ictal', 'preictal', 'interictal'], 
                                            batch_size=batch_size)

test_batches = datagen.flow_from_directory(directory=test_data_dir, 
                                           target_size=(img_rows,img_cols), 
                                           classes=['ictal', 'preictal', 'interictal'], 
                                           batch_size=batch_size, 
                                           shuffle=False) # shuffle=False only for test_batches because later, when we plot the evaluation results from the model to a confusion matrix, 
                                                          # we'll need to able to access the unshuffled labels for the test set. By default, the data sets are shuffled.

# Visualize the Data
imgs, labels = next(train_batches)

# We then use this plotting function obtained from TensorFlow's documentation to plot the processed images within our Jupyter notebook.

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(imgs)
print(labels)

# Build model

# Model 1

'''model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(img_rows, img_cols, 3), padding='same'))
model.add(Activation(activationFunction))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, (3, 3), padding='same'))
model.add(Activation(activationFunction))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(3))
model.add(Activation('softmax'))'''

# Model 2
model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(img_rows, img_cols, 3), padding='same'))
model.add(Activation(activationFunction))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, (3, 3), padding='same'))
model.add(Activation(activationFunction))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3, 3), padding='same'))
model.add(Activation(activationFunction))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation(activationFunction))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))

# We can check out a summary of the model
model.summary()

# Now that the model is built, we compile the model using the an optimizer with a specified learning rate, a loss of categorical_cross_entropy because we have more than 2 classes, and we'll look at accuracy as our performance metric.
opt = Adam()

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# We train our model
history = model.fit(x=train_batches,
    steps_per_epoch=len(train_batches), # steps_per_epoch should be set to the number of steps (batches of samples) to yield from the training set before declaring one epoch finished and starting the next epoch. 
                                        # This is typically set to be equal to the number of samples in our training set divided by the batch size
    validation_data=valid_batches,
    validation_steps=len(valid_batches), # same as with steps_per_epoch
    epochs=epochs,
    verbose=1,
    class_weight=class_weight
)

def my_metrics(y_true, y_pred):
    accuracy=accuracy_score(y_true, y_pred)
    precision=precision_score(y_true, y_pred,average='weighted')
    f1Score=f1_score(y_true, y_pred, average='weighted') 
    print("Accuracy  : {}".format(accuracy))
    print("Precision : {}".format(precision))
    print("f1Score : {}".format(f1Score))
    cm=confusion_matrix(y_true, y_pred)
    print(cm)
    return accuracy, precision, f1Score

### Predicting On The Test Data ###
# Now we'll use our previously built model and call model.predict() to have the model predict on the test set.
predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=1) # Similar to steps_per_epoch, steps specifies how many batches to yield from the test set before declaring one prediction round complete.

# After running the predictions, we can print our the rounded predictions see what they look like.
np.round(predictions)

### Plotting Predictions With A Confusion Matrix ###
cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))

# Note, we can access the unshuffled true labels for the test set by calling test_batches.classes

#test_batches.classes

# We transform the one-hot encoded predicted labels to be in the same format as the true labels by only selecting the element with the highest value for each prediction

#np.argmax(predictions, axis=-1)

# We then define the plot_confusion_matrix() function that is copied directly from scikit-learn.

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# We can then inspect the class_indices for the labels so that we know in which order to pass them to our confusion matrix.
test_batches.class_indices

# Finally, we plot the confusion matrix.
cm_plot_labels = ['ictal','preictal', 'interictal']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

# More metrics
print("==============TEST RESULTS============")
testAcc, testPrec, testFScore = my_metrics(test_batches.classes, np.argmax(predictions, axis=-1))

print(classification_report(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1)))

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# We save our model
model.save( data_path + "model.h5py")

# We zip our "outout" folder
!zip -r chb18_30s_cmexican.zip output &> /dev/null