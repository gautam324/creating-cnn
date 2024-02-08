

# Important library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from keras.preprocessing import image
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Read the labels.csv file and checking shape and records
labels_all = pd.read_csv("/content/drive/MyDrive/DATASETS/archive (9)/dogs.csv")
print(labels_all.shape)
labels_all.head()

# Visualize the number of each breeds
breeds_all = labels_all["labels"]
breed_counts = breeds_all.value_counts()
breed_counts.head()

CLASS_NAMES = ['Labrador', 'Golden Retriever', 'Poodle', 'German Shepherd', 'Bulldog', 'Beagle', 'Boxer', 'Dachshund', 'Yorkshire Terrier']
labels = labels_all[(labels_all['labels'].isin(CLASS_NAMES))]
labels = labels.reset_index()
labels.head()

import os

def extract_folder_names(directory):
    folder_names = []
    # Iterate over all entries in the specified directory
    for entry in os.scandir(directory):
        if entry.is_dir():
            folder_names.append(entry.name)
    return folder_names

# Example usage:
directory = "/content/drive/MyDrive/DATASETS/archive (9)/train"  # Specify your folder path here
folder_names = extract_folder_names(directory)
print("Folder names:", folder_names)

import os

def extract_folder_names(directory):
    folder_names = []
    # Iterate over all entries in the specified directory
    for entry in os.scandir(directory):
        if entry.is_dir():
            folder_names.append(entry.name)
    return folder_names

# Example usage:
directory = "/content/drive/MyDrive/DATASETS/archive (9)/train"  # Specify your folder path here
folder_names = extract_folder_names(directory)
print("Folder names:", folder_names)

import os
import numpy as np
from sklearn.preprocessing import label_binarize
from keras.preprocessing import image
from tqdm import tqdm


# Creating numpy matrix with zeros
X_data = np.zeros((len(labels), 224, 224, 3), dtype='float32')

# One hot encoding
Y_data = label_binarize(labels['labels'], classes=CLASS_NAMES)

# Reading and converting image to numpy array and normalizing dataset
base_directory = '/content/drive/MyDrive/DATASETS/archive (9)/train'

for i in tqdm(range(len(labels))):
    label = labels['labels'][i]
    label_folder = label.replace(' ', '_')
    img_path = os.path.join(base_directory, label_folder, '%03d.jpg' % (i + 1))
    print("Image path:", img_path)  # Print the image path for debugging
    if not os.path.exists(img_path):
        print(f"Skipping image {i+1} as it does not exist.")
        continue
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    x = np.expand_dims(img.copy(), axis=0)
    X_data[i] = x / 255.0

# Printing train image and one hot encode shape & size
print('\nTrain Images shape: ', X_data.shape, ' size: {:,}'.format(X_data.size))
print('One-hot encoded output shape: ', Y_data.shape, ' size: {:,}'.format(Y_data.size))

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Building the Model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(len(CLASS_NAMES), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

model.summary()

# Splitting the data set into training and testing data sets
X_train_and_val, X_test, Y_train_and_val, Y_test = train_test_split(X_data, Y_data, test_size = 0.1)
# Splitting the training data set into training and validation data sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train_and_val, Y_train_and_val, test_size = 0.2)

# Training the model
epochs = 10
batch_size = 128

history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs,
                    validation_data = (X_val, Y_val))

# Plot the training history
plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], color='r')
plt.plot(history.history['val_accuracy'], color='b')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'val'])

plt.show()

Y_pred = model.predict(X_test)
score = model.evaluate(X_test, Y_test)
print('Accuracy over the test set: \n ', round((score[1]*100), 2), '%')

# Plotting image to compare
plt.imshow(X_test[1,:,:,:])
plt.show()

# Finding max value from predition list and comaparing original value vs predicted
print("Originally : ",labels['breed'][np.argmax(Y_test[1])])
print("Predicted : ",labels['breed'][np.argmax(Y_pred[1])])
