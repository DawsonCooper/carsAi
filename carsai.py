import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
import numpy as np
import sys
import os
import cv2
import csv
    # Your code here
EPOCHS = 10
IMG_WIDTH = 128
IMG_HEIGHT = 128
NUM_CATEGORIES = 196
TEST_SIZE = 0.4
# set path var to the path of data folder
testPath = "./data/test"
trainPath = "./data/train"
validationPath = "./data/valid"
def main():

    # get numpy arrays representing image list and label list for training and testing and validation
    trainDataImages, trainDataLabels = load_data(trainPath)
    testDataImages, testDataLabels = load_data(testPath)
    validationDataImages, validationDataLables = load_data(testPath)

    # keras formatted labels
    print(trainDataLabels)
    trainDataLabels = tf.keras.utils.to_categorical(trainDataLabels)
    testDataLabels = tf.keras.utils.to_categorical(testDataLabels)
    validationDataLables = tf.keras.utils.to_categorical(validationDataLables)

    # get model
    model = get_model()

    # fit against training data
    model.fit(trainDataImages, trainDataLabels, epochs=EPOCHS)
    # evaluate against test data
    model.evaluate(testDataImages, testDataLabels, verbose=2)

    

def load_data(path):
    images = []
    labels = []
    imagePath = path + "/images"

    for image in os.listdir(imagePath):

        imagePath = os.path.join(imagePath, image)
        labelPath = os.path.join('./data/train/labels', image.replace(".jpg", ".txt"))

  
        if os.path.isfile(imagePath):
            data = cv2.imread(imagePath)
            if data is None:
                print("Image is None")
                continue
            data = cv2.resize(data, (IMG_WIDTH, IMG_HEIGHT))
            
            
            # read text file into an array
            if os.path.isfile(labelPath):
                with open(labelPath, "r") as file:
                    reader = csv.reader(file)
                    for row in reader:
                        if row is not None:
                            label = row
                    labels.append(label)
            images.append(data)
            
            
            
    return np.array(images), np.array(labels)  


def get_model():

    model = Sequential()


    # hidden layer (convolutional layers) 
    model.add(Conv2D(64, (3,3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # second hidden layer
    model.add(Conv2D(64, (3,3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # flatten and dense layer
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    
    # output layer
    model.add(Dense(NUM_CATEGORIES, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


if __name__ == "__main__":
    main()