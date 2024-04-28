import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from sklearn.preprocessing import MinMaxScaler
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
validPath = "./data/valid"
def main():

    # get numpy arrays representing image list and label list for training and testing and validation
    trainDataImages, trainDataLabels = load_data(trainPath, 'train')
    testDataImages, testDataLabels = load_data(testPath, 'test')
    validationDataImages, validationDataLables = load_data(validPath, 'valid')

    # keras formatted labels
    scaler = MinMaxScaler()
    trainDataLabels = scaler.fit_transform(trainDataLabels)
    testDataLabels = scaler.fit_transform(testDataLabels)   
    validationDataLables = scaler.fit_transform(validationDataLables)

    trainDataLabels = tf.keras.utils.to_categorical(trainDataLabels)
    testDataLabels = tf.keras.utils.to_categorical(testDataLabels)
    validationDataLables = tf.keras.utils.to_categorical(validationDataLables)

    # get model
    model = get_model()

    # fit against training data
    model.fit(trainDataImages, trainDataLabels, epochs=EPOCHS)
    # evaluate against test data
    model.evaluate(testDataImages, testDataLabels, verbose=2)



def load_data(path, group):
    images = []
    labels = []
    imagePath = path + "/images"
    for image in os.listdir(imagePath):

        imagePath = os.path.join(imagePath, image)
        labelPath = os.path.join(f'./data/{group}/labels', image.replace(".jpg", ".txt"))

  
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
                    # TODO: We need to preprocess the label data before appending it to the labels array current format ['int float float float'] -> [int, float, float, float] then pass to sklearns minmaxscaler
                    for row in reader:
                        if row is not None:
                            row = row[0].split(" ")
                            for item in row:
                                item = float(item)
                            label = row
                    print(label)
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

    return model


if __name__ == "__main__":
    main()