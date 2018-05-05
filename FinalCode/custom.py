from utils import resizeImages, getData, makeSubmission


bFolder = 'bird_dataset/'
imFolder = bFolder + "spects/"
imDFolder = bFolder + "resized_spect/"
essFolder = bFolder + "essential_data/"

rRow = 623
rCol = 128
num_species = 19


# resize the spectrograms
# resizeImages(imFolder,imDFolder,rRow,rCol)

# get train, test data
trainX, trainY, testX, testY, testIDs = getData(essFolder, imDFolder, rRow, rCol, num_species)


# Custom CNN Network
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Convolution2D, MaxPooling2D, Input
from keras.layers.normalization import BatchNormalization
import numpy as np

bn_axis=3
imgInput = Input(shape=(rRow,rCol,1))

concatInput = BatchNormalization(axis=bn_axis, name='bn_conv1')(imgInput)

concatInput = Convolution2D(16, 5, 5, subsample=(1, 1), activation='relu',
                  init="he_normal", border_mode="same", name='conv_1')(concatInput)
concatInput = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(concatInput)

concatInput = BatchNormalization(axis=bn_axis, name='bn_conv2')(concatInput)
concatInput = Convolution2D(32, 5, 5, subsample=(1, 1), activation='relu',
                  init="he_normal", border_mode="same", name='conv2')(concatInput)
concatInput = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(concatInput)

concatInput = BatchNormalization(axis=bn_axis, name='bn_conv3')(concatInput)
concatInput = Convolution2D(64, 5, 5, subsample=(1, 1), activation='relu',
                  init="he_normal", border_mode="same", name='conv3')(concatInput)
concatInput = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(concatInput)

concatInput = BatchNormalization(axis=bn_axis, name='bn_conv4')(concatInput)
concatInput = Convolution2D(128, 5, 5, subsample=(1, 1), activation='relu',
                  init="he_normal", border_mode="same", name='conv4')(concatInput)
concatInput = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(concatInput)

concatInput = BatchNormalization(axis=bn_axis, name='bn_conv5')(concatInput)
concatInput = Convolution2D(19, 4, 4, subsample=(1, 1), activation='relu',
                  init="he_normal", border_mode="same", name='conv5')(concatInput)

concatInput = BatchNormalization(axis=bn_axis, name='bn_dense')(concatInput)
concatInput = Flatten(name='flatten')(concatInput)

concatInput = Dropout(0.1)(concatInput)
concatInput = Dense(512, activation='relu', name='dense')(concatInput)

concatInput = Dropout(0.1)(concatInput)
concatInput = Dense(num_species, activation='softmax', name='softmax')(concatInput)

model = Model(imgInput, concatInput)
model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=['accuracy'])


hist = model.fit(trainX, trainY, batch_size=32, nb_epoch=125)
predictions = model.predict(testX)


# create submission.csv
submissionFolder = bFolder + 'Submission/'
modelName = 'custom'

makeSubmission(submissionFolder, modelName, num_species, testIDs, predictions)
