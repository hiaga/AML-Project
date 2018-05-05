from utils import resizeImages, getData, makeSubmission


bFolder = 'bird_dataset/'
imFolder = bFolder + "spects/"
imDFolder = bFolder + "resized_spect/"
essFolder = bFolder + "essential_data/"

rRow = 623
rCol = 128
num_species = 19


# resize the spectrograms
resizeImages(imFolder,imDFolder,rRow,rCol)

# get train, test data
trainX, trainY, testX, testY, testIDs = getData(essFolder, imDFolder, rRow, rCol, num_species)


# VGG-like Network
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, LSTM
# from keras.optimizers import SGD
import numpy as np

model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(rRow, rCol)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(19, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


trainX = np.reshape(trainX,(322,rRow,rCol))
testX = np.reshape(testX, (323,rRow,rCol))

hist = model.fit(trainX, trainY, batch_size=32, epochs=8)
predictions = model.predict(testX)


# create submission.csv
submissionFolder = bFolder + 'Submission/'
modelName = 'lstm_image_623_128'

makeSubmission(submissionFolder, modelName, num_species, testIDs, predictions)
