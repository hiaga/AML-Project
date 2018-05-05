from utils import resizeImages, getData, makeSubmission


bFolder = 'bird_dataset/'
imFolder = bFolder + "spects/"
imDFolder = bFolder + "resized_spect/"
essFolder = bFolder + "essential_data/"

rRow = 310
rCol = 64
num_species = 19


# resize the spectrograms
resizeImages(imFolder,imDFolder,rRow,rCol)

# get train, test data
trainX, trainY, testX, testY, testIDs = getData(essFolder, imDFolder, rRow, rCol, num_species)


# VGG-like Network
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, LSTM, Bidirectional
# from keras.optimizers import SGD
import numpy as np

model = Sequential()
model.add(Bidirectional(LSTM(32,return_sequences=True), input_shape=(rRow, rCol)))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(num_species, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


trainX = np.reshape(trainX,(322, rRow, rCol))
testX = np.reshape(testX, (323, rRow, rCol))

hist = model.fit(trainX, trainY, batch_size=32, epochs=8)
predictions = model.predict(testX)


# create submission.csv
submissionFolder = bFolder + 'Submission/'
modelName = 'bi_lstm_32_32_image_310_64'

makeSubmission(submissionFolder, modelName, num_species, testIDs, predictions)
