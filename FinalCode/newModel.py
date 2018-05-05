from utils import resizeImages, getData, makeSubmission, getAudioData
import numpy as np

bFolder = 'bird_dataset/'
essFolder = bFolder + "essential_data/"

num_species = 19

trainX, trainY, testX, testY, testIDs = getAudioData(essFolder, num_species)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import SGD

model = Sequential()

model.add(Conv1D(32, 4, activation='relu', input_shape=(193, 1)))
model.add(Conv1D(32, 4, activation='relu'))
model.add(MaxPooling1D(4))
model.add(Conv1D(64, 4, activation='relu'))
model.add(Conv1D(64, 4, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(num_species, activation='sigmoid'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


trainX = np.expand_dims(trainX, axis=2)
testX = np.expand_dims(testX, axis=2)

model.fit(trainX, trainY, batch_size=64, epochs=3000)

predictions = model.predict(testX)

# create submission.csv
submissionFolder = bFolder + 'Submission/'
modelName = 'conv1D_soft_3000_new'

makeSubmission(submissionFolder, modelName, num_species, testIDs, predictions)
