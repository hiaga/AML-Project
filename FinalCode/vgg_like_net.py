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
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import SGD


input_shape = (rRow, rCol, 1)

model = Sequential()

model.add(Conv2D(64, (4, 4), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(19, activation='sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd)

hist = model.fit(trainX, trainY, batch_size=32, epochs=400)

predictions = model.predict(testX)


# create submission.csv
submissionFolder = bFolder + 'Submission/'
modelName = 'vgg_1000'

makeSubmission(submissionFolder, num_species, testIDs, predictions)
