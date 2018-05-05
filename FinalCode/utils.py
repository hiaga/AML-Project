import numpy as np
import pandas as pd
import os
from PIL import Image
# import soundfile as sf



def resizeImages(baseFolder,targetFolder,nRow,nCol):
	
	fileList = os.listdir(baseFolder)

	for f in fileList:
		image = Image.open(baseFolder +f)
		rImage = image.resize((nRow,nCol))
		gImage = rImage.convert('L')
		gImage.save(targetFolder+f,"PNG")


def extract_feature(file_name):
    X, sample_rate = sf.read(file_name, dtype='float32')
    if X.ndim > 1:
        X = X[:,0]
    X = X.T

    stft = np.abs(librosa.stft(X))

    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz


def getAudioData(essFolder,num_species):
	recIdFileMap = pd.read_csv(essFolder + "rec_id2filename.txt", sep = ',')
	cvFoldMap = pd.read_csv(essFolder + "CVfolds_2.txt", sep= ',')
	labelsMap = pd.read_csv(essFolder + "rec_labels_test_hidden.txt", sep= ';')
	XY = np.zeros((len(labelsMap), num_species))

	for i in range(len(labelsMap)):
		row = labelsMap.iloc[i]
		speciesIDs = row[0].split(',')
		speciesIDs.pop(0)
		for j in speciesIDs:
			if(j!='?'):
				XY[i, (int)(j)] = 1

	XY= pd.DataFrame(XY)
	XY['rec_id'] = cvFoldMap.rec_id
	XY['fold'] = cvFoldMap.fold
	XY['filename'] = recIdFileMap.filename

	audioMatrix = np.load('feat.npy')

	trainXYSet = XY[XY.fold == 0]
	testXYSet =  XY[XY.fold == 1]

	trainX = audioMatrix[trainXYSet.index.values,:]
	testX = audioMatrix[testXYSet.index.values,:]

	trainY = trainXYSet.iloc[:,[i for i in range(19)]]
	testY = testXYSet.iloc[:,[i for i in range(19)]]

	trainY = trainY.values
	testY = testY.values

	return trainX, trainY, testX, testY, testXYSet.rec_id


def getData(essFolder, imgFolder, nRow, nCol, num_species):
	recIdFileMap = pd.read_csv(essFolder + "rec_id2filename.txt", sep = ',')
	cvFoldMap = pd.read_csv(essFolder + "CVfolds_2.txt", sep= ',')
	labelsMap = pd.read_csv(essFolder + "rec_labels_test_hidden.txt", sep= ';')
	XY = np.zeros((len(labelsMap), num_species))

	for i in range(len(labelsMap)):
		row = labelsMap.iloc[i]
		speciesIDs = row[0].split(',')
		speciesIDs.pop(0)
		for j in speciesIDs:
			if(j!='?'):
				XY[i, (int)(j)] = 1

	XY= pd.DataFrame(XY)
	XY['rec_id'] = cvFoldMap.rec_id
	XY['fold'] = cvFoldMap.fold
	XY['filename'] = recIdFileMap.filename
	XY['spec_file_name'] = [i+".png" for i in recIdFileMap.filename]

	imgMat = np.array([np.array(Image.open(imgFolder + img)).flatten()
              for img in XY['spec_file_name']],'f')

	trainXYSet = XY[XY.fold == 0]
	testXYSet =  XY[XY.fold ==1]

	imgMatTrain = np.array([np.array(Image.open(imgFolder + img)).flatten()
              for img in trainXYSet['spec_file_name']],'f')
	
	imgMatTest = np.array([np.array(Image.open(imgFolder + img)).flatten()
              for img in testXYSet['spec_file_name']],'f')


	trainY = trainXYSet.iloc[:,[i for i in range(19)]]
	testY = testXYSet.iloc[:,[i for i in range(19)]]

	trainY = trainY.values
	testY = testY.values

	trainX = imgMatTrain.reshape(imgMatTrain.shape[0], nRow, nCol,1)
	testX = imgMatTest.reshape(imgMatTest.shape[0], nRow, nCol,1)

	print(np.shape(trainX))
	print(np.shape(testX))

	trainX = trainX.astype('float32')
	testX = testX.astype('float32')

	trainX /= 255
	testX /= 255

	return trainX, trainY, testX, testY, testXYSet.rec_id

def getAudioFeaturesData(essFolder):
	pass


def makeSubmission(submissionFolder,modelName, num_species, testIDs, testPredictions):
	idCol = []
	probCol = []

	for bird in range(num_species):
		ids = np.array(testIDs)*100 + bird
		probs = testPredictions[:,bird]
		idCol += list(ids)
		probCol += list(probs)

	result = pd.DataFrame(idCol, columns=['Id'])
	result['Probability'] = probCol
	result.to_csv(submissionFolder+ 'submission_' + modelName + '.csv', index= False)
