from utils import resizeImages, getData, makeSubmission

import os
import pandas as pd
import librosa
import numpy as np
import soundfile as sf


def extract_feature(file_name):
    X, sample_rate = sf.read(file_name, dtype='float32')
    if X.ndim > 1:
        X = X[:,0]
    X = X.T

    # short term fourier transform
    stft = np.abs(librosa.stft(X))

    # mfcc
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    # chroma
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    # melspectrogram
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

    # spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)

    # combinedFeatures = np.empty((0,193))
    ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
    # combinedFeatures = np.vstack([combinedFeatures,ext_features])

    return ext_features


bFolder = 'bird_dataset/'
imFolder = bFolder + "spects/"
imDFolder = bFolder + "resized_spect/"
essFolder = bFolder + "essential_data/"

num_species = 19

XX = np.load('feat.npy')
print (np.shape(XX))

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
XY['spec_file_name'] = [i+".wav" for i in recIdFileMap.filename]

combinedFeatures = np.empty((0,193))

audioFolder = "./src_wavs/"

for f in XY['spec_file_name'].values:
    print (f)
    ff = extract_feature(audioFolder+f)
    combinedFeatures = np.vstack([combinedFeatures,ff])
    print np.shape(combinedFeatures)

# np.save('feat.npy',combinedFeatures)

