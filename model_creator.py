import librosa
import numpy as np
import joblib
import io
from six.moves.urllib.request import urlopen
#Trainng

import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

path = (os.path.abspath(os.path.dirname(__file__).replace("",""))+"/dataset.csv")
dt = pd.read_csv(path)

train, test = train_test_split(dt)

#print(train.shape)
#print(test.shape)

train_X = train[['chroma_stft','spec_cent','spec_bw','rolloff','zcr','mfcc']]# features
train_y = train.prognosis #labels

test_X = test[['chroma_stft','spec_cent','spec_bw','rolloff','zcr','mfcc']]
test_y = test.prognosis

#knn
model = KNeighborsClassifier(n_neighbors=3) #KNN Model for classification
model.fit(train_X, train_y)


joblib.dump(model, 'covid_cough_model.sav') #save model using joblib

print("model saved")

url = "https://firebasestorage.googleapis.com/v0/b/server-65459.appspot.com/o/neg-cough.wav?alt=media&token=bd7a2d9f-b5f4-41d8-869d-9d61fd31fe17"# an example url where the cough audio is manually uploaded
y, sr = librosa.load(io.BytesIO(urlopen(url).read()))
#print(y.shape)
#print(sr)

chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
zcr = librosa.feature.zero_crossing_rate(y)
mfcc = librosa.feature.mfcc(y=y, sr=sr)

#loading up variables 
chroma_stft = np.mean(chroma_stft)
spec_cent = np.mean(spec_cent)
spec_bw = np.mean(spec_bw)
rolloff = np.mean(rolloff)
zcr = np.mean(zcr)
mfcc = np.mean(mfcc)

model = joblib.load('covid_cough_model.sav')
#print(model)
val = np.array([chroma_stft,spec_cent,spec_bw,rolloff,zcr,mfcc])
val=val.reshape(1,-1)
prediction = model.predict(val)
print(prediction[0])

