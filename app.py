# heroku buildpacks:set heroku/python
# heroku buildpacks:add --index 1 heroku-community/apt
# heroku buildpacks
# Should show apt first, then python

# run these commands to setup the remote to install libs from Aptfile too!

from flask import Flask
from firebase import firebase
import os
import librosa
import numpy as np
import joblib
import io
from six.moves.urllib.request import urlopen

app = Flask(__name__)
app.debug = True
path = (os.path.abspath(os.path.dirname(__file__).replace("",""))+"//asset//model.sav")
print(path)
firebase = firebase.FirebaseApplication("https://drding-26dcc.firebaseio.com/", None)

@app.route('/')
def home():
    return "Running"

@app.route('/predict/<phno>')
def upload(phno):
    
    req_id = '%s' % phno
    
    data = firebase.get('/drding-covid/'+ req_id, '')
    url = data["audio"]
    
    y, sr = librosa.load(io.BytesIO(urlopen(url).read()))
    #print(sr)
    #print(y)


    #features
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
    
    model = joblib.load(path)
    val = np.array([chroma_stft,spec_cent,spec_bw,rolloff,zcr,mfcc])
    val=val.reshape(1,-1)
    prediction = model.predict(val)

    return prediction[0]
if __name__ == "__main__":
    app.run()
    
