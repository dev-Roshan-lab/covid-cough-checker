# covid-cough-checker
Now diagnose covid-19 with cough sound!

---
## How to set up the project for your personal use
>- Downlod the [dataset](https://github.com/virufy/covid/tree/master/data)
>- split the negative and positive cough samples to two different folder _**pos**_ and _**neg**_
>- run _**model_trainer.py**__ this creates the dataset for you
>- next we create a KNN model and for that we run _**model_creator.py**
>- Prediction time!
```python
  
  import joblib
  import numpy as np
  
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
  
  model = joblib.load('path_to_model')
  #print(model)
  val = np.array([chroma_stft,spec_cent,spec_bw,rolloff,zcr,mfcc])# make an array of features
  val=val.reshape(1,-1)
  prediction = model.predict(val)
  print(prediction[0])
```



