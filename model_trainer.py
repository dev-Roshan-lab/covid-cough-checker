from csv import writer
import os
import librosa
import numpy as np

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)
       
for count, filename in enumerate(os.listdir('pos')): # we run the same for neagtive i.e change pos to neg, "pos" and "neg" are folder names containg positive and negative samples
    audio = "pos/"+filename
    print(audio)
    
    y, sr = librosa.load(audio, mono=True, duration=1)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
    chroma_stft = np.mean(chroma_stft)
    spec_cent = np.mean(spec_cent)
    spec_bw = np.mean(spec_bw)
    rolloff = np.mean(rolloff)
    zcr = np.mean(zcr)
    mfcc = np.mean(mfcc)
    
    contents = [chroma_stft,spec_cent,spec_bw,rolloff,zcr,mfcc,"positive"]
    
    append_list_as_row('dataset.csv', contents)
    print(str(count)+ "Done")
   
