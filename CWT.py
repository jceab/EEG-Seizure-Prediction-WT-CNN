from google.colab import drive
from pathlib import Path
import sys

drive.mount('/content/gdrive', force_remount=True)

base = Path('gdrive/My Drive/TFM/data/')
sys.path.append(str(base))

# unzipper from drive to local colab disk

zip_path = base/'epilep/edfs.zip'
!cp "{zip_path}" .
!unzip -q edfs.zip
!rm edfs.zip

# Library used to calculate CWT
!pip install spkit mne

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, find_events, pick_types, set_eeg_reference
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from mne import viz

import pywt

import numpy as np
import matplotlib.pyplot as plt

import spkit
print('spkit-version ', spkit.__version__)
import spkit as sp
from spkit.cwt import ScalogramCWT
from spkit.cwt import compare_cwt_example

%matplotlib inline

# We read the CSV with all the timestamps for the patients
df_seizures = pd.read_csv('seizures summary.csv',delimiter=';', encoding='utf8')
#list(df_seizures.columns)

%%capture
# Path with the recordings
myPath = 'edfs/'
# single subject
data = os.listdir(myPath)

waveletName = 'cMaxican'
#sample rate
fs = 256
#Output path were the scalograms will be saved
outputPath = 'scalograms_cmexican_30s/'

for i in range(len(data)):

    # We load the signal
    signals = np.load(data[i][:-4] + '.npy')
    
    # Selection of segments for each phase
    paciente = str(data[i][:-7])
    nRecording = str(data[i][5:-4])
    
    #fase = 'interictal'
    #fase = 'preictal'
    fases = ['ictal','preictal','interictal']
    
    for fase in fases:
      # Path
      directory = paciente+'/' + nRecording+'/' + fase
      path = os.path.join(outputPath, directory)
  
      # We create the directories
      os.makedirs(path)

      # We define the start and end time of the phase we are calculating                
      start = int(df_seizures.loc[df_seizures['npy'] == data[i][:-4], fase + ' inicio'].iloc[0])
      end = int(df_seizures.loc[df_seizures['npy'] == data[i][:-4], fase + ' fin'].iloc[0])
      length = end - start

      # Segment length
      segment_duration = 7680 # 256=1s, 2560=10s, 7680=30s
      
      for j in range(int(length/segment_duration)):
          signal = signals[start:start+segment_duration,:]
          t = np.arange(len(signal))/fs
      #We create the scalograms for each channel
          for k in range(len(signals[0])):
              XW,S = ScalogramCWT(signal[:,k],t,fs=fs,wType=waveletName,PlotPSD=True)
              imageName = 0
              # We save the scalogram as PNG image
              plt.imshow(np.abs(XW),origin='lower',cmap='jet',interpolation='sinc', aspect='auto')
              plt.axis('off')
              plt.savefig(outputPath + paciente + '/'+nRecording+'/' + fase +'/'+ paciente + nRecording +'_'+ str(j)+'_'+str(k)+'.png', bbox_inches='tight',pad_inches = 0)
              plt.close()
              imageName += 1
      
          start = start + (segment_duration+1)
      
      # We calculate the remaining segment in case the total length is not a multiple of the segment length we are using
      if end - start > 0:
          signal = signals[start:end,:]
          t = np.arange(len(signal))/fs
          #We create the scalograms for each channel
          for l in range(len(signal[0])):
            XW,S = ScalogramCWT(signal[:,l],t,fs=fs,wType=waveletName,PlotPSD=True)
            imageName = 0
            #We save the scalogram as PNG image
            plt.imshow(np.abs(XW),origin='lower',cmap='jet',interpolation='sinc',aspect='auto')
            plt.axis('off')
            plt.savefig(outputPath + paciente + '/'+nRecording+'/' + fase +'/'+ paciente + nRecording +'__'+ str(l)+'_'+str(l)+'.png', bbox_inches='tight',pad_inches = 0)
            plt.close()
            imageName += 1