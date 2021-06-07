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
    dataPath = os.path.join(myPath, data[i])
    # We read our EDF recording
    raw = read_raw_edf(dataPath, preload=True, stim_channel='auto', verbose=False)
    # We show the data type
    print('Data type: {}\n\n{}\n'.format(type(raw), raw))
    # We convert the EDF file into a dataframe
    raw_df = raw.to_data_frame()
    # We dropped the columns that are not needed
    df = raw_df.loc[:,~raw_df.columns.str.startswith('.-')]
    df = df.loc[:,~df.columns.str.startswith('--')]
    df = df.loc[:,~df.columns.str.startswith('time')]
    # We convert the dataframe into a numpy array
    edf_np = df.to_numpy(dtype='float64')
    # We save the numpy array as an .npy file
    np.save(data[i][:-4], edf_np)
