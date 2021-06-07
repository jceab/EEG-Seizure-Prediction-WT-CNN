# EEG-Seizure-Prediction-WT-CNN
The project is part of my Master's Thesis for the Data Science Master degree at the Universitat Oberta de Catalunya (UOC). It discusses the possibility to learn different features from multi-channel scalp EEG signals that will allow distinguishing between interictal (normal) and preictal (pre-seizure) phases. To achieve this, a time-frequency analysis will be applied to EEG recordings from the CHB-MIT Scalp EEG database by using Wavelet Transformations (WT) which will later serve asinput for a Convolutional Neural Network (CNN).

# Description of the scripts

1. EDA.ipynb: Exploratory Analysis of the EEG data.
2. Preprocessing.py: The EEG recordings are converted from the raw EDF file format first into a Pandas DataFrame for data preparation and cleaning, and then to a NumPy array to be able to apply Continuous Wavelet Transformations to the signals
3. CWT.py: This script splits the Numpy Array in fixed-length segments and then applies CWT to the signal segments in order to obtain the corresponding Scalograms to them.
4. CNN.py: CNN architecture used to predict the onset of an epileptic seizure.

Additional files:

- seizure-summary.csv: File containing the timestamps for the different phases (ictal, preictal and interictal) used to create the segments for the EEG recordings.
