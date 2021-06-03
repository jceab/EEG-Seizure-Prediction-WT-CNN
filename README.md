# EEG-Seizure-Prediction-WT-CNN
The project is about applying CNNs to EEG data from CHB-MIT to predict seizure. It's the Master's Thesis for the Data Science Master degree at the Universitat Oberta de Catalunya (UOC). The objective of the project was to try to replicate the result obtained in the paper: Truong, Nhan Duy, et al. "Convolutional neural networks for seizure prediction using intracranial and scalp electroencephalogram." Neural Networks 105 (2018): 104-111.

The different scripts included consist to create scalograms (time-frequency representations) of EEG data from the CHB-MIT Epilepsy Database by applying Wavelet Transformation techniques and then use them as input to a CNN in order to predict the precital (pre-seizure) phase of a patient.

# Description of the scripts

1. EDA.py: Exploratory Analysis of the EEG data.
2. Preprocessing.py: The EEG recordings are converted from the raw EDF file format first into a Pandas DataFrame for data preparation and cleaning, and then to a NumPy array to be able to apply Continuous Wavelet Transformations to the signals
3. CWT.py: This script splits the Numpy Array in fixed-length segments and then applies CWT to the signal segments in order to obtain the corresponding Scalograms to them.
4. CNN.py: CNN architecture used to predict the onset of an epileptic seizure.

Additional files:

- seizure-summary.csv: File containing the timestamps for the different phases (ictal, preictal and interictal) used to create the segments for the EEG recordings.
