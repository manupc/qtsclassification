# qtsclassification

This repository contains the source code for the paper "Time Series Quantum Classifiers with Amplitude Embedding". The code will be available once the article has been accepted for publication.

The structure of the code is as follows:
- Files QXXXXX.py are the main scripts to run the experimentation with the Hybrid Classic/Quantum neural network for the XXXXX dataset.
- Files TXXXXX.py are the main scripts to run the experimentation with the Traditional (classic) feedforward neural network for the XXXXX dataset.
- Files QTools.py, Dataset.py, TrainingTools.py contain helping functions and classes to load the data, make experiments and save/load results.
- Files QXXXXX_R.pkl / TXXXXX_R.pkl are Pickle files containing the results of all executions performed with the hybrid (Q) and Traditional (T) neural networks, for the problem XXXXX.

The source data files for the datasets are not included here, since they belong to the UEA & UCR Time Series Classification Repository website at https://www.timeseriesclassification.com . If you plan to run these scripts successfully, we suggest to download the datasets from the aforementioned repository and put the TRAIN and TEST arff files into the dataset folder.
