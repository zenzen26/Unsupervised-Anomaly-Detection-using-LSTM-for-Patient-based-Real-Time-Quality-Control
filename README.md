# Repository Details
This repository contains the experiment and result for patient based real time quality control. The models developed include unsupervised LSTM autoencoder and moving average. The analytes that are experimented on is sodium and creatinine. The goal is to detect anomalies and evaluate the accuracy, false positive rate and false negative rate.

# Folders and Files Description

## Datasets
This folder contains the dataset used in this experiment.
- DeIdentified_Data.xlsx : The original datasets that consists all analyte results
- sodium.csv : Derived from DeIdentified_Data.xlsx, consists sodium related data
- creatinine.csv:  Derived from DeIdentified_Data.xlsx, consists creatinine related data

## LSTM Models
This folder consists of the trained model for unsupervised LSTM.
- LSTM_Creatinie_No_MA_1.h5 : LSTM without Moving Average Model for Creatinine
- LSTM_Creatinine_MA.h5 : LSTM with Moving Average Model for Creatinine
- LSTM_Sodium_MA_2.h5 :	LSTM with Moving Average Model for Sodium
- LSTM_Sodium_No_MA_1.h5 : LSTM without Moving Average for sodium

## LSTM Experiment
This folder consists of the training process for unsupervised LSTM, it does not include results and predictions. 
- LSTM_Creatinine_Final.ipynb : LSTM without Moving Average Model Training for Creatinine
- LSTM_Creatinine_MA_Final.ipynb : LSTM with Moving Average Model Training for Creatinine
- LSTM_Sodium_Final.ipynb : LSTM without Moving Average Model Training for Sodium
- LSTM_Sodium_MA_Final.ipynb : LSTM with Moving Average Model Training for Sodium

## LSTM Experiment Result
This folder consists the prediction result of LSTM. The result you see in each python notebook is only for one of the biases, to view the results for other bias rate, please change the bias value in the notebook.
- LSTM_Creatinine_MA_Result.ipynb : The prediction result of LSTM with Moving Average Model for Creatinine
- LSTM_Creatinine_Result.ipynb : The prediction result of LSTM without Moving Average Model for Creatinine
- LSTM_Sodium_MA_Result.ipynb : The prediction result of LSTM with Moving Average Model for Sodium
- LSTM_Sodium_Result.ipynb : LSTM without Moving Average Model for Sodium

## Moving Average Experiments
This folder consists of the "training" process for moving average, and the results of _all_ biases.
- MA_Creatinine_Final.ipynb	: Moving Average for Creatinine
- MA_Sodium_Final.ipynb	: Moving Average for Sodium

# Parameters that can be Tuned
Several parameters that can be tuned to view more results are as follows. Please also change the directories found in the notebooks to your own directory when you need to run it yourself. 

**For notebooks in LSTM Experiments:**
- window_size : The window size for MA (only found in LSTM with Moving Average notebooks)
- TIMESTEP : The size of time step when generating sequence for LSTM training

**For notebooks in LSTM Experiment Result:**

_Note: If you change the TIMESTEP or window_size previously during training, you will need to save the model again during training, and use the updated model for predictions. In short, ensure that the tunings are the same when you want to make a prediciton._
- window_size : The window size for MA, ensure that it is the same as the trained model (only found in LSTM with Moving Average notebooks)
- TIMESTEP : The size of time step sequence,  ensure that it is the same as the trained model
- bias : The rate of bias
- threshold: The value of threshold

**For notebooks in Moving Average Experiments:**
- window_size : The window size for moving average
- bias : The rate of bias
- threshold: The value of threshold

