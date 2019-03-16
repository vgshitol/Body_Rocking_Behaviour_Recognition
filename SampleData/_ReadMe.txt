This folder contains sample data for one trial. You will be provided with several trials for training. Your task will be predict the 'detection' in the same format as it provided here.

The data 'time.txt' contains the time stamps of the measurements and detections in seconds. The 'armIMU.txt' file contains the accelerometer (first three columns) and gyroscope (last three columns) for the arm. The 'wristIMU.txt' contains similar measurements for the wrist. The 'detection.txt' is the corresponding detections with a value of '1' when the rocking motion was observed. When predicting you will be given everything except for 'detection.txt', which you will have to produce. Remember that it must have the same number of rows as 'time.txt'

When analyzing time-series (and in particular inertial data), it is often useful to extract features over a window of time corresponding to a few seconds (anything around 3 to 5 seconds is often useful). These features can either be time based (e.g., mean, variance, correlation between channels) or frequency domain (e.g., look at specific frequency bands or maximum frequency response). This publication list some features that may be useful:

https://www.mdpi.com/2411-5134/2/4/32

The function "_matlab_getFeaturesIMU.m" provides an illustration of how features can be computed over a time window. For the second part of the project, I will encourage you to learn features from the data using a CNN.
