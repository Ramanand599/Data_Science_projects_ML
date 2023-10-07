# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:54:45 2023

@author: raman
"""


import numpy as np
import pickle

# Load the scaler object
loaded_rfc=pickle.load(open("C:/Users/raman/Downloads/ML_Projects/Pima Indians Diabetes Prediction/rfc.pkl","rb"))

# Load the rfc (RandomForestClassifier) object
loaded_scaler=pickle.load(open("C:/Users/raman/Downloads/ML_Projects/Pima Indians Diabetes Prediction/scaler.pkl","rb"))


# Define your input data
input_data = (6, 148, 72, 35, 79, 33.6, 0.627, 50)
input_data_as_numpy_array = np.array(input_data)
input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

# Scale the input data using the loaded scaler
std_data = loaded_scaler.transform(input_data_reshape)

# Make predictions using the loaded RandomForestClassifier
prediction = loaded_rfc.predict(std_data)

# Print the result
if prediction == 0:
    print("No Diabetes")
else:
    print("Diabetes")
