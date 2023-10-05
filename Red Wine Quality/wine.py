# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 13:22:28 2023

@author: raman
"""


import numpy as np
import pickle

# Load the saved model
loaded_model = pickle.load(open("C:/Users/raman/Downloads/ML_Projects/Red Wine Quality/random_forest.sav", 'rb'))

# Load the scaler object (assuming you previously saved it)
scaler = pickle.load(open("C:/Users/raman/Downloads/ML_Projects/Red Wine Quality/scaler_object.pkl", 'rb'))

input_data=(7.4,0.70,0.00,1.9,0.076,11.0,34.0,0.9987,3.51,0.56,9.4)
input_data_as_numpy_array=np.array(input_data)
input_data_reshape=input_data_as_numpy_array.reshape(1,-1)

#save the scaler object to a .pkl file
with open("scaler_object.pkl","wb") as scaler_file:
    pickle.dump(scaler,scaler_file)

std_data=scaler.transform(input_data_reshape)
print(std_data)
prediction=loaded_model.predict(std_data)
print("Predicted Class:",prediction)