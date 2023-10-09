# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 13:39:31 2023

@author: raman
"""


import numpy as np
import pickle
import streamlit as st

loaded_rf = pickle.load(open("C:/Users/raman/OneDrive/Documents/ML_Projects/Medical Cost Prediction/rf.pkl", 'rb'))
loaded_scaler = pickle.load(open("C:/Users/raman/OneDrive/Documents/ML_Projects/Medical Cost Prediction/scaler.pkl", 'rb'))


def medical_cost_prediction(input_data):
    # Changing the input data into a numpy array
    input_data_as_numpy_array = np.array(input_data)
    # Reshaping the numpy array into 2D for instance (1, n_features)
    input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

    # Standardize the input data using the loaded scaler
    std_data = loaded_scaler.transform(input_data_reshape)

    # Making the prediction on the basis of the loaded model
    prediction = loaded_rf.predict(std_data)
    return prediction[0]

def main(debug=False):
    # Setting the Title
    st.title("Medical Cost Prediction")
    
    # Getting the input from the user
    age = st.text_input("Age")
    sex = st.text_input("Sex")
    bmi = st.text_input("BMI")
    children = st.text_input("Children")
    smoker = st.text_input("Smoker")
    region = st.text_input("Region")
    
    # Code for prediction
    cost = ""
    
    # Creating a button for prediction
    if st.button("Medical Cost"):
        input_data = [float(age), sex, float(bmi), int(children), smoker, region]
        cost = medical_cost_prediction(input_data)
        st.write("Predicted Medical Cost:", cost)
    
if __name__ == "__main__":
    main(debug=True)