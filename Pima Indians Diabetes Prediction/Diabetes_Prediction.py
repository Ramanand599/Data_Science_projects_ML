# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 13:02:27 2023

@author: raman
"""


import numpy as np
import pickle
import streamlit as st

# Load the scaler object
loaded_rfc=pickle.load(open("C:/Users/raman/Downloads/ML_Projects/Pima Indians Diabetes Prediction/rfc.pkl","rb"))

def diabeted_prediction():
    # Load the rfc (RandomForestClassifier) object
    loaded_scaler=pickle.load(open("C:/Users/raman/Downloads/ML_Projects/Pima Indians Diabetes Prediction/scaler.pkl","rb"))
    
    # Define your input data
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
        
        
def main(debug=False):
    input_data=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']
    # setting the title
    st.title("Diabetes Prediction")
    
    # Getting the input data from the user
    Pregnancies = st.text_input("Pregnancies")
    Glucose = st.text_input("Glucose")
    BloodPressure = st.text_input("BloodPressure")
    SkinThickness = st.text_input("SkinThickness")
    Insulin = st.text_input("Insulin")
    BMI = st.text_input("BMI")
    DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction")
    Age = st.text_input("Age")
    
    
    #code for prediction
    diabeted=""
    
    # creating a button for prediction
    if st.button("Diabetes Prediction"):
        diabeted=diabeted_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    st.success(diabeted)
    
if __name__=="__main__":
    main(debug=True)