# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 15:06:00 2023

@author: raman
"""


import numpy as np
import pickle
import streamlit as st

loaded_rf = pickle.load(open("C:/Users/raman/OneDrive/Documents/ML_Projects/House Price Prediction/best_rf.pkl", 'rb'))
loaded_scaler = pickle.load(open("C:/Users/raman/OneDrive/Documents/ML_Projects/House Price Prediction/scaler.pkl", 'rb'))
def house_price_prediction(input_data):
    input_data_as_numpy_array=np.array(input_data)
    input_data_reshape=input_data_as_numpy_array.reshape(1,-1)
    std_data=loaded_scaler.transform(input_data_reshape)
    print(std_data)

    # prediction
    prediction=loaded_rf.predict(std_data)
    return prediction[0]
    
    
def main():
    # Setting the title
    st.title("House Price Prediction")

    # Getting the input data from the user
    bedrooms = st.number_input("Number of bedrooms")
    bathrooms = st.number_input("Number of bathrooms")
    sqft_living = st.number_input("Square footage of living space")
    sqft_lot = st.number_input("Square footage of lot")
    floors = st.number_input("Number of floors")
    waterfront = st.number_input("Waterfront (0 for no, 1 for yes)")
    view = st.number_input("View (0-4)")
    condition = st.number_input("Condition (1-5)")
    grade = st.number_input("Grade (1-13)")
    sqft_above = st.number_input("Square footage above ground")
    sqft_basement = st.number_input("Square footage of basement")
    age = st.number_input("Age of the house")
    renovated = st.number_input("Renovated (0 for no, 1 for yes)")
    zipcode = st.number_input("Zipcode")
    lat = st.number_input("Latitude")
    long = st.number_input("Longitude")
    sqft_living15 = st.number_input("Square footage of living space (2015)")
    sqft_lot15 = st.number_input("Square footage of lot (2015)")

    input_data = [
        bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view,
        condition, grade, sqft_above, sqft_basement, age, renovated, zipcode,
        lat, long, sqft_living15, sqft_lot15
    ]

    if st.button("Predict House Price"):
        prediction = house_price_prediction(input_data)
        st.success(f"Predicted Price Of The House Is: ${prediction:,.2f}")
if __name__ == "__main__":
    main()