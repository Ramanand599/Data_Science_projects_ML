import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open("C:/Users/raman/Downloads/ML_Projects/Red Wine Quality/random_forest.sav", 'rb'))

# Load the scaler object (assuming you previously saved it)
scaler = pickle.load(open("C:/Users/raman/Downloads/ML_Projects/Red Wine Quality/scaler_object.pkl", 'rb'))

# Creating a function for wine quality prediction
def wine_Quality_prediction(input_data):
    # Standardize the input data using the loaded scaler
    std_data = scaler.transform(input_data)

    # Make predictions using the loaded Random Forest Classifier
    prediction = loaded_model.predict(std_data)
    
    return prediction

def main():
    # Giving the title
    st.title("Wine Quality Web App")
    
    # Getting the input from the user
    fixed_acidity = st.text_input("Enter Fixed Acidity")
    volatile_acidity = st.text_input("Enter Volatile Acidity")
    citric_acid = st.text_input("Enter Citric Acid")
    residual_sugar = st.text_input("Enter Residual Sugar")
    chlorides = st.text_input("Enter Chlorides")
    free_sulfur_dioxide = st.text_input("Enter Free Sulfur Dioxide")
    total_sulfur_dioxide = st.text_input("Enter Total Sulfur Dioxide")
    density = st.text_input("Enter Density")
    pH = st.text_input("Enter pH")
    sulphates = st.text_input("Enter Sulphates")
    alcohol = st.text_input("Enter Alcohol")
    
    # Code for Prediction
    wine_prediction = ""
    
    # Creating a button for prediction
    if st.button("Predict Wine Quality"):
        # Create a list with the input values
        input_data = [float(fixed_acidity), float(volatile_acidity), float(citric_acid),
                      float(residual_sugar), float(chlorides), float(free_sulfur_dioxide),
                      float(total_sulfur_dioxide), float(density), float(pH),
                      float(sulphates), float(alcohol)]
        wine_prediction = wine_Quality_prediction([input_data])
        st.write(f"Predicted Wine Quality: {wine_prediction[0]}")

if __name__ == "__main__":
    main()
