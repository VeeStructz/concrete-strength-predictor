import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the trained CatBoost model
model = joblib.load('catboost_tuned_model.pkl')

# Title of the app
st.title("Concrete Compressive Strength Prediction")

# Input fields for user to enter feature values
st.header("Input Features")
cement = st.number_input("Cement (kg/m³)", min_value=0.0, value=540.0)
slag = st.number_input("Blast Furnace Slag (kg/m³)", min_value=0.0, value=0.0)
flyash = st.number_input("Fly Ash (kg/m³)", min_value=0.0, value=0.0)
water = st.number_input("Water (kg/m³)", min_value=0.0, value=162.0)
superplasticizer = st.number_input("Superplasticizer (kg/m³)", min_value=0.0, value=2.5)
coarseaggregate = st.number_input("Coarse Aggregate (kg/m³)", min_value=0.0, value=1040.0)
fineaggregate = st.number_input("Fine Aggregate (kg/m³)", min_value=0.0, value=676.0)
age = st.number_input("Age (days)", min_value=0, value=28)

# Create a feature array from user inputs
input_data = np.array([cement, slag, flyash, water, superplasticizer, coarseaggregate, fineaggregate, age]).reshape(1, -1)

# Make a prediction
if st.button("Predict"):
    # Predict compressive strength
    prediction = model.predict(input_data)
    st.success(f"Predicted Compressive Strength: {prediction[0]:.2f} MPa")

    # Generate stress-strain curve
    st.header("Stress-Strain Curve")
    st.write("The stress-strain curve is generated using Hognestad's parabolic relationship.")

    # Parameters for the stress-strain curve
    fc = prediction[0]  # Predicted compressive strength (MPa)
    epsilon_0 = 0.002  # Strain at peak stress
    epsilon_u = 0.0035  # Ultimate strain (concrete fails)

    # Generate strain values
    strain = np.linspace(0, epsilon_u, 100)

    # Hognestad's parabolic stress-strain relationship
    stress = np.where(
        strain <= epsilon_0,
        fc * (2 * (strain / epsilon_0) - (strain / epsilon_0) ** 2),  # Parabolic region
        fc * (1 - 0.15 * ((strain - epsilon_0) / (epsilon_u - epsilon_0)))  # Linear descending branch
    )

    # Plot the stress-strain curve
    fig, ax = plt.subplots()
    ax.plot(strain, stress, label="Stress-Strain Curve", color="blue")
    ax.set_xlabel("Strain")
    ax.set_ylabel("Stress (MPa)")
    ax.set_title("Stress-Strain Curve for Concrete")
    ax.legend()
    ax.grid(True)

    # Display the plot in Streamlit
    st.pyplot(fig)