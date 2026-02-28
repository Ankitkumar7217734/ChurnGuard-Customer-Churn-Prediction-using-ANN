import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

#load the model
model = tf.keras.models.load_model('model.h5')

#load the scaler and encoders
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder_geography.pkl', 'rb') as file:
    one_hot_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
    
# Define the Streamlit app
st.title("Customer Churn Prediction")

#user input
gender = st.selectbox("Gender",label_encoder_gender.classes_)
geography = st.selectbox("Geography", one_hot_encoder.categories_[0])
age=st.slider("Age", 18, 92)
balance=st.number_input("Balance", min_value=0.0)
credits_score=st.slider("Credit Score", 350, 850)
estimated_salary=st.number_input("Estimated Salary", min_value=0.0)
tenure=st.slider("Tenure", 0, 10)
num_of_products=st.slider("Number of Products", 1, 4)
has_cr_card=st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member=st.selectbox("Is Active Member", ["Yes", "No"])

# Preprocess the input data
gender_encoded = label_encoder_gender.transform([gender])[0]

input_data =pd.DataFrame({
    "CreditScore": [credits_score],
    "Geography": [geography],
    "Gender": [gender_encoded],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [1 if has_cr_card == "Yes" else 0],
    "IsActiveMember": [1 if is_active_member == "Yes" else 0],
    "EstimatedSalary": [estimated_salary]
})

# One-hot encode the geographyinput
geography_encoded = one_hot_encoder.transform(input_data[["Geography"]])
if hasattr(geography_encoded, "toarray"):
    geography_encoded = geography_encoded.toarray()
geography_encoded_df = pd.DataFrame(geography_encoded, columns=one_hot_encoder.get_feature_names_out(["Geography"]))


# conbine the encoded geography with the rest of the input data
input_data = pd.concat([input_data.drop("Geography", axis=1), geography_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Make prediction
prediction = model.predict(input_data_scaled)
predicted_churn = prediction[0][0]

st.write(f"Predicted Churn Probability: {predicted_churn:.2f}")

if predicted_churn > 0.5:
    st.write("The customer is likely to churn.")
else:   
    st.write("The customer is unlikely to churn.")