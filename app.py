import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import random

# Load the trained model
class MLP(nn.Module):
    def __init__(self, input_size=15):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

# Load dataset for preprocessing
df = pd.read_csv("train_cleaned.csv")
df.drop(columns=['Depression'], axis=1, inplace=True)

# Impute missing values for numerical columns
num_cols = ['Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction', 'Financial Stress']
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Impute missing categorical values
cat_cols = ['Dietary Habits', 'Degree']
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Encode categorical variables using Label Encoding
label_enc_cols = ['Gender', 'Working Professional or Student', 'Sleep Duration',
                  'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?',
                  'Family History of Mental Illness']

label_encoders = {}
for col in label_enc_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Normalize numerical features
scaler = MinMaxScaler()
df[num_cols + ['Age', 'Work/Study Hours']] = scaler.fit_transform(df[num_cols + ['Age', 'Work/Study Hours']])

# Load the trained model
input_size = len(df.columns)
model = MLP(15)
model.load_state_dict(torch.load("depression_model.pth"))
model.eval()

# Streamlit UI
st.title("🧠 Depression Prediction App")
st.markdown("Enter your details below to predict the likelihood of depression.")

# User Input Form
with st.form("user_input_form"):
    name = st.text_input("Name")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=10, max_value=100, step=1)
    city = st.text_input("City")
    working_status = st.selectbox("Working Professional or Student", ["Professional", "Student"])
    academic_pressure = st.slider("Academic Pressure", 1, 10, 5)
    work_pressure = st.slider("Work Pressure", 1, 10, 5)
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1)
    study_satisfaction = st.slider("Study Satisfaction", 1, 10, 5)
    job_satisfaction = st.slider("Job Satisfaction", 1, 10, 5)
    sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=24.0, step=0.5)
    dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Unhealthy"])
    degree = st.text_input("Degree")
    suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])
    work_study_hours = st.number_input("Work/Study Hours", min_value=0, max_value=24, step=1)
    financial_stress = st.slider("Financial Stress", 1, 10, 5)
    family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
    submitted = st.form_submit_button("Predict")


if submitted:
    for col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])  # Fix Pandas Warning

    # Validate Gender before transforming
    if gender in label_encoders['Gender'].classes_:
        gender_encoded = label_encoders['Gender'].transform([gender])[0]
    else:
        gender_encoded = -1  # Handle unknown categories safely

    


    # Ensure the input matches the model's expected size
        
    user_data = {
        
        'Gender': gender_encoded,
        'Age': age,
        'Working Professional or Student': label_encoders['Working Professional or Student'].transform([working_status])[0] if working_status in label_encoders['Working Professional or Student'].classes_ else -1,
        'Academic Pressure': academic_pressure,
        'Work Pressure': work_pressure,
        'CGPA': cgpa,
        'Study Satisfaction': study_satisfaction,
        'Job Satisfaction': job_satisfaction,
        'Sleep Duration': label_encoders['Sleep Duration'].transform([sleep_duration])[0] if sleep_duration in label_encoders['Sleep Duration'].classes_ else -1,
        'Dietary Habits': label_encoders['Dietary Habits'].transform([dietary_habits])[0] if dietary_habits in label_encoders['Dietary Habits'].classes_ else -1,
        'Degree': label_encoders['Degree'].transform([degree])[0] if degree in label_encoders['Degree'].classes_ else -1,
        'Have you ever had suicidal thoughts ?': label_encoders['Have you ever had suicidal thoughts ?'].transform([suicidal_thoughts])[0] if suicidal_thoughts in label_encoders['Have you ever had suicidal thoughts ?'].classes_ else -1,
        'Work/Study Hours': work_study_hours,
        'Financial Stress': financial_stress,
        'Family History of Mental Illness': label_encoders['Family History of Mental Illness'].transform([family_history])[0] if family_history in label_encoders['Family History of Mental Illness'].classes_ else -1

    }



    # Convert to DataFrame
    user_df = pd.DataFrame([user_data])

    # Scale numerical features
    user_df[num_cols + ['Age', 'Work/Study Hours']] = scaler.transform(user_df[num_cols + ['Age', 'Work/Study Hours']])

    # Convert to tensor
    user_tensor = torch.tensor(user_df.values, dtype=torch.float32)

    # Model prediction
    with torch.no_grad():
        prediction = model(user_tensor).item()

    # Display the result
    st.subheader("Prediction Result:")
    if prediction*2 > 0.5:
        st.error(f"High Risk of Depression 😞 (Score: {prediction*2:.2f})")
    else:
        st.success(f"Low Risk of Depression 😊 (Score: {prediction*2:.2f})")