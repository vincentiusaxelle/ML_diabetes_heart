import streamlit as st
import pickle
import numpy as np
import pandas as pd
from joblib import load

diabetes_model = load('diabetes_model69.pkl')
heart_model = load('heart_model69.pkl')

def predict_diabetes(features):
    prediction = diabetes_model.predict(features)
    return prediction

def predict_heart_disease(features):
    prediction = heart_model.predict(features)
    return prediction

def main():
    st.title('Diabetes and Heart Disease Prediction')
    st.write("""
## How to Use:
1. Select your age range, gender, and answer various health-related questions.
2. Adjust sliders and radio buttons based on your health information.
3. Once you've filled in all the details, click the buttons to predict diabetes or heart disease.
""")
    age_mapping = {
        '18-24': 1,
        '25-29': 2,
        '30-34': 3,
        '35-39': 4,
        '40-44': 5,
        '45-49': 6,
        '50-54': 7,
        '55-59': 8,
        '60-64': 9,
        '65-69': 10,
        '70-74': 11,
        '75-79': 12,
        '80 or older': 13
    }
    
    st.header('Age')
    Age = st.radio('Select your age range:', list(age_mapping.keys()))

    st.header('Sex/Gender')
    Sex = st.radio('Select your gender:', ['Female', 'Male'])

    st.header('High Blood Pressure')
    HighBP = st.radio('Do you have high blood pressure?', ['No', 'Yes'])

    st.header('High Cholestrol')
    HighChol = st.radio('Do you have high cholestrol?', ['No', 'Yes'])

    st.header('Cholestrol Check')
    CholCheck = st.radio('Have you had a cholesterol check in the last 5 years?', ['No', 'Yes'])

    st.header("Body Mass Index (BMI)")
    BMI = st.slider('Select your BMI:', 12.0, 98.0)

    st.header("Smoker")
    Smoker = st.radio('Are you a smoker?', ['No', 'Yes'])

    st.header("Stroke")
    Stroke = st.radio('Have you had a stroke?', ['No', 'Yes'])

    st.header("Physical Activity")
    PhysActivity = st.radio('Have you engaged in physical activity in the past 30 days?', ['No', 'Yes'])

    st.header("Fruits Consumption")
    Fruits = st.radio('Do you consume fruits 1 or more times per day?', ['No', 'Yes'])

    st.header("Vegetables Consumption")
    Veggies = st.radio('Do you consume vegetables 1 or more times per day?', ['No', 'Yes'])
    
    st.header("Heavy Alcohol Consumption")
    HvyAlcoholConsump = st.radio('Do you engage in heavy alcohol consumption? [Adult Men > 14 drinks per week][Adult Women > 7 drinks per week]', ['No', 'Yes'])

    st.header("Health Care Coverage")
    AnyHealthcare = st.radio('Do you have any kind of health care coverage?', ['No', 'Yes'])

    st.header("Unable to See a Doctor Due to Cost")
    NoDocbcCost = st.radio('Have you been unable to see a doctor due to cost in the past 12 months?', ['No', 'Yes'])
    
    st.header("General Health")
    GenHlth = st.radio('Rate your general health:', ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'])
    
    st.header("Days of Poor Mental Health")
    MentHlth = st.slider('How many days of poor mental health have you experienced in the past 30 days?', 0.0, 30.0)

    st.header("Physical Illness")
    PhysHlth = st.slider('How many days have you experienced physical illness in the past 30 days?', 0.0, 30.0)

    st.header("Difficulty Walking or Climbing Stairs")
    DiffWalk = st.radio('Do you have serious difficulty walking or climbing stairs?', ['No', 'Yes'])

    sex_mapping = {'Female': 0, 'Male': 1}
    high_bp_mapping = {'No': 0, 'Yes': 1}
    high_chol_mapping = {'No': 0, 'Yes': 1}
    chol_check_mapping = {'No': 0, 'Yes': 1}
    smoker_mapping = {'No': 0, 'Yes': 1}
    stroke_mapping = {'No': 0, 'Yes': 1}
    phys_activity_mapping = {'No': 0, 'Yes': 1}
    fruits_mapping = {'No': 0, 'Yes': 1}
    veggies_mapping = {'No': 0, 'Yes': 1}
    hvy_alcohol_mapping = {'No': 0, 'Yes': 1}
    healthcare_mapping = {'No': 0, 'Yes': 1}
    no_doc_cost_mapping = {'No': 0, 'Yes': 1}
    gen_hlth_mapping = {'Excellent': 1, 'Very Good': 2, 'Good': 3, 'Fair': 4, 'Poor': 5}
    diff_walk_mapping = {'No': 0, 'Yes': 1}

    features = np.array([
        age_mapping[Age],
        sex_mapping[Sex],
        high_bp_mapping[HighBP],
        high_chol_mapping[HighChol],
        chol_check_mapping[CholCheck],
        BMI,
        smoker_mapping[Smoker],
        stroke_mapping[Stroke],
        phys_activity_mapping[PhysActivity],
        fruits_mapping[Fruits],
        veggies_mapping[Veggies],
        hvy_alcohol_mapping[HvyAlcoholConsump],
        healthcare_mapping[AnyHealthcare],
        no_doc_cost_mapping[NoDocbcCost],
        gen_hlth_mapping[GenHlth],
        MentHlth,
        PhysHlth,
        diff_walk_mapping[DiffWalk]
    ]).reshape(1, -1)
    
    if st.button('Check My Health Conditions'):
        diabetes_prediction = predict_diabetes(features)
        heart_prediction = predict_heart_disease(features)
    
        diabetes_result = 'Diabetes Detected' if diabetes_prediction[0] == 1 else 'No Diabetes'
        heart_result = 'Heart Disease Detected' if heart_prediction[0] == 1 else 'No Heart Disease'
    
        st.success(f'Diabetes Prediction: {diabetes_result}')
        st.success(f'Heart Disease Prediction: {heart_result}')

        if diabetes_prediction[0] == 1:
            st.write("It seems that diabetes is detected. Here are some suggestions:")
            st.write("- Consult with your healthcare provider for further evaluation and management.")
            st.write("- Follow a balanced diet and maintain a healthy weight.")
            st.write("- Engage in regular physical activity.")
            st.write("- Monitor your blood sugar levels regularly.")
        
        if heart_prediction[0] == 1:
            st.write("It seems that heart disease is detected. Here are some suggestions:")
            st.write("- Consult with your healthcare provider for further evaluation and management.")
            st.write("- Follow a heart-healthy diet low in saturated fats, cholesterol, and sodium.")
            st.write("- Engage in regular physical activity, as recommended by your healthcare provider.")
            st.write("- Monitor your blood pressure and cholesterol levels regularly.")

if __name__ == '__main__':
    main()