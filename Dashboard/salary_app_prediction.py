import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf

st.header("""
# Your avg salary worth App

This app predicts how much you are worth

""")

st.sidebar.header('Tell us more about you')

st.sidebar.markdown("""
Please input the data
""")

# Collects user input features into dataframe
def user_input_features():
    age =  st.sidebar.number_input('How old are you?',min_value=16,step=1)
    abilities = st.sidebar.multiselect(
    'What are your abilities?',
    ['Python','R','Spark','AWS','Excel'])
    Sum_ab = len(abilities)
    seniority = st.sidebar.selectbox('Got any degrees?',options=['Junior', 'Senior','None'])
    if seniority == 'Junior':
        seniority = 1
    elif seniority == 'Senior':
        seniority = 2
    else:
        seniority = 0

    job_simp_analyst = 0
    job_simp_data_engineer = 0
    job_simp_data_scientist= 0
    job_simp_director = 0
    job_simp_manager = 0
    job_simp_mle = 0 
    job_simp_na = 0

    job_simp = st.sidebar.selectbox('What is your dream job?',options=['Analyst', 'Data Engineer','Data Scientist','Director','Manager','Machine Learning'])

    if job_simp == 'Analyst':
        job_simp_analyst =1
    elif job_simp == 'Data Engineer':
        job_simp_data_engineer = 1
    elif job_simp == 'Data Scientist':
        job_simp_data_scientist = 1
    elif job_simp == 'Director':
        job_simp_director = 1
    elif job_simp == 'Manager':
        job_simp_manager = 1
    elif job_simp == 'Machine Learning':
        job_simp_mle = 1
    else:
        job_simp_na = 1

    same_state = 1 if st.sidebar.checkbox('Able to work out of state') else 0

    data = {'same_state':same_state,
            'age' : age,
            'seniority' : seniority,
            'Sum_ab':Sum_ab,
            'job_simp_analyst' : job_simp_analyst,
            'job_simp_data_engineer' : job_simp_data_engineer,
            'ob_simp_data_scientist' : job_simp_data_scientist,
            'job_simp_director' : job_simp_director,
            'job_simp_manager' : job_simp_manager,
            'job_simp_mle' : job_simp_mle,
            'job_simp_na' : job_simp_na,
            }
    return pd.DataFrame(data, index=[0])
df = user_input_features()
# Load the trained TensorFlow model
model = tf.keras.models.load_model('/Users/alfahwun/GMC/ANN/model1.h5')
# Make predictions on the features
predictions = model.predict(df)
# Print the predictions
if st.button('Get your avg salary'):
    st.write(f'your average salary is around {round(predictions[0][0])}K $ per year')