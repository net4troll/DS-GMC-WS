import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

st.header("""
# Kidney Prediction App

This app predicts if the patient **Has CKD or not**

""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
Please input the data
""")

# Collects user input features into dataframe
def user_input_features():
    age=st.sidebar.number_input('age',step=1)
    blood_pressure=st.sidebar.slider('blood_pressure',50.0,180.0)
    specific_gravity=st.sidebar.slider('specific_gravity',1.005,1.025)
    albumin=st.sidebar.slider('albumin',0.0,5.0)
    sugar=st.sidebar.slider('sugar',0.0,5.0)
    blood_glucose_random=st.sidebar.slider('blood_glucose_random',22.0,490.0)
    blood_urea=st.sidebar.slider('blood_urea',1.5,391.0)
    serum_creatinine=st.sidebar.slider('serum_creatinine',0.4,76.0)
    sodium=st.sidebar.slider('sodium',4.5,163.0)
    potassium=st.sidebar.slider('potassium',2.5,47.0)
    hemoglobin=st.sidebar.slider('hemoglobin',3.1,17.8)
    
    red_blood_cells=st.sidebar.selectbox('red_blood_cells',options=['normal', 'abnormal']),
    pus_cell=st.sidebar.selectbox('pus_cell',options=['normal', 'abnormal']),
    pus_cell_clumps=st.sidebar.selectbox('pus_cell_clumps',options=['notpresent', 'present']),
    bacteria=st.sidebar.selectbox('bacteria',options=['notpresent', 'present']),
    packed_cell_volume=st.sidebar.selectbox('packed_cell_volume',options=['44', '38', '31', '32', '35', '39', '36', '33', '29', '28', '41', '16', '24', '37', '30', '34', '40', '45', '27', '48', '52', '14', '22', '18', '42', '17', '46', '23', '19', '25', '26', '15', '21', '43', '20', '47', '9', '49', '50', '53', '51', '54']),
    white_blood_cell_count=st.sidebar.selectbox('white_blood_cell_count',options=['7800', '6000', '7500', '6700', '7300', '9800', '6900', '9600', '12100', '4500', '12200', '11000', '3800', '11400', '5300', '9200', '6200', '8300', '8400', '10300', '9100', '7900', '6400', '8600', '18900', '21600', '4300', '8500', '11300', '7200', '7700', '14600', '6300', '7100', '11800', '9400', '5500', '5800', '13200', '12500', '5600', '7000', '11900', '10400', '10700', '12700', '6800', '6500', '13600', '10200', '9000', '14900', '8200', '15200', '5000', '16300', '12400', '10500', '4200', '4700', '10900', '8100', '9500', '2200', '12800', '11200', '19100', '12300', '16700', '2600', '26400', '8800', '7400', '4900', '8000', '12000', '15700', '4100', '5700', '11500', '5400', '10800', '9900', '5200', '5900', '9300', '9700', '5100', '6600']),
    red_blood_cell_count=st.sidebar.selectbox('red_blood_cell_count',options=['5.2', '3.9', '4.6', '4.4', '5', '4.0', '3.7', '3.8', '3.4', '2.6', '2.8', '4.3', '3.2', '3.6', '4', '4.1', '4.9', '2.5', '4.2', '4.5', '3.1', '4.7', '3.5', '6.0', '5.0', '2.1', '5.6', '2.3', '2.9', '2.7', '8.0', '3.3', '3.0', '3', '2.4', '4.8', '5.4', '6.1', '6.2', '6.3', '5.1', '5.8', '5.5', '5.3', '6.4', '5.7', '5.9', '6.5']),
    hypertension=st.sidebar.selectbox('hypertension',options=['yes', 'no']),
    diabetes_mellitus=st.sidebar.selectbox('diabetes_mellitus',options=['yes', 'no']),
    coronary_artery_disease=st.sidebar.selectbox('coronary_artery_disease',options=['no', 'yes']),
    appetite=st.sidebar.selectbox('appetite',options=['good', 'poor']),
    pedal_edema=st.sidebar.selectbox('pedal_edema',options=['no', 'yes']),
    anemia=st.sidebar.selectbox('anemia',options=['no', 'yes'])
    
    data = {'age' : age,
            'blood_pressure' : blood_pressure,
            'specific gravity' : specific_gravity,
            'albumin' : albumin,
            'sugar' : sugar,
            'red blood cells' : red_blood_cells,
            'pus cell' : pus_cell,
            'pus cell clumps' : pus_cell_clumps,
            'bacteria' : bacteria,
            'blood glucose random' : blood_glucose_random,
            'blood urea' : blood_urea,
            'serum creatinine' : serum_creatinine,
            'sodium' : sodium,
            'potassium' : potassium,
            'hemoglobin' : hemoglobin,
            'packed cell volume' : packed_cell_volume,
            'white blood cell count' : white_blood_cell_count,
            'red blood cell count' : red_blood_cell_count,
            'hypertension' : hypertension,
            'diabetes mellitus' : diabetes_mellitus,
            'coronary artery disease' : coronary_artery_disease,
            'appetite' : appetite,
            'pedal edema' : pedal_edema,
            'anemia' : anemia}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Combines user input features with entire dataset
# This will be useful for the encoding phase
data_raw = pd.read_csv('/Users/alfahwun/misc/kidney_raw.csv')
df = pd.concat([input_df,data_raw],axis=0)

# Encoding of ordinal features
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
def feature_encoder(*labels):
  for label in labels:
    df[label] = encoder.fit_transform(df[label])
feature_encoder('red blood cells','pus cell','pus cell clumps','bacteria','hypertension','diabetes mellitus','coronary artery disease','appetite','pedal edema','anemia')


# Reads in saved classification model
load_rf = pickle.load(open('/Users/alfahwun/misc/kidney_rf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_rf.predict(df)
prediction_proba = load_rf.predict_proba(df)

if prediction[0] == 1:
    ckd = 'patient has kidney disease'
else:
    ckd = 'patient has no kidney disease'
labels=['Prob. Not CKD', 'Prob. CKD']
values= prediction_proba[0]
fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=20,
                  marker=dict(colors=['green', 'red'], line=dict(color='#000000', width=2)))


st.subheader('Prediction')
st.write(ckd)

st.subheader('Prediction Probability')
st.write(fig)