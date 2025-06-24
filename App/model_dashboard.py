## Importing necessary libraries
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from category_encoders import OneHotEncoder
from sklearn.preprocessing import StandardScaler, FunctionTransformer, MinMaxScaler, PowerTransformer

from sklearn.svm import SVR

import pickle

from datetime import datetime
import streamlit as st

import os
this_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(this_dir, 'x_train.csv')

st.set_page_config(layout='wide')

df = pd.read_csv(file_path)

## Required custom function
def clean_data(matrix):
    data = matrix.copy()
    col_names = ['soil_contamination','long_term_assessment','remediation_ind','local_datetime',
                'onshore_state_abbreviation','location_type','incident_area_type',
                'system_part_involved', 'could_be_hca','cause','commodity_released_type','unintentional_release_bbls',
                'release_type','ignite_ind','shutdown_datetime','restart_datetime','on_site_datetime']
    
    df = pd.DataFrame(data=data, columns=col_names)

    df['local_datetime'] = pd.to_datetime(df['local_datetime'])
    df['on_site_datetime'] = pd.to_datetime(df['on_site_datetime'])
    df['response_time'] = (df['on_site_datetime'] - df['local_datetime']).dt.total_seconds()
    df.loc[df['response_time'] <= 0, 'response_time'] = 0

    df["shutdown_datetime"] = pd.to_datetime(df["shutdown_datetime"], format='mixed')
    df['restart_datetime'] = pd.to_datetime(df["restart_datetime"], format='mixed')
    df['shutdown_period'] = ((df['restart_datetime'] - df['shutdown_datetime']).dt.total_seconds()).fillna(value=0)

    df['could_be_hca'] = df['could_be_hca'].replace({"Yes":1, "No":0})
    df['ignite_ind'] = df['ignite_ind'].replace({"Yes":1, "No":0})
    df['soil_contamination'] = df['soil_contamination'].replace({"Yes":1, "No":0})
    df['long_term_assessment'] = df['long_term_assessment'].replace({"Yes":1, "No":0})
    df['remediation_ind'] = df['remediation_ind'].replace({"Yes":1, "No":0})
    
    df.drop(columns=['local_datetime','on_site_datetime','shutdown_datetime','restart_datetime'], inplace=True)
    
    return df


## Loading my model
model_path = os.path.join(this_dir, "my_sv_model.pkl")
with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)


## Creating page
_, center, right = st.columns([1,3,1])
with center:
    image_path = os.path.join(this_dir, "sv_model.png")
    st.image(image_path, caption="Here is a visualization of the model", use_container_width=True)
with right:
    st.subheader("")
    st.subheader("This model's accuracy:")
    st.write("Model Mean Absolute Error: 15.78%")
    st.write("Model Median Absolute Error: 1.33%")
    st.write("")
    st.write("So while the MAE is large at 15.78%, the MEDEA is very low at 1.33%, meaning that the typical prediction error is below 1.33%")
    st.write("")
    st.write("Meaning: The model is very precise with a typical absolute error of 1.33%")

st.header("Welcome - Time to test this model :)")

left, right = st.columns([3,1])

with left:
    st.subheader("Impute the features for the test you want to predict into the form below: ")
    with st.form("Test features:"):
        st.markdown("#### **Please input the required dates and times below:**")
        col_1, col_2, col_3, col_4 = st.columns(4)
        with col_1:
            local_date = st.date_input("Incident date")
            local_time = st.time_input("Incident time")
            local_datetime = datetime.combine(local_date, local_time)
        
        with col_2:
            onsite_date = st.date_input("Site arrival date")
            onsite_time = st.time_input("Arrival time")
            onsite_datetime = datetime.combine(onsite_date, onsite_time)

        with col_3:
            shutdown_date = st.date_input("Facility shutdown date")
            shutdown_time = st.time_input("Shutdown time")
            shutdown_datetime = datetime.combine(shutdown_date, shutdown_time)

        with col_4:
            restart_date = st.date_input("Facility restart date")
            restart_time = st.time_input("Facility restart time")
            restart_datetime = datetime.combine(restart_date, restart_time)

        st.markdown("#### **Please input the other required features below (location, details, impact):**")
        col_5, col_6, col_7 = st.columns(3)
        with col_5:
            state = st.selectbox("State", df['onshore_state_abbreviation'].unique())
            location_type = st.selectbox("Location type", df['location_type'].unique())
            area_type = st.selectbox("Area type", df['incident_area_type'].unique())
            system_part = st.selectbox("System part", df['system_part_involved'].unique())
        
        with col_6:
            released_bbls = st.number_input("Release volume (bbl)")
            cause = st.selectbox("Cause", df['cause'].unique())
            commodity = st.selectbox("Commodity", df['commodity_released_type'].unique())
            release_type = st.selectbox("Release type", df['release_type'].unique())
            could_be_hca = st.selectbox("HCA", df['could_be_hca'].unique())
        
        with col_7:
            ignite = st.selectbox("Ignited", df['ignite_ind'].unique())
            soil = st.selectbox("Soil contamination", df['soil_contamination'].unique())
            remedy = st.selectbox("Remediation", df['remediation_ind'].unique())
            long_term_assess = st.selectbox("Long-term Assessment", df['long_term_assessment'].unique())

        submitted = st.form_submit_button("Create")

with right:
    if submitted:
        test = pd.DataFrame(
            {
                'local_datetime': local_datetime, 'on_site_datetime': onsite_datetime, 'onshore_state_abbreviation': state,
                'location_type': location_type, 'incident_area_type': area_type, 'system_part_involved': system_part,
                'could_be_hca': could_be_hca, 'cause': cause, 'commodity_released_type': commodity,
                'unintentional_release_bbls': released_bbls, 'release_type': release_type, 'ignite_ind': ignite, 
                'soil_contamination': soil, 'long_term_assessment': long_term_assess, 'remediation_ind': remedy,
                'shutdown_datetime': shutdown_datetime, 'restart_datetime': restart_datetime
            }, index=[0]
        )
        st.write('This is what what you want to predict on: ')
        st.write(test)
    else:
        st.subheader("Please fill the form on the left 👈👈")

    if st.button("Make prediction"):
        test = pd.DataFrame(
            {
                'local_datetime': local_datetime, 'on_site_datetime': onsite_datetime, 'onshore_state_abbreviation': state,
                'location_type': location_type, 'incident_area_type': area_type, 'system_part_involved': system_part,
                'could_be_hca': could_be_hca, 'cause': cause, 'commodity_released_type': commodity,
                'unintentional_release_bbls': released_bbls, 'release_type': release_type, 'ignite_ind': ignite, 
                'soil_contamination': soil, 'long_term_assessment': long_term_assess, 'remediation_ind': remedy,
                'shutdown_datetime': shutdown_datetime, 'restart_datetime': restart_datetime
            }, index=[0]
        )
        st.write(test)
        st.write("Prediction (Percentage to be recovered):")
        pred = (np.sin(loaded_model.predict(test))**2) 
        st.success(f'{pred[0] * 100:.2f} %')
        st.write("Volume in bbls:")
        bbls = float(pred * (test['unintentional_release_bbls'].values[0]))
        st.write(f'{bbls:.2f} bbls')
