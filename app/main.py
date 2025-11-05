# use streamlit to create the app
import streamlit as st
import pickle
import pandas as pd
import math
import plotly.graph_objects as go
import numpy as np

def get_clean_data():
    data = pd.read_csv("data/data.csv")
    # drop unnamed and id columns
    data = data.drop(["Unnamed: 32", 'id'], axis=1)

    #Diagnosis (M = malignant, B = benign)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data

def add_sidebar():
    st.sidebar.header("Cell Nuclei Details")
    data = get_clean_data()

    slider_labels = [
        ('Mean Radius', 'radius_mean'),
        ('Mean Texture', 'texture_mean'),
        ('Mean Perimeter', 'perimeter_mean'),
        ('Mean Area', 'area_mean'),
        ('Mean Smoothness', 'smoothness_mean'),
        ('Mean Compactness', 'compactness_mean'),
        ('Mean Concavity', 'concavity_mean'),
        ('Mean Concave Points', 'concave points_mean'),
        ('Mean Symmetry', 'symmetry_mean'),
        ('Mean Fractal Dimension', 'fractal_dimension_mean'),
        ('Radius (Standard Error)', 'radius_se'),
        ('Texture (Standard Error)', 'texture_se'),
        ('Perimeter (Standard Error)', 'perimeter_se'),
        ('Area (Standard Error)', 'area_se'),
        ('Smoothness (Standard Error)', 'smoothness_se'),
        ('Compactness (Standard Error)', 'compactness_se'),
        ('Concavity (Standard Error)', 'concavity_se'),
        ('Concave Points (Standard Error)', 'concave points_se'),
        ('Symmetry (Standard Error)', 'symmetry_se'),
        ('Fractal Dimension (Standard Error)', 'fractal_dimension_se'),
        ('Worst Radius', 'radius_worst'),
        ('Worst Texture', 'texture_worst'),
        ('Worst Perimeter', 'perimeter_worst'),
        ('Worst Area', 'area_worst'),
        ('Worst Smoothness', 'smoothness_worst'),
        ('Worst Compactness', 'compactness_worst'),
        ('Worst Concavity', 'concavity_worst'),
        ('Worst Concave Points', 'concave points_worst'),
        ('Worst Symmetry', 'symmetry_worst'),
        ('Worst Fractal Dimension', 'fractal_dimension_worst')
    ]

    # create dictionary to return 
    input_dict = {}

    for label, key in slider_labels:

        # make the slide bar range
        min_value=float(0),
        max_value=float(data[key].max()),
        max_label = max_value[0]
        if max_label < 1:
            max_label = 1.0
        else:
            max_label = float(math.ceil(max_label / 10) * 10)
        min_label = min_value[0]
        max_value = (max_label,)
        
        # create the sidebar slider
        input_dict[key] = {
            "value": st.sidebar.slider(
                label,
                min_value=min_label,
                max_value=max_label,
                value=float(data[key].mean()),
            ),
            "max": max_label,
            "min": min_label
        }
    
    return input_dict

# scaled the data for our radar graph
# make all data inrange of 0 and 1
# Scaling by the min and max from the slider 
def get_scaled_values(input_dict):

    scaled_dict = {}

    for key, value in input_dict.items():
        actual_value = input_dict[key]["value"]
        max_val = input_dict[key]["max"]
        min_val = input_dict[key]["min"]
        scaled_value = (actual_value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict 

def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                  'Smoothness', 'Compactness', 'Concavity', 
                  'Concave Points', 'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'],
            input_data['texture_mean'],
            input_data['perimeter_mean'],
            input_data['area_mean'],
            input_data['smoothness_mean'],
            input_data['compactness_mean'],
            input_data['concavity_mean'],
            input_data['concave points_mean'],
            input_data['symmetry_mean'],
            input_data['fractal_dimension_mean'],
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'],
            input_data['texture_se'],
            input_data['perimeter_se'],
            input_data['area_se'],
            input_data['smoothness_se'],
            input_data['compactness_se'],
            input_data['concavity_se'],
            input_data['concave points_se'],
            input_data['symmetry_se'],
            input_data['fractal_dimension_se'],
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'],
            input_data['texture_worst'],
            input_data['perimeter_worst'],
            input_data['area_worst'],
            input_data['smoothness_worst'],
            input_data['compactness_worst'],
            input_data['concavity_worst'],
            input_data['concave points_worst'],
            input_data['symmetry_worst'],
            input_data['fractal_dimension_worst'],
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )

    return fig

# Create the prediction part using pickle
def add_prediction(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_array = np.array([v["value"] for v in input_data.values()]).reshape(1, -1)
    
    # This will make the mean is 0 or make it become reference point
    input_array_scaled = scaler.transform(input_array)

    # create the prediction
    prediction = model.predict(input_array_scaled)

    if prediction[0] == 0:
        st.write("Benign")
    else:
        st.write("Malignant")

    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])








def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor Application",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # create the side bar
    input_data = add_sidebar()

    with st.container():
        st.title("🩺 Breast Cancer Predictor Application")
        st.write("The Breast Cancer Predictor is a machine-learning–powered tool that helps estimate whether a tumor is likely benign or malignant, based on patient data. It's built using the Breast Cancer Wisconsin (Diagnostic) dataset from Kaggle.")

    # create 2 columns in the main page
    col1, col2 = st.columns([4,1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    
    with col2:
        add_prediction(input_data)



if __name__== "__main__":
    main()