# use streamlit to create the app
import streamlit as st
import pickle
import pandas as pd
import math

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
        
        st.sidebar.slider(
            label,
            min_value=min_label,
            max_value=max_label,
            value=float(data[key].mean())
        )


def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor Application",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # create the side bar

    add_sidebar()

    with st.container():
        st.title("🩺 Breast Cancer Predictor Application")
        st.write("The Breast Cancer Predictor is a machine-learning–powered tool that helps estimate whether a tumor is likely benign or malignant, based on patient data. It's built using the Breast Cancer Wisconsin (Diagnostic) dataset from Kaggle.")

    # create 2 columns in the main page
    col1, col2 = st.columns([4,1])

    with col1:
        st.write("This is column 1")
    
    with col2:
        st.write("This is column 2")



if __name__== "__main__":
    main()