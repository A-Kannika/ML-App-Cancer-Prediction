# Breast Cancer Prediction ML App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ml-app-cancer-prediction.streamlit.app/)

A simple web application built with Streamlit to predict breast cancer (Malignant or Benign) based on diagnostic features.
It's built using the Breast Cancer Wisconsin (Diagnostic) dataset from Kaggle.
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data


## 🚀 Live Demo

You can access and try the live application here:
**[https://ml-app-cancer-prediction.streamlit.app/](https://ml-app-cancer-prediction.streamlit.app/)**

## 📋 Project Overview

This project uses a trained machine learning model to classify breast cancer tumors. The web app provides an interactive interface where users can input various features (e.g., radius, texture, perimeter) from a biopsy, and the model will return a prediction in real-time.

The model (`cancer_prediction_model.pkl`) was trained on the `Breast_Cancer.csv` dataset using `scikit-learn`.

## ✨ Features

* **Interactive Interface:** User-friendly sliders and input fields to enter 30 different diagnostic features.
* **Real-Time Prediction:** Instantly classifies a tumor as **Malignant** or **Benign** based on the input data.
* **Data-Driven:** Built using a `scikit-learn` model trained on a standard breast cancer dataset.
* **Accessible:** Deployed as a public web app using Streamlit Cloud.

## 💻 Technology Stack

* **Python:** Core programming language.
* **Streamlit:** For building and deploying the interactive web app.
* **Scikit-learn (sklearn):** For loading and using the pre-trained machine learning model.
* **Pandas:** For data manipulation and creating the input DataFrame.
* **Numpy:** For numerical operations.

## 🛠️ How to Run Locally

To run this application on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/A-Kannika/ML-App-Cancer-Prediction.git](https://github.com/A-Kannika/ML-App-Cancer-Prediction.git)
    cd ML-App-Cancer-Prediction
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    Your browser should automatically open to `http://localhost:8501`.

    Learning resources
    https://www.youtube.com/watch?v=NfwfiyMi1lk
