import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
import requests
import pickle

# GitHub file links
MODEL_URL = "https://github.com/Isha055014/ANN--Digital-Marketing-Campaign-Success-Prediction--DLM/blob/25d8f8a6524e9aa6751b9a8d92dd712eadee32c8/ig14_ann_model.h5"
DATASET_URL = "https://github.com/Isha055014/ANN--Digital-Marketing-Campaign-Success-Prediction--DLM/blob/25d8f8a6524e9aa6751b9a8d92dd712eadee32c8/digital_marketing_campaigns_smes_.csv"
HISTORY_URL = "https://github.com/Isha055014/ANN--Digital-Marketing-Campaign-Success-Prediction--DLM/blob/25d8f8a6524e9aa6751b9a8d92dd712eadee32c8/ig14_history.pkl"

MODEL_PATH = "ig14_ann_model.h5"
DATASET_PATH = "digital_marketing_campaigns_smes_.csv"
HISTORY_PATH = "ig14_history.pkl"

# Download model if not found
@st.cache_resource
def load_ig14_model():
    if not os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "wb") as f:
            response = requests.get(MODEL_URL)
            f.write(response.content)
    return tf.keras.models.load_model(MODEL_PATH)

ig14_model = load_ig14_model()

# Download dataset if not found
if not os.path.exists(DATASET_PATH):
    with open(DATASET_PATH, "wb") as f:
        response = requests.get(DATASET_URL)
        f.write(response.content)

# File ID from Google Drive link
ig14_file_id = '1-mdmgqVhtp3fqMKuDK1yBzV6MH2OiXYj'

# Construct the download URL
ig14_download_url = f'https://drive.google.com/uc?id={ig14_file_id}'

# Load the dataset
ig14_DigitalMarketingCampaigns_data = pd.read_csv(ig14_download_url)

# Step 2: Data Preprocessing
ig14_DigitalMarketingCampaigns_data.drop(columns=['campaign_id'], inplace=True)  # Remove unnecessary ID column

# Encode categorical variables using Ordinal Encoding
ig14_categorical_cols = ['company_size', 'industry', 'marketing_channel', 'target_audience_area','target_audience_age',
                          'region', 'device', 'operating_system', 'browser','success']
ig14_ordinal_encoder = OrdinalEncoder()
ig14_DigitalMarketingCampaigns_data[ig14_categorical_cols] = ig14_ordinal_encoder.fit_transform(ig14_DigitalMarketingCampaigns_data[ig14_categorical_cols])

# Normalize numerical features using Min-Max Scaling
ig14_numerical_cols = ['ad_spend', 'duration', 'engagement_metric', 'conversion_rate',
                        'budget_allocation', 'audience_reach', 'device_conversion_rate',
                        'os_conversion_rate', 'browser_conversion_rate']
ig14_scaler = MinMaxScaler()
ig14_DigitalMarketingCampaigns_data[ig14_numerical_cols] = ig14_scaler.fit_transform(ig14_DigitalMarketingCampaigns_data[ig14_numerical_cols])

# Step 3: Split dataset into training and testing
ig14_X = ig14_DigitalMarketingCampaigns_data.drop(columns=['success'])
ig14_y = ig14_DigitalMarketingCampaigns_data['success']
ig14_X_train, ig14_X_test, ig14_y_train, ig14_y_test = train_test_split(ig14_X, ig14_y, test_size=0.2, random_state=5504714)

# Download and load training history
if not os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, "wb") as f:
        response = requests.get(HISTORY_URL)
        f.write(response.content)

with open(HISTORY_PATH, "rb") as f:
    ig14_history = pickle.load(f)

# Streamlit Dashboard
def ig14_run_dashboard():
    st.title("Digital Marketing Campaign Success Prediction")

    # **Sidebar Filters (Hyperparameter Selection)**
    epochs = st.sidebar.slider("Epochs", 10, 100, step=10, value=50)
    batch_size = st.sidebar.slider("Batch Size", 8, 64, step=8, value=16)
    learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.1, step=0.0001, format="%.4f", value=0.001)
    activation_function = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh", "softmax"], index=0)
    num_layers = st.sidebar.slider("Number of Layers", 1, 5, step=1, value=3)
    neurons_per_layer = st.sidebar.slider("Neurons per Layer", 8, 128, step=8, value=32)

    # Model Accuracy
    accuracy = ig14_model.evaluate(ig14_X_test, ig14_y_test, verbose=0)[1]
    st.metric(label="Model Accuracy", value=f"{accuracy*100:.2f}%")

    # ADDING VISUALIZATIONS BELOW

    ## **Model Accuracy Over Epochs**
    st.subheader("Model Accuracy Over Epochs")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ig14_history['accuracy'], label='Train Accuracy', color='blue')
    ax.plot(ig14_history['val_accuracy'], label='Validation Accuracy', color='red')
    ax.legend()
    ax.set_title("Training vs Validation Accuracy")
    st.pyplot(fig)

    ## **Feature Importance Visualization**
    st.subheader("Feature Importance")
    try:
        feature_importance = np.mean(np.abs(ig14_model.get_weights()[0]), axis=1)
        feature_names = ig14_X.columns
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(x=feature_importance, y=feature_names, ax=ax)
        ax.set_title("Feature Importance based on ANN Weights")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Feature Importance could not be calculated: {e}")

    ## **Prediction Probability Distribution**
    st.subheader("Prediction Probability Distribution")
    ig14_y_prob = ig14_model.predict(ig14_X_test)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(ig14_y_prob, bins=20, kde=True, ax=ax)
    ax.set_title("Predicted Probability Distribution")
    ax.set_xlabel("Predicted Probability of Success")
    st.pyplot(fig)

    ## **Confusion Matrix**
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pd.crosstab(ig14_y_test, (ig14_model.predict(ig14_X_test) > 0.5).astype("int32").ravel()), annot=True, fmt='d', ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# Run Streamlit
if __name__ == "__main__":
    ig14_run_dashboard()
