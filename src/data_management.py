import streamlit as st
import pandas as pd
import joblib

@st.cache_data
def load_original_housing_data():
    df = pd.read_csv("outputs/datasets/collection/HousingPrices.csv")
    return df

def load_cleaned_housing_data():
    df = pd.read_csv("outputs/datasets/cleaned/HousingPrices.csv")
    return df

def load_pkl_file(file_path):
    return joblib.load(filename=file_path)