import streamlit as st
import pandas as pd
from src.data_management import load_pkl_file
import matplotlib.pyplot as plt

def page_model_analysis():
    version = 'v1'
    pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_sale_price/{version}/reg_pipeline.pkl")
    tenure_labels_map = load_pkl_file(
        f"outputs/ml_pipeline/predict_sale_price/{version}/mappings.pkl")
    price_feat_importance = plt.imread(
        f"outputs/ml_pipeline/predict_sale_price/{version}/features_importance.png")
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{version}/X_train.csv")
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{version}/X_test.csv")
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{version}/y_train.csv")
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{version}/y_test.csv")

    st.write("### ML Pipeline: Predict House Price")

    st.write("---")

    # show pipeline steps
    st.write("* ML pipeline to predict house prices")
    st.write(pipe)
    st.write("---")

    # show best features
    st.write("* The features the model was trained and their importance.")
    st.write(X_train.columns.to_list())
    st.image(price_feat_importance)
    st.write("---")
