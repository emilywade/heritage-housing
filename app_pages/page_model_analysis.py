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
    
    st.info(
        f"Initially we considered various regression models including linear regression"
        f" methods and ensemble tree-based approaches to predict house sale prices. We started"
        f" by training an ElasticNet model, but on evaluating the performance it was clear the"
        f" skew in sale price was not being handled. \n\n"
        f"After handling target imbalance, LinearRegression and Ridge models performed the best."
        f" We decided to opt for a Ridge model due to its regularisation capabilities. \n\n"
        f"We used GridSearch CV to optimise the Ride regression hyperparameters. The model achieved"
        f"reasonable performance, although it is important to acknowledge that potential feature "
        f"engineering limitations (eg. missing data) remains a challenge that can affect prediction "
        f"quality. "
    )
    
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

    eval_text_path = f"outputs/ml_pipeline/predict_sale_price/{version}/regression_performance.txt"
    with open(eval_text_path, "r") as file:
        eval_text = file.read()
    st.write("### Regression Evaluation Metrics")
    st.code(eval_text)

    st.write("---")

    eval_plot_path = f"outputs/ml_pipeline/predict_sale_price/{version}/regression_evaluation_plots.png"
    st.write("### Regression Evaluation Plots")
    st.image(eval_plot_path)