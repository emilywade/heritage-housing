import streamlit as st
import pandas as pd
from src.data_management import load_pkl_file

def page_house_price_predictor_body():

    version = 'v1'  
    price_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_sale_price/{version}/reg_pipeline.pkl"
    )

    df_inherited = pd.read_csv("outputs/datasets/collection/InheritedHouses.csv")

    st.write("### Inherited Houses Price Prediction")
    st.info(
        f"* This page allows you to predict the **SalePrice** of inherited houses using your trained model. "
        f"These predictions are based on pre-processed features and model logic defined during your project.\n\n"
        f"Below you can inspect the unseen dataset, generate predictions, and download the results."
    )

    if st.checkbox("Inspect Inherited Houses Data"):
        st.write(f"Dataset shape: {df_inherited.shape[0]} rows × {df_inherited.shape[1]} columns")
        st.write(df_inherited.head(10))

    st.write("---")

    # run prediction
    if st.button("Run Price Prediction"):
        expected_features = pd.read_csv(f"outputs/ml_pipeline/predict_sale_price/{version}/X_train.csv").columns.to_list()

        df_subset = df_inherited[expected_features]

        predictions = price_pipe.predict(df_subset)

        # add predictions to the DataFrame
        df_inherited['Predicted_SalePrice'] = predictions

        # display
        st.success("Predictions generated successfully!")
        st.write(df_inherited.head(10))

