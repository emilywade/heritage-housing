import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from src.data_management import load_pkl_file


def summarize_data(df, numeric_cols, categorical_cols):
    numeric_summary = df[numeric_cols].mean().round(2)

    categorical_summary_values = []
    for col in categorical_cols:
        mode_series = df[col].mode(dropna=True)
        if not mode_series.empty:
            categorical_summary_values.append(mode_series.iloc[0])
        else:
            categorical_summary_values.append(pd.NA)

    categorical_summary = pd.Series(categorical_summary_values, index=categorical_cols)

    summary_series = pd.concat([numeric_summary, categorical_summary], axis=0)
    
    return summary_series


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
        f"Below you can inspect the unseen dataset, generate predictions and compare to the rest of the market."
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
        
        st.metric("Min Predicted Sale Price", f"${df_inherited['Predicted_SalePrice'].min():,.0f}")
        st.metric("Max Predicted Sale Price", f"${df_inherited['Predicted_SalePrice'].max():,.0f}")
        st.metric("Average Predicted Sale Price", f"${df_inherited['Predicted_SalePrice'].mean():,.0f}")


        df_inherited = df_inherited.reset_index(drop=True)
        df_market = pd.read_csv(f"outputs/datasets/cleaned/HousingPrices.csv")

        df_inherited['Property'] = [f"Property {i+1}" for i in range(len(df_inherited))]

        fig = px.bar(
            df_inherited,
            x='Property',
            y='Predicted_SalePrice',
            text='Predicted_SalePrice',
            labels={'Property': 'Property', 'Predicted_SalePrice': 'Predicted Sale Price ($)'},
            title='Predicted Sale Prices for Inherited Properties',
            height=500
        )

        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(yaxis=dict(tickprefix="$"))

        st.plotly_chart(fig)

        st.write("---")

        numeric_cols = df_market.select_dtypes(include='number').columns.intersection(expected_features).tolist()
        categorical_cols = df_market.select_dtypes(include='object').columns.intersection(expected_features).tolist()

        market_summary = summarize_data(df_market, numeric_cols, categorical_cols)
        market_summary.name = 'Market Average'

        inherited_summary = summarize_data(df_inherited, numeric_cols, categorical_cols)
        inherited_summary.name = 'Inherited Average'

        inherited_props = df_inherited[numeric_cols + categorical_cols].copy()
        inherited_props = inherited_props.reset_index(drop=True)
        inherited_props.columns = numeric_cols + categorical_cols

        inherited_props = inherited_props.T
        inherited_props.columns = [f"Property {i+1}" for i in range(inherited_props.shape[1])]

        combined_summary = pd.concat([
            market_summary,
            inherited_summary,
            inherited_props
        ], axis=1)

        st.write("### Combined Summary Table")
        st.info(
            f"Key Insights:\n"
            f"* Inherited homes tend to have smaller living spaces but larger lot sizes compared to the average market home.\n"
            f"* Condition of inherited homes is average or slightly better, but overall quality scores are slightly lower.\n"
            f"* House age is somewhat younger on average, which likely rises the sale price of the inherited properties.\n"
            f"* Individual properties vary notably, showing diversity in size and layout (e.g., presence or absence of second floors)."
        )
        st.dataframe(combined_summary)
