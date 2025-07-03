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


def create_summary_with_prices(df_inherited, df_market, expected_features):
    numeric_cols = df_market.select_dtypes(include='number').columns.intersection(expected_features).tolist()
    categorical_cols = df_market.select_dtypes(include='object').columns.intersection(expected_features).tolist()

    market_summary = summarize_data(df_market, numeric_cols, categorical_cols)
    inherited_summary = summarize_data(df_inherited, numeric_cols, categorical_cols)

    market_avg_price = df_market['SalePrice'].mean() if 'SalePrice' in df_market else None
    inherited_avg_price = df_inherited['Predicted_SalePrice'].mean()

    property_names = df_inherited['Property'].tolist()

    summary_df = pd.concat([market_summary, inherited_summary], axis=1)
    summary_df.columns = ['Market Average', 'Inherited Average']

    prop_data = df_inherited[expected_features].copy()
    prop_data['Property'] = property_names
    prop_summary = prop_data.set_index('Property').T

    final_df = pd.concat([summary_df, prop_summary], axis=1)

    sale_price_values_numeric = [market_avg_price, inherited_avg_price] + list(df_inherited['Predicted_SalePrice'])
    sale_price_series = pd.Series(data=sale_price_values_numeric, index=final_df.columns, name='Predicted Sale Price')
    final_df = pd.concat([sale_price_series.to_frame().T, final_df])
    final_df.index = final_df.index.astype(str) 

    def format_value(val):
        if pd.isna(val):
            return val
        if isinstance(val, (int, float, np.number)):
            return f"{val:,.2f}"
        return val

    formatted_df = final_df.applymap(format_value)

    formatted_df.loc['Predicted Sale Price'] = formatted_df.loc['Predicted Sale Price'].apply(
        lambda v: f"${v}" if pd.notnull(v) else v
    )
    styled_df = formatted_df.style

    return styled_df




def page_house_price_predictor_body():

    version = 'v1'  
    price_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_sale_price/{version}/reg_pipeline.pkl"
    )

    df_inherited = pd.read_csv("outputs/datasets/collection/InheritedHouses.csv")

    st.write("### House Price Prediction")
    st.info(
        f"The client is interested in predicting the sale price of their inherited properties, "
        f"as well as any other property in the area."
    )
    st.info(
        "* Choose whether to predict prices for inherited houses or to enter your own custom house data below."
    )

    prediction_mode = st.radio(
        "Select prediction mode:",
        ("Inherited Houses", "Custom House")
    )
    expected_features = pd.read_csv(f"outputs/ml_pipeline/predict_sale_price/{version}/X_train.csv").columns.to_list()
    
    if prediction_mode == "Inherited Houses":
        st.subheader("Inherited Houses Price Prediction")

        if st.checkbox("Inspect Inherited Houses Data"):
            st.write(f"Dataset shape: {df_inherited.shape[0]} rows × {df_inherited.shape[1]} columns")
            st.write(df_inherited.head(10))


        if st.button("Run Price Prediction"):
            

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

            styled_summary = create_summary_with_prices(df_inherited, df_market, expected_features)


            st.write("### Combined Summary Table")
            st.info(
                f"Key Insights:\n"
                f"* Inherited homes tend to have smaller living spaces but larger lot sizes compared to the average market home.\n"
                f"* Condition of inherited homes is average or slightly better, but overall quality scores are slightly lower.\n"
                f"* House age is somewhat younger on average, which likely rises the sale price of the inherited properties.\n"
                f"* Individual properties vary notably, showing diversity in size and layout (e.g., presence or absence of second floors)."
            )
            st.dataframe(styled_summary)
            st.info(
                f"Comments on Predicted Prices:\n"
                f"* Property 1: low predicted price due to very small living space, no second floor, and high age.\n"
                f"* Property 2: despite large lot and better condition, lack of second floor and old age keep price"
                f" moderate — slightly higher than Property 1.\n"
                f"* Property 3: higher price driven by large size, newer age, and second floor. even though quality is slightly "
                f"lower.\n"
                f"* Property 4: large size and new age push price up, but smaller lot and other possible features keep it "
                f"slightly below Property 3.\n"
            )

    else:
        st.subheader("Predict Price for a Custom House")

        st.write("Enter your custom house features below:")

        first_flr_sf = st.number_input("1st Floor SF", min_value=0, value=1000)
        second_flr_sf = st.number_input("2nd Floor SF", min_value=0, value=500)
        gr_liv_area = st.number_input("Above Ground Living Area", min_value=0, value=1500)
        lot_area = st.number_input("Lot Area", min_value=0, value=10000)
        overall_cond = st.slider("Overall Condition", 1, 10, value=5)
        overall_qual = st.slider("Overall Quality", 1, 10, value=5)
        house_age = st.number_input("House Age (years)", min_value=0, value=30)
        total_sf = st.number_input("Total SF", min_value=0, value=2500)
        above_grade_sf = st.number_input("Above Grade SF", min_value=0, value=1500)

        input_dict = {
            "1stFlrSF": first_flr_sf,
            "2ndFlrSF": second_flr_sf,
            "GrLivArea": gr_liv_area,
            "LotArea": lot_area,
            "OverallCond": overall_cond,
            "OverallQual": overall_qual,
            "HouseAge": house_age,
            "TotalSF": total_sf,
            "AboveGradeSF": above_grade_sf,
        }

        input_df = pd.DataFrame([input_dict])

        for col in expected_features:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[expected_features]

        if st.button("Predict Custom House Price"):
            predicted_price = price_pipe.predict(input_df)[0]
            st.success(f"🏡 Predicted Sale Price: **${predicted_price:,.2f}**")