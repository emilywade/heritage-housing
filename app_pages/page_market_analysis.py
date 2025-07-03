import streamlit as st
from src.data_management import load_original_housing_data, load_cleaned_housing_data
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px


def page_market_analysis_body():

    # load data
    df_original = load_original_housing_data()
    df_cleaned = load_cleaned_housing_data()

    # variables found relevant during EDA
    vars_to_study = ['AboveGradeSF', 'GarageArea', 'GrLivArea', 'HouseAge', 'OverallQual', 'TotalSF']

    st.write("### House Price Study")
    st.info(
        f"* The client is interested in understanding which property characteristics "
        f"are most strongly associated with higher or lower sale prices."
    )

    # inspect data
    if st.checkbox("Inspect Original Dataset"):
        st.write(
            f"* The dataset has {df_original.shape[0]} rows and {df_original.shape[1]} columns. "
            f"Below are the first 10 rows:")
        st.write(df_original.head(10))

    st.write("---")

    st.info(
        f"The dataset required some initial cleaning before investigating  "
        f"correlations between variables.\n"
        f"Below is the cleaned dataset after:\n"
        f"* Dropping WoodDeckSF and EnclosedPorch due to severe levels of missing data\n"
        f"* Imputing missing values in LotFrontage, BedroomAbvGr and MasVnrArea with median values\n"
        f"* Imputing missing values in GarageFinish, BsmtFinType1 and BsmtExposure with 'missing'\n"
    )

    # inspect cleaned dataset
    if st.checkbox("Inspect Cleaned Dataset"):
        st.write(
            f"* The dataset has {df_cleaned.shape[0]} rows and {df_cleaned.shape[1]} columns. "
            f"Below are the first 10 rows:")
        st.write(df_cleaned.head(10))

    st.write("---")

    # correlation study summary
    st.write(
        f"* An exploratory analysis was conducted to study correlations with SalePrice. \n"
        f"The variables most strongly correlated with SalePrice are: **{vars_to_study}**."
    )

