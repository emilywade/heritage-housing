import streamlit as st
from src.data_management import load_original_housing_data, load_cleaned_housing_data
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import numpy as np


def page_market_analysis_body():

    # load data
    df_original = load_original_housing_data()
    df_cleaned = load_cleaned_housing_data()

    # variables found relevant during EDA
    vars_to_study = ['AboveGradeSF', 'GarageArea', 'GrLivArea', 'HouseAge', 'OverallQual', 'TotalSF', 'BsmtExposure', 'GarageFinish', 'KitchenQual']

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


    num_vars = ['AboveGradeSF', 'GarageArea', 'GrLivArea', 'HouseAge', 'OverallQual', 'TotalSF']
    cat_vars = ['BsmtExposure', 'GarageFinish', 'KitchenQual']
    
    
    st.info(
        f"**Key Observations:**\n\n"
        f"- **OverallQual** has the strongest positive correlation with **SalePrice**, meaning that houses with better overall quality ratings tend to sell for significantly higher prices.\n"
        f"- **TotalSF** (Total Square Footage) is also strongly positively correlated with **SalePrice**, indicating that larger houses generally command higher prices.\n"
        f"- **HouseAge** shows an inverse relationship with **SalePrice**, suggesting that newer houses tend to be more expensive, while older houses typically have lower sale prices."
    )
    

    # correlation matrix
    if st.checkbox("Correlation Matrix"):
        corr = df_cleaned[num_vars + ['SalePrice']].corr()
        threshold = 0.0
        mask = np.abs(corr) < threshold
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", mask=mask, cbar=True, ax=ax)
        plt.title(f"Correlation Matrix")
        st.pyplot(fig)
    
    

    # numerical variable plots
    if st.checkbox("SalePrice vs Numerical Variables"):
        for var in num_vars:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(x=var, y='SalePrice', data=df_cleaned, ax=ax)
            sns.regplot(x=var, y='SalePrice', data=df_cleaned, scatter=False, color='red', ax=ax)
            plt.title(f'SalePrice vs {var}')
            st.pyplot(fig)
