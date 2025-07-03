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
        f"**Key Variable Insights:**\n\n"
        f"- **OverallQual**: Homes with consistently superior overall quality achieve higher sale prices.\n\n"
        f"- **TotalSF**: Larger total living area (including basement) is strongly linked to higher sale prices.\n\n"
        f"- **KitchenQual**: Higher kitchen quality ratings predict higher sale prices. This suggests kitchen condition is a major factor for buyers.\n\n"
        f"- **GrLivArea**: Larger above-ground living spaces are strongly correlated with higher prices, but are slightly less uniquely predictive than total size.\n\n"
        f"- **GarageArea**: Larger garages increase home value. However, garage space appears less important to price than overall living area.\n\n"
        f"- **HouseAge**: Newer houses generally sell for more. Age is negatively related to price.\n\n"
        f"- **RemodAge**: Homes that have been more recently remodeled tend to achieve higher sale prices."
    )

    
    # correlation matrix
    if st.checkbox("Correlation Matrix"):
        plot_corr_matrix(df_cleaned, num_vars)

    # numerical variable plots
    if st.checkbox("SalePrice vs Numerical Variables"):
        plot_numerical_vars(df_cleaned, num_vars)

    # categorical variable plots
    if st.checkbox("SalePrice vs Categorical Variables"):
        plot_categorical_vars(df_cleaned, cat_vars, target='SalePrice')

    

def plot_corr_matrix(df, num_vars):
    corr = df[num_vars + ['SalePrice']].corr()
    threshold = 0.0
    mask = np.abs(corr) < threshold
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", mask=mask, cbar=True, ax=ax)
    plt.title(f"Correlation Matrix")
    st.pyplot(fig)
    
    
def plot_numerical_vars(df, num_vars):
    for var in num_vars:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=var, y='SalePrice', data=df, ax=ax)
        sns.regplot(x=var, y='SalePrice', data=df, scatter=False, color='red', ax=ax)
        plt.title(f'SalePrice vs {var}')
        st.pyplot(fig)

def plot_categorical_vars(df, cat_vars, target='SalePrice'):
    for col in cat_vars:
        median_order = df.groupby(col)[target].median().sort_values()
        order = list(median_order.index)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x=col, y=target, data=df, order=order, ax=ax)
        ax.set_title(f"{target} Distribution by {col} (ordered by median)")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)