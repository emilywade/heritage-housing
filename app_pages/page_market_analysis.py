import streamlit as st
from src.data_management import load_housing_data
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px


def page_market_analysis_body():

    # load data
    df = load_housing_data()

    # variables found relevant during EDA
    vars_to_study = ['OverallQual', 'TotalSF', 'KitchenQual', 'GrLivArea', 'GarageArea', 'HouseAge', 'RemodAge']

    st.write("### House Price Study")
    st.info(
        f"* The client is interested in understanding which property characteristics "
        f"are most strongly associated with higher or lower sale prices."
    )

    # inspect data
    if st.checkbox("Inspect Dataset"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns. "
            f"Below are the first 10 rows:")
        st.write(df.head(10))

    st.write("---")
