import streamlit as st

def page_summary_body():

    st.write("### Quick Project Summary")

    st.info(
        f"**Project Terms & Context**\n"
        f"* A **house sale** refers to a transaction where a residential property is sold.\n"
        f"* The **SalePrice** is the final price at which the house was sold.\n"
        f"* Various **property features** such as overall quality, living area size, garage finish, basement finish type, and neighborhood "
        f"can influence the sale price.\n\n"
        f"**Project Dataset**\n"
        f"* The dataset represents detailed information about residential house sales in Ames, Iowa.\n"
        f"* It includes **20+ features** covering physical attributes (e.g., overall quality, year built, number of rooms), "
        f"location (e.g., neighborhood), and additional amenities (e.g., fireplaces, garage type, basement quality).\n"
        f"* The primary goal is to **predict the SalePrice** of houses based on these features."
    )

    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/emilywade/heritage-housing)."
    )
    
    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - The client is interested in understanding which property features have the strongest impact on house prices. "
        f"This will help guide renovation or investment decisions.\n"
        f"* 2 - The client wants to accurately estimate the sale price of new or unseen houses, enabling better market positioning and "
        f"more effective pricing strategies."
    )
