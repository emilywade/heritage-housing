import streamlit as st

def page_project_hypothesis_body():

    st.write("### Project Hypotheses and Validation")

    st.info(
        f"In this section, we outline the main hypotheses formulated at the beginning of the project and examine whether they were validated using data analysis and modeling results."
    )

    st.success(
        f"**Hypothesis 1:** Larger living area (GrLivArea) and total square footage (TotalSF) have a strong positive impact on house sale price.\n\n"
        f"✅ *Validated.* Feature importance and correlation analysis confirmed that bigger homes typically achieve higher prices.\n\n"
        
        f"**Hypothesis 2:** Higher overall quality (OverallQual) significantly increases sale price.\n\n"
        f"✅ *Validated.* OverallQual was consistently the most influential factor in both the exploratory data analysis and predictive model feature importance plots.\n\n"
        
        f"**Hypothesis 3:** Newer houses (lower HouseAge) tend to sell at higher prices.\n\n"
        f"⚖️ *Partially validated.* Although newer houses generally show higher prices, the relationship is not strictly linear; other features (like quality) can outweigh age effects.\n\n"
        
        f"**Hypothesis 4:** Larger lot area (LotArea) leads to higher sale price.\n\n"
        f"⚖️ *Partially validated.* Lot area does show a positive correlation with price, but the impact is weaker compared to living area or quality. This suggests buyers value interior space and quality more than outdoor space.\n\n"
        
        f"**Hypothesis 5:** Overall condition (OverallCond) has a moderate positive effect on price.\n\n"
        f"⚖️ *Partially validated.* Condition has an effect but is not as strong as quality. In some cases, even average condition houses achieved high prices if other features were favorable."
    )


