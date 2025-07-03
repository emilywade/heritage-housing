# Heritage Housing Issues

## Dataset Content
This dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/codeinstitute/housing-prices-data). The dataset contains detailed information on residential properties sold in Ames, Iowa from 2006 to 2010. It includes 1460 records and 24 features describing various aspects of the houses, such as:

- Physical attributes: Lot size, living area, basement details, number of bedrooms, garage info, etc.
- Quality and condition: Overall condition and quality ratings, year built, remodeling history.
- Sale details: Sale price.


## Business Requirements
The client has shared the dataset to be used to help answer their business requirements:
- Show how house features relate to sale prices using clear data visualizations.
- Predict sale prices for Lydia’s four inherited houses accurately.
- Allow price prediction for any house in Ames, Iowa.

## Hypothesis
- Hypothesis 1: Larger living area (GrLivArea) and total square footage (TotalSF) have a strong positive impact on house sale price.
- Hypothesis 2: Higher overall quality (OverallQual) significantly increases sale price.
- Hypothesis 3: Newer houses (lower HouseAge) tend to sell at higher prices.
- Hypothesis 4: Larger lot area (LotArea) leads to higher sale price.
- Hypothesis 5: Overall condition (OverallCond) has a moderate positive effect on price.

To answer these, we will use:
- Feature Importance
- Correlation analysis 
- Exploratory data analysis
- Predictive model feature importance plots

## Mapping Business Requirements to Data Visualisations
1. Visualizing relationships between house features and sale prices:
We will create clear and informative visualisations such as scatter plots, correlation heatmaps, and feature importance charts to help the client understand which house attributes most strongly impact sale prices in Ames, Iowa.

2. Predicting sale prices for the inherited houses:
The dashboard will include a dedicated section to input the features of Lydia’s four inherited houses and use a trained machine learning model to accurately predict their sale prices.

3. Allowing price prediction for any house in Ames:
A user-friendly interface will be implemented to allow input of custom house features for any property in Ames, enabling price predictions beyond the inherited houses and supporting future property valuation needs.

## ML Business Case
- We want a regression model to predict house sale prices so Lydia can use this for understanding the value of her inherited properties, as well as any other property in the area. The target variable is a continous numeric value, so we consider a regression model which is supervised and uni-dimensional.
- The ideal outcome is to provide Lydia with confidence in her ability to value properties in the area.
- The model success metric is R2 > 0.7. We will also evaluate the model using MAE, MSE and RMSE.
- The input for the model is house characteristics (either manually entered or inferred from the inherited houses dataset provided). The output is a sale value for the given property.
- Multiple regression techniques will be explored and evaluated before deciding on the best model.
- ML Pipeline will include data cleaning, feature engineering, scaling and regression. 

## Dashboard Design
Page 1: Quick project summary
- Purpose: Give the client a plain-language explanation of the project
- Objective of the project
- Data source and what it contains
- Jargon
- What was done (cleaning, modeling, dashboard)
- How this helps the client (insights, predictions, pricing strategy)
- Limitations and future improvements

Page 2: Attribute Correlations and Market Analysis
- Purpose: Directly address the business objective “How do house attributes correlate with sale price?”
- Based on market dataset only
- Insights drawn to help explain the pricing behind Ames properties
- Analysis on the housing market in Ames as a whole to allow the client to contextualise against previous housing market knowledge  
- Toggle to allow the client to visualise where their inherited properties fit in the wider Ames housing market

Page 3: Project Hypothesis and Validation
- Purpose: Explain the hypotheses and whether or not they were met. 

Page 4: House Price Prediction Tool
- Purpose: Directly addresses the business objective: “What are the predicted sales prices of the inherited properties and any other property in Ames?”
- Based off a trained model
- Two modes: one mode for the inherited properties, one mode with slider or input fields to allow client to predict other houses in the area
- “Run predictive analysis” button that serves the prospect data to our ML pipelines
- Outputs a predicted price and a confidence interval 

Page 5: Model Performance 
- Purpose: Show transparency in model capabilities and limits
- Considerations and conclusions after the pipeline is trained
- Present ML pipeline steps
- Feature importance 
- Pipeline performance

