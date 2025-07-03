import streamlit as st
from app_pages.multipage import MultiPage

from app_pages.page_summary import page_summary_body
from app_pages.page_market_analysis import page_market_analysis_body
from app_pages.page_predict import page_house_price_predictor_body

app = MultiPage(app_name="Sale Price Predictor")

app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Housing Prices Market Analysis", page_market_analysis_body)
app.add_page("Housing Price Predictor", page_house_price_predictor_body)

app.run()