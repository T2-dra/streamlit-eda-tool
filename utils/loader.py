import pandas as pd
import streamlit as st

@st.cache_data
def load_data(_file, file_name):
    _file.seek(0)
    df = pd.read_csv(_file)
    return df