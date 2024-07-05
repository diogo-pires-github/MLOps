import streamlit as st

def NavigationBar():
    with st.sidebar:
        st.page_link('index.py', label='Homepage', icon='🪙')
        st.page_link('pages/data.py', label='Check the data', icon='📊')
        st.page_link('pages/pipeline.py', label='Visualize the pipelines', icon='🪠')
        st.page_link('pages/run_pipeline.py', label='Run the pipelines', icon='🪄')
        st.page_link('pages/mlflow.py', label='Check the logs', icon='📜')
        st.page_link('pages/data_drift.py', label='Simulate data drift', icon='🔓')
