import streamlit as st
from modules.nav import NavigationBar

# Page configuration
st.set_page_config(
    page_title="Dashboard tool",
    page_icon="./images/pastel-de-nata.png")

NavigationBar()

if 'kedro_viz_started' not in st.session_state:
    st.session_state['kedro_viz_started'] = False

if 'mlflow_started' not in st.session_state:
    st.session_state['mlflow_started'] = False

# Page content
st.markdown(
    '''
    # Welcome to our humble dashboard.

    The sidebar on the left allows you to navigate throught different chunks
    of our project.
    '''
)
