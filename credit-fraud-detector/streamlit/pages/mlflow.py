
import os
import requests
import subprocess

import streamlit as st
import streamlit.components.v1 as components

from time import sleep
from pathlib import Path
from dotenv import load_dotenv
from kedro.framework.project import configure_project
from modules.nav import NavigationBar

# Page configuration
load_dotenv()

pkg = Path(__file__).parent.name
configure_project(pkg)

st.set_page_config(
    page_title="Pipelines",
    page_icon="./images/pastel-de-nata.png",
    layout='wide')

NavigationBar()

def run_mlflow():
    try:
        server = requests.get(os.environ.get('MLFLOW_SERVER'))
        if server.status_code == 200:
            st.session_state['mlflow_started'] = True
    except:
        if not st.session_state['mlflow_started']:
            st.session_state['mlflow_started'] = True
            subprocess.Popen(['mlflow', 'server'], cwd='../', shell=False,
                            stderr=subprocess.PIPE, stdout=subprocess.PIPE)

def show_mlflow_ui():
    st.subheader('MLFlow server')

    reporter = st.info('Starting server...')

    run_mlflow()

    sleep(5)
    resp = requests.get(os.environ.get('MLFLOW_SERVER'))
    while not resp and resp.status_code == 200:
        reporter.info('Waiting for server start...')
        resp = requests.get(os.environ.get('MLFLOW_SERVER'))

    reporter.empty()


    if st.session_state['mlflow_started']:
        st.caption('This is interactive')
        components.iframe(os.environ.get('MLFLOW_SERVER'), width=1400, height=900)


# Page content
st.markdown(
    '''
    # Here you can check our MLFlow Client, with our logs.
    '''
)


show_mlflow_ui()
