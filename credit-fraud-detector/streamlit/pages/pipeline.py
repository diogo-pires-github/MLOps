import os
import requests
import subprocess

import streamlit as st
import streamlit.components.v1 as components

from time import sleep
from pathlib import Path
from dotenv import load_dotenv
from modules.nav import NavigationBar
from kedro.framework.project import configure_project

# Page configuration
load_dotenv()

pkg = Path(__file__).parent.name
configure_project(pkg)

st.set_page_config(
    page_title="Pipelines",
    page_icon="./images/pastel-de-nata.png",
    layout='wide')

NavigationBar()

def run_kedro_viz():
    if not st.session_state['kedro_viz_started']:
        st.session_state['kedro_viz_started'] = True
        subprocess.Popen(['kedro', 'viz', '--no-browser'], cwd='../', shell=False,
                         stderr=subprocess.PIPE, stdout=subprocess.PIPE)

def show_pipeline_viz():
    st.subheader('PIPELINE VISUALIZATION')

    reporter = st.info('Starting server...')

    run_kedro_viz()

    sleep(5)
    resp = requests.get(os.environ.get('KEDRO_VIZ'))
    while not resp and resp.status_code == 200:
        reporter.info('Waiting for server start...')
        resp = requests.get(os.environ.get('KEDRO_VIZ'))

    reporter.empty()


    if st.session_state['kedro_viz_started']:
        st.caption('This is interactive')
        components.iframe(os.environ.get('KEDRO_VIZ'), width=1400, height=900)


# Page content
st.markdown(
    '''
    # Here you can check our pipelines, developed with Kedro.
    '''
)


show_pipeline_viz()
