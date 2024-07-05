import subprocess
import streamlit as st
from time import sleep

def DisplayPipeExecution(pipe_tag, delete_after_execution=False):
    command = ['kedro', 'run', '--tags', pipe_tag]
    process = subprocess.Popen(command, cwd='../', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

    placeholder = st.empty()
    with placeholder.container(height=250):
        while process.poll() is None:
            line = process.stdout.readline()
            if not line:
                continue
            st.write(line.strip())

    if delete_after_execution:
        placeholder.success('Pipeline ran succesfully! Loading the result:')
        sleep(3)
        placeholder.empty()
