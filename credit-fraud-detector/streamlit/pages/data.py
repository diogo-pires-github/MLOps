import os
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from modules.nav import NavigationBar
from modules.data import get_data, plot_feats

# Page configuration
st.set_page_config(
    page_title="Data sources",
    page_icon="./images/pastel-de-nata.png",
    layout='wide')

NavigationBar()


# Page content
st.markdown(
    '''
    # Here you can see our data during different phases of our pipeline.

    Go ahead, try it:
    '''
)

# Choose data
choice = st.radio('Select the level:', ('Raw', 'Preprocessed', 'From feature store'))

# Get and show the data as table
df = get_data(choice)
st.dataframe(df.head(5))

# Visualization of Class
st.markdown('## Class')

fig, ax = plt.subplots(figsize=(15, 7))
sns.countplot(x='Class', data=df, ax=ax, palette=['blue', 'red'])
ax.set_title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
for p in ax.patches:
    x = p.get_bbox().get_points()[:,0]
    y = p.get_bbox().get_points()[1,1]
    ax.annotate(f'{100.*y/df.shape[0]:.2f}%', (x.mean(), y),
            ha='center', va='bottom')

st.pyplot(fig)

# Visualization of Features
st.markdown('## Features')
plot_feats(df)
