import streamlit as st
import tensorflow as tf
import torch
import numpy
from predict import *

#import matplotlib.pyplot as plt
#import seaborn as sns


#sns.set_theme(style="darkgrid")
#sns.set()
#st.title('Check Text for Political Bias')

text_input = st.text_area('Input text here:') 

model = load_model()

scores = score_bias(model, text_input)

