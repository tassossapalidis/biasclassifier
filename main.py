import streamlit as st
import torch
import numpy as np
from predict import *

import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="darkgrid")
sns.set()
st.title('Check Text for Political Bias')

model = load_model()

text_input = st.text_area('Input text here:') 
if len(text_input) != 0:
    scores = score_bias(model, text_input)

    print("Score for", text_input[:10], scores)

    logits = scores.logits.detach().numpy()
    odds = np.exp(logits)
    probs = odds / (1 + odds)
    # preds = np.argmax(probs, axis = 1)
    print(probs)

    st.text('The predictions are: {}'.format(probs))