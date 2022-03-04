from flask import escape
import functions_framework
import requests
import json
import numpy as np
import os
from utils import *
from predict import *
from scraper import *

@functions_framework.http
def polarity_func(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    request_json = request.get_json(silent=True)

    if not os.path.exists("/tmp/model.bin"):
        MODEL_URL = 'https://polarity-model.s3.us-west-1.amazonaws.com/pytorch_model_100.bin'
        r = requests.get(MODEL_URL)

        #Cloud function vm is a read only s/m. The only writable place is the tmp folder
        file = open("/tmp/model.bin", "wb")
        file.write(r.content)
        file.close()

    model = load_model('/tmp/model.bin')
    url = str(request_json['url'])
    min_length = 100

    raw_title, chunked_passages = preprocess(url, min_length)
    input_passages = prepare_inference_input(chunked_passages, raw_title)
    probs = predict(model, input_passages)

    # return summary dict
    result = aggregate_model_outputs(probs) 
  
    # identify passages that contributed to classification
    hotspot_idxs = identify_attributions(probs, result['med_class'], 0.90)
    hotspot_idxs = hotspot_idxs[:min(3, len(hotspot_idxs))]
    result['bias_sentences'] = [chunked_passages[int(i)] for i in hotspot_idxs]
    return json.dumps(result)

def preprocess(url, min_length):
    """
    Preprocesses text input
    """
    raw_text = scraper.get_content(url)
    raw_title = raw_text[0]
    raw_content = raw_text[1]
    chunked_passages = chunk_sentences(split_text_to_sentences(raw_content), min_length)

    return (raw_title, chunked_passages)

def predict(model, input_passages):
    """Performs custom prediction.
    """
    scores = score_bias(model, input_passages)
    logits = scores.logits.detach().numpy()
    odds = np.exp(logits)
    probs = odds / (1 + odds)
    return probs.tolist()