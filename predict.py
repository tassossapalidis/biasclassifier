import torch

#from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification


def load_model():
	with torch.no_grad():
		model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
		model_dict = torch.load('./pytorch_model.bin', map_location = 'cpu')
		model.load_state_dict(model_dict)
		model = model.eval()
		return model

def score_bias(model, text_input):
	pass

