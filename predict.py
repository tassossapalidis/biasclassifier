import torch

from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification

def load_model():
	'''
	Loads DistilBert model with our parameters
	'''
	with torch.no_grad():
		model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
		model_dict = torch.load('./pytorch_model.bin', map_location = 'cpu')
		model.load_state_dict(model_dict)
		model = model.eval()
		return model

def score_bias(model, text_input):
	'''
	Evaluate input text using our model
	Output: probability for each class
	'''
	tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

	model_input = tokenizer(text_input,
 	          	stride=128,
 	          	truncation=True,
 	          	max_length=384,
 	          	padding=True,
				return_tensors='pt')

	score = model(**model_input)
	return score
