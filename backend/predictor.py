import numpy as np
import os

class Preprocessor(object):

    def preprocess(self, url):
        """
        Preprocesses text input
        """
        text_input = get_content(url)
        text_input = text_input[0] + '<TOKEN>' + text_input[1]

        return text_input

class MyPredictor(object):
    """An example Predictor for an AI Platform custom prediction routine."""

    def __init__(self, model, preprocessor):
        """Stores artifacts for prediction. Only initialized via `from_path`.
        """
        self._model = model
        self._preprocessor = preprocessor

    def predict(self, instances, **kwargs):
        """Performs custom prediction.
        """
        inputs = np.asarray(instances)
        preprocessed_inputs = self._preprocessor.preprocess(inputs)
        scores = score_bias(self._model, preprocessed_inputs)
        logits = scores.logits.detach().numpy()
        odds = np.exp(logits)
        probs = odds / (1 + odds)

        return probs.tolist()

    @classmethod
    def from_path(cls, model_dir):
        """Creates an instance of MyPredictor using the given path.

        This loads artifacts that have been copied from your model directory in
        Cloud Storage. MyPredictor uses them during prediction.

        Args:
            model_dir: The local directory that contains the trained Keras
                model and the pickled preprocessor instance. These are copied
                from the Cloud Storage model directory you provide when you
                deploy a version resource.

        Returns:
            An instance of `MyPredictor`.
        """
        model_path = os.path.join(model_dir, 'pytorch_model.bin')
        model = load_model(model_path)

        preprocessor = Preprocessor()

        return cls(model, preprocessor)