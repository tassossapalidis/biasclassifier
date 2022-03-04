# pip install tensorflow-addons
# git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow && pip install -e

import torch
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
import tensorflow as tf
import argparse
import scraper


if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Convert from PyTorch to TF')
        parser.add_argument('--model_dir', type = str, help="input pytorch model dir")
        parser.add_argument('--save_dir_tf', type = str, help="tensorflow output dir")


        args = parser.parse_args()
        tf_model = TFDistilBertForSequenceClassification.from_pretrained(args.model_dir, from_pt=True)
        tokenizer = DistilBertTokenizer.from_pretrained(tf_model)

        dummy_input = tokenizer("Hello, World. My Name is Tensorflow.",
                        stride=128,
                        truncation=True,
                        max_length=384,
                        padding=True,
                        return_tensors='pt')['input_ids']

        tf.saved_model.save(tf_model, args.save_dir_tf, signatures=None, options=None)

        tf_model_reload = tf.saved_model.load(args.save_dir_tf)
        print(tf_model_reload.predict(dummy_input))