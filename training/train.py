# Import all the necessary dependencies
import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
from torch import optim
from torch import nn

import csv
from collections import OrderedDict
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
from datasets import Dataset
from tqdm.notebook import tqdm
import util
from args import get_train_test_args

# dataset class for news articles
class NewsDataset(Dataset):

    def __init__(self, data, labels):
        self.X = data
        self.y = np.zeros((len(labels), 2))
        self.y[np.arange(len(labels)),labels] = 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input = np.asarray(self.X[idx])
        output = np.asarray(self.y[idx,:])
        return (input, output)

class Trainer():
    def __init__(self, args, log):
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.eval_every = args.eval_every
        self.path = os.path.join(args.save_dir, 'checkpoint')
        self.num_visuals = args.num_visuals
        self.save_dir = args.save_dir
        self.log = log
        self.visualize_predictions = args.visualize_predictions
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save(self, model):
        model.save_pretrained(self.path)

    def evaluate(self, model, data_loader, split='validation'):
        device = self.device

        # Does model need to change in any way for evaluation?

        all_logits = []
        all_labels = []

        with torch.no_grad(), tqdm(total=len(data_loader.dataset)) as progress_bar:
            for batch in data_loader:
                # Setup for forward
                inputs, labels = batch

                batch_size = len(inputs)

                logits = model(input_ids = inputs.to(device)).logits

                all_logits.append(logits)
                all_labels.append(labels)
                progress_bar.update(batch_size)

        logits = torch.cat(all_logits).cpu().numpy()
        labels = torch.cat(all_labels).cpu().numpy()

        # Need to implement scoring
        odds = np.exp(logits)
        probs = odds / (1 + odds)
        preds = np.argmax(probs, axis = 1)
        truth = np.argmax(labels, axis = 1)
        acc = np.sum(1 - abs(preds - truth)) / preds.size
        conf = np.sum(abs(probs[:, 0] - probs[:, 1])) / preds.size

        results_list = [('accuracy', acc),
                        ('confidence', conf)]

        results = OrderedDict(results_list)

        return results


    def train(self, model, train_dataloader, val_dataloader):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        device = self.device
        model.to(device)

        optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        best_scores = {'accuracy': -1.0, 'confidence': -1.0}

        for epoch in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch}')
            i = 0
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                for batch in train_dataloader:
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = batch

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize (model's loss is binary cross entropy)
                    outputs = model(input_ids = inputs.to(device))
                    loss = criterion(outputs[0], labels.to(device))

                    loss.backward()
                    optimizer.step()

                    # print statistics
                    #running_loss += loss.item()

                    if i % self.eval_every == self.eval_every - 1:    # print every 20 mini-batches
                        self.log.info(f'Evaluating at epoch {epoch} step {i}...')
                        scores = self.evaluate(model, val_dataloader)

                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in scores.items())
                        self.log.info('Visualizing in TensorBoard...')
                        for k, v in core.items():
                            tbx.add_scalar(f'val/{k}', v, i)
                        self.log.info(f'Eval {results_str}')

                        if scores['accuracy'] >= best_scores['accuracy']:
                            best_scores = scores
                            self.save(model)
                    i += 1

        return best_scores


def main():

    # Get arguments
    args = get_train_test_args()

    # Set a seed

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    if args.load_dir:
        model = DistilBertForSequenceClassification.from_pretrained(args.load_dir)

    if args.do_train:
        if not os.path.exists(args.save_dir):
            args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
            log = util.get_logger(args.save_dir, 'log_train')
            log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')

            log.info("Preparing Training Data...")

            args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

            trainer = Trainer(args, log)

            df_train = pd.read_csv("data/train.csv")

            # Prep training dataset
            train_input = tokenizer(list(df_train.content.array),
                stride=128,
                truncation=True,
                max_length=384,
                padding=True)

            train_labels = list(df_train.label.array)
            train_dataset = NewsDataset(train_input['input_ids'], train_labels)

            # Prep validation dataset
            log.info("Preparing Validation Data...")
            df_val = pd.read_csv("data/validation.csv")

            val_input = tokenizer(list(df_val.content.array),
                stride=128,
                truncation=True,
                max_length=384,
                padding=True)

            val_labels = list(df_val.label.array)
            val_dataset = NewsDataset(val_input['input_ids'], val_labels)

            # Data Loaders
            train_loader = torch.utils.data.DataLoader(train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=2)

            val_loader = torch.utils.data.DataLoader(val_dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=2)

            best_scores = trainer.train(model, train_loader, val_loader)

    if args.do_eval:
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        split_name = 'test'
        log = util.get_logger(args.save_dir, f'log_{split_name}')

        trainer = Trainer(args, log)
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint')

        log.info("Preparing Test Data...")

        model.to(args.device)
        df_test = pd.read_csv("data/test.csv")

        test_input = tokenizer(list(df_test.content.array),
                stride=128,
                truncation=True,
                max_length=384,
                padding=True)
        test_labels = list(df_test.label.array)
        test_dataset = NewsDataset(test_input['input_ids'], test_labels)

        test_loader = torch.utils.data.DataLoader(test_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=2)

        scores = self.evaluate(model, test_loader)

        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in scores.items())
        log.info(f'Eval {results_str}')


if __name__ == '__main__':
    main()

