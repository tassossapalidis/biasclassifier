import argparse

def get_train_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num-visuals', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='save/') # folder to save model checkpoints
    parser.add_argument('--load-dir', type=str) # folder from which to load model weights
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--run-name', type=str, default='multitask_distilbert')
    parser.add_argument('--recompute-features', action='store_true')
    parser.add_argument('--train-file', type=str, default='train.csv') # file used to train the model
    parser.add_argument('--val-file', type=str, default='validation.csv') # file used to validate model
    parser.add_argument('--eval-file', type=str, default='test.csv') # file used to test model
    parser.add_argument('--do-train', action='store_true')
    parser.add_argument('--do-eval', action='store_true')
    parser.add_argument('--sub-file', type=str, default='')
    parser.add_argument('--visualize-predictions', action='store_true')
    parser.add_argument('--eval-every', type=int, default=5000)
    args = parser.parse_args()
    return args
