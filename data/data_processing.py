import argparse
import glob
import numpy as np
import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split

## USAGE ##
# use process_input function to format an input for inference
# run this file as a script to produce training data as a csv
### if you run as script, output will be a csv, with col0 being the content
### and col1 being the bias label
###########

liberal_news = ['Buzzfeed News','CNN','Vox']
conservative_news = ['Breitbart','Fox News','National Review']

def process_input(text):
    return [clean_sentence(sentence, '') for sentence in split_sentences(text)]

def load_data(folder):
    data = pd.DataFrame()

    # get csv files from folder
    dir = os.path.join(folder, '*.csv')
    data_files = glob.glob(dir)

    assert len(data_files) > 0, "data files or folder not found at {}".format(folder)

    # iterate through csv files
    for data_file in data_files:
        curr_data = pd.read_csv(data_file)
        data = pd.concat([data, curr_data])

    # get cleaned dataframe
    data = prepare_dataframe(data)

    return data


def prepare_train_data(dataframe, min_length = 20):

    # get list of sentences and labels
    X = []
    y = []
    for _, row in dataframe.iterrows():
        title = clean_sentence(row['title'], row['publication'])
        sentences = [clean_sentence(sentence, row['publication']) for sentence in split_sentences(row['content'], min_length)]

        # append the title (first segment) to each sentence and add it to the data
        for sentence in sentences:
          title_sentence = ' [SEP] '.join([title, sentence])
          X += [title_sentence]

        y += [row['bias']] * len(sentences)

    return X, y

###############################################################################
# helper functions
###############################################################################

def prepare_dataframe(data):
    data_transform = data[data['publication'].isin(liberal_news) | 
                     data['publication'].isin(conservative_news)]
    data_transform = data_transform.drop(
        ['Unnamed: 0', 'Unnamed: 0.1', 'id','author','date','year','month','url'],
        axis = 1)
    data_transform['bias'] = np.where(data_transform['publication'].isin(liberal_news), 0, 1)
    data_transform = data_transform.reset_index()

    return data_transform

# from: https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences
def split_sentences(text, min_length = 20):
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"

    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(". ",".<stop>")
    text = text.replace("? ","?<stop>")
    text = text.replace("! ","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>") # split at the stop token.

    if len(sentences) > 1:
      sentences = sentences[:-1] # what's the point of this line?

    # combine consecutive sentences until we have reached min number of words
    sentences_combo = []
    idx = 0
    while idx < len(sentences):
      s_in = sentences[idx]
      combo_len = len(s_in.split(" "))
      combo_sents = [s_in]
      while combo_len < min_length and idx < len(sentences) - 1:
        combo_sents += [sentences[idx+1]] # needs to be list, otherwise string will be interpreted as list
        combo_len += len(sentences[idx+1].split(" "))
        idx += 1
      s_out = ' '.join(combo_sents) # join to the next sentence

      sents_length = sum([len(s.split(" ")) for s in sentences])
      if (combo_len >= min_length):
        sentences_combo += [s_out]
      idx += 1
    sentences_combo = [s.strip() for s in sentences_combo]

    return sentences_combo


def clean_sentence(sentence, news_source):
    # make lowercase
    sentence = sentence.lower()
    # remove location intro
    sentence = sentence.split('—')[-1]
    # remove excess spaces
    sentence = [word for word in sentence.split(' ') if len(word) > 0]
    # handle some intros
    source_idx = [sentence.index(word) for word in sentence if word == '(' + news_source.lower() + ')']
    if (len(source_idx) > 0 and source_idx[0] < 5):
        sentence = sentence[source_idx[0] + 1:]
    sentence = ' '.join(sentence)

    # remove punctuation (don't do this?)
    #sentence = re.sub(r'[^\w\s]', '', sentence)

    # remove news_source
    if len(news_source) > 0:
        sentence = re.sub(r'\b' + re.escape(news_source.lower()) + r'\b','<source_token>', sentence)

    return sentence

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data.')
    parser.add_argument('--data_folder', type = str, help = 'folder containing .csv training data files; e.g. ./data')
    parser.add_argument('--train_data_filename', type = str, help = 'name of train data file')
    parser.add_argument('--val_data_filename', type = str, help = 'name of validation data file')
    parser.add_argument('--test_data_filename', type = str, help = 'name of test data file')

    parser.add_argument('--min_length', type = int, help = 'minimum number of text words in a single example')

    args = parser.parse_args()

    full_df = load_data(args.data_folder)
    X_train, X_test, y_train, y_test = train_test_split(full_df[['content', 'publication', 'title']], 
        full_df['bias'],
        test_size=0.2, 
        random_state=42)

    # 80% of data to train
    train_df = pd.concat([X_train, y_train], axis=1)
    print("Number of Articles in Training", len(y_train))
    print("Percentage Conservative Articles in Training", float(sum(y_train)) / len(y_train))

    # 20% of data to evaluation
    eval_df = pd.concat([X_test, y_test], axis=1)
    X_val, X_test, y_val, y_test = train_test_split(eval_df[['content', 'publication', 'title']], 
        eval_df['bias'],
        test_size=0.8, 
        random_state=42)

    # 4% of data to training during validation
    val_df = pd.concat([X_val, y_val], axis=1)
    print("Number of Articles in Validation", len(y_val))
    print("Percentage Conservative Articles in Validation", float(sum(y_val)) / len(y_val))


    # 16% of data to test 
    test_df = pd.concat([X_test, y_test], axis=1)
    print("Number of Articles in Test", len(y_test))
    print("Percentage Conservative Articles in Test", float(sum(y_test)) / len(y_test))

    print("Preparing Training Data...")
    x_tr, y_tr = prepare_train_data(train_df, args.min_length)
    train_data = pd.DataFrame(list(zip(x_tr, y_tr)),
                    columns =['content', 'label'])

    print("Number of Passages in Training", len(y_tr))
    print("Percentage Conservative Passages in Training", float(sum(y_tr)) / len(y_tr))


    print("Preparing Validation Data...")
    x_va, y_va = prepare_train_data(val_df, args.min_length)
    val_data = pd.DataFrame(list(zip(x_va, y_va)),
                    columns =['content', 'label'])
    print("Number of Passages in Validation", len(y_va))
    print("Percentage Conservative Passages in Validation", float(sum(y_va)) / len(y_va))


    print("Preparing Test Data...")
    x_te, y_te = prepare_train_data(test_df, args.min_length)
    test_data = pd.DataFrame(list(zip(x_te, y_te)),
                    columns =['content', 'label'])
    print("Number of Passages in Test", len(y_te))
    print("Percentage Conservative Passages in Test", float(sum(y_te)) / len(y_te))


    train_data.to_csv(args.train_data_filename, index = False, header = True)
    val_data.to_csv(args.val_data_filename, index = False, header = True)
    test_data.to_csv(args.test_data_filename, index = False, header = True)
