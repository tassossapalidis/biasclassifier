import argparse
import glob
import numpy as np
import os
import pandas as pd
import re

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

def prepare_train_data(folder):
    data = pd.DataFrame()

    # get csv files from folder
    dir = os.path.join(folder, '*.csv')
    data_files = glob.glob(dir)
    # iterate through csv files
    for data_file in data_files:
        curr_data = pd.read_csv(data_file)
        data = pd.concat([data, curr_data])

    # get cleaned dataframe
    data = prepare_dataframe(data)

    # get list of sentences and labels
    X = []
    y = []
    for _, row in data.iterrows():
        title = [clean_sentence(sentence, row['publication']) for sentence in split_sentences(row['title'])]
        sentences = [clean_sentence(sentence, row['publication']) for sentence in split_sentences(row['content'])]
        X += title
        X += sentences
        y += [row['bias']] * len(title)
        y += [row['bias']] * len(sentences)

    return X, y

###############################################################################
# helper functions
###############################################################################

def prepare_dataframe(data):
    data_transform = data[data['publication'].isin(liberal_news) | 
                     data['publication'].isin(conservative_news)]
    data_transform = data_transform.drop(
        ['Unnamed: 0', 'id','author','date','year','month','url'],
        axis = 1)
    data_transform['bias'] = np.where(data_transform['publication'].isin(liberal_news), 0, 1)
    data_transform = data_transform.reset_index()

    return data_transform

# from: https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences
def split_sentences(text):
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
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]

    return sentences


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
    # remove punctuation
    sentence = re.sub(r'[^\w\s]', '', sentence)
    # remove news_source
    if len(news_source) > 0:
        sentence = re.sub(r'\b' + re.escape(news_source.lower()) + r'\b','<source_token>', sentence)

    return sentence

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data.')
    parser.add_argument('data_folder', type = str, help = 'folder containing .csv training data files; e.g. ./data')
    parser.add_argument('train_data_filename', type = str, help = 'name of output data file')

    args = parser.parse_args()

    x, y = prepare_train_data(args.data_folder)
    data = pd.DataFrame(list(zip(x, y)),
               columns =['content', 'label'])
    
    data.to_csv(args.train_data_filename, index = False, header = False)
