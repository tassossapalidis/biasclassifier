import numpy as np
import re
import scraper

def clean_sentence(sentence, news_source = ''):
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

def split_text_to_sentences(text):
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

    return sentences


def chunk_sentences(sentences, min_length = 20):
    # combine consecutive sentences until we have reached min number of words
    sentences_chunked = []
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
        sentences_chunked += [s_out]

      idx += 1

    sentences_chunked = [s.strip() for s in sentences_chunked]

    return sentences_chunked

# URL to Model Inputs
def prepare_inference_input(url: str, min_length: int):
  raw_text = scraper.get_content(url)
  raw_title = raw_text[0]
  raw_content = raw_text[1]

  title = clean_sentence(title)
  raw_sentences = split_text_to_sentences(content)
  sentences = [clean_sentence(sentence) for sentence in chunk_sentences(raw_sentences, min_length)]

  input_passages = []

  # append the title (first segment) to each sentence and add it to the data
  for sentence in sentences:
    title_sentence = ' [SEP] '.join([title, sentence])
    input_passages += [title_sentence]

  return input_sentences

	
def aggregate_model_outputs(probs: list):
  probs = np.array(probs)
  avg_probs = np.mean(probs, axis=0)
  med_probs = np.median(probs, axis=0)
  max_probs = np.max(probs, axis=0)

  agg_result = {"avg_class": np.argmax(avg_probs),
                "avg_conf": np.max(avg_probs),
                "med_class": np.argmax(med_probs),
                "med_conf": np.max(med_probs),
                "max_class": np.argmax(max_probs),
                "max_conf": np.max(max_probs)}
  return agg_result

def identify_attributions(probs: list, class_label: int, thresh: float):
  class_probs = np.array(probs)[:, class_label]

  if (thresh > np.max(class_probs)):
    thresh = np.max(class_probs)

  return np.where(class_probs >= thresh)


