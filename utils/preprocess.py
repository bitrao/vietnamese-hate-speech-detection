
from vncorenlp import VnCoreNLP

vncorenlp = VnCoreNLP("models/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 

from pyvi import ViTokenizer

import re
import numpy as np

STOPWORDS = 'data/vietnamese-stopwords.txt'
with open(STOPWORDS, "r") as ins:
    stopwords = []
    for line in ins:
        dd = line.strip('\n')
        stopwords.append(dd)
    stopwords = set(stopwords)

def filter_stop_words(train_sentences, stop_words = stopwords):
    new_sent = [word for word in train_sentences.split() if word not in stop_words]
    train_sentences = ' '.join(new_sent)     
    return train_sentences

def deEmojify(text):
    # Define the emoji pattern using Unicode ranges
    emoji_pattern = re.compile(
        r"[\U0001F600-\U0001F64F]|"  # emoticons
        r"[\U0001F300-\U0001F5FF]|"  # symbols & pictographs
        r"[\U0001F680-\U0001F6FF]|"  # transport & map symbols
        r"[\U0001F1E0-\U0001F1FF]|"  # flags (iOS)
        r"[\U00002702-\U000027B0]|"  # Dingbats
        r"[\U000024C2-\U0001F251]|"  # Enclosed characters
        r"[\U0001F900-\U0001F9FF]|"  # Enclosed characters
        r"[\U0001FA70-\U0001FAFF]|"  # Enclosed characters
        r"[\U000024C2-\U0001F251]|"  # Enclosed characters
        r"[=:](\)+|\]+)*"  # Your custom pattern for smiley faces
        , flags=re.UNICODE
    )
    # Substitute emojis with an empty string
    return emoji_pattern.sub(r"", text)

def tokenizer(text, option=1):
    if option == 1:
        return vncorenlp.tokenize(text)
    else: 
        return ViTokenizer.tokenize(text)

def preprocess(text, tokenized=True, lowercased=True):

    text = filter_stop_words(text, stopwords)
    text = deEmojify(text)
    text = text.lower() if lowercased else text
    if tokenized:
        pre_text = ""
        sentences = tokenizer(text)
        for sentence in sentences:
            pre_text += " ".join(sentence)
        text = pre_text
    print(text)
    return text

def pre_process_features(X, y, tokenized=True, lowercased=True):
    print(X)
    X = [preprocess(str(p), tokenized=tokenized, lowercased=lowercased) for p in list(X)]
    print(X)
    for idx, ele in enumerate(X):
        if not ele:
            np.delete(X, idx)
            np.delete(y, idx)
    return X, y

