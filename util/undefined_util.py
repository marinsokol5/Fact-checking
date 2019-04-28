# Imports from external libraries
from tqdm import tqdm
from os import listdir
import json
import nltk
from nltk.corpus import stopwords as get_stopwords
from nltk.tokenize import word_tokenize
import string
import en_core_web_sm
nlp = en_core_web_sm.load()

# Imports from internal libraries



def tokenize(sentence, stopwords=None):
    if stopwords is None:
        stopwords = get_stopwords.words("english")

    if type(sentence) is str:
        sentence_no_punctuation = sentence.translate(str.maketrans('', '', string.punctuation))
        words = sentence_no_punctuation.split()
        lowercase_words = [w.lower() for w in words]
        lowercase_words_no_stopwords = [w for w in lowercase_words if w not in stopwords]
        return lowercase_words_no_stopwords
    elif type(sentence) in [list, set]:
        sequences = [w.translate(str.maketrans('', '', string.punctuation)).strip() for w in sentence]
        sequences_lowercase = [s.lower() for s in sequences]
        sequences_lowercase_no_stopwords = [" ".join([w for w in s.split() if w not in stopwords]) for s in
                                            sequences_lowercase]
        return sequences_lowercase_no_stopwords


def get_entities(line):
    return {e.text for e in nlp(line).ents}

















