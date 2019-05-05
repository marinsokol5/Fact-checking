# Imports from external libraries
from tqdm import tqdm
from nltk.corpus import stopwords as get_stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string
import en_core_web_sm
from nltk import word_tokenize
from nltk.tag.perceptron import PerceptronTagger

# Imports from internal libraries


def clean_sentences(sentences, stopwords=None, remove_stopwords=True, lowercase=True, remove_punctuation=True, punctuation=None, stem=False, stemmer=None, lemmatize=False, lemmer=None, tokenize=False, tagger=None, only_nouns=False, only_entities=False, entities=None):
    if stopwords is None and remove_stopwords:
        stopwords = get_stopwords.words("english")
    if lemmer is None and lemmatize:
        lemmer = WordNetLemmatizer()
    if stemmer is None and stem:
        stemmer = PorterStemmer()
    if punctuation is None and remove_punctuation:
        punctuation = string.punctuation
    if tagger is None and only_nouns:
        tagger = PerceptronTagger()
    if only_entities:
        scipy_nlp = en_core_web_sm.load()
    if entities is None:
        # https://spacy.io/api/annotation#section-named-entities
        entities = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART']

    result = []
    for sentence in tqdm(sentences):
        if only_entities:
            doc = scipy_nlp(sentence)
            sentence = " ".join([X.text for X in doc if X.ent_type_ in entities])

        if only_nouns:
            sentence = " ".join([w for w, t in tagger.tag(word_tokenize(sentence)) if t.startswith("NN")])

        if remove_punctuation:
            sentence = sentence.translate(str.maketrans('', '', punctuation))
        if lowercase:
            sentence = sentence.lower()

        # words = sentence.split()
        words = word_tokenize(sentence)

        if remove_stopwords:
            words = [w for w in words if w not in stopwords]
        if lemmatize:
            words = [lemmer.lemmatize(w) for w in words]
        if stem:
            words = [stemmer.stem(w) for w in words]

        if tokenize:
            result.append(words)
        else:
            result.append(" ".join(words))

    return result


def clean_sentence(sentence, stopwords=None, remove_stopwords=True, lowercase=True, remove_punctuation=True, punctuation=None, stem=False, stemmer=None, lemmatize=False, lemmer=None, tokenize=False, tagger=None, only_nouns=False, only_entities=False, entities=None):
    return clean_sentences([sentence], stopwords, remove_stopwords, lowercase, remove_punctuation, punctuation, stem, stemmer, lemmatize, lemmer, tokenize, tagger, only_nouns, only_entities, entities)[0]















