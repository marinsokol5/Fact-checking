# Imports from external libraries
from unicodedata import normalize
import pickle

# Imports from internal libraries

def normalize_nfd(text: str) -> str:
    return normalize("NFD", text)


def pickle_load(path: str):
    return pickle.load(open(path, "rb"))


def pickle_save(object, path: str):
    pickle.dump(object, open(path, "wb"))


