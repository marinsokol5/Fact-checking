# Imports from external libraries
from unicodedata import normalize
import pickle
from typing import List

# Imports from internal libraries

def normalize_nfd(text: str) -> str:
    return normalize("NFD", text)


def pickle_load(path: str):
    return pickle.load(open(path, "rb"))


def pickle_save(object, path: str):
    pickle.dump(object, open(path, "wb"))


def combine_sets(sets: List[set]) -> set:
    combined = set()
    for one_set in sets:
        combined.update(one_set)
    return combined


def combine_nested_sets(sets: List[List[set]]) -> List[set]:
    combined: List[set] = []
    for list_of_sets in sets:
        combined.append(combine_sets(list_of_sets))
    return combined
