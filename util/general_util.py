# Imports from external libraries
from unicodedata import normalize
import pickle
from typing import List
from collections import Counter

# Imports from internal libraries

def normalize_nfd(text: str) -> str:
    return normalize("NFD", text)


def pickle_load(path: str):
    return pickle.load(open(path, "rb"))


def pickle_save(object, path: str):
    pickle.dump(object, open(path, "wb"))


def dict_ratio(dictionary):
    sum_of_values = sum(dictionary.values())
    return {k: v/sum_of_values for k, v in dictionary.items()}


def print_data(data, data_name="Data"):
    print(f"{data_name} size: {len(data)}")
    counter = Counter([d[2] for d in data])
    print(f"{data_name} ratio: {dict_ratio(counter)}")


def print_ratios(train, validation, test):
    print_data(train, "Train")
    print()
    print_data(validation, "Validation")
    print()
    print_data(test, "Test")


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


def inverse_dictionary(dictionary: dict):
    return {v: k for k, v in dictionary.items()}
