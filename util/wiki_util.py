# Imports from external libraries
import json
from tqdm import tqdm
from os import listdir
import pickle

# Imports from internal libraries
from classes.wiki_article import WikiArticle
from classes.constants import *
from util.general_util import *


def read_wiki_dump(path: str, decode_brackets=True):
    wiki = [json.loads(line) for line in open(path, "r")]
    return [WikiArticle.from_json(w, decode_brackets=decode_brackets) for w in wiki if len(w['id'])]


def pickle_wiki_dump(path=wiki_dump_folder_path, pickle_path=wiki_pickle_folder_path, dictionary_path=article_id_to_pickle_dictionary_path, decode_brackets=True):
    article_to_path = dict()
    for file_name in tqdm(listdir(path)):
        wiki_file = f"{path}/{file_name}"
        wiki = read_wiki_dump(wiki_file, decode_brackets=decode_brackets)

        file_name_raw = file_name.split(".")[0]
        pickle_file = f"{pickle_path}/{file_name_raw}.pkl"
        pickle.dump(wiki, open(pickle_file, "wb"))

        for w in wiki:
            article_to_path[w.id] = pickle_file

    pickle.dump(article_to_path, open(f"{dictionary_path}.pkl", "wb"))


def find_wiki_article(article_id: str, dictionary_path=article_id_to_pickle_dictionary_path):
    dictionary = pickle_load(dictionary_path)

    if article_id not in dictionary:
        return None

    wiki = pickle_load(dictionary[article_id])
    for w in wiki:
        if w.id == article_id:
            return w




