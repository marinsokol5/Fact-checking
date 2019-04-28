# Imports from external libraries
import json
from tqdm import tqdm
from os import listdir
import pickle

# Imports from internal libraries
from classes.wiki_article import WikiArticle
from classes.constants import *
from util.general_util import *
from classes.claim import Claim


def read_claims(path: str, decode_brackets=True):
    claims = [json.loads(line) for line in open(path, "r")]
    return [Claim.from_json(c, decode_brackets=decode_brackets) for c in claims if len(str(c['id']))]


def pickle_claims(path=claims_train_set_path, pickle_path=claims_pickle_train_set_path, decode_brackets=True):
    claims = read_claims(path, decode_brackets)
    pickle_save(claims, pickle_path)


def pickle_all_claims(decode_brackets=True):
    pickle_claims(claims_train_set_path, claims_pickle_train_set_path, decode_brackets)
    pickle_claims(claims_validation_set_path, claims_pickle_validation_set_path, decode_brackets)
    # pickle_claims(claims_test_set_path, claims_pickle_test_set_path, decode_brackets)


def load_claims_train():
    return pickle_load(claims_pickle_train_set_path)


def load_claims_validation():
    return pickle_load(claims_pickle_validation_set_path)


