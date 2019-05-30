# Imports from external libraries
from torch.utils.data import Dataset
import sqlite3
import pickle
from pytorch_pretrained_bert import BertTokenizer
import torch
from typing import List
import pandas as pd
import csv
import numpy as np
from nltk import word_tokenize
import torch.nn as nn

# Imports from internal libraries

from classes.claim import Claim
from util.general_util import inverse_dictionary


class GoogleDataset(Dataset):
    TRAIN_DATA = "./data/google_train.pkl"
    VALIDATION_DATA = "./data/google_validation.pkl"
    TEST_DATA = "./data/google_test.pkl"
    ALL_DATA = "./data/google_all.pkl"

    LABEL_MAP = {
        Claim.SUPPORTS: 0,
        Claim.REFUTES: 1,
        "Unlabelled": -1
    }
    LABEL_MAP_INVERSE = inverse_dictionary(LABEL_MAP)

    def __init__(self, google_result):
        self.google_result = google_result
        self.data = [(claim, result) for claim, result, label in google_result]
        self.labels = [label for claim, result, label in google_result]

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class GoogleDatasetRaw(GoogleDataset):
    def __init__(self, google_result):
        super().__init__(google_result)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def from_pickle(pickle_path: str):
        google_result = pickle.load(open(pickle_path, "rb"))
        return GoogleDatasetRaw(google_result)


class GoogleDatasetBert(GoogleDataset):
    def __init__(self, google_result, max_sequence_length: int = 512):
        super().__init__(google_result)
        self.max_sequence_length = max_sequence_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_len=self.max_sequence_length)

    def __getitem__(self, index):
        tokenized_claim = self.tokenizer.tokenize(self.data[index][0])
        tokenized_google_text = self.tokenizer.tokenize(self.data[index][1])
        tokenized_google_text_truncated = tokenized_google_text[:(self.max_sequence_length - 3 - len(tokenized_claim))]
        tokens_bert = ["[CLS]"] +\
                      tokenized_claim +\
                      ["[SEP]"] +\
                      tokenized_google_text_truncated +\
                      ["[SEP]"]

        segment_ids = [0] * (len(tokenized_claim) + 2)
        segment_ids += [1] * (len(tokenized_google_text_truncated) + 1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens_bert)
        input_mask = [1] * len(tokens_bert)

        padding = [0] * (self.max_sequence_length - len(tokens_bert))
        segment_ids += padding
        input_ids += padding  # padding has an id of 0
        input_mask += padding

        assert len(segment_ids) == self.max_sequence_length
        assert len(input_ids) == self.max_sequence_length
        assert len(input_mask) == self.max_sequence_length

        label_id = GoogleDataset.LABEL_MAP[self.labels[index]]

        return torch.tensor(input_ids), \
               torch.tensor(input_mask), \
               torch.tensor(segment_ids), \
               torch.tensor(label_id)

    def __len__(self):
        return len(self.data)

    def get_as_input(self, include_label=False):
        input_ids = torch.LongTensor([t[0].numpy() for t in self])
        input_masks = torch.LongTensor([t[1].numpy() for t in self])
        segment_ids = torch.LongTensor([t[2].numpy() for t in self])
        if not include_label:
            return input_ids, segment_ids, input_masks

        label_ids = torch.LongTensor([t[3] for t in self])
        return input_ids, segment_ids, input_masks, label_ids

    def get_one_as_input(self, index, include_label=False):
        ticket = self[index]
        input_ids = torch.LongTensor([ticket[0].numpy()])
        input_masks = torch.LongTensor([ticket[1].numpy()])
        segment_ids = torch.LongTensor([ticket[2].numpy()])
        if not include_label:
            return input_ids, segment_ids, input_masks

        label_ids = torch.LongTensor([ticket[3]])
        return input_ids, segment_ids, input_masks, label_ids

    def get_one_as_text(self, index, include_padding=False):
        word_ids = self[index][0].numpy()
        words = self.tokenizer.convert_ids_to_tokens(word_ids)
        # if not include_padding:
        #     words = words[:words.index("[SEP]")]
        return " ".join(words)

    def get_vocabulary(self):
        return [word for word, index in self.tokenizer.vocab.items() if not word.startswith("[unused")]

    @staticmethod
    def from_pickle(pickle_path: str, max_sequence_length: int = 512):
        google_result = pickle.load(open(pickle_path, "rb"))
        return GoogleDatasetBert(google_result, max_sequence_length)


class GoogleDatasetGlove(GoogleDataset):
    GLOVE_6B_200 = f"data/embeddings/glove.6B.200d.txt"

    def __init__(self, google_result, glove_path):
        super().__init__(google_result)
        self.glove_matrix = pd.read_table(glove_path, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

    def glove(self, word: str):
        if not self.glove_contains_word(word):
            return None
        return self.glove_matrix.loc[word].values

    def glove_contains_word(self, word: str):
        return word in self.glove_matrix.index

    def glove_tokens(self, tokens: List[str]):
        return np.array([self.glove(t) for t in tokens if self.glove_contains_word(t)])

    def embedding_size(self):
        return self.glove_matrix.shape[1]

    def __getitem__(self, index):
        claim: str = self.data[index][0]
        google_result: str = self.data[index][1]

        claim_tokenized = word_tokenize(claim.lower())
        google_result_tokenized = word_tokenize(google_result.lower())

        claim_embedded = self.glove_tokens(claim_tokenized)
        google_result_embedded = self.glove_tokens(google_result_tokenized)

        label_id = GoogleDataset.LABEL_MAP[self.labels[index]]

        return torch.from_numpy(claim_embedded), \
               torch.from_numpy(google_result_embedded), \
               label_id

    def __len__(self):
        return len(self.data)

    @staticmethod
    def from_pickle(pickle_path: str, glove_path):
        google_result = pickle.load(open(pickle_path, "rb"))
        return GoogleDatasetGlove(google_result, glove_path)

    @staticmethod
    def collate(batch: List[tuple]):
        # each element in a batch is (claim, google_result, label) all embedded
        claims_batch = [b[0] for b in batch]
        google_result_batch = [b[1] for b in batch]
        labels_batch = torch.LongTensor([b[2] for b in batch])

        claims_padded = nn.utils.rnn.pad_sequence(claims_batch, batch_first=True)
        google_results_padded = nn.utils.rnn.pad_sequence(google_result_batch, batch_first=True)

        return claims_padded, google_results_padded, labels_batch


if __name__ == '__main__':
    import os
    os.chdir("..")
    dat = GoogleDatasetGlove.from_pickle(GoogleDataset.TRAIN_DATA, GoogleDatasetGlove.GLOVE_6B_200)
    print(dat[0])

