#!/usr/bin/env python
# coding: utf-8

# In[1]:

# In[2]:


from dataset.google_dataset import *

from util.general_util import pickle_save

from bert_serving.client import BertClient


if __name__ == '__main__':
    which_dataset = int(input("Koji dataset -> 1, 2 ili 3: "))
    if which_dataset == 1:
        dataset = GoogleDatasetRaw.from_pickle(pickle_path=GoogleDataset.TRAIN_DATA)
        pickle_path = GoogleDatasetBertPickle.BERT_PICKLED_TRAIN
    elif which_dataset == 2:
        dataset = GoogleDatasetRaw.from_pickle(pickle_path=GoogleDataset.VALIDATION_DATA)
        pickle_path = GoogleDatasetBertPickle.BERT_PICKLED_VALIDATION
    else:
        dataset = GoogleDatasetRaw.from_pickle(pickle_path=GoogleDataset.TEST_DATA)
        pickle_path = GoogleDatasetBertPickle.BERT_PICKLED_TEST
    print(f"Pickling into {pickle_path}.")

    bc = BertClient()

    labels = []
    bert_client_to_encode = []

    for text, label in dataset:
        claim, google_result = text
        bert_client_to_encode.append(f"{claim} ||| {google_result}")
        labels.append(label)

    encoded = bc.encode(bert_client_to_encode)

    pickle_save(list(zip(encoded, labels)), pickle_path)

