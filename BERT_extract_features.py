#!/usr/bin/env python
# coding: utf-8

# In[1]:

# In[2]:


from dataset.google_dataset import *
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from pytorch_pretrained_bert.modeling import BertConfig
import copy

from util.general_util import pickle_save
from util.modelling_util import save_bert_model, load_bert_model, logits_to_percentages, freeze_only_first_n_layers, get_trainable_parameters, get_best_possible_threshold
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler


if __name__ == '__main__':
    which_dataset = int(input("Koji dataset -> 1, 2 ili 3: "))
    if which_dataset == 1:
        dataset = GoogleDatasetBert.from_pickle(pickle_path=GoogleDataset.TRAIN_DATA)
        pickle_path = GoogleDatasetBertPickle.BERT_PICKLED_TRAIN
    elif which_dataset == 2:
        dataset = GoogleDatasetBert.from_pickle(pickle_path=GoogleDataset.VALIDATION_DATA)
        pickle_path = GoogleDatasetBertPickle.BERT_PICKLED_VALIDATION
    else:
        dataset = GoogleDatasetBert.from_pickle(pickle_path=GoogleDataset.TEST_DATA)
        pickle_path = GoogleDatasetBertPickle.BERT_PICKLED_TEST
    print(f"Pickling into {pickle_path}.")

    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.eval()
    bert_model.to(device)

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=sampler)

    bert_pickled = []
    for batch in tqdm(dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        all_encoded_layers, polled_output = bert_model(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            output_all_encoded_layers=True
        )
        print(all_encoded_layers.size())
        print(polled_output.size())
        print(label_ids.size())
        raise Exception()

        # bert_pickled.append()

    pickle_save(bert_pickled, pickle_path)


