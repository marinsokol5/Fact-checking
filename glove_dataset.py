from dataset.google_dataset import *
import torch
from util.general_util import pickle_save
from tqdm import tqdm

which_dataset = int(input("Koji dataset -> 1, 2 ili 3: "))
if which_dataset == 1:
    glove_dataset = GoogleDatasetGlove.from_pickle(pickle_path=GoogleDataset.TRAIN_DATA, glove_path=GoogleDatasetGlove.GLOVE_6B_200)
    pickle_path = GoogleDatasetGlovePickle.GLOVE_PICKLED_TRAIN
elif which_dataset == 2:
    glove_dataset = GoogleDatasetGlove.from_pickle(pickle_path=GoogleDataset.VALIDATION_DATA, glove_path=GoogleDatasetGlove.GLOVE_6B_200)
    pickle_path = GoogleDatasetGlovePickle.GLOVE_PICKLED_VALIDATION
else:
    glove_dataset = GoogleDatasetGlove.from_pickle(pickle_path=GoogleDataset.TEST_DATA, glove_path=GoogleDatasetGlove.GLOVE_6B_200)
    pickle_path = GoogleDatasetGlovePickle.GLOVE_PICKLED_TEST

print(f"Pickle path is: {pickle_path}")

glove_pickled = []
for glove_output in tqdm(glove_dataset):
    glove_pickled.append(glove_output)
pickle_save(glove_pickled, pickle_path)