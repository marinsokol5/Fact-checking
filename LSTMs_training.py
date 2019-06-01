

from dataset.google_dataset import *
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import copy
from util.modelling_util import save_model, logits_to_percentages, freeze_only_first_n_layers, get_trainable_parameters, get_best_possible_threshold
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score, confusion_matrix, classification_report
import json
import visdom
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F

from models.LSTMs import LSTMs





# In[4]:


def visdom_plot_line_initialize(visdom_vision, metric, visdom_plot_title, scores_on_train_data, scores_on_validation_data):
    plot = visdom_vision.line(
        Y=np.array([scores_on_train_data[metric]]),
        X=np.array([0]),
        opts={
            'legend': ['Train'],
            'title': f"{visdom_plot_title} {metric}",
            'xlabel': "Epochs",
            'ylabel': f"{metric}",
            'markers': True,
            'markersize': 5
        },
        env="apt"
    )
    visdom_vision.line(
        Y=np.array([scores_on_validation_data[metric]]),
        X=np.array([0]),
        win=plot,
        update="append",
        name="Validation", 
        opts={'markers':True, 'markersize':5},
        env="apt"
    )
    return plot


# In[5]:


def visdom_plot_line(visdom_vision, window, metric, epoch_index, scores_on_train_data, scores_on_validation_data):
    visdom_vision.line(
        Y=np.array([scores_on_train_data[metric]]),
        X=np.array([epoch_index + 1]),
        win=window,
        update="append",
        name="Train", 
        opts={'markers':True, 'markersize':5},
        env="apt"
    )
    visdom_vision.line(
        Y=np.array([scores_on_validation_data[metric]]),
        X=np.array([epoch_index + 1]),
        win=window,
        update="append",
        name="Validation", 
        opts={'markers':True, 'markersize':5},
        env="apt"
    )


# In[6]:


def train(original_model, train_data, validation_data, max_epochs=100, early_stop_epochs = 50, save_every_n = 10, model_name="model1",visdom_vision=None, visdom_plot_title=None):
    model = copy.deepcopy(original_model)
    model.to(device)
    
    scores_on_train_data = evaluate(model, train_data)
    scores_on_validation_data = evaluate(model, validation_data)
    
    loss_plot = visdom_plot_line_initialize(visdom_vision, 'loss', visdom_plot_title, scores_on_train_data, scores_on_validation_data)
    accuracy_plot = visdom_plot_line_initialize(visdom_vision, 'accuracy', visdom_plot_title, scores_on_train_data, scores_on_validation_data)
    auc_plot = visdom_plot_line_initialize(visdom_vision, 'auc', visdom_plot_title, scores_on_train_data, scores_on_validation_data)
    balanced_accuracy_plot = visdom_plot_line_initialize(visdom_vision, 'balanced_accuracy', visdom_plot_title, scores_on_train_data, scores_on_validation_data)
    sensitivity_plot = visdom_plot_line_initialize(visdom_vision, 'sensitivity', visdom_plot_title, scores_on_train_data, scores_on_validation_data)
    specificity_plot = visdom_plot_line_initialize(visdom_vision, 'specificity', visdom_plot_title, scores_on_train_data, scores_on_validation_data)

    best_model = copy.deepcopy(model)
    best_balanced_accuracy = scores_on_validation_data['balanced_accuracy']
    best_accuracy = scores_on_validation_data['accuracy']
    best_loss = scores_on_validation_data['loss']
    validation_score_hasnt_improved = 0
    save_iter = 0

    # for epoch_index in trange(max_epochs, desc="Epoch"):
    for epoch_index in range(max_epochs):
        model.train()

        dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                collate_fn=GoogleDatasetGlove.collate, drop_last=True)
        # for batch in tqdm(dataloader, desc="Iteration"):
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            claim_batch, google_result_batch, label_batch = batch
            
            optimizer.zero_grad()

            logits = model((claim_batch, google_result_batch))
            batch_loss = criterion(logits.squeeze(), label_batch)

            batch_loss.backward()
            optimizer.step()
        
        scores_on_train_data = evaluate(model, train_data)
        scores_on_validation_data = evaluate(model, validation_data)
        
        visdom_plot_line(visdom_vision, loss_plot, 'loss', epoch_index, scores_on_train_data, scores_on_validation_data)
        visdom_plot_line(visdom_vision, accuracy_plot, 'accuracy', epoch_index, scores_on_train_data, scores_on_validation_data)
        visdom_plot_line(visdom_vision, auc_plot, 'auc', epoch_index, scores_on_train_data, scores_on_validation_data)
        visdom_plot_line(visdom_vision, balanced_accuracy_plot, 'balanced_accuracy', epoch_index, scores_on_train_data, scores_on_validation_data)
        visdom_plot_line(visdom_vision, sensitivity_plot, 'sensitivity', epoch_index, scores_on_train_data, scores_on_validation_data)
        visdom_plot_line(visdom_vision, specificity_plot, 'specificity', epoch_index, scores_on_train_data, scores_on_validation_data)

        current_balanced_accuracy = scores_on_validation_data['balanced_accuracy']
        current_accuracy = scores_on_validation_data['accuracy']
        current_loss = scores_on_validation_data['loss']
        if any([
            current_balanced_accuracy > best_balanced_accuracy,
            np.isclose(current_balanced_accuracy, best_balanced_accuracy) and current_accuracy > best_accuracy,
            np.isclose(current_balanced_accuracy, best_balanced_accuracy) and np.isclose(current_accuracy, best_accuracy) and current_loss < best_loss
        ]):
            best_model = copy.deepcopy(model)
            best_balanced_accuracy = current_balanced_accuracy
            best_accuracy = current_accuracy
            best_loss = current_loss
            validation_score_hasnt_improved = 0
        else:
            validation_score_hasnt_improved += 1

        if validation_score_hasnt_improved >= early_stop_epochs:
            return best_model
        save_iter += 1
        if save_iter % save_every_n == 0:
            save_iter = 0
            save_model(model, model_name)

    return best_model


def evaluate(model, evaluation_data, result_file_name=None, find_best_threshold=False):
    model.to(device)
    model.eval()
    
    actual_classes = np.array([], dtype=np.int)
    predicted_classes = np.array([], dtype=np.int)
    predicted_probabilities = np.array([], dtype=np.float)
    loss_sum, loss_size = 0, 0

    dataloader = DataLoader(evaluation_data, batch_size=batch_size, shuffle=True, num_workers=4,
                            collate_fn=GoogleDatasetGlove.collate, drop_last=True)

    for batch in tqdm(dataloader, desc="Evaluating"):
    # for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        claim_batch, google_result_batch, label_batch = batch

        with torch.no_grad():
            logits = model((claim_batch, google_result_batch))
            logits = logits.squeeze()
            batch_loss = criterion(logits, label_batch)

            percentages = F.softmax(logits, dim=1).cpu().numpy()
            batch_predicted_classes = np.argmax(percentages, axis=1)
            batch_predicted_probabilities = percentages[:, 1]

            actual_classes = np.concatenate([actual_classes, label_batch.cpu().numpy()], axis=0)
            predicted_classes = np.concatenate([predicted_classes, batch_predicted_classes], axis=0)
            predicted_probabilities = np.concatenate([predicted_probabilities, batch_predicted_probabilities], axis=0)

            loss_sum += batch_loss.item()
            loss_size += 1

    report = classification_report(actual_classes, predicted_classes, output_dict=True)
    if find_best_threshold:
        threshold = get_best_possible_threshold(actual_classes, predicted_classes, metric=accuracy_score)[0]
        predicted_classes = np.array([0 if p < threshold else 1 for p in predicted_probabilities])

    result = {
        'loss': loss_sum/loss_size,
        'accuracy': accuracy_score(actual_classes, predicted_classes),
        'balanced_accuracy': balanced_accuracy_score(actual_classes, predicted_classes),
        'sensitivity': report['1']['recall'],
        'specificity': report['0']['recall'],
        'f1-score': f1_score(actual_classes, predicted_classes),
        'auc': roc_auc_score(actual_classes, predicted_probabilities),
        'confusion_matrix': confusion_matrix(actual_classes, predicted_classes).tolist()
    }
    
    if result_file_name is not None:
        file = f"./saved_models/{result_file_name}.txt"
        with open(file, "w+") as f:
            f.write(json.dumps(result))
    
    return result


if __name__ == '__main__':
    # glove_file = GoogleDatasetGlove.GLOVE_6B_200
    train_data = GoogleDatasetGlovePickle(GoogleDatasetGlovePickle.GLOVE_PICKLED_TRAIN)
    validation_data = GoogleDatasetGlovePickle(GoogleDatasetGlovePickle.GLOVE_PICKLED_VALIDATION)
    test_data = GoogleDatasetGlovePickle(GoogleDatasetGlovePickle.GLOVE_PICKLED_TEST)

    embedding_size = train_data[0][0].size()[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    visdom_vision = visdom.Visdom()

    batch_size = 16
    lstms_model = LSTMs(
        emb_dim=embedding_size,
        hidden_dim=250,
        num_layers=1,
        batch_size=batch_size,
        device=device
    )
    # num_train_optimization_steps = int(len(train_data) / batch_size) * max_epochs

    optimizer = Adam(lstms_model.parameters(), lr=0.0003)
    criterion = nn.CrossEntropyLoss()

    max_epochs = 100
    early_stop_epochs = 30
    save_every_n = 10

    model_name = "lstms_1"
    lstms_trained = train(lstms_model, train_data, validation_data, max_epochs, early_stop_epochs, save_every_n, model_name, visdom_vision, visdom_plot_title=f"{model_name}")
    evaluate(lstms_trained, test_data, f"{model_name}_results")
    save_model(lstms_model, f"{model_name}")


