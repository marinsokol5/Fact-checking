# Imports from external libraries
import torch
from pytorch_pretrained_bert import BertConfig, BertForSequenceClassification
from torch.nn.functional import softmax
import numpy as np

# Imports from internal libraries


def save_model(model, name):
    model_path = f"./saved_models/{name}.pt"
    config_path = f"./saved_models/{name}.config"
    torch.save(model.state_dict(), model_path)
    with open(config_path, 'w+') as f:
        f.write(model.config.to_json_string())


def load_bert_model(name, num_labels=2, freeze=True):
    model_path = f"./saved_models/{name}.pt"
    config_path = f"./saved_models/{name}.config"
    config = BertConfig(config_path)
    model = BertForSequenceClassification(config, num_labels=num_labels)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    if freeze:
        freeze_only_first_n_layers(model, 2)
    return model


def interpret_bert_output(logits):
    return softmax(logits, dim=1)


def logits_to_percentages(logits):
    return interpret_bert_output(logits)


def freeze_only_first_n_layers(model, n):
    for i, layer in enumerate(model.children()):
        for param in layer.parameters():
            if i < n:
                param.requires_grad = False
            else:
                param.requires_grad = True


def get_trainable_parameters(model, named=False):
    if named:
        return ((n,p) for n,p in model.named_parameters() if p.requires_grad)
    else:
        return (p for p in model.parameters() if p.requires_grad)


def get_best_possible_threshold(y_true, y_predicted_probabilities, metric):
    thresholds = np.linspace(0, 1, num=100)
    best_threshold = None
    best_score = None

    for threshold in thresholds:
        y_predicted = [0 if probability < threshold else 1 for probability in y_predicted_probabilities]
        score = metric(y_true, y_predicted)
        if best_score is None or score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold, best_score


