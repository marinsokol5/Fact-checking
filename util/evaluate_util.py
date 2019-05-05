# Imports from external libraries
from typing import List

# Imports from internal libraries
from util.general_util import combine_sets, combine_nested_sets


def f_betta(precision, recall, betta):
    if precision == 0.0 and recall == 0.0:
        return 0.0
    betta_squared = betta ** 2
    return ((1 + betta_squared) * precision * recall) / (betta_squared * precision + recall)


def evaluate_documents_retrieval(actual_documents_set: List[set], predicted_documents_set: List[set]):
    average_precision, average_recall, average_f1, average_f2, counter = 0, 0, 0, 0, 0
    for actual_documents, predicted_documents in zip(actual_documents_set, predicted_documents_set):
        evaluation = evaluate_document_retrieval(actual_documents, predicted_documents)
        average_precision += evaluation['precision']
        average_recall += evaluation['recall']
        average_f1 += evaluation['f1_score']
        average_f2 += evaluation['f2_score']
        counter += 1

    return {
        'average_precision': average_precision / counter,
        'average_recall': average_recall / counter,
        'average_f1_score': average_f1 / counter,
        'average_f2_score': average_f2 / counter,
    }


def evaluate_documents_retrieval_full(actual_documents_sets: List[List[set]], predicted_documents_set: List[set], verbose=True):
    average_precision, average_recall, average_f1, average_f2, counter, oracle_accuracy = 0, 0, 0, 0, 0, 0
    if verbose:
        document_results = []
    for actual_documents, predicted_documents in zip(actual_documents_sets, predicted_documents_set):
        evaluation = evaluate_document_retrieval_full(actual_documents, predicted_documents)
        average_precision += evaluation['precision']
        average_recall += evaluation['recall']
        average_f1 += evaluation['f1_score']
        average_f2 += evaluation['f2_score']
        counter += 1
        oracle_accuracy += evaluation['oracle_accuracy']

        if verbose:
            document_results.append(evaluation)

    result = {
        'average_precision': average_precision / counter,
        'average_recall': average_recall / counter,
        'average_f1_score': average_f1 / counter,
        'average_f2_score': average_f2 / counter,
        'oracle_accuracy': oracle_accuracy / counter,
    }

    if verbose:
        return result, document_results
    return result


def evaluate_document_retrieval(actual_documents: set, predicted_documents: set):
    true_positives = len(actual_documents.intersection(predicted_documents))
    false_positives = len(predicted_documents.difference(actual_documents))
    false_negatives = len(actual_documents.difference(predicted_documents))

    if len(predicted_documents) == 0 or len(actual_documents) == 0:
        if len(actual_documents) == 0 and len(predicted_documents) == 0:
            precision, recall, f1_score, f2_score = 1, 1, 1, 1
        else:
            precision, recall, f1_score, f2_score = 0, 0, 0, 0
    else:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = f_betta(precision, recall, 1)
        f2_score = f_betta(precision, recall, 2)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'f2_score': f2_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def evaluate_document_retrieval_full(actual_documents_sets: List[set], predicted_documents: set):
    found = 0
    for actual_documents in actual_documents_sets:
        if len(actual_documents) == 0:
            if len(predicted_documents) == 0:
                found = 1
            continue
        if len(actual_documents.difference(predicted_documents)) == 0:
            found = 1

    results = evaluate_document_retrieval(combine_sets(actual_documents_sets), predicted_documents)
    results['oracle_accuracy'] = found

    return results


