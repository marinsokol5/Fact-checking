# Imports from external libraries

# Imports from internal libraries


def f_betta(precision, recall, betta):
    if precision == 0.0 and recall == 0.0:
        return 0.0
    betta_squared = betta ** 2
    return ((1 + betta_squared) * precision * recall) / (betta_squared * precision + recall)


def evaluate_document_retrieval(actual_documents: set, predicted_documents: set):
    precision = len(actual_documents.intersection(predicted_documents)) / len(predicted_documents)
    recall = len(actual_documents.intersection(predicted_documents)) / len(actual_documents)
    f1_score = f_betta(precision, recall, 1)
    f2_score = f_betta(precision, recall, 2)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'f2_score': f2_score
    }