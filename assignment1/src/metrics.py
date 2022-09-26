from sklearn.utils import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import make_scorer


# Adapted from https://github.com/JonathanTay/CS-7641-assignment-1/blob/master/helpers.py
def balanced_accuracy(truth, pred):
    wts = compute_sample_weight('balanced', truth)
    return accuracy_score(truth, pred, sample_weight=wts)


def f1_accuracy(truth, pred):
    wts = compute_sample_weight('balanced', truth)
    return f1_score(truth, pred, average="binary", sample_weight=wts)

scorer = make_scorer(balanced_accuracy)
f1_scorer = make_scorer(f1_accuracy)

