from jiwer import wer
from Levenshtein import distance as levenshtein_distance
from sklearn.metrics import precision_score, recall_score, f1_score


def cer(pred_text, target_text):
    return levenshtein_distance(pred_text, target_text) / len(target_text)

def compute_precision_recall_f1(pred_chars, target_chars):
    max_len = max(len(pred_chars), len(target_chars))
    pred_chars += [''] * (max_len - len(pred_chars))
    target_chars += [''] * (max_len - len(target_chars))

    precision = precision_score(target_chars, pred_chars, average='weighted', zero_division=0)
    recall = recall_score(target_chars, pred_chars, average='weighted', zero_division=0)
    f1 = f1_score(target_chars, pred_chars, average='weighted', zero_division=0)

    return precision, recall, f1