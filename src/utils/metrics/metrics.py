def calculate_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0
    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score

def calculate_recall(TP: int, FN: int) -> float:
    if TP + FN == 0:
        return 0
    recall_score = TP / (TP + FN)
    return recall_score

def calculate_precision(TP: int, FP: int) -> float:
    if TP + FP == 0:
        return 0
    precision_score = TP / (TP + FP)
    return precision_score

def calculate_accuracy(TP: int, TN: int, FP: int, FN: int) -> float:
    accuracy_score = (TP + TN) / (TP + TN + FP + FN)
    return accuracy_score

def calculate_total_samples(TP: int, TN: int, FP: int, FN: int) -> int:
    total_samples = TP + TN + FP + FN
    return total_samples

def calculate_results(prediction: list, y_true: list) -> tuple[float, float, float, float]:
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(prediction)):
        if prediction[i] == 1 and y_true[i] == 1:
            TP += 1
        elif prediction[i] == 0 and y_true[i] == 0:
            TN += 1
        elif prediction[i] == 1 and y_true[i] == 0:
            FP += 1
        elif prediction[i] == 0 and y_true[i] == 1:
            FN += 1
    return TP, TN, FP, FN

