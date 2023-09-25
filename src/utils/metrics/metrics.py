from sklearn.metrics import classification_report as classification_report_

def classification_report(*args, **kwargs):
    return classification_report_(*args, **kwargs)

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

