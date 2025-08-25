from sklearn.metrics import f1_score, accuracy_score

def evaluate_all(golden_label, output_label,n_digits = 3):
    accuracy = round(accuracy_score(golden_label, output_label) * 100, n_digits)
    weighted_f1 = round(f1_score(golden_label, output_label, average='weighted') * 100, n_digits)
    micro_f1 = round(f1_score(golden_label, output_label, average='micro') * 100, n_digits)
    macro_f1 = round(f1_score(golden_label, output_label, average='macro') * 100, n_digits)
    result_dict = {"accuracy": accuracy, "weighted_F1": weighted_f1, "micro_F1": micro_f1, "macro_F1": macro_f1,}
    return result_dict