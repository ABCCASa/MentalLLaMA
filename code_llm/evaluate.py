import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    print(f"Adding '{parent_dir}' to PYTHONPATH")
    sys.path.append(parent_dir)

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from typing import List
from transformers import AutoTokenizer
from huggingface_hub import login

import IMHI_dataset

def load_outputs(root):
    outputs = {}
    for file in os.listdir(root):
        if not file.endswith(".csv"):
            continue
        dataset_name = file.split('.')[0]
        data = pd.read_csv(f"{root}/{file}", dtype=str)
        outputs[dataset_name] = data
    return outputs


def extract_label_index(text, valid_labels: List):
    found = []
    for label_index, item in enumerate(valid_labels):
        if isinstance(item, list):
            for label in item:
                pos_index = text.rfind(label)
                if pos_index > 0:
                    found.append((pos_index, label_index))
        else:
            pos_index = text.rfind(item)
            if pos_index > 0:
                found.append((pos_index, label_index))
    if len(found)<= 0:
        return -1
    return max(found, key=lambda x: x[0])[1]

def get_label_index(label: str, all_labels: List):
    return all_labels.index(label)


def evaluate_output(dataset_name, output_df, tokenizer, result_dict):
    golden_label_index = []
    output_label_index = []
    count = 0

    search_labels = IMHI_dataset.get_search_labels(dataset_name)
    standard_labels = IMHI_dataset.get_standard_labels(dataset_name)

    output_token_count = []
    for index, row in output_df.iterrows():
        output_token_count.append(len(tokenizer(row["response"])["input_ids"]))
        golden_label_index.append(get_label_index(row["label"], standard_labels))
        output_an = row["response"].lower()
        output_id = extract_label_index(output_an, search_labels)
        if output_id == -1:
            count += 1
            output_id = 0
        output_label_index.append(output_id)

    avg_accuracy = round(accuracy_score(golden_label_index, output_label_index) * 100, 4)
    weighted_f1 = round(f1_score(golden_label_index, output_label_index, average='weighted') * 100, 4)
    micro_f1 = round(f1_score(golden_label_index, output_label_index, average='micro') * 100, 4)
    macro_f1 = round(f1_score(golden_label_index, output_label_index, average='macro') * 100, 4)

    max_token = max(output_token_count)
    mean_token = sum(output_token_count) / len(output_token_count)

    result = f"Dataset: {dataset_name}, average acc:{avg_accuracy}, weighted F1 {weighted_f1}, micro F1 {micro_f1}, macro F1 {macro_f1}, OOD count: {count}, max token: {max_token}, mean token: {int(mean_token)}\n"

    result_dict["dataset"].append(dataset_name)
    result_dict["average acc"].append(avg_accuracy)
    result_dict["weighted F1"].append(weighted_f1)
    result_dict["micro F1"].append(micro_f1)
    result_dict["macro F1"].append(macro_f1)
    result_dict["OOD count"].append(count)
    result_dict["max token"].append(max_token)
    result_dict["mean token"].append(int(mean_token))
    print(result)

def save_result(result_df, output_path):
    os.makedirs("../model_result/", exist_ok=True)
    result_df.to_csv(f"../model_result/{output_path}.csv",  index=False)

def main(output_path: str, model_path:str = None):
    if input('Enter "y" if you want login: ') == "y":
        login()
    cache_dir = "../my_model_cache"
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    outputs =  load_outputs(f"../model_output/{output_path}")
    result_dict = {
        "dataset": [],
        "average acc": [],
        "weighted F1": [],
        "micro F1": [],
        "macro F1": [],
        "OOD count": [],
        "max token": [],
        "mean token": [],
    }
    for dataset_name, outputs_per_dataset in outputs.items():
        evaluate_output(dataset_name, outputs_per_dataset, tokenizer, result_dict)

    result_df = pd.DataFrame(result_dict)
    save_result(result_df, output_path)


if __name__ == "__main__":
    #main("Llama-3.1-8B_few_shot", "meta-llama/Llama-3.1-8B-Instruct")
    main("Llama-2-14B_zero_shot", "meta-llama/Llama-2-13b-chat-hf")
