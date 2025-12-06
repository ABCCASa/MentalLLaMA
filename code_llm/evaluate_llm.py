import os
import sys
from bert_score import score
import evaluate

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    print(f"Adding '{parent_dir}' to PYTHONPATH")
    sys.path.append(parent_dir)
import pandas as pd
import our_metrics
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
    text = text.lower()
    texts = text.split("label:")
    if len(texts)>1:
        text = texts[-1]
        text = text.split("\n")[0]

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
    return min(found, key=lambda x: x[0])[1]

def get_label_index(label: str, all_labels: List):
    label = label.lower()
    return all_labels.index(label)

def evaluate_output(dataset_name, output_df):
    golden_label_index = []
    output_label_index = []
    golden_response =[]
    output_response=[]


    count = 0

    search_labels = IMHI_dataset.get_search_labels(dataset_name)
    standard_labels = IMHI_dataset.get_standard_labels(dataset_name)


    for index, row in output_df.iterrows():

        golden_label_index.append(get_label_index(row["label"], standard_labels))
        output_an = row["response"].lower()
        output_id = extract_label_index(output_an, search_labels)
        if output_id == -1:
            count += 1
            output_id = 0
        output_label_index.append(output_id)

        golden_response.append(f"Label: {row["label"]} \nExplanation: {row["reason"]}")
        output_response.append(row["response"])


    result_dict = {"dataset": dataset_name}
    result_dict.update(our_metrics.evaluate_all(golden_label_index, output_label_index))
    result_dict["OOD_count"] = count

    #bert_P, bert_R, bert_F1 = score(output_response, golden_response, lang="en")  # lang 改成对应语言


    #result_dict["bert_P"] = round(bert_P.mean().item()*100,3)
    #result_dict["bert_R"] = round(bert_R.mean().item()*100,3)
    #result_dict["bert_F1"] = round(bert_F1.mean().item()*100,3)

    bleu = evaluate.load("bleu")
    bleu_result = bleu.compute(predictions=output_response, references=golden_response)['bleu']
    result_dict["bleu"] = round(bleu_result * 100, 3)


    print(", ".join([f"{k}: {v}" for k, v in result_dict.items()]))
    return result_dict

def save_result(result_df, output_path):
    os.makedirs("../model_result/", exist_ok=True)
    result_df.to_csv(f"../model_result/{output_path}.csv",  index=False)

def main(output_path: str):
    outputs =  load_outputs(f"../model_output/{output_path}")
    result_dict = []
    for dataset_name, outputs_per_dataset in outputs.items():
        result_dict.append(evaluate_output(dataset_name, outputs_per_dataset))

    result_df = pd.DataFrame(result_dict)
    save_result(result_df, output_path)


if __name__ == "__main__":
    #main("Llama-3.1-8B_few_shot", "meta-llama/Llama-3.1-8B-Instruct")
    main("bert_llm2_zero_shot_cot_label")
