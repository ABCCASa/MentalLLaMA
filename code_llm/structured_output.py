import os
import sys
from llm_tools import LLM
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    print(f"Adding '{parent_dir}' to PYTHONPATH")
    sys.path.append(parent_dir)
import pandas as pd
import IMHI_dataset
import argparse
def load_outputs(root):
    outputs = {}
    for file in os.listdir(root):
        if not file.endswith(".csv"):
            continue
        dataset_name = file.split('.')[0]
        data = pd.read_csv(f"{root}/{file}", dtype=str)
        outputs[dataset_name] = data
    return outputs

def evaluate_output(dataset_name, df, llm:LLM, batch_size, retry_count):
    llm_label = []
    llm_explain =[]
    contents = []
    for index, row in df.iterrows():
        contents.append(row["response"])
    structures= llm.batch_structured_output(contents, 10000, IMHI_dataset.get_noneable_model(dataset_name), batch_size, retry_count=retry_count)
    for structure in structures:
        if structure is not None:
            llm_label.append(structure["label"])
            llm_explain.append(structure["explanation"])
        else:
            llm_label.append(None)
            llm_explain.append(None)

    df["llm_label"] = llm_label
    df["llm_explain"] = llm_explain
    return df

def main(data_path: str, model_path:str, batch_size: int, retry_count:int):
    cache_dir = "../my_model_cache"
    llm = LLM(model_path, "cuda", cache_dir)
    os.makedirs(f"../model_struct_output/{data_path}/", exist_ok=True)
    for dataset_name, outputs_per_dataset in load_outputs(f"../model_output/{data_path}").items():
        outputs_per_dataset = evaluate_output(dataset_name, outputs_per_dataset, llm, batch_size, retry_count)
        outputs_per_dataset.to_csv(f"../model_struct_output/{data_path}/{dataset_name}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--retry_count', type=int, default=3)
    args = parser.parse_args()
    main(**vars(args))

    # cd code_llm
    # python structured_output.py --data_path Llama-3.1-8B_one_shot --batch_size 24 --retry_count 3
    # python structured_output.py --model_path Qwen/Qwen3-0.6B --data_path llm2_zero_shot --batch_size 2 --retry_count 3

