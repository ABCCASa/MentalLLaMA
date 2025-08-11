import os
import argparse
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import json
import re

def load_test_data(root, prompt_path):
    test_data = {}
    for file in os.listdir(root):
        if not file.endswith(".csv"):
            continue
        dataset_name = file.split('.')[0]

        with open(f"{prompt_path}/{dataset_name}.txt", "r", encoding="utf-8") as f:
            template_prompt = f.read()

        queries = []
        labels = []
        for index, row in pd.read_csv(f"{root}/{file}", dtype=str).iterrows():
            def replace_placeholder(match):
                key = match.group(1)
                return str(row[key])
            query = re.sub(r"\[([^\]]+)\]", replace_placeholder, template_prompt)
            queries.append(query)
            labels.append(row["label"])
        test_data[dataset_name] = {
            "query": queries,
            "label": labels
        }
    return test_data


def generate_response(model, tokenizer, query):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_length=16384)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(query, response, "\n")
    return response


def save_output(output, dataset_name, output_path):
    if not os.path.exists("model_output/"):
        os.mkdir("model_output/")
    if not os.path.exists("model_output/"+output_path):
        os.mkdir("model_output/"+output_path)
    output = pd.DataFrame(output, index=None)
    output.to_csv(f"model_output/{output_path}/{dataset_name}.csv",  index=False)


def main(model_path: str, data_path: str,  prompt_path: str,output_path: str, device):
    test_data = load_test_data(data_path, prompt_path)
    cache_dir = "my_model_cache"
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device, cache_dir=cache_dir)
    for dataset_name, data in test_data.items():
        print(f"Start Dataset: {dataset_name}")
        responses = []
        for query in data["query"]:
            response = generate_response(model, tokenizer, query)
            responses.append(response)
        data["response"] = responses
        save_output(data, dataset_name, output_path)


if __name__ == "__main__":
    main("Qwen/Qwen3-1.7B", "test_data/small", "prompt_templates/zero_shot", "Qwen3-1.7B_zero_shot", torch.device("cuda"))
