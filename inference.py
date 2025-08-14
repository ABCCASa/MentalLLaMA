import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login, logout
import re
import argparse
import math

def load_test_data(root, prompt_type, ignore_dataset = None):
    test_data = {}
    for file in os.listdir(root):
        if not file.endswith(".csv"):
            continue
        dataset_name = file.split('.')[0]

        if ignore_dataset is not None and dataset_name in ignore_dataset:
            continue



        with open(f"prompt_templates/{prompt_type}/{dataset_name}.txt", "r", encoding="utf-8") as f:
            template_prompt = f.read()

        queries = []
        labels = []

        df = pd.read_csv(f"{root}/{file}", dtype=str)

        for row_index, row in df.iterrows():
            def replace_placeholder(match):
                key = match.group(1).strip()
                if ":" in key:
                    key, idx_str =  key.split(":", 1)
                    key, idx_str = key.strip(), idx_str.strip()
                    if idx_str.startswith(("+", "-")):
                        idx = (row_index + int(idx_str)) % len(df)
                    else:
                        idx = int(idx_str) % len(df)
                    return df[key][idx]
                else:
                    return str(row[key])
            query = re.sub(r"\[([^\]]+)\]", replace_placeholder, template_prompt)
            queries.append(query)
            labels.append(row["label"])
        test_data[dataset_name] = {
            "query": queries,
            "label": labels
        }
    return test_data





def batch_generate_response(model, tokenizer, queries):
    tokenizer.padding_side = "left"
    messages = [[{"role": "user", "content": query}] for query in queries]
    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_length=16384)

    output_ids = generated_ids[:,len(model_inputs.input_ids[0]):]
    responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    return responses


def batch_generate_response_for_dataset(model, tokenizer, queries, dataset_name, batch_size, print_freq):
    responses = []
    total_batch = math.ceil(len(queries) / batch_size)
    progress = 0
    for i in range(0, len(queries), batch_size):
        batch_data = queries[i: min(i + batch_size, len(queries))]
        batch_response = batch_generate_response(model, tokenizer, batch_data)
        responses += batch_response
        progress += 1
        if progress % print_freq == 0:
            print(batch_data[0], batch_response[0])
            print(f"[{dataset_name}] {progress}/{total_batch}\n")

    return responses


def save_output(output, dataset_name, output_path):
    if not os.path.exists("model_output/"):
        os.mkdir("model_output/")
    if not os.path.exists("model_output/"+output_path):
        os.mkdir("model_output/"+output_path)
    output = pd.DataFrame(output, index=None)
    output.to_csv(f"model_output/{output_path}/{dataset_name}.csv",  index=False)


def main(model_path: str, data_path: str,  prompt_type: str, output_path: str, batch_size: int, print_freq: int):

    # find the dataset that already have result
    exsited_dataset = []
    for file in os.listdir(f"model_output/{output_path}"):
        if not file.endswith(".csv"):
            continue
        dataset_name = file.split('.')[0]
        exsited_dataset.append(dataset_name)

    # load dataset
    test_data = load_test_data(data_path, prompt_type, exsited_dataset)

    # login hugging face
    try:

        login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
    except:
        print("Please set HF_TOKEN environment variable")

    # load model
    cache_dir = "my_model_cache"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=torch.bfloat16, cache_dir=cache_dir)

    #if tokenizer.pad_token_id is None:
    #    tokenizer.pad_token = tokenizer.eos_token

    # start inference
    for dataset_name, data in test_data.items():
        print(f"Start Dataset: {dataset_name}")
        responses = batch_generate_response_for_dataset(model, tokenizer, data["query"], dataset_name, batch_size, print_freq)
        data["response"] = responses
        save_output(data, dataset_name, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--prompt_type', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--print_freq', type=int, default=50)

    #main(**vars(parser.parse_args()))
    main("Qwen/Qwen3-0.6B", "test_data/small", "zero_shot", "Qwen3-8B_zero_shot_small", 2,1)
