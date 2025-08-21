import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import re
import argparse
import math
import time
import gc


def load_all_dataset(dataset_path, prompt_type, ignore_dataset = None):
    test_data = {}
    for file in os.listdir(dataset_path):
        if not file.endswith(".csv"):
            continue
        dataset_name = file.split('.')[0]
        if ignore_dataset is not None and dataset_name in ignore_dataset:
            continue

        with open(f"prompt_templates/{prompt_type}/{dataset_name}.txt", "r", encoding="utf-8") as f:
            template_prompt = f.read()

        queries = []
        labels = []
        df = pd.read_csv(f"{dataset_path}/{file}", dtype=str)
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
        test_data[dataset_name] = {"query": queries, "label": labels}
    return test_data


def generate_batch_responses(model, tokenizer, queries, max_length):
    if getattr(tokenizer, "chat_template", None):
        messages = [[{"role": "user", "content": query}] for query in queries]
        model_inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(model.device)
    else:
        model_inputs = tokenizer(queries, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(**model_inputs, max_new_tokens=max_length)
    output_ids = generated_ids[:, len(model_inputs.input_ids[0]):]

    responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return responses

def generate_responses_for_dataset(model, tokenizer, queries, dataset_name, batch_size, max_length, print_freq):
    start_time = time.time()
    responses = []
    total_batch = math.ceil(len(queries) / batch_size)
    progress = 0
    for i in range(0, len(queries), batch_size):
        batch_data = queries[i: min(i + batch_size, len(queries))]
        batch_response = generate_batch_responses(model, tokenizer, batch_data, max_length)
        responses += batch_response
        progress += 1
        if progress % print_freq == 0 or progress == 1 or progress == total_batch:
            print(batch_data[0], batch_response[0])
            print(f"[{dataset_name}] {progress}/{total_batch}, {int(time.time()-start_time)}s\n")
    gc.collect()
    torch.cuda.empty_cache()
    return responses


def save_output(output, dataset_name, output_path):
    if not os.path.exists("model_output/"):
        os.mkdir("model_output/")
    if not os.path.exists(f"model_output/{output_path}"):
        os.mkdir(f"model_output/{output_path}")
    output = pd.DataFrame(output, index=None)
    output.to_csv(f"model_output/{output_path}/{dataset_name}.csv",  index=False)


def main(model_path: str, data_path: str,  prompt_type: str, output_path: str, device: str, batch_size: int, max_length:int , print_freq: int):

    # find the dataset that already have result
    exsited_dataset = []
    if os.path.exists(f"model_output/{output_path}"):
        for file in os.listdir(f"model_output/{output_path}"):
            if not file.endswith(".csv"):
                continue
            dataset_name = file.split('.')[0]
            exsited_dataset.append(dataset_name)

    # load dataset
    test_data = load_all_dataset(data_path, prompt_type, exsited_dataset)

    # login hugging face
    if input('Enter "y" if you want login: ') == "y":
        login()

    cache_dir = "my_model_cache"
    device = torch.device(device)
    print("current device:", device)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        print("[warning] tokenizer does not have pad_token, use eos_token to instead.")
        tokenizer.pad_token = tokenizer.eos_token

    # load model
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, cache_dir=cache_dir).to(device).eval()
    if model.generation_config is not None and model.generation_config.pad_token_id is None:
        print("[warning] model does not have pad_token, auto set")
        model.generation_config.pad_token_id = tokenizer.pad_token_id


    # start inference
    for dataset_name, data in test_data.items():
        print(f"Start Dataset: {dataset_name}")
        responses = generate_responses_for_dataset(model, tokenizer, data["query"], dataset_name, batch_size,
                                                   max_length, print_freq)
        data["response"] = responses
        save_output(data, dataset_name, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--prompt_type', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--max_length', type=int, default=5000)

    main(**vars(parser.parse_args()))
    #main("Qwen/Qwen3-0.6B", "test_data/our", "zero_shot", "Qwen3-1.7B_zero_shot_small", "cuda" ,2, 1000,1)
