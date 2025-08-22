import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import argparse
import math
import time
import gc
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    print(f"Adding '{parent_dir}' to PYTHONPATH")
    sys.path.append(parent_dir)
from IMHI_dataset import get_dict_dataset

def load_all_datasets(dataset_dir, prompt_dir, ignore_dataset = None):
    test_datasets = {}
    for file in os.listdir(dataset_dir):
        if not file.endswith(".csv"):
            continue
        dataset_name = file.split('.')[0]
        if ignore_dataset is not None and dataset_name in ignore_dataset:
            continue
        full_dataset_file = f"{dataset_dir}/{file}"
        full_prompt_file =f"{prompt_dir}/{dataset_name}.txt"
        current_dataset = get_dict_dataset(full_dataset_file, full_prompt_file)

        #current_dataset = {k: v[:2] for k, v in current_dataset.items()}

        test_datasets[dataset_name] = current_dataset


    return test_datasets


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


def save_output(output, dataset_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output = pd.DataFrame(output, index=None)
    output.to_csv(f"{output_dir}/{dataset_name}.csv",  index=False)


def main(model_path: str, data_dir: str,  prompt_dir: str, output_dir: str, device: str, batch_size: int, max_length:int , print_freq: int):

    # find the dataset that already have result, and ignore them
    exsited_datasets = []
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if not file.endswith(".csv"):
                continue
            dataset_name = file.split('.')[0]
            exsited_datasets.append(dataset_name)

    # load dataset
    test_datasets = load_all_datasets(data_dir, prompt_dir, exsited_datasets)

    # login hugging face
    if input('Enter "y" if you want login: ') == "y":
        login()


    device = torch.device(device)
    print("current device:", device)

    cache_dir = "../my_model_cache"
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
    for dataset_name, data in test_datasets.items():
        print(f"Start Dataset: {dataset_name}")
        responses = generate_responses_for_dataset(model, tokenizer, data["query"], dataset_name, batch_size, max_length, print_freq)
        data["response"] = responses
        save_output(data, dataset_name, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--prompt_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--print_freq', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=5000)

    main(**vars(parser.parse_args()))
    # cd code_llm
    # python inference.py --model_path Qwen/Qwen3-0.6B --data_dir ../dataset/test --prompt_dir ../prompt_templates/zero_shot --output_dir ../model_output/Qwen3-0.6B_zero_shot  --device cuda --batch_size 24 --max_length 5000 --print_freq 5

