import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    print(f"Adding '{parent_dir}' to PYTHONPATH")
    sys.path.append(parent_dir)

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import argparse
import math
import time
import gc
from IMHI_dataset import get_dataset


def generate_batch_responses(model, tokenizer, datas, max_length):
    if getattr(tokenizer, "chat_template", None):
        if "system" in datas.keys():
            messages = [
                [{"role": "system", "content": system}, {"role": "user", "content": query}]
                for query, system in zip(datas["query"], datas["system"])]
        else:
            messages = [[{"role": "user", "content": query}] for query in datas["query"]]
        messages = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,)
    else:
        messages = datas["query"]

    model_inputs = tokenizer(messages, return_tensors="pt", padding=True).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**model_inputs, max_new_tokens=max_length)
    output_ids = generated_ids[:, len(model_inputs.input_ids[0]):]

    responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return messages, responses

def generate_responses_for_dataset(model, tokenizer, dataset, dataset_name, batch_size, max_length, print_freq):
    start_time = time.time()
    model_inputs = []
    responses = []
    total_batch = math.ceil(len(dataset) / batch_size)
    progress = 0
    for i in range(0, len(dataset), batch_size):
        batch_data = dataset[i: min(i + batch_size, len(dataset))]
        batch_model_inputs, batch_responses = generate_batch_responses(model, tokenizer, batch_data, max_length)
        responses += batch_responses
        model_inputs += batch_model_inputs
        progress += 1
        if progress % print_freq == 0 or progress == 1 or progress == total_batch:
            print(batch_model_inputs[i], batch_responses[0])
            print(f"[{dataset_name}] {progress}/{total_batch}, {int(time.time()-start_time)}s\n")
    gc.collect()
    torch.cuda.empty_cache()
    return model_inputs, responses


def inference_one_dataset(model, tokenizer, dataset_name, dataset_file, prompt_file, output_file, batch_size: int, max_length, print_freq):
    dataset = get_dataset(dataset_file, prompt_file)

    model_inputs, responses = generate_responses_for_dataset(model, tokenizer, dataset, dataset_name, batch_size,
                                                         max_length, print_freq)
    dataset = dataset.to_dict()
    dataset["response"] = responses
    dataset["model_input"] = model_inputs

    # check dir exist
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # save result
    output = pd.DataFrame(dataset, index=None)
    output.to_csv(output_file, index=False)


def main(model_path: str, data_dir: str, prompt_dir: str, output_dir: str, device: str, batch_size: int,
         max_length:int , print_freq: int):

    # login hugging face
    if input('Enter "y" if you want login: ') == "y":
        login()

    # load tokenizer and model
    device = torch.device(device)
    print("current device:", device)

    cache_dir = "../my_model_cache"
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        print("[warning] tokenizer does not have pad_token, use eos_token to instead.")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, cache_dir=cache_dir).to(device).eval()
    if model.generation_config is not None and model.generation_config.pad_token_id is None:
        print("[warning] model does not have pad_token, auto set")
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    os.makedirs(output_dir, exist_ok=True)
    # inference each dataset
    for file in os.listdir(data_dir):
        if not file.endswith(".csv"):
            continue
        dataset_name = file.split('.')[0]
        dataset_file = os.path.join(data_dir, file)
        prompt_file = os.path.join(prompt_dir, f"{dataset_name}.txt")
        output_file = os.path.join(output_dir, f"{dataset_name}.csv")
        if os.path.exists(output_file):
            print(f"Output file {output_file} already exists, skipping.")
            continue
        print(f"Start Dataset: {dataset_name}")
        inference_one_dataset(model, tokenizer, dataset_name, dataset_file, prompt_file, output_file, batch_size, max_length, print_freq)



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
    args = parser.parse_args()
    main(**vars(args))

    # main("Qwen/Qwen3-0.6B", "../dataset/test","../prompt_templates/zero_shot", "../model_output/Qwen3-0.6B_zero_shot", "cuda", 2, 5000, 1)


    # cd code_llm
    # python inference.py --model_path Qwen/Qwen3-0.6B --data_dir ../dataset/test --prompt_dir ../prompt_templates/zero_shot --output_dir ../model_output/Qwen3-0.6B_zero_shot  --device cuda --batch_size 24 --max_length 5000 --print_freq 5

