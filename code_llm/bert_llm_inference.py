import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    print(f"Adding '{parent_dir}' to PYTHONPATH")
    sys.path.append(parent_dir)

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForSequenceClassification
from huggingface_hub import login
import argparse
import math
import time
import gc
from IMHI_dataset import get_dataset
import IMHI_dataset

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
            print(batch_model_inputs[0], batch_responses[0])
            print(f"[{dataset_name}] {progress}/{total_batch}, {int(time.time()-start_time)}s\n")
    gc.collect()
    torch.cuda.empty_cache()
    return model_inputs, responses

def inference_one_dataset(call_bert, model, tokenizer, dataset_name, dataset_file, llm_prompt_file, output_file, batch_size: int, max_length, print_freq):
    dataset = get_dataset(dataset_file, llm_prompt_file, {"bert": call_bert})
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


def main(model_path: str, bert_dir:str, data_dir: str, bert_prompt_dir:str, llm_prompt_dir: str,bert_out:str,
         output_dir: str, device: str, batch_size: int, max_length:int , print_freq: int):


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
        llm_prompt_file = os.path.join(llm_prompt_dir, f"{dataset_name}.txt")
        bert_prompt_file = os.path.join(bert_prompt_dir, f"{dataset_name}.txt")
        output_file = os.path.join(output_dir, f"{dataset_name}.csv")
        if os.path.exists(output_file):
            print(f"Output file {output_file} already exists, skipping.")
            continue
        print(f"Start Dataset: {dataset_name}")

        with open(bert_prompt_file, "r") as f:
            bert_prompt = f.read()
        label_set = IMHI_dataset.get_standard_labels(dataset_name)
        bert_tokenizer = AutoTokenizer.from_pretrained(f"{bert_dir}/{dataset_name}")
        bert_model = AutoModelForSequenceClassification.from_pretrained(f"{bert_dir}/{dataset_name}", num_labels=len(label_set)).to(device).eval()
        def call_bert(**kwargs):
            query = IMHI_dataset.apply_single_prompt(kwargs, bert_prompt)["query"]
            inputs = bert_tokenizer([query], truncation=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = bert_model(**inputs)
                logits = outputs.logits
            if bert_out == "probs":
                probs = torch.softmax(logits, dim=-1).tolist()[0]
                text_probs = ", ".join([f"{name}: {round(prob, 2)}" for name, prob in zip(label_set, probs) ])
                return text_probs
            elif bert_out =="label":
                label_index = outputs.logits.argmax(dim=-1).tolist()[0]
                return label_set[label_index]
            elif bert_out == "conf":
                probs = torch.softmax(logits[0], dim=-1)
                max_vals, max_indices = torch.max(probs, dim=-1)
                prob = max_vals.item()
                label = label_set[max_indices.item()]
                mean = 1/len(label_set)
                prob = (prob - mean)/(1-mean)
                if prob > 0.75:
                    conf = "high"
                elif prob > 0.25:
                    conf = "medium"
                else:
                    conf = "low"
                return f"{label} ({conf} confidence)"

            else:
                return ""


        inference_one_dataset(call_bert, model, tokenizer, dataset_name, dataset_file, llm_prompt_file, output_file, batch_size, max_length, print_freq)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--bert_dir', type=str, default="../fine-tuned_model/bert")
    parser.add_argument('--data_dir', type=str, default="../dataset/test")
    parser.add_argument('--bert_prompt_dir', type=str, default="../prompt_templates/classifier")
    parser.add_argument('--llm_prompt_dir', type=str)
    parser.add_argument('--bert_out', type=str, choices=["probs","conf","label"],default="")
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--print_freq', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=5000)
    args = parser.parse_args()
    main(**vars(args))

    # main("Qwen/Qwen3-0.6B", "../dataset/test","../prompt_templates/zero_shot", "../model_output/Qwen3-0.6B_zero_shot", "cuda", 2, 5000, 1)


    # cd code_llm
    #python bert_llm_inference.py --model_path Qwen/Qwen3-0.6B --llm_prompt_dir ../prompt_templates/bert_llm_one_shot_cot --bert_out conf --output_dir ../model_output/bert_llm2_one_shot_cot --batch_size 1



