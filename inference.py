import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login, logout
import re
import argparse

def load_test_data(root, prompt_type):
    test_data = {}
    for file in os.listdir(root):
        if not file.endswith(".csv"):
            continue
        dataset_name = file.split('.')[0]


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


def generate_response(model, tokenizer, query):
    messages = [{"role": "user", "content": query}]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_length=16384)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    response = tokenizer.decode(output_ids, skip_special_tokens=True)

    return response


def save_output(output, dataset_name, output_path):
    if not os.path.exists("model_output/"):
        os.mkdir("model_output/")
    if not os.path.exists("model_output/"+output_path):
        os.mkdir("model_output/"+output_path)
    output = pd.DataFrame(output, index=None)
    output.to_csv(f"model_output/{output_path}/{dataset_name}.csv",  index=False)


def main(model_path: str, data_path: str,  prompt_type: str,output_path: str, print_freq: int):

    test_data = load_test_data(data_path, prompt_type)

    # login hugging face
    try:
        login(token=os.environ["HF_TOKEN"])
    except:
        print("Please set HF_TOKEN environment variable")

    # load model
    cache_dir = "my_model_cache"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=torch.bfloat16, cache_dir=cache_dir)


    # start inference
    for dataset_name, data in test_data.items():
        print(f"Start Dataset: {dataset_name}")
        responses = []
        for index, query in enumerate(data["query"]):
            response = generate_response(model, tokenizer, query)
            responses.append(response)
            if index % print_freq == 0:
                print(query, response)
                print(f"[{dataset_name}] {index+1}/{len(data["query"])}\n")

        data["response"] = responses
        save_output(data, dataset_name, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--prompt_type', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--print_freq', type=int, default=50)

    main(**vars(parser.parse_args()))
    #main("Qwen/Qwen3-8B", "test_data/our", "few_shot", "Qwen3-8B_few_shot", 50)
