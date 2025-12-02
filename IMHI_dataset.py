import pandas as pd
import re
import random
from datasets import Dataset
import uuid
import os
from datasets import concatenate_datasets

def extract_prompts(text):
    tag_pat = re.compile(r'<([^\>]+)>')
    matches = list(tag_pat.finditer(text))
    out = {}
    for i, m in enumerate(matches):
        key = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[key] = text[start:end].strip()
    return out

def random_number_exclude(start, end, exclude):
    # [start, end)
    choices = [i for i in range(start, end) if i not in exclude]
    return random.choice(choices)


def get_full_dataset(dataset_dir, prompt_dir, include = None):
    datasets = []
    dataset_names = []
    for file in os.listdir(dataset_dir):
        if not file.endswith(".csv"):
            continue

        dataset_name = file.split('.')[0]
        if include is not None and dataset_name not in include:
            continue
        dataset_file = os.path.join(dataset_dir, file)
        prompt_file = os.path.join(prompt_dir, f"{dataset_name}.txt")
        dataset = get_dataset(dataset_file, prompt_file)
        dataset = dataset.add_column("dataset_name", [dataset_name]*len(dataset))

        datasets.append(dataset)
        dataset_names.append(dataset_name)
    return concatenate_datasets(datasets), dataset_names


def get_dataset(dataset_file, prompt_file, functions=None):
    with open(prompt_file, "r", encoding="utf-8") as f:
        template_prompt = f.read()
        template_prompts = extract_prompts(template_prompt)

    dataset = {k:[] for k in template_prompts.keys()}

    df = pd.read_csv(dataset_file, dtype=str)
    for row_index, row in df.iterrows():
        random_keys = {"current": row_index}  # add current to randoms_keys to avoid pick the current row

        def replace_placeholder(match):
            key = match.group(1).strip()

            if ":" in key:
                key, f_str = key.split(":", 1)
                key, f_str = key.strip(), f_str.strip()

                if key.startswith("@"): #function call
                    params = [str(row[k]) for k in f_str.strip(",").strip()]
                    return functions[key[1:]](*params)

                if f_str.startwith("r"): # random
                    if f_str == "r":
                        idx = random_number_exclude(0, len(df), random_keys.values())
                        random_keys[str(uuid.uuid4())] = idx
                    elif f_str in random_keys.keys():
                        idx = random_keys[f_str]
                    else:
                        idx = random_number_exclude(0, len(df), random_keys.values())
                        random_keys[f_str] = idx
                elif f_str.startswith(("+", "-")): # offset
                    idx = (row_index + int(f_str)) % len(df)
                else:   # num index
                    idx = int(f_str) % len(df)
                return df[key][idx]


            if key.startswith("@"): #function call
                return functions[key[1:]]()

            return str(row[key])

        for k, v in template_prompts.items():
            dataset[k].append( re.sub(r"\{([^\]]+)\}", replace_placeholder, v))

    return Dataset.from_dict(dataset)


def get_standard_labels(dataset_name: str):
    if dataset_name == 'swmh':
        standard_labels = ['no mental disorder', 'suicide', 'depression', 'anxiety', 'bipolar']
    elif dataset_name == 't-sid':
        standard_labels = ['no mental disorder', 'depression', 'suicide or self-harm', 'ptsd']
    elif dataset_name in ['CLP', 'DR', 'dreaddit', 'loneliness', 'Irf', 'MultiWD']:
        standard_labels = ["false", "true"]
    elif dataset_name == 'SAD':
        standard_labels = ['other causes', 'school', 'financial problem', 'family issues', 'social relationships', 'work', 'health issues', 'emotional turmoil', 'everyday decision making']
    elif dataset_name == "CAMS":
        standard_labels = ['none', 'bias or abuse', 'jobs and career', 'medication', 'relationship', 'alienation']
    else:
        raise NameError(f"{dataset_name} is not a valid dataset name")

    return standard_labels


def get_search_labels(dataset_name: str):
    if dataset_name == 'swmh':
        search_labels = ['no mental', 'suicide', 'depression', 'anxiety', 'bipolar']
    elif dataset_name == 't-sid':
        search_labels = ['no mental', 'depression', ['suicide', 'self-harm'], 'ptsd']
    elif dataset_name in ['CLP', 'DR', 'dreaddit', 'loneliness', 'Irf', 'MultiWD']:
        search_labels = ["false", "true"]
    elif dataset_name == 'SAD':
        search_labels = ['other', 'school', 'financial', 'family issue', 'social', 'work', 'health', 'emotional', 'decision']
    elif dataset_name == "CAMS":
        search_labels = [['none', 'no causes'], ['bias', 'abuse'], ['job', 'career'], 'medication', 'relationship', 'alienation']
    else:
        raise NameError(f"{dataset_name} is not a valid dataset name")
    return search_labels