import pandas as pd
import re
import random

def random_number_exclude(start, end, exclude):
    # [start, end)
    choices = [i for i in range(start, end) if i not in exclude]
    return random.choice(choices)

def get_dict_dataset(dataset_file, prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        template_prompt = f.read()
    queries = []
    labels = []
    df = pd.read_csv(dataset_file, dtype=str)
    for row_index, row in df.iterrows():
        random_keys = {"current": row_index}  # add current to randoms_keys to avoid pick the current row
        def replace_placeholder(match):
            key = match.group(1).strip()
            if ":" in key:
                key, f_str = key.split(":", 1)
                key, f_str = key.strip(), f_str.strip()
                if f_str.startswith(("+", "-")):
                    idx = (row_index + int(f_str)) % len(df)
                elif f_str == "r":
                    idx = random_number_exclude(0, len(df), [row_index])
                elif f_str.startwith("r"):
                    if f_str in random_keys.keys():
                        idx = random_keys[f_str]
                    else:
                        idx = random_number_exclude(0, len(df), random_keys.values())
                        random_keys[f_str] = idx
                else:
                    idx = int(f_str) % len(df)
                return df[key][idx]
            else:
                return str(row[key])

        query = re.sub(r"\[([^\]]+)\]", replace_placeholder, template_prompt)
        queries.append(query)
        labels.append(row["label"])
    return  {"query": queries, "label": labels}


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