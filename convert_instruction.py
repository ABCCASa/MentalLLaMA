import pandas as pd
import os
from typing import List, Union

def extract_label_indexes(text:str, valid_labels: List, map_labels)-> List:
    extracted_labels = []
    for index, item in enumerate(valid_labels):
        if isinstance(item, list):
            for label in item:
                if label in text:
                    extracted_labels.append(map_labels[index])
                    break
        elif item in text:
            extracted_labels.append(map_labels[index])
    return extracted_labels


def convert(root, source_folder, target_folder, filename):
    df = pd.read_csv(f"{root}/{source_folder}/{filename}")
    dataset_name = filename.removesuffix(".csv")

    if dataset_name == 'swmh':
        valid_labels = ['no mental', 'suicide', 'depression', 'anxiety', 'bipolar']
        map_labels = ['no mental disorder', 'suicide', 'depression', 'anxiety', 'bipolar']
    elif dataset_name == 't-sid':
        valid_labels = ['no mental', 'depression', 'suicide or self-harm', 'ptsd']
        map_labels = ['no mental disorder', 'depression', 'suicide or self-harm', 'ptsd']
    elif dataset_name in ['CLP', 'DR', 'dreaddit', 'loneliness', 'Irf', 'MultiWD']:
        valid_labels = ["yes", "no"]
        map_labels = ["true", "false"]
    elif dataset_name == 'SAD':
        valid_labels = ['other', 'school', 'financial problem', 'family issue', 'social relationship', 'work', 'health issue', ['emotional turmoil', "emotion turmoil"], 'everyday decision making']
        map_labels = ['other causes', 'school', 'financial problem', 'family issues', 'social relationships', 'work', 'health issues', 'emotional turmoil', 'everyday decision making']
    elif dataset_name == "CAMS":
        valid_labels = [['none','no causes'], 'bias or abuse', 'jobs and career', 'medication', 'relationship', 'alienation']
        map_labels = ['none', 'bias or abuse', 'jobs and career', 'medication', 'relationship', 'alienation']
    else:
        raise NameError(f"{dataset_name} is not a valid dataset name")


    posts = []
    labels = []
    reasons = []
    aspects = []

    # extract label and reason
    for index, row in df.iterrows():
        if "gpt-3.5-turbo" in df.columns:
            output =  row["gpt-3.5-turbo"]
        else:
            output = row["response"]
        label_and_reason = output.split("Reasoning:", 1)
        label_text = label_and_reason[0].lower()
        reason = label_and_reason[1].strip()
        if reason == "":
            print(f"[{dataset_name}] Can't obtain a reasoning step from: row {index}")
            continue


        extracted_labels = extract_label_indexes(label_text, valid_labels, map_labels)

        if len(extracted_labels) != 1:
            print(f"[{dataset_name}] Can't obtain a unique label from: {label_text}, get{extracted_labels}")
            continue


        query = row["query"]
        post = query.split('" Question:')[0].removeprefix('Consider this post: "').strip()
        if dataset_name == "MultiWD":
            aspect = query.split('Question: Does the')[1].replace('wellness dimension exist in the post?', "").strip()
            aspects.append(aspect)
        elif dataset_name == "Irf":
            aspect = query.split('Question: Does the post show risk of')[1].replace('?', "").strip()
            aspects.append(aspect)

        label = extracted_labels[0]
        posts.append(post)
        labels.append(label)
        reasons.append(reason)


    # create new dataframe
    data = pd.DataFrame()
    data["post"] = posts
    if dataset_name == "MultiWD" or dataset_name == "Irf":
        data["aspect"] = aspects
    data["label"] = labels
    data["reason"] = reasons

    if not os.path.exists(f"{root}/{target_folder}"):
        os.makedirs(f"{root}/{target_folder}")

    data.to_csv(f"{root}/{target_folder}/{filename}", index=False)


def convert_all(root, source_folder, target_folder):
    for file in os.listdir(f"{root}/{source_folder}"):
        if file.endswith(".csv"):
            convert(root, source_folder, target_folder, file)


if __name__ == "__main__":
    convert_all("test_data", "instruction", "our2")
