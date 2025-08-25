import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,  get_linear_schedule_with_warmup
import argparse
from torch.utils.data import DataLoader
import json
import math
from torch.optim import AdamW
from tqdm import tqdm
import sys
import os
from torch.nn.utils import clip_grad_norm_
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    print(f"Adding '{parent_dir}' to PYTHONPATH")
    sys.path.append(parent_dir)
import IMHI_dataset
import our_metrics


def train_one_epoch(model, dataloader, optimizer, scheduler, max_norm):
    model.train()
    losses = 0
    total = 0
    device = model.device
    for batch in tqdm(dataloader, leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        loss = outputs.loss
        loss.backward()

        clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        batch_count = len(batch["labels"])
        losses += loss.item() * batch_count
        total += batch_count
    avg_loss = losses / total
    lr = scheduler.get_last_lr()[0]
    return {"train_loss": avg_loss, "lr": lr}


def evaluate_model(model, dataloader):
    model.eval()
    device = model.device
    total = 0
    losses = 0
    golden_label = []
    output_label = []
    with torch.inference_mode():
        for batch in tqdm(dataloader, leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            batch_count = len(batch["labels"])
            total += batch_count
            # loss

            losses += outputs.loss.item() * batch_count

            # accuracy
            golden_label += batch["labels"].tolist()
            output_label += outputs.logits.argmax(dim=-1).tolist()


    loss = losses / total if total > 0 else 0.0
    result_dict = our_metrics.evaluate_all(golden_label, output_label)
    result_dict["loss"] = loss
    result_dict["output_label"] = output_label
    return result_dict


def train_one_dataset(model_path: str, dataset_name: str, train_data_file: str, valid_data_file: str, test_data_file: str,
                      prompt_file: str, batch_size: int, device: torch.device, output_dir: str, lr: float, epochs: int,
                      warmup_ratio: float, max_norm: float):

    label_set = IMHI_dataset.get_standard_labels(dataset_name)

    cache_dir = "../my_model_cache"
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, cache_dir=cache_dir, num_labels=len(label_set)).to(device)

    train_dataset = IMHI_dataset.get_dataset(train_data_file, prompt_file)
    valid_dataset = IMHI_dataset.get_dataset(valid_data_file, prompt_file)
    test_dataset = IMHI_dataset.get_dataset(test_data_file, prompt_file)

    def format_example(ex):
        out = tokenizer(ex["query"], max_length=512, truncation=True)
        out["labels"] = label_set.index(ex["label"])
        return out

    train_tokenized = train_dataset.map(format_example,  remove_columns=train_dataset.column_names)
    valid_tokenized = valid_dataset.map(format_example, remove_columns=valid_dataset.column_names)
    test_tokenized = test_dataset.map(format_example, remove_columns=test_dataset.column_names)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        train_tokenized,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator
    )
    valid_loader = DataLoader(
        valid_tokenized,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )
    test_loader = DataLoader(
        test_tokenized,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * math.ceil(len(train_loader))
    num_warmup_steps = int(warmup_ratio * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


    epochs_logs = []
    print(f"[{dataset_name}] start training")
    for epoch in range(1, epochs+1):
        print(f"Epoch: {epoch}/{epochs}:")

        #train
        train_log = train_one_epoch(model, train_loader, optimizer, scheduler, max_norm)
        print(f"    [Train Log]", ", ".join([f"{k}: {v}" for k, v in train_log.items()]))

        #validate
        valid_log = evaluate_model(model, valid_loader)
        valid_log.pop("output_label")
        print(f"    [Valid Log]", ", ".join([f"{k}: {v}" for k, v in valid_log.items()]))

        # save log
        epochs_logs.append({"train_log": train_log, "valid_log": valid_log})

    test_log = evaluate_model(model, test_loader)
    pred_labels = test_log.pop("output_label")
    print(f"[Test Log]", ", ".join([f"{k}:{v}" for k, v in test_log.items()]))


    # save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # save log
    log = {"epoch_log": epochs_logs, "test_log": test_log}
    with open(os.path.join(output_dir, "log.json"), "w") as f:
        json.dump(log, f)

    # save output for test
    pred_labels = [label_set[label_idx] for label_idx in pred_labels]
    test_dataset = test_dataset.to_dict()
    test_dataset["pred_labels"] = pred_labels
    output = pd.DataFrame(test_dataset, index=None)
    output.to_csv(os.path.join(output_dir, "test_output.csv"), index=False)

def main(model_path:str, train_data_dir: str, valid_data_dir: str, test_data_dir:str, prompt_dir: str, output_dir: str, batch_size: int,
         device: torch.device, lr: float, epochs: int, warmup_ratio: float, max_norm:float):

    device = torch.device(device)
    for file in os.listdir(train_data_dir):
        if not file.endswith(".csv"):
            continue
        dataset_name = file.split('.')[0]

        train_data_file = os.path.join(train_data_dir, file)
        valid_data_file = os.path.join(valid_data_dir, file)
        test_data_file = os.path.join(test_data_dir, file)
        if not os.path.exists(valid_data_file) or not os.path.exists(test_data_file):
            print(f"{dataset_name} does not exist, skipping")
            continue
        prompt_file = os.path.join(prompt_dir, f"{dataset_name}.txt")
        output_dir_per_dataset = os.path.join(output_dir, f"{dataset_name}")
        if os.path.exists(output_dir_per_dataset):
            print(f"{dataset_name} is already trained, skipping")
            continue

        train_one_dataset(model_path, dataset_name, train_data_file, valid_data_file, test_data_file, prompt_file, batch_size, device,
                          output_dir_per_dataset, lr, epochs, warmup_ratio, max_norm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--train_data_dir', type=str)
    parser.add_argument('--valid_data_dir', type=str)
    parser.add_argument('--test_data_dir', type=str)
    parser.add_argument('--prompt_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup_ratio', type=float, default=0.08)
    parser.add_argument('--max_norm', type=float, default=1)

    args = parser.parse_args()
    main(**vars(args))

    # cd code_bert
    # python classifier_train.py --model_path google-bert/bert-base-cased --train_data_dir ../dataset/train --valid_data_dir ../dataset/valid --test_data_dir ../dataset/test --prompt_dir ../prompt_templates/classifier --output_dir ../fine-tuned_model/bert --device cuda --batch_size 32 --epochs 4

