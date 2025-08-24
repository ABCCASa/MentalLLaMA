from datasets import Dataset
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

def train_one_epoch(model, dataloader, optimizer, scheduler, max_norm):
    model.train()
    losses = 0
    total = 0
    device = model.device
    for batch in tqdm(dataloader):
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


def valid_model(model, dataloader):
    model.eval()
    device = model.device
    total = 0
    correct = 0
    losses = 0
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            # loss
            batch_count = len(batch["labels"])
            losses += outputs.loss.item() * batch_count

            # accuracy
            labels = batch["labels"]
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()

            total += batch_count

    acc = correct / total if total > 0 else 0.0
    loss = losses / total if total > 0 else 0.0
    return {"valid_loss": loss, "accuracy": acc}


def train_one_dataset(model_path: str, dataset_name: str, train_data_file: str, valid_data_file: str, prompt_file: str,
         batch_size: int, device: torch.device, output_dir: str, lr: float, epochs: int, warmup_ratio: float, max_norm: float):

    labels = IMHI_dataset.get_standard_labels(dataset_name)

    cache_dir = "../my_model_cache"
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, cache_dir=cache_dir, num_labels=len(labels)).to(device)

    train_dataset = IMHI_dataset.get_dict_dataset(train_data_file, prompt_file)
    train_dataset = Dataset.from_dict(train_dataset)

    valid_dataset = IMHI_dataset.get_dict_dataset(valid_data_file, prompt_file)
    valid_dataset = Dataset.from_dict(valid_dataset)

    def format_example(ex):
        out = tokenizer(ex["query"], max_length=512, truncation=True)
        out["labels"] = labels.index(ex["label"])
        return out

    train_dataset = train_dataset.map(format_example,  remove_columns=train_dataset.column_names)
    valid_dataset = valid_dataset.map(format_example, remove_columns=valid_dataset.column_names)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * math.ceil(len(train_loader))
    num_warmup_steps = int(warmup_ratio * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


    logs = []


    for epoch in range(epochs):
        print(f"[{dataset_name}] Epochs: {epoch+1}/{epochs}")
        log = train_one_epoch(model, train_loader, optimizer, scheduler, max_norm)
        print(f"train loss: {log['train_loss']: .4f}, lr: {log['lr']: .5e}")
        log.update(valid_model(model, valid_loader))
        print(f"valid loss: {log['valid_loss']: .4f}, accuracy: {log['accuracy']: .4f}")
        logs.append(log)


    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "log.json"), "w") as f:
        json.dump(logs, f)


def main(model_path:str, train_data_dir: str, valid_data_dir: str, prompt_dir: str, output_dir: str, batch_size: int,
         device: torch.device, lr: float, epochs: int, warmup_ratio: float, max_norm:float):

    device = torch.device(device)
    for file in os.listdir(train_data_dir):
        if not file.endswith(".csv"):
            continue
        dataset_name = file.split('.')[0]
        train_data_file = os.path.join(train_data_dir, file)
        valid_data_file = os.path.join(valid_data_dir, file)
        if not os.path.exists(valid_data_file):
            print(f"File {valid_data_file} does not exist, skipping")
            continue
        prompt_file = os.path.join(prompt_dir, f"{dataset_name}.txt")
        output_dir_per_dataset = os.path.join(output_dir, f"{dataset_name}")
        train_one_dataset(model_path, dataset_name, train_data_file, valid_data_file, prompt_file, batch_size, device,
                          output_dir_per_dataset, lr, epochs, warmup_ratio, max_norm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--train_data_dir', type=str)
    parser.add_argument('--valid_data_dir', type=str)
    parser.add_argument('--prompt_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--max_norm', type=float, default=1)

    args = parser.parse_args()
    main(**vars(args))

    # python classifier_train.py --model_path google-bert/bert-base-cased --train_data_dir ../dataset/train --valid_data_dir ../dataset/valid --prompt_dir ../prompt_templates/classifier --output_dir ../fine-tuned_model/bert --device cuda --batch_size 32 --epochs 5

