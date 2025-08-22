from datasets import Dataset
import torch
from transformers import AutoTokenizer, BertForSequenceClassification, DataCollatorWithPadding,  get_linear_schedule_with_warmup

from torch.utils.data import DataLoader
import json
import math
from torch.optim import AdamW
from tqdm import tqdm
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    print(f"Adding '{parent_dir}' to PYTHONPATH")
    sys.path.append(parent_dir)
import IMHI_dataset

def train_one_epoch(model, dataloader, optimizer, scheduler):
    model.train()
    losses = 0
    total = 0
    device = model.device
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
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

            # forward (no labels so outputs.logits 一定存在)
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



def main(model_path: str, dataset_name: str, batch_size: int, device: str, save_dir: str, lr= 2e-4, epochs = 10, warmup_ratio = 0.1):
    labels = IMHI_dataset.get_standard_labels(dataset_name)

    cache_dir = "../my_model_cache"
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    model = BertForSequenceClassification.from_pretrained(model_path, cache_dir=cache_dir, num_labels=len(labels)).to(torch.device(device))


    prompt_path = f"../prompt_templates/classifier/{dataset_name}.txt"
    train_dataset = IMHI_dataset.get_dict_dataset(f"../dataset/train/{dataset_name}.csv", prompt_path)
    train_dataset = Dataset.from_dict(train_dataset)

    valid_dataset = IMHI_dataset.get_dict_dataset(f"../dataset/valid/{dataset_name}.csv", prompt_path)
    valid_dataset = Dataset.from_dict(valid_dataset)


    def format_example(ex):
        out = tokenizer(ex["query"], max_length=512, truncation=True)
        out["label"] = labels.index(ex["label"])
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
        print(f"Epochs: {epoch+1}/{epochs}")
        log = train_one_epoch(model, train_loader, optimizer, scheduler)
        print(f"train loss: {log['train_loss']}, lr: {log['lr']}")
        log.update(valid_model(model, valid_loader))
        print(f"valid loss: {log['valid_loss']}, accuracy: {log['accuracy']}")
        logs.append(log)


    save_dir = f"{save_dir}/{dataset_name}"

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    with open(f"{save_dir}/log.json", "w") as f:
        json.dump(logs, f)

if __name__ == "__main__":
    main("google-bert/bert-base-uncased",  "SAD", 10, "cuda", "../fine-tuned model/test")




