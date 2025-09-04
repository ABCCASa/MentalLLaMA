import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding,  get_linear_schedule_with_warmup
import argparse
from torch.utils.data import DataLoader
import json
import math
from torch.optim import AdamW
from trl import SFTConfig, SFTTrainer
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    print(f"Adding '{parent_dir}' to PYTHONPATH")
    sys.path.append(parent_dir)
import IMHI_dataset




def main(model_path:str, train_data_dir: str, valid_data_dir:str, prompt_dir: str, output_dir: str, device: torch.device, print_freq: int):

    device = torch.device(device)
    train_dataset, include = IMHI_dataset.get_full_dataset(train_data_dir, prompt_dir)
    valid_dataset, _ = IMHI_dataset.get_full_dataset(valid_data_dir, prompt_dir, include)
    def format_example(ex):

        if "system" in train_dataset.column_names:
            return {
                "prompt": [{"content":ex["system"],"role": "system"}, {"content":ex["query"],"role": "user"}],
                "completion": [{"content": ex["answer"] ,"role": "assistant" }]}
        else:
            return {
                "prompt": [{"content": ex["query"], "role": "user"}],
                "completion": [{"content": ex["answer"], "role": "assistant"}]}

    train_dataset = train_dataset.map(format_example,  remove_columns=train_dataset.column_names)
    valid_dataset = valid_dataset.map(format_example, remove_columns=valid_dataset.column_names)

    cache_dir = "../my_model_cache"
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir, torch_dtype=torch.bfloat16).to(device)
    model.train()
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=10 ,
        per_device_train_batch_size = 6,
        per_device_eval_batch_size = 6,
        learning_rate=1e-5,
        gradient_accumulation_steps=6,
        warmup_ratio =0.03,
        logging_steps=print_freq,
        save_strategy = "epoch",
        eval_strategy= "epoch",
        save_total_limit = 2,
        completion_only_loss =True,
        max_length= 2048,
        packing =False
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--train_data_dir', type=str)
    parser.add_argument('--valid_data_dir', type=str)
    parser.add_argument('--prompt_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--print_freq', type=int, default=10)


    args = parser.parse_args()
    main(**vars(args))

    # cd code_llm
    # python train.py --model_path meta-llama/Llama-3.2-1B-Instruct --train_data_dir ../dataset/train --valid_data_dir ../dataset/valid --prompt_dir ../prompt_templates/QA --output_dir ../fine-tuned_model/llama-3.2-1B --device cuda --print_freq 5

