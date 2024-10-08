import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftConfig
from peft import PeftModel
from trl import SFTTrainer
from trl import setup_chat_format
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          pipeline,
                          logging)
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from huggingface_hub import notebook_login
import bitsandbytes as bnb
import os

# Define the prompt generation functions
def generate_prompt(data_point):
    cta_types = np.load('cta_types.npy', allow_pickle=True)
    return f"""
            Classify the text {",".join(cta_types)}, and return the answer as the corresponding column type.
text: {data_point["values"]}
label: {data_point["type"]}""".strip()

def generate_test_prompt(data_point):
    cta_types = np.load('cta_types.npy', allow_pickle=True)
    return f"""
            Classify the text into {",".join(cta_types)}, and return the answer as the corresponding column type.
#text: {data_point["values"]}
    text: {",".join(data_point.values)}
label: """.strip()

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def predict_domain(prompt, model, tokenizer):
    y_pred = []
    with open('cta_types_domain_reduced_5domain.json', 'r') as file:
        cta_type_domain = json.load(file)

    pipe = pipeline(task="text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=2,
                    temperature=0.1)

    result = pipe(prompt)
    answer = result[0]['generated_text'].split("Domain:")[-1].strip()

    return answer

def predict(test, model, tokenizer, domain):
    y_pred = []
    with open('cta_types_domain_reduced_5domain.json', 'r') as file:
        cta_type_domain = json.load(file)

    categories = cta_type_domain[domain]
    
    for i in tqdm(range(len(test))):
        prompt = test.iloc[i]["prediction"]
        pipe = pipeline(task="text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=2,
                        temperature=0.1)

        result = pipe(prompt)
        answer = result[0]['generated_text'].split("label:")[-1].strip()

        # Determine the predicted category
        for category in categories:
            if category.lower() in answer.lower():
                y_pred.append(answer)
                break
        else:
            y_pred.append("none")

    return y_pred

def predict_old(test, model, tokenizer):
    y_pred = []
    categories = np.load('cta_types.npy', allow_pickle=True)
    
    for i in tqdm(range(len(test))):
        prompt = test.iloc[i]["prediction"]
        pipe = pipeline(task="text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=2,
                        temperature=0.1)

        result = pipe(prompt)
        answer = result[0]['generated_text'].split("label:")[-1].strip()

        # Determine the predicted category
        for category in categories:
            if  answer.lower() in category.lower():
                y_pred.append(answer)
                break
        else:
            y_pred.append("none")

    return y_pred

def evaluate(y_true, y_pred):
    labels = np.load('cta_types.npy', allow_pickle=True)
    mapping = {label: idx for idx, label in enumerate(labels)}

    def map_func(x):
        return mapping.get(x, -1)  # Map to -1 if not found, but should not occur with correct data

    y_true_mapped = np.vectorize(map_func)(y_true)
    y_pred_mapped = np.vectorize(map_func)(y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true_mapped, y_pred=y_pred_mapped)
    print(f'Accuracy: {accuracy:.3f}')

    # Generate accuracy report
    unique_labels = set(y_true_mapped)  # Get unique labels

    for label in unique_labels:
        label_indices = [i for i in range(len(y_true_mapped)) if y_true_mapped[i] == label]
        label_y_true = [y_true_mapped[i] for i in label_indices]
        label_y_pred = [y_pred_mapped[i] for i in label_indices]
        label_accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {labels[label]}: {label_accuracy:.3f}')

    # Generate classification report
    class_report = classification_report(y_true=y_true_mapped, y_pred=y_pred_mapped, target_names=labels, labels=list(range(len(labels))))
    print('\nClassification Report:')
    print(class_report)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true_mapped, y_pred=y_pred_mapped, labels=list(range(len(labels))))
    print('\nConfusion Matrix:')
    print(conf_matrix)

def load_pretrained_model(fine_tuned_model):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        fine_tuned_model,
        device_map="auto",
        torch_dtype="float16",
        quantization_config=bnb_config,
    )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model)
    
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def generate_train_val_data(train_sample=5, val_sample=5):
    
    X_train = pd.read_parquet('raw/train_values.parquet', engine='pyarrow')
    X_val   = pd.read_parquet('raw/val_values.parquet', engine='pyarrow')
    y_train = pd.read_parquet('raw/train_labels.parquet', engine='pyarrow')
    y_val   = pd.read_parquet('raw/val_labels.parquet', engine='pyarrow')
    
    X_train = X_train.assign(type=y_train.type.values)
    X_train[X_train['values'].apply(lambda col: len(",".join(col)) > 512)] = np.nan
    X_train.dropna()
    X_train = X_train.groupby('type').sample(n=train_sample)
    
    X_val = X_val.assign(type=y_val.type.values)
    X_val[X_val['values'].apply(lambda col: len(",".join(col)) > 512)] = np.nan
    X_val.dropna()
    X_val = X_val.groupby('type').sample(n=val_sample)
    
    # Generate prompts for training and evaluation data
    X_train.loc[:,'prediction'] = X_train.apply(generate_prompt, axis=1)
    X_val.loc[:,'prediction'] = X_val.apply(generate_prompt, axis=1)
    
    # Convert to datasets
    train_data = Dataset.from_pandas(X_train[["prediction"]])
    eval_data = Dataset.from_pandas(X_val[["prediction"]])

    return train_data, eval_data

def initiate_trainer(model, train_data, eval_data, tokenizer, modules, output_dir):
    peft_config = LoraConfig(
                          lora_alpha=16,
                          lora_dropout=0,
                          r=64,
                          bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=modules,
                        )
    
    training_arguments = TrainingArguments(
                                          output_dir=output_dir,                    # directory to save and repository id
                                          num_train_epochs=1,                       # number of training epochs
                                          per_device_train_batch_size=1,            # batch size per device during training
                                          gradient_accumulation_steps=8,            # number of steps before performing a backward/update pass
                                          gradient_checkpointing=True,              # use gradient checkpointing to save memory
                                          optim="paged_adamw_32bit",
                                          logging_steps=1,
                                          learning_rate=2e-4,                       # learning rate, based on QLoRA paper
                                          weight_decay=0.001,
                                          fp16=True,
                                          bf16=False,
                                          max_grad_norm=0.3,                        # max gradient norm based on QLoRA paper
                                          max_steps=-1,
                                          warmup_ratio=0.03,                        # warmup ratio based on QLoRA paper
                                          group_by_length=False,
                                          lr_scheduler_type="cosine",               # use cosine learning rate scheduler
                                        #   report_to="wandb",                  # report metrics to w&b
                                          eval_strategy="steps",              # save checkpoint every epoch
                                          eval_steps = 0.2
                                        )
    
    trainer = SFTTrainer(
                          model=model,
                          args=training_arguments,
                          train_dataset=train_data,
                          eval_dataset=eval_data,
                          peft_config=peft_config,
                          dataset_text_field="prediction",
                          tokenizer=tokenizer,
                          max_seq_length=512,
                          packing=False,
                          dataset_kwargs={
                          "add_special_tokens": False,
                          "append_concat_token": False,
                          }
                        )
    return trainer

def initiate_base_model(base_model_name):
    bnb_config = BitsAndBytesConfig(
                                        load_in_4bit=True,
                                        bnb_4bit_use_double_quant=False,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype="float16",
                                    )
                                    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype="float16",
        quantization_config=bnb_config,
    )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer
