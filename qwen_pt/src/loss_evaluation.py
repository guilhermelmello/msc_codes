from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from trainer import evaluate_clm

import torch

MODEL_NAME = 'Qwen/Qwen3-0.6B'
DATASET_PATH = 'datasets/qwen3-06b'
# MODEL_NAME = 'guilhermelmello/qwen-pt-base-unigram-8k'
# DATASET_PATH = 'datasets/clm-1024-unigram-pt-8k' 
BATCH_SIZE = 16
NUM_WORKERS = 8



print('>>> Loading dataset')
dataset = load_from_disk(DATASET_PATH)
print(dataset)


print('>>> Loading Tokenizer')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(tokenizer)


print('>>> Loading Model')
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model = torch.compile(model)
model.to('cuda')
print(model)


print('>>> Loss Evaluation')
loss_train = evaluate_clm(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset['train'],
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)
print('Train Loss:', loss_train)

loss_val = evaluate_clm(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset['validation'],
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)
print('Validation Loss:', loss_val)
