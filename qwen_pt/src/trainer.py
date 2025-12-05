import os
import torch
import utils

from copy import deepcopy
from datasets import Dataset, load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
    PreTrainedModel,
)


def compute_loss(logits, labels):
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()

    loss_fn = CrossEntropyLoss()
    loss = loss_fn(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1)
    )
    return loss


def get_optimizer_parameters(model, weight_decay, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


def train_clm(
    model: PreTrainedModel,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    n_epochs: int,
    num_workers: int=None,
):
    # dataloaders
    train_dataset.set_format('torch')
    validation_dataset.set_format('torch')
    train_dataloader = DataLoader(
        train_dataset, # type: ignore
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    validation_dataloader = DataLoader(
        validation_dataset, # type: ignore
        batch_size=batch_size,
        pin_memory=True,
    )

    # optimizer and weight decay
    optimizer_parameters = get_optimizer_parameters(model, weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=learning_rate)

    # lr scheduler
    num_training_steps = n_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    # move model to GPU, if available
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)
    model.to(device) # type: ignore

    # track best model
    best_model_loss = float('inf')
    best_model_state = {}

    # training loop
    print('>>> Model Training')
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(n_epochs):
        progress_bar.set_description(f'Epoch {epoch+1}/{n_epochs}')

        # start training mode
        model.train()
        total_train_loss = 0
        for batch in train_dataloader:
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                att_mask = batch['attention_mask'].to(device, non_blocking=True)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=att_mask,
                )
                loss = compute_loss(outputs.logits, input_ids)
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            total_train_loss += loss.item()
            del batch

        # start evaluation mode
        model.eval()
        eval_steps = len(validation_dataloader)
        eval_progress_bar = tqdm(range(eval_steps), leave=False)
        total_validation_loss = 0

        # evaluation loop (validation)
        for batch in validation_dataloader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            att_mask = batch['attention_mask'].to(device, non_blocking=True)
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=att_mask,
                )                
            total_validation_loss += compute_loss(outputs.logits, input_ids).item()
            eval_progress_bar.update(1)
            del batch
        eval_progress_bar.close()

        # report metrics
        train_loss = total_train_loss / len(train_dataloader)
        validation_loss = total_validation_loss / len(validation_dataloader)

        epoch_results = {
            'epoch': epoch + 1,
            'loss': train_loss,
            'val_loss': validation_loss,
        }
        print(epoch_results)

        # update best model
        if validation_loss < best_model_loss:
            best_model_state = deepcopy(model.state_dict())
            best_model_loss = validation_loss

    model.load_state_dict(best_model_state)
    del best_model_state

    return model


def evaluate_clm(model, tokenizer, dataset, batch_size):
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)
    model.to(device)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

    total_loss = 0
    progress_bar = tqdm(range(len(dataloader)), leave=False)

    model.eval()
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        att_mask = batch['attention_mask'].to(device, non_blocking=True)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=att_mask,
            )
        total_loss += compute_loss(outputs.logits, input_ids).item()
        progress_bar.update(1)
        del batch

    progress_bar.close()

    loss = total_loss / len(dataloader)
    return loss


if __name__ == '__main__':
    print('>>> Loading CLM Dataset from Disk')
    dataset = load_from_disk('/work/gmello/datasets/clm-1024-unigram-pt-10k/validation')
    assert isinstance(dataset, Dataset)

    print('>>> Creating train and test splits for hyperparameter search.')
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    print(dataset)

    print('>>> Loading Tokenizer')
    tokenizer = AutoTokenizer.from_pretrained('guilhermelmello/tokenizer-unigram-pt-10k')
    print(tokenizer)

    print('>>> Loading Model')
    model = utils.initialize_clm_from_config('Qwen/Qwen3-0.6B', tokenizer)
    model = torch.compile(model)
    print(model)

    model = train_clm(
        model=model,
        train_dataset=dataset['train'],
        validation_dataset=dataset['test'],
        batch_size=16,
        learning_rate=0.00001,
        weight_decay=0.1,
        warmup_steps=100,
        n_epochs=10,
        num_workers=16,
        max_steps=100,
    )

    loss = evaluate_clm(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset['test'],
        batch_size=16,
    )
    print(f'final loss: {loss}')

    output_path = './models/qwen-pt-unigram'
    print(f'Saving model to {output_path}')

    os.makedirs(output_path)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
