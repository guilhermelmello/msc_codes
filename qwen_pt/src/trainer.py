import argparse
import os
import torch
import utils

from copy import deepcopy
from datasets import Dataset, DatasetDict, load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
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
    num_epochs: int,
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
        num_workers=num_workers,
    )

    # optimizer and weight decay
    optimizer_parameters = get_optimizer_parameters(model, weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=learning_rate)

    # lr scheduler
    num_training_steps = num_epochs * len(train_dataloader)
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
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(num_epochs):
        progress_bar.set_description(f'Epoch {epoch+1}/{num_epochs}')

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


def evaluate_clm(model, tokenizer, dataset, batch_size, num_workers):
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)
    model.to(device)

    dataset.set_format('torch')
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )

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


def _parse_arguments():
    parser = argparse.ArgumentParser(description="Causal LM Pretraining")
    parser.add_argument(
        '--dataset-path', type=str, required=True,
        help='Path of the tokenized dataset.'
    )
    parser.add_argument(
        "--tokenizer-name", type=str, required=True,
        help="Name or path of the tokenizer to load."
    )
    parser.add_argument(
        "--model-name", type=str, required=True,
        help="Name or path of the model config to load."
    )
    parser.add_argument(
        "--save-path", type=str, required=True,
        help="Local path to save pretrained model."
    )
    parser.add_argument(
        "--init-mode", type=utils.InitMode, required=True,
        help="Initializes a random model using base-config or default-config."
    )
    parser.add_argument(
        "--batch-size", type=int, required=True,
        help="Size of the training batch."
    )
    parser.add_argument(
        "--learning-rate", type=float, required=True,
        help="Learning rate value."
    )
    parser.add_argument(
        "--weight-decay", type=float, required=True,
        help="Weight decay value."
    )
    parser.add_argument(
        "--warmup-steps", type=int, required=True,
        help="Warmup steps value."
    )
    parser.add_argument(
        "--num-epochs", type=int, required=True,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--num-workers", type=int, required=True,
        help="Number of workers for loading data in parallel."
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_arguments()

    print('>>> Loading CLM Dataset from Disk:', args.dataset_path)
    dataset = load_from_disk(args.dataset_path)
    assert isinstance(dataset, DatasetDict)

    # print('>>> Creating train and test splits.')
    # dataset = dataset.train_test_split(test_size=0.1, seed=42)
    # print(dataset)

    print('>>> Loading Tokenizer:', args.tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    print(tokenizer)

    print('>>> Loading Model:', args.model_name)
    model = utils.initialize_clm(
        model_name=args.model_name,
        tokenizer=tokenizer,
        init_mode=args.init_mode
    )
    model = torch.compile(model)
    utils.print_model_size(model)
    print(model)

    print('>>> Model Training')
    model = train_clm(
        model=model,
        train_dataset=dataset['train'],
        validation_dataset=dataset['validation'],
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
    )

    print('>>> Model Evaluation')
    loss = evaluate_clm(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset['validation'],
        batch_size=16,
        num_workers=args.num_workers,
    )
    print(f'final loss: {loss}')

    print(f'>>> Saving model to {args.save_path}')
    os.makedirs(args.save_path, exist_ok=True)
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    print('Training completed with success.')
