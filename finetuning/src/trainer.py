import torch

from copy import deepcopy
from datasets import ClassLabel, Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm.auto import tqdm
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    get_scheduler,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from typing import Optional
from .uitls import ComputeMetricsCallback


def train(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    batch_size: int,
    learning_rate: float,
    n_epochs: int,
    compute_metrics: Optional[ComputeMetricsCallback] = None,
    objective_metric_name: str = 'val_loss',
    greater_is_better: bool = False,
):
    # data preparation
    train_dataset.set_format('torch')
    validation_dataset.set_format('torch')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=True) # type: ignore
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, collate_fn=data_collator) # type: ignore

    # optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = n_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name='linear',
        num_warmup_steps=0,
        optimizer=optimizer,
        num_training_steps=num_training_steps,
    )

    # move model to GPU, if available
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)
    model.to(device) # type: ignore

    # track best model
    best_model_metric = float('-inf') if greater_is_better else float('inf')
    best_model_state = {}

    # target
    label_feature = train_dataset.features['label']
    is_classification = isinstance(label_feature, ClassLabel)
    if is_classification:
        num_labels = train_dataset.features['label'].num_classes
    else:
        num_labels = 1

    # loss function for logging
    # using loss from models output may result in different values when
    # avereaging mini batches losses using different batch sizes for training
    # and evaluating. Using 'sum' as loss reduction is more stable.
    # Examples:
    # 'mean' -> total_loss / len(dataloader) : unstable
    # 'sum' -> total_loss / len(dataloader.dataset) : more stable
    loss_fn = CrossEntropyLoss(reduction='sum') if num_labels > 1 else MSELoss(reduction='sum')

    # training loop
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(n_epochs):
        progress_bar.set_description(f'Epoch {epoch+1}/{n_epochs}')

        # start training mode
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            del batch

        # start evaluation mode
        model.eval()

        eval_steps = len(train_dataloader) + len(validation_dataloader)
        eval_progress_bar = tqdm(range(eval_steps), leave=False)

        # training loss
        total_train_loss = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            total_train_loss += loss_fn(outputs.logits, batch['labels']).item()
            eval_progress_bar.update(1)
            del batch

        # evaluation loop (validation)
        total_validation_loss = 0
        for batch in validation_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            total_validation_loss += loss_fn(outputs.logits, batch['labels']).item()

            if compute_metrics is not None:
                compute_metrics(EvalPrediction(
                    predictions=outputs.logits,
                    label_ids=batch['labels'],
                ))
            eval_progress_bar.update(1)
        eval_progress_bar.close()

        # report metrics
        val_metrics = compute_metrics() if compute_metrics is not None else {}
        val_metrics = val_metrics or {}
        val_metrics = {f'val_{k}': v for k, v in val_metrics.items()}

        train_loss = total_train_loss / len(train_dataloader.dataset) # type: ignore
        validation_loss = total_validation_loss / len(validation_dataloader.dataset) # type: ignore
        epoch_results = {
            'epoch': epoch + 1,
            'loss': train_loss,
            'val_loss': validation_loss,
            **val_metrics,
        }
        print(epoch_results)

        # update best model
        epoch_metric = epoch_results[f'val_{objective_metric_name}']
        update_best_model = (
            (greater_is_better and epoch_metric > best_model_metric) or
            (not greater_is_better and epoch_metric < best_model_metric)
        )
        if update_best_model:
            best_model_state = deepcopy(model.state_dict())
            best_model_metric = epoch_metric

    model.load_state_dict(best_model_state)
    del best_model_state

    return model


def evaluate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    batch_size: int = 16,
    compute_metrics: Optional[ComputeMetricsCallback] = None,
):
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)
    model.to(device) # type: ignore

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator) # type: ignore

    # target
    label_feature = dataset.features['label']
    is_classification = isinstance(label_feature, ClassLabel)
    if is_classification:
        num_labels = dataset.features['label'].num_classes
    else:
        num_labels = 1

    loss_fn = CrossEntropyLoss(reduction='sum') if num_labels > 1 else MSELoss(reduction='sum')

    total_loss = 0
    eval_progress_bar = tqdm(range(len(dataloader)), leave=False)

    model.eval()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        total_loss += loss_fn(outputs.logits, batch['labels']).item()

        if compute_metrics is not None:
            compute_metrics(EvalPrediction(
                predictions=outputs.logits,
                label_ids=batch['labels'],
            ))
        eval_progress_bar.update(1)
    eval_progress_bar.close()

    loss = total_loss / len(dataloader.dataset) # type: ignore
    metrics = compute_metrics() if compute_metrics is not None else {}
    metrics = metrics or {}

    return {
        'loss': loss,
        **metrics,
    }