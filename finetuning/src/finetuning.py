"""
Finetuning script
"""
from typing import Optional

from datasets import DatasetDict
from huggingface_hub import HfApi
from transformers import (
    PreTrainedTokenizerBase,
    TrainingArguments,
    Trainer,
)

import gc
import pprint
import time
import torch

from .downstream_tasks import DownstreamTaskBase


def finetune(
    model_name: str,
    train_epochs: int,
    hyperparameters: dict,
    dataset: DatasetDict,
    task: DownstreamTaskBase,
    tokenizer: PreTrainedTokenizerBase,
    seed: int=42,
    hub_namespace: Optional[str]=None,
    push_to_hub: bool=False,
):
    '''Model Finetuning.'''
    # hyperparameters
    batch_size = hyperparameters['per_device_train_batch_size']
    lr = hyperparameters['learning_rate']

    api = None
    collection = None

    if push_to_hub:
        print('>>> Loading HuggingFace Hub Collection')
        api = HfApi()
        collection = api.create_collection(
            title=task.name,
            namespace=hub_namespace,
            exists_ok=True,
            private=True,
        )
        print(f'Model Collection: {collection.slug}')

    model = None
    trainer = None
    try:
        mname = model_name.split('/')[-1]
        model_id = f"{hub_namespace}/{mname}-{task.name}-{seed}"
        print(f'Training {model_id}')

        # Pretrained Model
        num_labels = dataset['train'].features['label'].num_classes
        model = task.load_pretrained_model(
            model_name=model_name,
            num_labels=num_labels
        )

        # Training Arguments
        train_args = TrainingArguments(
            seed=seed,
            num_train_epochs=train_epochs,
            # hyperparameters
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            # report and saving
            report_to="none",
            push_to_hub=push_to_hub,
            hub_private_repo=True,
            hub_model_id=model_id,
            load_best_model_at_end=True,
            metric_for_best_model=task.objective_metric_name,
            greater_is_better=task.is_maximization,
            logging_strategy='epoch',
            eval_strategy='epoch',
            save_strategy='epoch',
        )

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            compute_metrics=task.compute_metrics,
            processing_class=tokenizer,
        )
        trainer.train()

        print('Final Results')
        results = trainer.evaluate()
        pprint.pprint(results)

        if push_to_hub and api is not None and collection is not None:
            api.add_collection_item(
                collection.slug,
                item_id=model_id,
                item_type="model",
                exists_ok=True,
            )
    finally:
        del model
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)
