"""
Finetuning script
"""
from . import trainer
from .tasks import TaskBase
from datasets import DatasetDict
from transformers import PreTrainedTokenizerBase
from typing import Optional

import gc
import pprint
import torch


def finetune(
    model_name: str,
    n_epochs: int,
    hyperparameters: dict,
    dataset: DatasetDict,
    task: TaskBase,
    tokenizer: PreTrainedTokenizerBase,
    seed: Optional[int]=None,
):
    '''Model Finetuning.'''
    if seed is not None:
        torch.manual_seed(seed)

    # hyperparameters
    batch_size = hyperparameters['batch_size']
    lr = hyperparameters['learning_rate']

    try:
        # load pretrained model
        num_labels = dataset['train'].features['label'].num_classes
        model = task.load_pretrained_model(
            model_name=model_name,
            num_labels=num_labels
        )

        model = trainer.train(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset['train'],
            validation_dataset=dataset['validation'],
            n_epochs=n_epochs,
            # best model definition
            compute_metrics=task.compute_metrics,
            objective_metric_name=task.objective_metric_name,
            greater_is_better=task.is_maximization,
            # hyperparameters
            batch_size=batch_size,
            learning_rate=lr,
        )

        print('Best Model Results')
        results = trainer.evaluate(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset['validation'],
            batch_size=batch_size,
            compute_metrics=task.compute_metrics
        )
        pprint.pprint(results)
        return model
    finally:
        torch.cuda.empty_cache()
        gc.collect()
