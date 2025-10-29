"""
Finetuning script
"""
from typing import Callable, Union, List, Optional

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
    TrainingArguments,
    Trainer,
)

import gc
import pprint
import time
import torch

from downstream_tasks import DownstreamTaskBase
import downstream_tasks as tasks
import hp_search


def get_tokenizer_map(
    tokenizer: PreTrainedTokenizerBase,
    is_sentence_pair : bool
) -> Callable[[Dataset], BatchEncoding]:
    '''Defines the tokenization mapper'''
    if is_sentence_pair:
        return lambda examples: tokenizer(
            text=examples['text'],
            text_pair=examples['text_pair'],
            padding="max_length",
            truncation=True,
        )

    return lambda examples: tokenizer(
        text=examples['text'],
        padding="max_length",
        truncation=True,
    )


def finetune(
    model_name: str,
    train_epochs: int,
    hyperparameters: dict,
    dataset: DatasetDict,
    task: DownstreamTaskBase,
    tokenizer: PreTrainedTokenizerBase,
    seeds: List[int]=[42],
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

    for seed in seeds:
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


def run(
    task_name: str,
    model_name: str,
    hub_namespace: str,
    n_trials: int = 3,
    train_epochs: int = 5,
    push_to_hub: bool=True,
    seeds: Union[int, List[int]]=42,
):
    print(f'>>> Running Task: {task_name}')
    task = tasks.load_task(task_name)

    print('>>> Loading Dataset')
    dataset = task.load_dataset()
    print(dataset)

    print('>>> Dataset Tokenization')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(tokenizer)

    is_sentence_pair = 'text_pair' in dataset['train'].features.keys()
    tokenizer_map = get_tokenizer_map(tokenizer, is_sentence_pair)
    dataset = dataset.map(tokenizer_map, batch_size=True)
    print(dataset)

    print('>>> Hyperparameter Search')
    hyperparameters = hp_search.optuna_search(
        model_name=model_name,
        n_trials=n_trials,
        train_epochs=train_epochs,
        dataset=dataset,
        tokenizer=tokenizer,
        task=task,
    )
    print('Selected Hyperparameters')
    print(hyperparameters)

    print('>>> Model Finetuning')
    seeds = [seeds] if isinstance(seeds, int) else seeds
    finetune(
        seeds=seeds,
        task=task,
        model_name=model_name,
        dataset=dataset,
        tokenizer=tokenizer,
        train_epochs=train_epochs,
        hyperparameters=hyperparameters,
        push_to_hub=push_to_hub,
        hub_namespace=hub_namespace,
    )


if __name__ == '__main__':
    run(
        task_name='assin-rte',
        model_name='google-bert/bert-base-uncased',
        hub_namespace='guilhermelmello',
        push_to_hub=True,
        n_trials=3,
        train_epochs=5,
        seeds=[42],
    )
