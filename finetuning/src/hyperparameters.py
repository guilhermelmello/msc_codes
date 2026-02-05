from . import trainer
from .tasks import TaskBase
from datasets import ClassLabel, DatasetDict
from optuna import Trial
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedTokenizerBase,
)
from typing import List, Optional

import gc
import optuna
import torch


def get_model_initializer(model_name: str, num_labels: int):
    '''Returns a function that creates new model instances.'''
    return lambda _: AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_name,
        num_labels=num_labels,
    )


def search(
    model_name: str,
    num_trials: int,
    num_epochs: int,
    lr_values: List[float],
    batch_size_values: List[int],
    task: TaskBase,
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    seed: Optional[int]=None
) -> dict[str, float]:
    '''Hyperparameter Search with Optuna.

    Optuna's `Trainer.hyperparameter_search` only returns the last epoch metric
    as trial objective value. This behavior limits evaluation when model overfits
    and last epoch results in worst metrics. This implementation allows a custom
    objective implementation that loads the best model at the end of the search
    and reports the evaluation metrics of best trial.
    '''
    if seed is not None:
        torch.manual_seed(seed)

    def optuna_objective(trial: Trial):
        print(f'=== Trial {trial.number}', '=' * 40)

        model = None
        try:
            # defines search space
            lr = trial.suggest_categorical('learning_rate', lr_values)
            batch_size = trial.suggest_categorical('batch_size', batch_size_values)

            print(f'Running with hyperparameters: {{')
            print(f'\tlearning_rate: {lr}')
            print(f'\tbatch_size: {batch_size}')
            print(f'}}')

            # target
            label_feature = dataset['train'].features['label']
            is_classification = isinstance(label_feature, ClassLabel)
            if is_classification:
                num_labels = dataset['train'].features['label'].num_classes
            else:
                num_labels = 1

            # model initialization
            model = task.load_pretrained_model(
                model_name=model_name,
                num_labels=num_labels,
            )

            if model_name == 'Qwen/Qwen3-0.6B':
                model.config.pad_token_id = tokenizer.pad_token_id # fix

            model = trainer.train(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset['train'],
                validation_dataset=dataset['validation'],
                num_epochs=num_epochs,
                # best model definition
                compute_metrics=task.compute_metrics,
                objective_metric_name=task.objective_metric_name,
                greater_is_better=task.is_maximization,
                # hyperparameters
                batch_size=batch_size,
                learning_rate=lr,
            )

            results = trainer.evaluate(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset['validation'],
                batch_size=batch_size,
                compute_metrics=task.compute_metrics
            )
            return results[task.objective_metric_name]

        finally:
            del model
            torch.cuda.empty_cache()
            gc.collect()

    direction = 'maximize' if task.is_maximization else 'minimize'
    study = optuna.create_study(direction=direction)
    study.optimize(
        optuna_objective,
        n_trials=num_trials,
        gc_after_trial=True,
    )
    return study.best_params
