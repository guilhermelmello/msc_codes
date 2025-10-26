from typing import Union, List

from .downstream_tasks import DownstreamTaskBase
from datasets import DatasetDict
from optuna import Trial
from transformers import AutoModelForSequenceClassification
from transformers import EvalPrediction
from transformers import PreTrainedTokenizerBase
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import BestRun

import gc
import optuna
import time
import torch


def get_model_initializer(model_name: str, num_labels: int):
    '''Returns a function that creates new model instances.'''
    return lambda _: AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_name,
        num_labels=num_labels,
    )


def search(
    model_name: str,
    n_trials: int,
    train_epochs: int,
    task: DownstreamTaskBase,
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
) -> Union[BestRun, List[BestRun]]:
    '''Hyperparameter Search'''
    num_labels = dataset['train'].features['label'].num_classes
    model_init = get_model_initializer(model_name, num_labels)

    training_args = TrainingArguments(
        num_train_epochs=train_epochs,
        logging_strategy='epoch',
        eval_strategy='epoch',
        push_to_hub=False,
        report_to="none",
    )

    trainer = Trainer(
        model=None,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        compute_metrics=task.compute_metrics,
        processing_class=tokenizer,
        model_init=model_init,
    )

    def optuna_hp_space(trial: Trial) -> dict[str, float]:
        return {
            'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True),
            'per_device_train_batch_size': trial.suggest_categorical('per_device_train_batch_size', [16, 32, 64]),
        }

    def compute_objective(metrics: dict[str, float]) -> float:
        return metrics[f'eval_{task.objective_metric_name}']

    best_trials = trainer.hyperparameter_search(
        direction='maximize' if task.is_maximization else 'minimize',
        compute_objective=compute_objective,
        backend='optuna',
        hp_space=optuna_hp_space,
        n_trials=n_trials,
        gc_after_trial=True
    )

    return best_trials


def optuna_search(
    model_name: str,
    n_trials: int,
    train_epochs: int,
    task: DownstreamTaskBase,
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
) -> Union[dict[str, float], None]:
    '''Hyperparameter Search with Optuna.

    `Trainer.hyperparameter_search` only returns the last epoch metric as
    trial objective value. This behavior limits evaluation when model overfits
    and last epoch results in worst metrics. This implementation allows a
    custom objective implementation that loads the best model at the end of the
    trial and report its evaluation metrics for the trial.
    '''
    def optuna_objective(trial: Trial):
        model = None
        trainer = None

        try:
            # defines search space
            lr = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
            batch_size = trial.suggest_categorical('per_device_train_batch_size', [16, 32, 64])

            # model initialization
            num_labels = dataset['train'].features['label'].num_classes
            model = task.load_pretrained_model(
                model_name=model_name,
                num_labels=num_labels,
            )

            training_args = TrainingArguments(
                num_train_epochs=train_epochs,
                logging_strategy='epoch',
                eval_strategy='epoch',
                push_to_hub=False,
                report_to="none",
                # best model definition
                save_strategy='epoch',
                load_best_model_at_end=True,
                metric_for_best_model=task.objective_metric_name,
                greater_is_better=task.is_maximization,
                # hyper parameters
                per_device_train_batch_size=batch_size,
                learning_rate=lr,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset['train'],
                eval_dataset=dataset['validation'],
                compute_metrics=task.compute_metrics,
                processing_class=tokenizer,
            )

            trainer.train()
            results = trainer.evaluate()
            return results[f'eval_{task.objective_metric_name}']
        finally:
            del model
            del trainer
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(2)

    direction = 'maximize' if task.is_maximization else 'minimize'
    study = optuna.create_study(direction=direction)
    study.optimize(
        optuna_objective,
        n_trials=n_trials,
        gc_after_trial=True,
    )
    return study.best_params
