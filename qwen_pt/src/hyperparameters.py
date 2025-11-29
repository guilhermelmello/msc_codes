from datasets import Dataset, DatasetDict, load_from_disk
from optuna import Trial
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerBase
from typing import Optional

import argparse
import gc
import optuna
import pprint
import torch

import trainer
import utils


def search(
    model_name: str,
    n_trials: int,
    n_epochs: int,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    weight_decay: float,
    warmup_steps: int,
    seed: Optional[int]=None
) -> dict[str, float]:
    '''Hyperparameter Search with Optuna.'''
    if seed is not None:
        torch.manual_seed(seed)

    def optuna_objective(trial: Trial):
        print(f'=== Trial {trial.number}', '=' * 40)

        model = None
        try:
            # defines search space
            lr = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
            batch_size = trial.suggest_categorical('batch_size', [5])

            print(f'Running with hyperparameters: {{')
            print(f'\tlearning_rate: {lr}')
            print(f'\tbatch_size: {batch_size}')
            print(f'}}')

            model = utils.initialize_clm_from_config(model_name, tokenizer)
            model = trainer.train_clm(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                learning_rate=lr,
                batch_size=batch_size,
                weight_decay=weight_decay,
                warmup_steps=warmup_steps,
                n_epochs=n_epochs,
            )

            loss = trainer.evaluate_clm(
                model=model,
                tokenizer=tokenizer,
                dataset=validation_dataset,
                batch_size=batch_size,
            )
            print(f'Trial {trial.number} finished with value: {loss}')
            return loss

        finally:
            utils.print_gpu_usage()
            del model
            torch.cuda.empty_cache()
            gc.collect()

    study = optuna.create_study(direction='minimize')
    study.optimize(
        optuna_objective,
        n_trials=n_trials,
        gc_after_trial=True,
    )
    return study.best_params


def _parse_arguments():
    parser = argparse.ArgumentParser(description="Hyperparameter Search")
    parser.add_argument(
        "--model-name", type=str, required=True,
        help="Name or path of the model."
    )
    parser.add_argument(
        "--tokenizer-name", type=str, required=True,
        help="Name or path of the tokenizer."
    )
    parser.add_argument(
        "--dataset-path", type=str, required=True,
        help="Path of the CLM ready dataset (must return a Dataset, not a DatasetDict)."
    )
    # parser.add_argument(
    #     "--max-seq-length", type=int, required=True,
    #     help="Maximum sequence length for training."
    # )
    # parser.add_argument(
    #     "--dataset-batch-size", type=int, required=True,
    #     help="Batch size for data processing."
    # )
    # parser.add_argument(
    #     "--dataset-num-proc", type=int, required=True,
    #     help="Number of process for parallel data processing."
    # )
    parser.add_argument(
        "--weight-decay", type=float, required=True,
        help="Weight Decay value."
    )
    parser.add_argument(
        "--warmup-steps", type=int, required=True,
        help="Number of steps to increase the learning rate."
    )
    parser.add_argument(
        "--n-trials", type=int, required=True,
        help="Number of trials."
    )
    parser.add_argument(
        "--n-epochs", type=int, required=True,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (default: None)."
    )
    return parser.parse_args()


if __name__ == '__main__':
    print('>>> HYPERPARAMETER SEARCH')
    args = _parse_arguments()

    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)
    print('Training Device:', device)

    print('>>> Loading CLM Dataset from Disk')
    dataset = load_from_disk(args.dataset_path)
    assert isinstance(dataset, Dataset)
    
    print('>>> Creating train and test splits for hyperparameter search.')
    dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
    print(dataset)

    print('>>> Loading Tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    print(tokenizer)

    print('>>> Hyperparameter Search')
    hparams = search(
        model_name=args.model_name,
        n_trials=args.n_trials,
        n_epochs=args.n_epochs,
        train_dataset=dataset['train'],
        validation_dataset=dataset['test'],
        tokenizer=tokenizer,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        seed=42,
    )
    print('\n\n\n', '=' * 40)
    print('Selected Hyperparameters')
    pprint.pprint(hparams)
    print('=' * 40)
