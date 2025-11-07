from src import tasks
from src import finetuning
from src import hyperparameters
from transformers import AutoTokenizer
from typing import Optional

import argparse
import os
import pprint

def read_arguments():
    """
    Read command-line arguments for model finetuning.

    Returns
    -------
    argparse.Namespace
        An object containing:
            - task_name (str): name of the downstream task (e.g. 'assin-rte')
            - model_name (str): model identifier from Hugging Face hub
            - n_trials (int): number of hyperparameter trials
            - n_epochs (int): number of training epochs
            - seed (Optional[int]): random seed for reproducibility
            - save_dir (Optional[str]): output directory for checkpoints
    """
    parser = argparse.ArgumentParser(
        description="Finetune a pre-trained model from HuggingFace's Hub on a given NLP task."
    )

    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="Name of the downstream task (e.g., 'assin-rte', 'assin-sts')."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name or path (e.g., 'neuralmind/bert-base-portuguese-cased')."
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=3,
        help="Number of hyperparameter trials to run (default: 3)."
    )

    parser.add_argument(
        "--n-epochs",
        type=int,
        default=5,
        help="Number of epochs to train each model (default: 5)."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)."
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save model checkpoints (default: None). If 'None', the model wont be saved."
    )

    args = parser.parse_args()
    return args


def run(
    task_name: str,
    model_name: str,
    n_trials: int = 3,
    n_epochs: int = 5,
    seed: Optional[int]=None,
    save_dir: Optional[str]=None,
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
    tokenizer_map = tasks.get_tokenizer_map(tokenizer, is_sentence_pair)
    text_columns = ['text', 'text_pair'] if is_sentence_pair else ['text']
    dataset = dataset.map(tokenizer_map, batched=True, remove_columns=text_columns)
    print(dataset)

    print('>>> Hyperparameter Search')
    hparams = hyperparameters.search(
        seed=seed,
        model_name=model_name,
        n_trials=n_trials,
        n_epochs=n_epochs,
        dataset=dataset,
        tokenizer=tokenizer,
        task=task,
    )
    print('Selected Hyperparameters')
    pprint.pprint(hparams)

    print('>>> Model Finetuning')
    model = finetuning.finetune(
        seed=seed,
        task=task,
        model_name=model_name,
        dataset=dataset,
        tokenizer=tokenizer,
        n_epochs=n_epochs,
        hyperparameters=hparams,
    )

    if save_dir != None:
        print(f'Saving model and tokenizer at {save_dir}')
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)


if __name__ == '__main__':
    args = read_arguments()
    run(
        task_name=args.task_name,
        model_name=args.model_name,
        n_trials=args.n_trials,
        n_epochs=args.n_epochs,
        save_dir=args.save_dir,
        seed=args.seed,
    )
