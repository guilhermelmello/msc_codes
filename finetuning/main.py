from src import tasks
from src import trainer
from src import finetuning
from src import hyperparameters
from transformers import AutoTokenizer
from typing import List, Optional

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
            - num_hp_trials (int): number of hyperparameter trials
            - num_hp_epochs (int): number of epochs for each hyperparameter trials
            - num_training_epochs (int): number of training epochs
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
        "--num-hp-trials",
        type=int,
        default=3,
        help="Number of hyperparameter trials to run (default: 3)."
    )
    parser.add_argument(
        "--num-hp-epochs",
        type=int,
        default=5,
        help="Number of epochs for each hyperparameter trial (default: 5)."
    )
    parser.add_argument(
        "--num-training-epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)."
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
    hp_lr_values: List[float],
    hp_bs_values: List[int],
    num_hp_trials: int = 3,
    num_hp_epochs: int = 5,
    num_training_epochs: int = 5,
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
        num_trials=num_hp_trials,
        num_epochs=num_hp_epochs,
        lr_values=hp_lr_values,
        batch_size_values=hp_bs_values,
        dataset=dataset,
        tokenizer=tokenizer,
        task=task,
    )

    print('=== Results ========================================')
    print('>>> Selected Hyperparameters')
    pprint.pprint(hparams)

    print('>>> Model Finetuning')
    model = finetuning.finetune(
        seed=seed,
        task=task,
        model_name=model_name,
        dataset=dataset,
        tokenizer=tokenizer,
        num_epochs=num_training_epochs,
        hyperparameters=hparams,
    )

    if save_dir != None:
        print(f'Saving model and tokenizer at {save_dir}')
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    print('>>> Results on Test Dataset')
    results = trainer.evaluate(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset['test'],
        compute_metrics=task.compute_metrics,
    )
    pprint.pprint(results)


if __name__ == '__main__':
    args = read_arguments()
    run(
        task_name=args.task_name,
        model_name=args.model_name,
        num_hp_trials=args.num_hp_trials,
        num_hp_epochs=args.num_hp_epochs,
        hp_lr_values=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        hp_bs_values=[8, 16],
        num_training_epochs=args.num_training_epochs,
        save_dir=args.save_dir,
        seed=args.seed,
    )
