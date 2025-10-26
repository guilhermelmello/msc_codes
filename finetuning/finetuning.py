"""
Finetuning script
"""
from typing import Callable
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase, BatchEncoding

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


def run():
    # ARGUMENTS
    task_name = 'assin_rte'
    model_name = 'google-bert/bert-base-uncased'
    n_trials = 3
    train_epochs = 5
    # END ARGUMENTS

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
    best_trial = hp_search.optuna_search(
        model_name=model_name,
        n_trials=n_trials,
        train_epochs=train_epochs,
        dataset=dataset,
        tokenizer=tokenizer,
        task=task,
    )

    print(best_trial)

    # TODO:
    # finetuning()
    # save_model()



if __name__ == "__main__":
    run()
