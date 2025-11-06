from typing import Union, List
from transformers import AutoTokenizer

from src import downstream_tasks as tasks
from src import finetuning
from src import hyperparameters


def run(
    task_name: str,
    model_name: str,
    n_trials: int = 3,
    n_epochs: int = 5,
    seed: Union[int, List[int]]=42,
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
        model_name=model_name,
        n_trials=n_trials,
        n_epochs=n_epochs,
        dataset=dataset,
        tokenizer=tokenizer,
        task=task,
    )
    print('Selected Hyperparameters')
    print(hparams)

    print('>>> Model Finetuning')
    seed = [seed] if isinstance(seed, int) else seed
    for seed_i in seed:
        finetuning.finetune(
            task=task,
            model_name=model_name,
            dataset=dataset,
            tokenizer=tokenizer,
            n_epochs=n_epochs,
            hyperparameters=hparams,
        )


if __name__ == '__main__':
    run(
        task_name='assin-rte',
        model_name='google-bert/bert-base-uncased',
        n_trials=3,
        n_epochs=5,
        seed=42,
    )
