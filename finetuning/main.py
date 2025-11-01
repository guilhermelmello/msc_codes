from typing import Union, List
from transformers import AutoTokenizer

from src import downstream_tasks as tasks
from src import finetuning
from src import hp_search


def run(
    task_name: str,
    model_name: str,
    hub_namespace: str,
    n_trials: int = 3,
    train_epochs: int = 5,
    push_to_hub: bool=True,
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
    seed = [seed] if isinstance(seed, int) else seed
    for seed_i in seed:
        finetuning.finetune(
            seed=seed_i,
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
        seed=42,
    )
