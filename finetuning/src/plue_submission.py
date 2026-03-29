from enum import Enum
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding

import argparse
import os
import torch
import tasks


class PlueTask(Enum):
    RTE = 'rte'
    WNLI = 'wnli'
    MRPC = 'mrpc'
    STSB = 'sts-b'


def read_arguments():
    """
    Read command-line arguments to create a PLUE Benchmark's submission file.
    """
    parser = argparse.ArgumentParser(
        description="Finetune a pre-trained model from HuggingFace's Hub on a given NLP task."
    )
    parser.add_argument(
        "--task",
        type=PlueTask,
        required=True,
        help="Name of the PLUE task."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name or path."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory to save the submision file."
    )
    
    args = parser.parse_args()
    return args


def load_plue_task(task: PlueTask) -> tasks.TaskBase:
    if task == PlueTask.MRPC:
        raise NotImplementedError('MRPC task is not implemented')
    if task == PlueTask.RTE:
        return tasks.plue.RecognizingTextualEntailment()
    if task == PlueTask.STSB:
        raise NotImplementedError('STS-B task is not implemented')
    if task == PlueTask.WNLI:
        return tasks.plue.WinogradNLI()


def get_file_name(task: PlueTask) -> int:
    if task == PlueTask.MRPC:
        return 'MRPC.tsv'
    if task == PlueTask.RTE:
        return 'RTE.tsv'
    if task == PlueTask.STSB:
        return 'STS-B.tsv'
    if task == PlueTask.WNLI:
        return 'WNLI.tsv'


def generate_predictions(model, tokenizer, dataset, batch_size=16):
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)
    model.to(device)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator) 

    model.eval()
    predictions = []

    progress_bar = tqdm(range(len(dataloader)), leave=False)

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        preds = torch.argmax(outputs.logits, dim=-1)
        predictions.append(preds.cpu().numpy())
    
        progress_bar.update(1)
    progress_bar.close()

    return predictions


def run(plueTask: PlueTask, model_name: str, save_dir: str):
    # carregar os dados da task (usar o que já tem implementado)
    print('>>> Loading Task')
    task = load_plue_task(plueTask)
    features, test_dataset = task.load_test_dataset()
    print(features)
    print(test_dataset)

    # carregar modelo e tokenizer
    print('>>> Loading Model')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = task.load_pretrained_model(model_name)

    # aplicar tokenizer nos dados
    print('>>> Dataset tokenization')
    is_sentence_pair = 'text_pair' in test_dataset.features.keys()
    tokenizer_map = tasks.get_tokenizer_map(tokenizer, is_sentence_pair)
    remove_columns = ['index', 'text', 'label']
    if is_sentence_pair:
        remove_columns.append('text_pair')
    tokenized_dataset = test_dataset.map(
        tokenizer_map,
        batched=True,
        remove_columns=remove_columns)
    print(tokenized_dataset)

    # fazer predições
    print('>>> Generating Predictions')
    predictions = generate_predictions(
        model, tokenizer, tokenized_dataset
    )

    # converte para labels
    print('>>> Converting Labels')
    predicted_labels = [
        features['label'].int2str(pred)
        for preds in predictions
        for pred in preds.tolist()
    ]

    # gerar arquivo
    print('>>> Creating Submition file')
    os.makedirs(save_dir, exist_ok=True)
    fpath = os.path.join(save_dir, get_file_name(plueTask))
    print(f'Saving predictions at {fpath}')
    with open(fpath, 'w') as f:
        f.write('index\tprediction')
        for idx, label in zip(test_dataset['index'], predicted_labels):
            f.write(f'\n{idx}\t{label}')

    print('DONE')


if __name__ == "__main__":
    args = read_arguments()

    run(
        args.task,
        args.model_name,
        args.save_dir,
    )

    