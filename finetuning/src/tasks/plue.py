from .task_base import TaskBase
from datasets import load_dataset
from transformers import EvalPrediction
from typing import Optional

from transformers import AutoModelForSequenceClassification
from transformers import PreTrainedModel

import evaluate
import re


class RecognizingTextualEntailment(TaskBase):
    '''PLUE dataset for Recognizing Textual Entailment (RTE).'''

    def __init__(self) -> None:
        self._f1_metric = evaluate.load('f1')
        self._acc_metric = evaluate.load('accuracy')

    @property
    def name(self) -> str:
        return 'plue-rte'

    @TaskBase.filtered_columns
    def load_dataset(self, split : Optional[str] = None):
        dataset = load_dataset('dlb/plue', 'RTE', split=split)

        # fix example
        dataset['train'] = dataset['train'].map(self._fix_train_example)
        dataset['test'] = dataset['test'].map(self._fix_test_example, batched=True, batch_size=1)

        dataset = dataset.class_encode_column('label')
        dataset['validation'] = dataset.pop('dev')

        # renamed columns
        return dataset.rename_columns({
            'sentence1': 'text',
            'sentence2': 'text_pair',
        })

    def load_test_dataset(self):
        dataset = load_dataset('dlb/plue', 'RTE')

        # fix example
        dataset['train'] = dataset['train'].map(self._fix_train_example)
        dataset['test'] = dataset['test'].map(self._fix_test_example, batched=True, batch_size=1)
        dataset = dataset.class_encode_column('label')

        # renamed columns
        return dataset['train'].features, dataset['test'].rename_columns({
            'sentence1': 'text',
            'sentence2': 'text_pair',
        })

    def _fix_train_example(self, example):
        if example['index'] == 2164:
            sentences = example['sentence1'].split('\t')
            return {
                'sentence1': sentences[0],
                'sentence2': sentences[1],
                'label': example['sentence2']
            }
        else:
            return example

    def _fix_test_example(self, example):
        if example['index'][0] == 1117:
            return self._fix_test_1117(example)
        else:
            return example

    def _fix_test_1117(self, example):
        tsv_str = str(example['index'][0]) + '\t' + example['sentence1'][0] + '\t' + example['sentence2'][0]
        splits = re.split('\n|\t', tsv_str)

        index = []
        sentence1 = []
        sentence2 = []
        label = []

        it = iter(splits)
        while True:
            try:
                index.append(int(next(it)))
                sentence1.append(next(it))
                sentence2.append(next(it))
                label.append(None)
            except StopIteration:
                break

        return {
            'index': index,
            'sentence1': sentence1,
            'sentence2': sentence2,
            'label': label,
        }

    def compute_metrics(self, eval_pred: Optional[EvalPrediction]=None) -> dict[str, float] | None:
        if eval_pred is None:
            f1 = self._f1_metric.compute(average='macro') or {}
            acc = self._acc_metric.compute() or {}
            return {**f1, **acc}

        logits, references = eval_pred
        predictions = logits.argmax(axis=-1) # type: ignore

        self._f1_metric.add_batch(
            predictions=predictions,
            references=references,
        )
        self._acc_metric.add_batch(
            predictions=predictions,
            references=references,
        )

    @property
    def objective_metric_name(self) -> str:
        return 'accuracy'
    
    @property
    def is_maximization(self) -> bool:
        return True

    def load_pretrained_model(self, model_name: str) -> PreTrainedModel:
        return AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name,
            num_labels=2,
        )


class WinogradNLI(TaskBase):
    '''PLUE dataset for Winograd Natural Language Inference (WNLI).'''

    def __init__(self) -> None:
        self._f1_metric = evaluate.load('f1')
        self._acc_metric = evaluate.load('accuracy')

    @property
    def name(self) -> str:
        return 'plue-wnli'

    @TaskBase.filtered_columns
    def load_dataset(self, split : Optional[str] = None):
        dataset = load_dataset('dlb/plue', 'WNLI', split=split)

        dataset = dataset.class_encode_column('label')
        dataset['validation'] = dataset.pop('dev')

        # renamed columns
        return dataset.rename_columns({
            'sentence1': 'text',
            'sentence2': 'text_pair',
        })

    def load_test_dataset(self):
        dataset = load_dataset('dlb/plue', 'WNLI')
        dataset = dataset.class_encode_column('label')

        # renamed columns
        return dataset['train'].features, dataset['test'].rename_columns({
            'sentence1': 'text',
            'sentence2': 'text_pair',
        })

    def compute_metrics(self, eval_pred: Optional[EvalPrediction]=None) -> dict[str, float] | None:
        if eval_pred is None:
            f1 = self._f1_metric.compute(average='macro') or {}
            acc = self._acc_metric.compute() or {}
            return {**f1, **acc}

        logits, references = eval_pred
        predictions = logits.argmax(axis=-1) # type: ignore

        self._f1_metric.add_batch(
            predictions=predictions,
            references=references,
        )
        self._acc_metric.add_batch(
            predictions=predictions,
            references=references,
        )

    @property
    def objective_metric_name(self) -> str:
        return 'accuracy'
    
    @property
    def is_maximization(self) -> bool:
        return True

    def load_pretrained_model(self, model_name: str) -> PreTrainedModel:
        return AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name,
            num_labels=2,
        )

