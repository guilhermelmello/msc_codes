from .task_base import TaskBase
from datasets import load_dataset, DatasetDict, ClassLabel
from transformers import EvalPrediction
from typing import Optional

from transformers import AutoModelForSequenceClassification
from transformers import PreTrainedModel

import evaluate


class OffensiveLanguageDetection(TaskBase):
    '''HateBR dataset for Offensive language detection.'''

    def __init__(self) -> None:
        self._f1_metric = evaluate.load('f1')

    @property
    def name(self) -> str:
        return 'hatebr-offensive'

    @TaskBase.filtered_columns
    def load_dataset(self, split : Optional[str] = None):
        dataset = load_dataset(
            'ruanchaves/hatebr',
            split=split,
            trust_remote_code=True
        )

        if isinstance(dataset, DatasetDict):
            for split in dataset:
                dataset[split].map(lambda example: {'offensive_language': int(example['offensive_language'])})
                dataset[split] = dataset[split].cast_column('offensive_language', ClassLabel(names=['False', 'True']))
        else:
            dataset.map(lambda example: {'offensive_language': int(example['offensive_language'])})
            dataset = dataset.cast_column('offensive_language', ClassLabel(names=['False', 'True']))

        # renamed columns
        return dataset.rename_columns({
            'instagram_comments': 'text',
            'offensive_language': 'label',
        })

    def compute_metrics(self, eval_pred: Optional[EvalPrediction]=None) -> dict[str, float] | None:
        if eval_pred is None:
            return self._f1_metric.compute(average='macro') or {}

        logits, references = eval_pred
        predictions = logits.argmax(axis=-1)

        self._f1_metric.add_batch(
            predictions=predictions,
            references=references,
        )

    @property
    def objective_metric_name(self) -> str:
        return 'f1'
    
    @property
    def is_maximization(self) -> bool:
        return True

    def load_pretrained_model(self, model_name: str) -> PreTrainedModel:
        return AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name,
            num_labels=2,
        )
