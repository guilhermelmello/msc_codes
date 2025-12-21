from transformers import EvalPrediction
from .task_base import TaskBase
from typing import Optional
from datasets import load_dataset

from transformers import AutoModelForSequenceClassification
from transformers import PreTrainedModel
from transformers.models.auto.auto_factory import _BaseAutoModelClass

import evaluate


class RecognisingTextualEntailment(TaskBase):
    '''Assin 2 dataset for Recognising Textual Entailment (RTE).'''

    def __init__(self):
        self._f1_metric = evaluate.load("f1")
        self._acc_metric = evaluate.load("accuracy")

    @property
    def name(self) -> str:
        return 'assin-rte'

    @TaskBase.filtered_columns
    def load_dataset(self, split : Optional[str] = None):
        dataset = load_dataset('nilc-nlp/assin2', 'default', split=split)

        # renamed columns
        return dataset.rename_columns({
            'premise': 'text',
            'hypothesis': 'text_pair',
            'entailment_judgment': 'label',
        })
    
    def compute_metrics(self, eval_pred: Optional[EvalPrediction]=None) -> dict[str, float] | None:
        if eval_pred is None:
            f1 = self._f1_metric.compute(average='binary') or {}
            acc = self._acc_metric.compute() or {}
            return {**f1, **acc}
        
        logits, references = eval_pred
        predictions = logits.argmax(axis=-1) # type: ignore

        self._f1_metric.add_batch(
            predictions=predictions,
            references=references
        )
        self._acc_metric.add_batch(
            predictions=predictions,
            references=references
        )
    
    @property
    def objective_metric_name(self) -> str:
        return 'f1'
    
    @property
    def is_maximization(self) -> bool:
        return True

    def load_pretrained_model(self, model_name: str, num_labels: int) -> _BaseAutoModelClass:
        return AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name,
            num_labels=num_labels,
        )


class SemanticTextualSimilarity(TaskBase):
    '''Assin 2 dataset for Semantic Textual Similarity (STS).'''

    def __init__(self) -> None:
        self._metrics = evaluate.combine(['mse', 'pearsonr'])

    @property
    def name(self) -> str:
        return 'assin2-sts'

    @TaskBase.filtered_columns
    def load_dataset(self, name: str = 'default', split : Optional[str] = None):
        dataset = load_dataset('nilc-nlp/assin2', name, split=split)

        # renamed columns
        return dataset.rename_columns({
            'premise': 'text',
            'hypothesis': 'text_pair',
            'relatedness_score': 'label'
        })

    def compute_metrics(self, eval_pred: Optional[EvalPrediction]=None) -> dict[str, float] | None:
        if eval_pred is None:
            return self._metrics.compute()

        logits, references = eval_pred
        self._metrics.add_batch(
            predictions=logits,
            references=references,
        )

    @property
    def objective_metric_name(self) -> str:
        return 'pearsonr'

    @property
    def is_maximization(self) -> bool:
        return True

    def load_pretrained_model(self, model_name: str, num_labels: int) -> PreTrainedModel:
        return AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name,
            num_labels=num_labels,
        )
