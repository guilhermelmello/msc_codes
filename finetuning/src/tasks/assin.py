from .task_base import TaskBase
from datasets import load_dataset
from transformers import EvalPrediction
from typing import Optional

from transformers import AutoModelForSequenceClassification
from transformers import PreTrainedModel

import evaluate


class RecognisingTextualEntailment(TaskBase):
    '''Assin dataset for Recognising Textual Entailment (RTE).'''

    def __init__(self):
        self._f1_metric = evaluate.load('f1')
        self._acc_metric = evaluate.load('accuracy')

    @property
    def name(self) -> str:
        return 'assin-rte'

    @TaskBase.filtered_columns
    def load_dataset(self, name: str = 'full', split : Optional[str] = None):
        dataset = load_dataset('nilc-nlp/assin', name, split=split)

        # renamed columns
        return dataset.rename_columns({
            'premise': 'text',
            'hypothesis': 'text_pair',
            'entailment_judgment': 'label',
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
        return 'f1'
    
    @property
    def is_maximization(self) -> bool:
        return True

    def load_pretrained_model(self, model_name: str, num_labels: int) -> PreTrainedModel:
        return AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name,
            num_labels=num_labels,
        )


# class SemanticTextualSimilarity(DownstreamTaskBase):
#     '''Assin dataset for Semantic Textual Similarity (STS).'''

#     @DownstreamTaskBase.filtered_columns
#     def load_dataset(self, name: str = 'full', split : Union[str, None] = None):
#         dataset = load_dataset('nilc-nlp/assin', name, split=split)

#         # renamed columns
#         return dataset.rename_columns({
#             'premise': 'text',
#             'hypothesis': 'text_pair',
#             'relatedness_score': 'label'
#         })

#     def compute_metrics(self, eval_pred: EvalPrediction) -> dict:
#         logits, references = eval_pred

#         metrics = evaluate.combine(['mse', 'pearsonr'])
#         return metrics.compute(
#             predictions=logits,
#             references=references,
#         )

#     @property
#     def objective_metric_name(self) -> str:
#         return 'pearsonr'
    
#     @property
#     def is_maximization(self) -> bool:
#         return True