from transformers import EvalPrediction
from .downstream_task_base import DownstreamTaskBase
from typing import Union
from datasets import load_dataset

from transformers import AutoModelForSequenceClassification
from transformers.models.auto.auto_factory import _BaseAutoModelClass

import evaluate
import numpy as np


class RecognisingTextualEntailment(DownstreamTaskBase):
    '''Assin 2 dataset for Recognising Textual Entailment (RTE).'''

    @property
    def name(self) -> str:
        return 'assin-rte'

    @DownstreamTaskBase.filtered_columns
    def load_dataset(self, split : Union[str, None] = None):
        dataset = load_dataset('nilc-nlp/assin2', 'default', split=split)

        # renamed columns
        return dataset.rename_columns({
            'premise': 'text',
            'hypothesis': 'text_pair',
            'entailment_judgment': 'label',
        })
    
    def compute_metrics(self, eval_pred: EvalPrediction) -> dict:
        f1_metric = evaluate.load("f1")
        acc_metric = evaluate.load("accuracy")
        
        logits, references = eval_pred
        predictions = np.argmax(logits, axis=-1)

        f1 = f1_metric.compute(
            predictions=predictions,
            references=references,
            average='binary'
        )
        acc = acc_metric.compute(
            predictions=predictions,
            references=references
        )

        return {'f1': f1, 'accuracy': acc}
    
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


# class SemanticTextualSimilarity(DownstreamTaskBase):
#     '''Assin 2 dataset for Semantic Textual Similarity (STS).'''

#     @DownstreamTaskBase.filtered_columns
#     def load_dataset(self, split : Union[str, None] = None):
#         dataset = load_dataset('assin2', 'default', split=split)

#         # renamed columns
#         return dataset.rename_columns({
#             'premise': 'text',
#             'hypothesis': 'text_pair',
#             'relatedness_score': 'label',
#         })
    
#     def compute_metrics(self, eval_pred: EvalPrediction) -> dict:
#         logits, references = eval_pred

#         metrics = evaluate.combine(['mse', 'pearsonr'])
#         return metrics.compute(
#             predictions=logits,
#             references=references,
#         )

