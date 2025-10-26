from abc import abstractmethod
from typing import NamedTuple
from transformers import EvalPrediction
from transformers import PreTrainedModel

from datasets import DatasetDict


class DownstreamTaskBase(NamedTuple):
    @staticmethod
    def filtered_columns(func):
        '''Decorator to remove datasets extra columns'''
        def wrapper(*args, **kwargs):
            dataset = func(*args, **kwargs)

            keep_columns = ['text', 'text_pair', 'label']
            column_names = (dataset['train'].column_names
                if isinstance(dataset, DatasetDict)
                else dataset.column_names)

            for col in column_names:
                if col not in keep_columns:
                    dataset = dataset.remove_columns(col)

            return dataset
        return wrapper

    @abstractmethod
    def load_dataset(self, **kwargs):
        '''Load and prepare dataset columns.

        Use `filtered_columns` decorator to remove extra columns.
        '''
        raise NotImplementedError

    @abstractmethod
    def compute_metrics(self, eval_pred: EvalPrediction) -> dict:
        '''Compute task evalutation metrics.'''
        raise NotImplementedError

    @property
    @abstractmethod
    def objective_metric_name(self) -> str:
        '''The metric name to watch during optimizations.'''
        raise NotImplementedError
    
    @property
    @abstractmethod
    def is_maximization(self) -> bool:
        '''Determines the direction of the objective metric optimization.'''
        raise NotImplementedError

    @abstractmethod
    def load_pretrained_model(self, model_name: str, **kwargs) -> PreTrainedModel:
        '''Auto Model for task'''
        raise NotImplementedError
