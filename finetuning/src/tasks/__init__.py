"""
Load datasets from huggingface with standard column names for training.
"""
from typing import Callable
from datasets import Dataset
from transformers import (
    BatchEncoding,
    PreTrainedTokenizerBase,
)

# datasets
from .task_base import TaskBase
from . import assin
from . import assin2


_tasks_map = {
    'assin-rte': assin.RecognisingTextualEntailment,
    # 'assin_sts': assin.SemanticTextualSimilarity(),
    'assin2-rte': assin2.RecognisingTextualEntailment,
    # 'assin2_sts': assin2.SemanticTextualSimilarity(),
    # 'bpsad_polarity': load_bpsad_polarity,
    # 'bpsad_rating': load_bpsad_rating,
}


def get_tokenizer_map(
    tokenizer: PreTrainedTokenizerBase,
    is_sentence_pair : bool
) -> Callable[[Dataset], BatchEncoding]:
    '''Defines the tokenization mapper'''
    if is_sentence_pair:
        return lambda examples: tokenizer(
            text=examples['text'],
            text_pair=examples['text_pair'],
            truncation=True,
        )

    return lambda examples: tokenizer(
        text=examples['text'],
        truncation=True,
    )


def load_task(name) -> TaskBase:
    '''Return a downstream task.'''
    try:
        TaskCls = _tasks_map[name]
        return TaskCls()
    except:
        raise ValueError(f'Task Not Found: {name}')
