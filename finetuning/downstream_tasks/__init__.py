"""
Load datasets from huggingface with standard column names for training.
"""

# datasets
from .downstream_task_base import DownstreamTaskBase
from . import assin
from . import assin2


_tasks_map = {
    'assin-rte': assin.RecognisingTextualEntailment(),
    # 'assin_sts': assin.SemanticTextualSimilarity(),
    'assin2-rte': assin2.RecognisingTextualEntailment(),
    # 'assin2_sts': assin2.SemanticTextualSimilarity(),
    # 'bpsad_polarity': load_bpsad_polarity,
    # 'bpsad_rating': load_bpsad_rating,
}


def load_task(name) -> DownstreamTaskBase:
    '''Return a downstream task.'''
    try:
        return _tasks_map[name]
    except:
        raise ValueError(f'Task Not Found: {name}')
