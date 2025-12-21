"""
Download and cache every asset needed to run in offline mode.

Links
-----
https://huggingface.co/docs/huggingface_hub/v1.0.0.rc6/en/package_reference/file_download#huggingface_hub.snapshot_download
"""

# from huggingface_hub import snapshot_download
from transformers import AutoModel
from datasets import load_dataset
import evaluate

def cache_metric(name, **kwargs):
    evaluate.load(name, **kwargs)

def cache_dataset(name, **kwargs):
    load_dataset(name, **kwargs)

def cache_model(name, **kwargs):
    AutoModel.from_pretrained(name, **kwargs)

# METRICS
cache_metric('f1')
cache_metric('accuracy')
cache_metric('pearsonr')
cache_metric('mse')

# DATASETS
cache_dataset('nilc-nlp/assin')
cache_dataset('nilc-nlp/assin2')

# MODELS
cache_model('google-bert/bert-base-uncased')
