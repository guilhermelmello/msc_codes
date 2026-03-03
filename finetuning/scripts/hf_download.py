"""
Download and cache every asset needed to run in offline mode.

Links
-----
https://huggingface.co/docs/huggingface_hub/v1.0.0.rc6/en/package_reference/file_download#huggingface_hub.snapshot_download
"""

# from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import evaluate

def cache_metric(name, **kwargs):
    print('>>> Caching metric:', name)
    evaluate.load(name, **kwargs)

def cache_dataset(name, **kwargs):
    print('>>> Caching dataset:', name)
    load_dataset(name, **kwargs)

def cache_model(name, **kwargs):
    print('>>> Caching model:', name)
    AutoModel.from_pretrained(name, **kwargs)
    AutoTokenizer.from_pretrained(name)

# METRICS
cache_metric('f1')
cache_metric('accuracy')
cache_metric('pearsonr')
cache_metric('mse')

# DATASETS
cache_dataset('nilc-nlp/assin')
cache_dataset('nilc-nlp/assin2')

# MODELS
cache_model('PORTULAN/albertina-100m-portuguese-ptbr-encoder')
cache_model('PORTULAN/albertina-900m-portuguese-ptbr-encoder')
cache_model('neuralmind/bert-base-portuguese-cased')
cache_model('neuralmind/bert-large-portuguese-cased')
cache_model('guilhermelmello/qwen-pt-bpe-8k')
cache_model('guilhermelmello/qwen-pt-base-bpe-8k')
cache_model('guilhermelmello/qwen-pt-unigram-8k')
cache_model('guilhermelmello/qwen-pt-base-unigram-8k')
