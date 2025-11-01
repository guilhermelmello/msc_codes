# see:
# https://huggingface.co/docs/huggingface_hub/v1.0.0.rc6/en/package_reference/file_download#huggingface_hub.snapshot_download

from huggingface_hub import snapshot_download

def cache_dataset(repo_id, **kwargs):
    snapshot_download(repo_id=repo_id, repo_type='dataset', **kwargs)

def cache_model(repo_id, **kwargs):
    snapshot_download(repo_id=repo_id, repo_type='model', **kwargs)

# cache_dataset('nilc-nlp/assin')
# cache_dataset('nilc-nlp/assin2')
# cache_model('google-bert/bert-base-uncased')