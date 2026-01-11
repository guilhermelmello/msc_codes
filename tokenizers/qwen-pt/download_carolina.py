from datasets import load_dataset

dataset = load_dataset(
    "carolina-c4ai/corpus-carolina",
    split="corpus",
    revision='v2.0.1',
)