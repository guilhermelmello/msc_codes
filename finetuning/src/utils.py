from typing import Optional, Protocol
from transformers import EvalPrediction


class ComputeMetricsCallback(Protocol):
    def __call__(self, eval_pred: Optional[EvalPrediction] = None) -> dict[str, float] | None:
        ...

