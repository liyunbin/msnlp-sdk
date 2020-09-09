#! -*- coding: utf-8 -*-
import logging
import numpy as np
from transformers import EvalPrediction
from typing import Dict


def compute_metrics_fn(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}
