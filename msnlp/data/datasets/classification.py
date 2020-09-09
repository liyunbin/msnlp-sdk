#! -*- coding: utf-8 -*-
import logging
import os
import time
from typing import List, Optional, Union, Dict

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset
from transformers import InputFeatures, PreTrainedTokenizer
from transformers.data.processors.glue import glue_convert_examples_to_features
from msnlp.data.processors import DATA_PROCESSOR
from msnlp.data.data_utils import Split

logger = logging.getLogger(__name__)


class ClassificationPredictSet(Dataset):

    def __init__(
        self,
        texts: List[str],
        id2label: Dict[str, int],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 128,
        task_name='calssification',
    ):
        self.processor = DATA_PROCESSOR[task_name]()
        examples = self.processor.get_predict_examples(texts)
        self.label_list = [label for _, label in id2label.items()]
        self.features = glue_convert_examples_to_features(
            examples,
            tokenizer,
            max_length=max_seq_length,
            label_list=self.label_list,
            output_mode='classification',
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list


class ClassificationDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """
    features: List[InputFeatures]
    label_list: List[str]

    def __init__(
        self,
        data_dir,
        tokenizer: PreTrainedTokenizer,
        label_list: List[str],
        limit_length: Optional[int] = None,
        mode: Union[str, Split]=Split.train,
        cache_dir: Optional[str] = None,
        max_seq_length: int = 128,
        overwrite_cache: bool = False,
        task_name='calssification',
    ):
        self.processor = DATA_PROCESSOR[task_name]()
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        logger.info(
            f"Creating features from dataset file at {data_dir}")

        examples = self.processor.get_examples(data_dir, mode)
        if limit_length is not None:
            examples = examples[:limit_length]
        self.features = glue_convert_examples_to_features(
            examples,
            tokenizer,
            max_length=max_seq_length,
            label_list=self.label_list,
            output_mode='classification',
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list
