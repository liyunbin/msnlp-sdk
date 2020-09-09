#! -*- coding: utf-8 -*-
import logging
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from msnlp.task.nlptask import NLPTask
from msnlp.data.datasets.token_classification import NerDataset
from msnlp.data.data_utils import Split
from msnlp.data.metrics.token_classification import compute_metrics_fn
from msnlp.data.metrics.token_classification import ner_metrics
from msnlp.task.nlptask import PredictionOutput

logger = logging.getLogger(__name__)


class TokenClassification(NLPTask):
    """Text classification task."""

    def __init__(self, task_name: str, is_train: bool = True, **kwargs):
        super().__init__(task_name, is_train=is_train, **kwargs)

    def train(self, data_dir, seed=None) -> PredictionOutput:
        # 构造数据集
        label2id = self.config.label2id
        train_dataset = NerDataset(
            data_dir=data_dir,
            tokenizer=self.tokenizer,
            label_map=label2id,
            pretrain_type=self.pretrain_type,
            mode=Split.train,
        )
        eval_dataset = NerDataset(
            data_dir=data_dir,
            tokenizer=self.tokenizer,
            label_map=label2id,
            pretrain_type=self.pretrain_type,
            mode=Split.dev,
        )

        # 交付统一训练
        eval_result = self.uni_train(
            data_dir,
            train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics_fn=compute_metrics_fn,
            seed=seed)

        return eval_result

    def test(self, data_dir) -> PredictionOutput:
        # 构造 test 数据集
        label2id = self.config.label2id
        test_dataset = NerDataset(
            data_dir,
            tokenizer=self.tokenizer,
            label_map=label2id,
            pretrain_type=self.pretrain_type,
            mode=Split.test,
        )

        # 提交统一测试
        eval_result = self.uni_test(
            data_dir, test_dataset, compute_metrics_fn=compute_metrics_fn,)
        predictions = eval_result.predictions
        label_ids = eval_result.label_ids
        metrics = ner_metrics(predictions, label_ids, self.config.id2label)
        for key, value in metrics.items():
            eval_result.metrics[key] = value
        return eval_result
