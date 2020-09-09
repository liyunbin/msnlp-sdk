#! -*- coding: utf-8 -*-
import logging
from msnlp.data.data_utils import Split
from msnlp.data.datasets.classification import ClassificationDataset
from msnlp.task.nlptask import NLPTask
from msnlp.task.nlptask import PredictionOutput
from msnlp.data.metrics.classification import compute_metrics_fn

logger = logging.getLogger(__name__)


class Classification(NLPTask):
    """Text classification task."""

    def __init__(self, task_name: str, is_train: bool = True, **kwargs):
        super().__init__(task_name, is_train=is_train, **kwargs)

    def train(self, data_dir, seed=None) -> PredictionOutput:
        # 构造数据集
        id2label = self.config.id2label
        label_list = [label for _, label in id2label.items()]
        train_dataset = ClassificationDataset(
            data_dir,
            tokenizer=self.tokenizer,
            label_list=label_list,
            mode=Split.train,
            task_name=self.task_name)
        eval_dataset = ClassificationDataset(
            data_dir,
            tokenizer=self.tokenizer,
            label_list=label_list,
            mode=Split.dev,
            task_name=self.task_name)

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
        id2label = self.config.id2label
        label_list = [label for _, label in id2label.items()]
        test_dataset = ClassificationDataset(
            data_dir,
            tokenizer=self.tokenizer,
            label_list=label_list,
            task_name=self.task_name,
            mode="test")

        # 提交统一测试
        eval_result = self.uni_test(
            data_dir, test_dataset, compute_metrics_fn=compute_metrics_fn)

        return eval_result
