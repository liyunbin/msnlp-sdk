#! -*- coding: utf-8 -*-
import logging
import os
from msnlp.utils import PREDICT_PROCESSOR
from msnlp.task import TASK_DICT
from msnlp.data.datasets import PREDICTSET_DICT

logger = logging.getLogger(__name__)


class NLPPredictor:
    """nlp predictor."""

    def __init__(self, task_name: str, **kwargs):
        """NLPTask 统一构造函数.
            Params
                task_name: NLP 任务名
                kwargs: 其他自定义入参
        """
        if task_name not in TASK_DICT:
            raise ValueError(
                '{} not supported right now. '
                'You could use `NLPPredictor.support_tasks()`'
                ' to see which supported models.'.format(task_name))

        self.nlptask = TASK_DICT[task_name](task_name, is_train=False, **kwargs)

    @classmethod
    def support_tasks(cls):
        r"""
        获取能支持的预训练模型的信息.
    """
        return TASK_DICT.keys()

    def predict_rest(self, json_data):
        texts = json_data.get("text", [])
        pred_result = self.nlptask.predict(texts)
        return pred_result

    def predict_testdataset(self, data_dir):
        return self.nlptask.test(data_dir)

    def predict_rest_topn(self, json_data, topn=1):
        texts = json_data.get("text", [])
        pred_result = self.nlptask.predict_topn(texts, topn)
        return pred_result
