#! -*- coding: utf-8 -*-
import logging
import os
import requests
from msnlp.config import (
    SUPPORT_MODELS,
    SUPPORT_MODEL_NAMES,
    PRETRAIN_TYPE_DICT,
)
from msnlp.task import TASK_DICT

logger = logging.getLogger(__name__)


class NLPTrainer:
    """Text classification task."""

    def __init__(self, task_name: str, **kwargs):
        """NLPTask 统一构造函数.
            Params
                task_name: NLP 任务名
                pretrain_name: 任务所需要的预训练模型
                pretrain_path: 需要加载的预训练模型的路径
                data_dir: 任务对应的数据集
                model_path: 输出的模型路径
                fp16: 是否使用混合精度模式
        """
        if task_name not in TASK_DICT:
            raise ValueError(
                '{} not supported right now. '
                'You could use `NLPTrainer.support_tasks()`'
                ' to see which supported models.'.format(task_name))

        # nlptask 构造函数
        self.nlptask = TASK_DICT[task_name](task_name, is_train=True, **kwargs)

    @classmethod
    def support_models(cls):
        r"""
        获取能支持的预训练模型的信息.
    """
        return SUPPORT_MODELS

    @classmethod
    def support_tasks(cls):
        r"""
        获取能支持的预训练模型的信息.
    """
        return TASK_DICT.keys()

    @classmethod
    def _call_back(
            cls,
            task_status: int=None,
            process: int=None,
            metrics: str=None,
            save_model_dir: str=None,
            metadata: str=None,
            en_callback: bool=True):
        """模型训练过程中回调apiserver。
        :param task_status: 任务状态（1‐训练中，2‐成功，3‐失败）
        :param process: 训练进度 0 - 100
        :param metrics: 模型metric
        :param save_model_dir: 模型保存目录
        :param metadata: 其它数据
        :return:
        """
        if not en_callback:
            return
        call_back_url = os.environ.get('trainCallBackUrl', 'None')
        if not call_back_url:
            logger.warning('callback url is None')
            return
        logger.info(f'callback url is : {call_back_url}')
        try:
            request_data = {
                'taskStatus' : task_status,
                'process': process,
                'index': metrics,
                'modelPaths' : save_model_dir,
                'metadata' : metadata
            }
            response = requests.request(
                "POST", call_back_url, timeout=5, json=request_data)
            if response.status_code == 200:
                logger.info(f'callback success. callback result:{response.json()}')
            else:
                logger.error(f'callback failed. msg: {response.json()}')
        except:
            logger.error('callback service error.', exc_info=True)

    def train(self, data_dir: str, model_path: str, en_callback: bool=True):
        try:
            self._call_back(task_status=1, process=0, en_callback=en_callback)
            eval_res = self.nlptask.train(data_dir)
            self._call_back(
                task_status=2,
                process=100,
                save_model_dir=model_path,
                en_callback=en_callback)
            return eval_res
        except Exception as e:
            logger.error('model train failed', exc_info=True)
            self._call_back(
                task_status=3,
                metadata=f'train failed, error msg: {e}',
                en_callback=en_callback)


if __name__ == '__main__':
    os.environ['trainCallBackUrl']='http://10.193.199.142:5610/api/v1/model/train/callback/650f520e9ed443edbd1990176de7b029'
    NLPTrainer._call_back(task_status=1,process=0)
