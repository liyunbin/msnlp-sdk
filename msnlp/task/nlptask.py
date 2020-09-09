#! -*- coding: utf-8 -*-
import logging
import random
import numpy as np
import os
import torch
from typing import NamedTuple
from transformers import EvalPrediction
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
    AutoModelForTokenClassification
)
from msnlp.config import TaskMode
from typing import Callable, Dict, Optional
from msnlp.utils import PREDICT_PROCESSOR
from msnlp.config import TaskName
from msnlp.config import PretrainType
from torch.utils.data.dataset import Dataset
from msnlp.config import SUPPORT_MODELS, SUPPORT_MODEL_NAMES
from msnlp.config import PRETRAIN_TYPE_DICT
from msnlp.data.processors import DATA_PROCESSOR
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)

MODEL_FUNC = {
    TaskName.text_classification: AutoModelForSequenceClassification,
    TaskName.ner: AutoModelForTokenClassification,
    TaskName.similarity: AutoModelForSequenceClassification,
    TaskName.sentiment2: AutoModelForSequenceClassification,
    TaskName.title_content_sent: AutoModelForSequenceClassification,
}

TOKENIZER_FUNC = {
    PretrainType.bert: AutoTokenizer,
    PretrainType.albert: BertTokenizer,
    PretrainType.roberta: BertTokenizer,
}


class PredictionOutput(NamedTuple):
    predictions: np.ndarray
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]


class NLPTask:
    """Base class for msxf nlp task."""

    def __init__(self, task_name: str, is_train: bool = True, **kwargs):
        """NLPTask 统一构造函数.
        Params
            task_name: NLP 任务名
            is_train: bool, 训练模式还是推理模式
            pretrain_name: 预训练模型名称
            pretrain_type: 任务所需要的预训练模型类型
            pretrain_path: 预训练模型的路径
            data_dir: NLP 任务对应的数据集
            model_path: NLP 任务的输出模型路径
            fp16: 是否使用混合精度模式
            use_cuda: 是否使用GPU
        """
        self.task_name = task_name
        self.pretrain_name = kwargs.pop('pretrain_name', None)
        self.pretrain_path = kwargs.pop('pretrain_path', None)
        self.pretrain_type = kwargs.pop('pretrain_type', None)
        self.data_dir = kwargs.pop('data_dir', None)
        self.model_path = kwargs.pop('model_path', None)
        self.fp16 = kwargs.pop('fp16', False)
        self.use_cuda = kwargs.pop('use_cuda', True)
        self.trainer = None

        if self.pretrain_type is None and self.pretrain_name in PRETRAIN_TYPE_DICT:
            self.pretrain_type = PRETRAIN_TYPE_DICT[self.pretrain_name]

        if is_train:
            task_mode = TaskMode.training
            if not self.pretrain_path:
                raise ValueError('NLPTask 训练模式需要 pretrain_path 参数.')
        else:
            task_mode = TaskMode.inference
            if not self.model_path:
                raise ValueError('NLPTask 推理模式需要 model_path 参数.')

        self._init_tokenizer(task_mode)
        self._init_model_config(task_mode)
        self._init_model(task_mode)

    @classmethod
    def support_models(cls):
        r"""
        获取能支持的预训练模型的信息.
    """
        return SUPPORT_MODELS

    def _init_tokenizer(self, task_mode: TaskMode):
        # init tokenizer
        if task_mode == TaskMode.training:
            tokenizer_path = self.pretrain_path
        else:
            tokenizer_path = self.model_path

        t_func = TOKENIZER_FUNC[self.pretrain_type]
        self.tokenizer = t_func.from_pretrained(tokenizer_path)
        return self.tokenizer

    def _init_model_config(self, task_mode: TaskMode):
        # init config
        if task_mode == TaskMode.training:
            config_path = self.pretrain_path
            processor = DATA_PROCESSOR[self.task_name]()
            label2id = processor.get_label2id(self.data_dir)
            id2label = processor.get_id2label(self.data_dir)
            num_labels = len(label2id)
            config = AutoConfig.from_pretrained(
                config_path,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
            )
        else:
            config_path = self.model_path
            config = AutoConfig.from_pretrained(config_path)
            id2label = {
                int(i): label for i, label in config.id2label.items()}

        self.id2label = id2label
        self.config = config
        return config

    def _init_model(self, task_mode: TaskMode):
        # init model
        if task_mode == TaskMode.training:
            model_path = self.pretrain_path
        else:
            model_path = self.model_path

        m_func = MODEL_FUNC[self.task_name]
        model = m_func.from_pretrained(model_path, config=self.config)

        if task_mode == TaskMode.inference:
            device = torch.device(
                "cuda" if torch.cuda.is_available() and self.use_cuda else "cpu")
            model.to(device)
            if self.fp16:
                from apex import amp
                model = amp.initialize(model, opt_level="O1")
            model.eval()

        self.model = model
        return model

    def init_training_args(self, model_path: str) -> TrainingArguments:
        r"""
        构造训练参数.
    """
        training_args = TrainingArguments(output_dir=model_path)
        training_args.logging_steps = 5000
        training_args.save_steps = 5000
        training_args.learning_rate = 2e-5
        training_args.num_train_epochs = 3
        training_args.per_device_train_batch_size = 32
        training_args.fp16 = self.fp16
        training_args.fp16_opt_level = "O1"
        return training_args

    def test(self, data_dir) -> PredictionOutput:
        raise NotImplementedError('nlp task not implement `test` method!')

    def train(self, data_dir, seed=None) -> PredictionOutput:
        raise NotImplementedError('nlp task not implement `train` method!')

    def uni_train(
            self,
            data_dir: str,
            train_dataset: Dataset,
            eval_dataset: Optional[Dataset]=None,
            compute_metrics_fn: Optional[Callable[[EvalPrediction], Dict]]=None,
            seed: int=None) -> PredictionOutput:
        r"""
        统一训练模块.
    """
        if not seed:
            seed = random.randint(0, 2020)
        set_seed(seed)

        # 构造训练参数
        training_args = self.init_training_args(self.model_path)

        # Initialize our Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_fn,
        )
        trainer.train(self.model_path)
        trainer.save_model()
        self.tokenizer.save_pretrained(self.model_path)
        self.trainer = trainer

        # Evaluation
        logger.info("*** Evaluate ***")
        trainer.compute_metrics = compute_metrics_fn
        eval_result = trainer.predict(test_dataset=eval_dataset)
        metrics = eval_result.metrics
        output_eval_file = os.path.join(
            self.model_path, f"eval_results_{self.task_name}.txt"
        )
        with open(output_eval_file, "w") as writer:
            logger.info(
                "***** Eval results {} *****"
                .format(self.task_name))
            for key, value in metrics.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

        return eval_result

    def uni_test(
        self,
        data_dir: str,
        test_dataset: Dataset,
        compute_metrics_fn: Optional[Callable[[EvalPrediction], Dict]]=None,
    ) -> PredictionOutput:
        r"""
        统一测试模块.
    """
        if not self.trainer:
            training_args = self.init_training_args(self.model_path)
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                compute_metrics=compute_metrics_fn,)
        logging.info("*** Test ***")
        eval_result = self.trainer.predict(test_dataset=test_dataset)
        inputs_list = [f.input_ids for f in test_dataset.features]
        # 保存预测结果
        # todo
        processor = PREDICT_PROCESSOR[self.task_name]
        predictions_logits = eval_result.predictions
        predictions = processor.post_processing(
            predictions_logits,
            self.config.id2label,
            tokenizer=self.tokenizer,
            input_list=inputs_list)

        output_test_file = os.path.join(
            self.model_path,
            f"test_results_{self.task_name}.txt"
        )
        with open(output_test_file, "w") as writer:
            logger.info(
                "***** Test results {} *****"
                .format(self.task_name))
            writer.write("index\tprediction\n")

            for index, pred_item in enumerate(predictions):
                writer.write("%d\t%s\n" % (index, pred_item))

        # 保存指标结果
        metrics = eval_result.metrics
        output_eval_file = os.path.join(
            self.model_path, f"test_metrics_{self.task_name}.txt"
        )
        with open(output_eval_file, "w") as writer:
            logger.info(
                "***** Eval results {} *****"
                .format(self.task_name))
            for key, value in metrics.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

        return eval_result

    def predict(self, texts):
        r"""
        统一预测模块，输入文本，输出结果.
    """
        predictions = []
        processor = PREDICT_PROCESSOR[self.task_name]
        features = processor.pre_processing(
            texts,
            self.tokenizer,
            self.pretrain_type,
            id2label=self.config.id2label,
            task_name=self.task_name)
        device = torch.device(
            "cuda" if torch.cuda.is_available() and self.use_cuda else "cpu")
        features = tuple(t.to(device) for t in features)
        inputs = {'input_ids': features[0],
                  'attention_mask': features[1],
                  'token_type_ids': features[2]}
        logits_list = []
        with torch.no_grad():
            outputs = self.model(**inputs)[0]
            logits_list = (outputs.data.cpu().numpy())
        input_list = features[0].cpu().numpy().tolist()
        predictions = processor.post_processing(
            logits_list,
            self.id2label,
            tokenizer=self.tokenizer,
            input_list=input_list,
            input_texts=texts)
        return {"data": predictions}

    def predict_topn(self, texts, topn=1):
        r"""
        统一预测模块，输入文本和topn，输出topn结果.
    """
        predictions = []
        processor = PREDICT_PROCESSOR[self.task_name]
        features = processor.pre_processing(
            texts,
            self.tokenizer,
            self.pretrain_type,
            id2label=self.config.id2label,
            task_name=self.task_name)
        device = torch.device(
            "cuda" if torch.cuda.is_available() and self.use_cuda else "cpu")
        features = tuple(t.to(device) for t in features)
        inputs = {'input_ids': features[0],
                  'attention_mask': features[1],
                  'token_type_ids': features[2]}
        logits_list = []
        with torch.no_grad():
            outputs = self.model(**inputs)[0]
            logits_list = (outputs.data.cpu().numpy())
        predictions = processor.post_processing_topn(
            logits_list,
            self.id2label,
            topn=topn,
            tokenizer=self.tokenizer,
            input_list=features[0])
        return {"data": predictions}
