#! -*- coding: utf-8 -*-
import unittest
import os
import shutil
from msnlp import NLPTrainer, NLPPredictor
from msnlp.config import (
    PRETRAIN_TYPE_DICT,
    TaskName
)

os.environ["CUDA_VISIBLE_DEVICES"] = ""

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# 内置在测试文件夹的预训练模型
PRETRAINED_NAME = "voidful-albert_chinese_tiny"
PRETRAINED_PATH = "{}/test_model/{}".format(BASE_DIR, PRETRAINED_NAME)

# 测试的样本数据集路径
TASK_SAMPLE_DATA_PATH = {
    TaskName.text_classification: "{}/data/task_samples/text_classification".format(BASE_DIR),
    TaskName.ner: "{}/data/task_samples/ner".format(BASE_DIR), 
    TaskName.similarity: "{}/data/task_samples/similarity".format(BASE_DIR), 
    TaskName.title_content_sent: "{}/data/task_samples/title_content_sent".format(BASE_DIR), 
}

# 测试输出的模型文件总目录
TASK_OUTPUT_ROOT = "{}/output".format(BASE_DIR)


class TestTrainer(unittest.TestCase):
    def setUp(self):
        if os.path.exists(TASK_OUTPUT_ROOT):
            shutil.rmtree(TASK_OUTPUT_ROOT)
        os.mkdir(TASK_OUTPUT_ROOT)

    def tearDown(self):
        if os.path.exists(TASK_OUTPUT_ROOT):
            shutil.rmtree(TASK_OUTPUT_ROOT)

    def test_trainer_pipeline(self):
        pretrain_path = PRETRAINED_PATH
        tasks = TASK_SAMPLE_DATA_PATH.keys()
        for task in tasks:
            if task not in TASK_SAMPLE_DATA_PATH:
                continue
            # test train pipeline
            data_dir = TASK_SAMPLE_DATA_PATH[task]
            model_path = "{}/{}".format(TASK_OUTPUT_ROOT, task)
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            trainner = NLPTrainer(
                task,
                pretrain_name=PRETRAINED_NAME,
                pretrain_path=pretrain_path,
                data_dir=data_dir,
                model_path=model_path)
            eval_res = trainner.train(data_dir, model_path, en_callback=False)
            self.assertLess(eval_res.metrics["eval_loss"], 5.0)

            # test predict pipeline
            predictor = NLPPredictor(
                task, model_path=model_path,
                pretrain_type=PRETRAIN_TYPE_DICT[PRETRAINED_NAME])
            eval_res = predictor.predict_testdataset(data_dir)
            if eval_res.metrics:
                self.assertLess(eval_res.metrics["eval_loss"], 5.0)
