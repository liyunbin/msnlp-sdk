import logging
import os
import numpy as np
import random
import sys
sys.path.append("..")
from msnlp.nlp_trainer import NLPTrainer
import argparse
parser = argparse.ArgumentParser()

parser.add_argument(
    "--task_name", help="task name: ner/text_classification/similarity", type=str, default="")
parser.add_argument(
    "--pretrain_name", help="model name:bert-base-chinese", type=str, default="")
parser.add_argument(
    "--data_dir", help="--train and eval data set dir path", type=str, default="")
parser.add_argument(
    "--model_path", help="output model dir path", type=str, default="")
parser.add_argument(
    "--pretrain_path", help="pretained model dir path", type=str, default="")
# 这里的bool是一个可选参数，返回给args的是 args.bool
args = parser.parse_args()

if __name__ == '__main__':
    task_name = args.task_name
    pretrain_name = args.pretrain_name
    data_dir = args.data_dir
    model_path = args.model_path
    pretrain_path = args.pretrain_path
    trainner = NLPTrainer(
        task_name,
        pretrain_name=pretrain_name,
        pretrain_path=pretrain_path,
        data_dir=data_dir,
        model_path=model_path)
    trainner.train(data_dir, model_path, en_callback=False)
