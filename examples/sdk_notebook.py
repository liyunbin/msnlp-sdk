#! -*- coding: utf-8 -*-
import msnlp
from msnlp import (
    MSPretrainedModel,
    Classification,
    TokenClassification,
    NLPPredictor,
    NLPTrainer,
)

"""根据中台已有的预训练模型，fine-tune 自己语料，得到新的预训练模型."""
pretrain_name = "bert-base-chinese"
pretrain_path = "/model/pretrained/{}".format(pretrain_name)
train_file = "/data/pretrained/train.txt"
eval_file = "/data/pretrained/eval.txt"
model_output = "/output/pretrained/"
pretrained_model = MSPretrainedModel(pretrain_name)
eval_result = pretrained_model.fine_tune(
    train_file, model_path=pretrain_path, output_dir=model_output, eval_file=eval_file)
print(eval_result)


# 输出预训练模型的句向量
sentence = "创业五载，继往开来"
sen_embedding = pretrained_model.sen_embeddings(sentence)


# 利用已有预训练模型，训练文本分类任务
pretrain_type = "bert"
pretrain_path = "/model/pretrained/{}".format(pretrain_name)
task_name = "text_classification"
data_dir = "/data/text_classification"
model_path = "/output/text_classification"
model = Classification(
    task_name,
    is_train=True,
    model_path=model_path,
    pretrain_type=pretrain_type,
    pretrain_path=pretrain_path,
    data_dir=data_dir)

model.train(data_dir)
text = [
    "几点到几点上班",
    "贷20000利息多少",
    "我不用了，怎么解绑",
    "立即还款里显示我没借过钱 我怎么还",
]
predictions = model.predict(text)
print(predictions)


# 使用统一训练封装，来训练不同任务
task_name = "text_classification"
pretrain_name = "bert-base-chinese"
data_dir = "/data/text_classification"
model_path = "/output/text_classification"
pretrain_path = "/model/pretrained/{}".format(pretrain_name)
trainner = NLPTrainer(
    task_name,
    pretrain_name=pretrain_name,
    pretrain_path=pretrain_path,
    data_dir=data_dir,
    model_path=model_path)
trainner.train(data_dir, model_path)


# 利用已有预训练模型，训练NER任务
pretrain_name = "voidful-albert_chinese_tiny"
pretrain_type = "albert"
pretrain_path = "../tests/test_model/{}".format(pretrain_name)
task_name = "ner"
# data_dir = "/data/ner"
# model_path = "/output/text_classification"
data_dir = "/home/tong.luo/zhongtai/data/msraner"
model_path = "/home/tong.luo/zhongtai/tmp/ner"

model = TokenClassification(
    task_name,
    is_train=True,
    pretrain_type=pretrain_type,
    pretrain_path=pretrain_path,
    data_dir=data_dir,
    model_path=model_path)

model.train(data_dir)
texts = ["中国的首都是北京，四川的省会是成都，张三的籍贯是上海。",
         "不知道他来自哪里，但是现在住在成都"]
predictions = model.predict(texts)
print("predictions:")
print(predictions)

# 加载已有 NLP 任务，并进行预测
model_path = "/home/data/nfs/model/msraner"
pretrain_type = "albert"
task_name = "ner"
json_data = {"text": ["中国的首都是北京，四川的省会是成都，张三的籍贯是上海。",
                      "不知道他来自哪里，但是现在住在成都"]
             }

predictor = NLPPredictor(task_name, model_path=model_path, pretrain_type=pretrain_type)
pres = predictor.predict_rest(json_data)
print(pres)
