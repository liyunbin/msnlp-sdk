# MSNLP SDK 使用说明

## 项目目标

本项目旨在提供一个基于预训练模型的 NLP SDK，方便算法工程师、开发工程师实现不同的 NLP 基础任务，快速进行实验验证。本项目基于 transformers 和 pytorch 两个开源项目，在此基础上进行了上层任务的封装，提供训练和预测的统一入口。

## 使用示例

分类任务的训练示例：

```python
# 使用统一训练封装函数，来训练不同任务
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
```

序列标注任务的预测示例：

```python
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
```

## 支持的任务和模型

目前支持以下任务：
- 文本分类
- 文本相似度
- 词相似度
- 情感分析

目前支持以下预训练模型：
- albert
- roberta
- bert
- xlnet
