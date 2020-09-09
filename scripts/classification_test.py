import sys
sys.path.append("..")
from msnlp.task.classification import Classification
import fire
from msnlp.nlp_trainer import NLPTrainer
from msnlp.nlp_predictor import NLPPredictor
import pandas as pd

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


def train_eng():
    task_name = "text_classification"
    pretrain_type = "bert"
    data_dir = "/home/dingnan.jin/projects/nlpsdk/data/classification/SST-2"
    model_path = "/home/dingnan.jin/projects/nlpsdk/testout/SST-2"
    pretrain_path = "/home/dingnan.jin/projects/nlpsdk/models/pretrained/bert-base-uncased"
    model = Classification(
        task_name,
        is_train=True,
        pretrain_type=pretrain_type,
        pretrain_path=pretrain_path,
        data_dir=data_dir,
        model_path=model_path)
    model.train(data_dir)


def train_sample():
    task_name = "text_classification"
    pretrain_type = "bert"
    data_dir = "/home/dingnan.jin/projects/nlpsdk/data/classification/SST-2_sample"
    model_path = "/home/dingnan.jin/projects/nlpsdk/testout/SST-2_sample"
    pretrain_path = "/home/dingnan.jin/projects/nlpsdk/models/pretrained/bert-base-uncased"
    model = Classification(
        task_name,
        is_train=True,
        pretrain_type=pretrain_type,
        pretrain_path=pretrain_path,
        data_dir=data_dir,
        model_path=model_path)
    model.train(data_dir)


def test_trainer():
    task_name = "text_classification"
    pretrain_name = "bert-base-chinese"
    data_dir = "/home/dingnan.jin/projects/nlpsdk/data/classification/SST-2_sample"
    model_path = "/home/dingnan.jin/projects/nlpsdk/testout/SST-2_sample"
    pretrain_path = "/home/dingnan.jin/projects/nlpsdk/models/pretrained/bert-base-uncased"
    trainner = NLPTrainer(
        task_name,
        pretrain_name=pretrain_name,
        pretrain_path=pretrain_path,
        data_dir=data_dir,
        model_path=model_path)
    trainner.train(data_dir, model_path, en_callback=False)


def train_xma():
    task_name = "text_classification"
    model_name = "bert-base-chinese"
    pretrain_type = "bert"
    data_dir = "/home/dingnan.jin/projects/nlpsdk/data/classification/xma"
    model_path = "/home/dingnan.jin/projects/nlpsdk/testout/xma"
    pretrain_path = "/home/dingnan.jin/projects/nlpsdk/models/pretrained/bert_base_chinese"
    model = Classification(
        task_name,
        is_train=True,
        pretrain_type=pretrain_type,
        pretrain_path=pretrain_path,
        data_dir=data_dir,
        model_path=model_path)
    model.train(data_dir)


def train_toutiao():
    task_name = "text_classification"
    model_name = "bert-base-chinese"
    pretrain_type = "bert"
    data_dir = "/home/dingnan.jin/projects/nlpsdk/data/classification/toutiao"
    model_path = "/home/dingnan.jin/projects/nlpsdk/testout/toutiao"
    pretrain_path = "/home/dingnan.jin/projects/nlpsdk/models/pretrained/bert_base_chinese"
    model = Classification(
        task_name,
        is_train=True,
        pretrain_type=pretrain_type,
        pretrain_path=pretrain_path,
        data_dir=data_dir,
        model_path=model_path)
    model.train(data_dir)


def test_toutiao():
    task_name = "text_classification"
    model_name = "bert-base-chinese"
    pretrain_type = "bert"
    data_dir = "/home/dingnan.jin/projects/nlpsdk/data/classification/toutiao"
    model_path = "/home/dingnan.jin/projects/nlpsdk/testout/toutiao"
    predictor = NLPPredictor(task_name, model_path=model_path, pretrain_type=pretrain_type)
    predictor.predict_testdataset(data_dir)


def test_xma():
    task_name = "text_classification"
    model_name = "bert-base-chinese"
    pretrain_type = "bert"
    data_dir = "/home/dingnan.jin/projects/nlpsdk/data/classification/xma"
    model_path = "/home/dingnan.jin/projects/nlpsdk/testout/xma"
    predictor = NLPPredictor(task_name, model_path=model_path, pretrain_type=pretrain_type)
    predictor.predict_testdataset(data_dir)


def train_sample():
    task_name = "text_classification"
    pretrain_type = "bert"
    data_dir = "/home/dingnan.jin/projects/nlpsdk/data/classification/SST-2_sample"
    model_path = "/home/dingnan.jin/projects/nlpsdk/testout/sst_sample"
    pretrain_path = "/home/dingnan.jin/projects/nlpsdk/models/pretrained/bert_base_chinese"
    model = Classification(
        task_name,
        is_train=True,
        pretrain_type=pretrain_type,
        pretrain_path=pretrain_path,
        data_dir=data_dir,
        model_path=model_path)
    model.train(data_dir)

def predict_sst():
    model_path = "/home/dingnan.jin/projects/nlpsdk/testout/SST-2"
    pretrain_type = "bert"
    task_name = "text_classification"
    texts = {
        "text": [
            "  it 's just incredibly dull",
            "it 's a charming and often affecting journey . ",
            "unflinchingly bleak and desperate",
            "allows us to hope that nolan is poised to embark a major career as a commercial yet inventive filmmaker .",]
    }
    predictor = NLPPredictor(task_name, model_path=model_path, pretrain_type=pretrain_type)
    pres = predictor.predict_rest(texts)
    print(pres)


def predict_xma():
    model_path = "/home/dingnan.jin/projects/nlpsdk/testout/xma"
    pretrain_type = "bert"
    task_name = "text_classification"
    texts = {
        "text": [
            "几点到几点上班",
            "贷20000利息多少",
            "我不用了，怎么解绑",
            "立即还款里显示我没借过钱 我怎么还",]
    }
    predictor = NLPPredictor(task_name, model_path=model_path, pretrain_type=pretrain_type)
    pres = predictor.predict_rest(texts)
    print(pres)

def predict_tnews():
    model_path = "/home/kuan.li/msnlp/output_models/tnews"
    pretrain_type = "bert"
    task_name = "text_classification"
    texts = {
        "text": [
            "以色列暗杀叙利亚7名“国家英雄”的行为该如何定性？",
            "小学生高校食堂写作业：不会的问大学生，他们也做错，只能靠自己",
            "芜湖有哪些上市公司？",
            "赵本山在东北的地位如何？最近怎么没有消息了呢？", ]
    }
    predictor = NLPPredictor(task_name, model_path=model_path, pretrain_type=pretrain_type)
    pres = predictor.predict_rest(texts)
    print(pres)

def predict_tnews_topn(topn=1):
    model_path = "/home/kuan.li/msnlp/output_models/tnews"
    pretrain_type = "bert"
    task_name = "text_classification"
    texts = {
        "text": [
            "以色列暗杀叙利亚7名“国家英雄”的行为该如何定性？",
            "小学生高校食堂写作业：不会的问大学生，他们也做错，只能靠自己",
            "芜湖有哪些上市公司？",
            "赵本山在东北的地位如何？最近怎么没有消息了呢？", ]
    }
    predictor = NLPPredictor(task_name, model_path=model_path, pretrain_type=pretrain_type)
    pres = predictor.predict_rest_topn(texts, topn)
    print(pres)

def train_next_intent():
    task_name = "text_classification"
    pretrain_name = "bert-base-chinese"
    pretrain_type = "bert"
    pretrain_path = "/home/kuan.li/msnlp/pretrained_models/bert_base_chinese"
    data_dir = "/home/kuan.li/msnlp/data/nextintent"
    model_path = "/home/kuan.li/msnlp/output_models/nextintent"

    trainner = NLPTrainer(
        task_name,
        pretrain_name=pretrain_name,
        pretrain_path=pretrain_path,
        data_dir=data_dir,
        model_path=model_path)
    trainner.train(data_dir, model_path, en_callback=False)

def predict_nextintent_topn(topn=1):
    model_path = "/home/kuan.li/msnlp/output_models/nextintent"
    pretrain_type = "bert"
    task_name = "text_classification"
    texts = {
        "text": [
            "放款未到账我要举报±贷款到账时间咨询§到银监会举报±投诉咨询§",
            "可以二次申请贷款吗？±二次申请§",
            "提现失败怎么办？±提现失败§贷款什么时候能到账？±贷款到账时间咨询§贷款审核需要多久？±审核时间要多久§？",
            "额度没变是不是意味着审核失败？±借款成功额度变化咨询§贷款审核状态查不了啊±审核是否通过查询§", ]
    }
    # 贷款到账时间咨询
    # 提现失败
    # 提现失败
    # 审核是否通过查询
    predictor = NLPPredictor(task_name, model_path=model_path, pretrain_type=pretrain_type)
    pres = predictor.predict_rest_topn(texts, topn)
    print(pres)

def predict_nextintent_file(topn=1):
    model_path = "/home/kuan.li/msnlp/output_models/nextintent"
    pretrain_type = "bert"
    task_name = "text_classification"
    data_dir = "/home/kuan.li/msnlp/data/nextintent"
    test_file = os.path.join(data_dir, "test.tsv")
    output_file = os.path.join(data_dir, "test_predict.tsv")
    test_set = pd.read_table(test_file, sep='\t')
    texts = {"text": test_set.text.tolist()}
    # texts = {
    #     "text": [
    #         "放款未到账我要举报±贷款到账时间咨询§到银监会举报±投诉咨询§",
    #         "可以二次申请贷款吗？±二次申请§",
    #         ]
    # }
    predictor = NLPPredictor(task_name, model_path=model_path, pretrain_type=pretrain_type)
    pres = predictor.predict_rest_topn(texts, topn)
    # test_set["predict_label"] = [','.join(p) for p in pres["data"]]
    test_set["predict_label"] = pres["data"]
    test_set.to_csv(path_or_buf=output_file, sep='\t', header=False, index=False)
    print("file saved.")
    for i in range(topn):
        num_correct = 0
        for v in test_set.values:
            top_set = set(v[2][:i+1])
            if v[1] in top_set:
                num_correct = num_correct + 1
        print("top_{topN} Accuracy:{acc}".format(topN=i+1, acc=float(num_correct)/len(test_set)))
    print("total lines:{}".format(len(test_set)))

def predict_sentiment_file():
    model_path = "/home/kuan.li/msnlp/output_models/sentiment"
    pretrain_type = "bert"
    task_name = "text_classification"
    data_dir = "/home/kuan.li/msnlp/data/sentiment"
    test_file = os.path.join(data_dir, "test.tsv")
    output_file = os.path.join(data_dir, "test_predict.tsv")
    test_set = pd.read_table(test_file, sep='\t')
    texts = {"text": test_set.text.tolist()}
    # texts = {
    #     "text": [
    #         "已经被你们坑了两次了",
    #         "草泥马逼的30次都没提现成功",
    #         "还光了吧我",
    #         ]
    # }
    predictor = NLPPredictor(task_name, model_path=model_path, pretrain_type=pretrain_type)
    pres = predictor.predict_rest(texts)
    test_set["predict_label"] = pres["data"]
    test_set.to_csv(path_or_buf=output_file, sep='\t', header=False, index=False)
    print("file saved.")
    num_correct = 0
    for v in test_set.values:
        if v[1] == v[2]:
            num_correct = num_correct + 1
    print("Accuracy:{}".format(float(num_correct)/len(test_set)))
    print("total lines:{}".format(len(test_set)))

if __name__ == '__main__':
    fire.Fire()
