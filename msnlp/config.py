#! -*- coding: utf-8 -*-

class TaskMode:
    training = 0
    inference = 1


class PretrainType:
    bert = "bert"
    albert = "albert"
    roberta = "roberta"


class TaskName:
    text_classification = "text_classification"
    similarity = "similarity"
    sentiment2 = "sentiment2"
    ner = "ner"
    title_content_sent = "title_content_sent"
    word_similarity = "word_similarity"


# Supported model list and discription
SUPPORT_MODELS = [
    {
        'pretrain_name': 'bert-base-chinese',
        'pretrain_type': PretrainType.bert,
        'lm_dataset': 'google zh wiki',
        'discription': 'https://github.com/brightmart/albert_zh',
    },
    {
        'pretrain_name': 'voidful-albert_chinese_tiny',
        'pretrain_type': PretrainType.albert,
        'lm_dataset': 'google zh wiki',
        'discription': '',
    },
    {
        'pretrain_name': 'hfl-chinese-roberta-wwm-ext',
        'pretrain_type': PretrainType.roberta,
        'lm_dataset': 'google zh wiki',
        'discription': '',
    },
    {
        'pretrain_name': 'hfl-chinese-roberta-wwm-ext-large',
        'pretrain_type': PretrainType.roberta,
        'lm_dataset': 'google zh wiki',
        'discription': '',
    },
]

SUPPORT_MODEL_NAMES = [s['pretrain_name'] for s in SUPPORT_MODELS]

# Supported pretrained model type
PRETRAIN_TYPE_DICT = {s['pretrain_name']: s['pretrain_type'] for s in SUPPORT_MODELS}

# word similarity task params
WORD_SIMILARITY_PATH = '/home/data/word_similarity/'
WORD_SIMILARITY_PARAMS = {
    # 上下文词相似计算
    'pre_train_model_path': 'bert_base_chinese/',
    'pre_train_max_seq_len': 64,

    # 基本词相似计算
    'word_embedding_file_name': 'Tencent_AILab_ChineseEmbedding.txt'
}