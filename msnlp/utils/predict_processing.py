#! -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import re
import unicodedata
from msnlp.data.datasets.token_classification import convert_texts_to_features
from msnlp.data.processors.token_classification import get_entities
from msnlp.data.datasets.classification import ClassificationPredictSet


class PredictProcessing:

    @classmethod
    def pre_processing(cls, texts, tokenizer, model_type, **kwargs):
        """
        :param texts: list of str
        :param labels: list of str
        :return: vectorizer, labelencoder, x, y
        """
        raise NotImplementedError('not implement pre_processing method!')

    @classmethod
    def post_processing(cls, logits, id2label, **kwargs):
        """
        :param text: str
        :param vectorizer: such as countvectorize  word2vec
        :return: numpy of shape
        """
        raise NotImplementedError('not implement pre_processing method!')


class TokenPredictprocess(PredictProcessing):

    @classmethod
    def pre_processing(cls, texts, tokenizer, model_type, **kwargs):
        """
        :param texts: list of sentences.
        :param labels: list of label.
        :param kwargs: model_params.
        :return:
        """
        features_raw = convert_texts_to_features(
            texts,
            tokenizer
        )
        features = []
        input_ids_list = []
        attention_mask_list = []
        token_ids_list = []
        for feature in features_raw:
            input_ids = feature.input_ids
            input_ids_list.append(input_ids)
            attention_mask_list.append(feature.attention_mask)
            token_ids_list.append(feature.token_type_ids)
        input_ids_list = torch.LongTensor(input_ids_list)
        attention_mask_list = torch.LongTensor(attention_mask_list)
        token_ids_list = torch.LongTensor(token_ids_list)
        features = [input_ids_list, attention_mask_list, token_ids_list]
        return features

    @staticmethod
    def clean_space(text):
        """"
        处理多余的空格
        """
        match_regex = re.compile(
            u'[\u4e00-\u9fa5。\.,，:：《》、\(\)（）]{1} +(?<![a-zA-Z])|\d+ +| +\d+|[a-z A-Z]+')
        should_replace_list = match_regex.findall(text)
        order_replace_list = sorted(
            should_replace_list, key=lambda i: len(i), reverse=True)
        for i in order_replace_list:
            if i == u' ':
                continue
            new_i = i.strip()
            text = text.replace(i, new_i)
        return text

    @staticmethod
    def stem(token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    @staticmethod
    def _is_control(ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_special(ch):
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    @classmethod
    def rematch(cls, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系
        args:
            text:模型预测结果
            tokens: 标签id到标签的映射字典
        return:
            tokens里每个token包含的字符在原始文本中的位置。
        example:
            假设使用bert-base的tokenizer
            text: 'how do you do,中华人民共和国成立于1949年，中国总人口超过1400000000人'
            tokens: ['how', 'do', 'you', 'do', ',', '中', '华',
                     '人', '民', '共', '和', '国', '成',
                     '立', '于', '1949', '年', '，', '中',
                     '国', '总', '人', '口', '超', '过',
                     '1400', '##000', '##000', '人']
            return: [[0, 1, 2], [4, 5], [7, 8, 9],
                     [11, 12], [13], [14],
                     [15], [16], [17], [18], [19],
                     [20], [21], [22], [23], [24, 25, 26, 27],
                     [28], [29], [30], [31], [32],
                     [33], [34], [35], [36], [37, 38, 39, 40],
                     [41, 42, 43], [44, 45, 46], [47]]
        """
        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            # lower case
            ch = unicodedata.normalize('NFD', ch)
            ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            ch = ch.lower()
            # lower case
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or cls._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if cls._is_special(token):
                token_mapping.append([])
            else:
                token = cls.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end

        return token_mapping

    @classmethod
    def post_processing(cls, logit_list, id2label, **kwargs):
        """
        由于ner返回需要提供实体在文本中的位置，因此输入需要包含原始文本
        args:
            logit_list:模型预测结果
            id2label: 标签id到标签的映射字典
            input_list:输入token转换后的ids
            input_texts:原始文本
            tokenizer:分词工具，辅助定位实体在原始文本中的位置
        """
        input_list = kwargs.get("input_list")
        input_texts = kwargs.get("input_texts", None)
        tokenizer = kwargs.get("tokenizer")
        result_list = []
        for i in range(len(input_list)):
            inputs = input_list[i]
            logits = logit_list[i]
            text = ""
            rematch_list = []
            if input_texts:
                text = input_texts[i]
                input_tokens = tokenizer.convert_ids_to_tokens(inputs)
                rematch_list = cls.rematch(text, input_tokens)
            pred_ids = np.argmax(logits, -1)
            input_ids = inputs
            labels = [id2label.get(i, "O") for i in pred_ids]
            chunk_list = get_entities(labels)
            # [('PER', 0, 1), ('LOC', 3, 3)]
            entity_list = []
            for chunk in chunk_list:
                entity_type = chunk[0]
                tokens = []
                begin = -1
                end = -1
                select_ids = []
                for i in range(chunk[1], chunk[2] + 1):
                    if input_ids[i] > 0:
                        tokens.append(
                            tokenizer._convert_id_to_token(input_ids[i]))
                        select_ids.append(i)
                if len(tokens) > 0:
                    entity = tokenizer.convert_tokens_to_string(tokens)
                    entity = cls.clean_space(entity)
                    if input_texts:
                        begin = rematch_list[select_ids[0]][0]
                        end = rematch_list[select_ids[-1]][-1]
                        entity = text[begin:(end + 1)].strip()
                    if entity:
                        entity_list.append([entity, entity_type, begin, end])

            result_list.append(entity_list)
        return result_list


class ClassificationPredictprocess(PredictProcessing):

    @classmethod
    def pre_processing(cls, texts, tokenizer, model_type, **kwargs):
        """
        :param texts: list of sentences.
        :param labels: list of label.
        :param kwargs: model_params.
        :return:
        """
        task_name = kwargs.get("task_name")
        id2label = kwargs.get("id2label")
        features_raw = ClassificationPredictSet(
            texts,
            id2label,
            tokenizer=tokenizer,
            task_name=task_name)
        features = []
        input_ids_list = []
        attention_mask_list = []
        token_ids_list = []
        for feature in features_raw:
            input_ids = feature.input_ids
            input_ids_list.append(input_ids)
            attention_mask_list.append(feature.attention_mask)
            token_ids_list.append(feature.token_type_ids)
        input_ids_list = torch.LongTensor(input_ids_list)
        attention_mask_list = torch.LongTensor(attention_mask_list)
        token_ids_list = torch.LongTensor(token_ids_list)
        features = [input_ids_list, attention_mask_list, token_ids_list]
        return features

    @classmethod
    def post_processing(cls, logits, id2label, **kwargs):
        predictions = np.argmax(logits, axis=1)
        predictions = [id2label[p] for p in predictions]
        return predictions

    @classmethod
    def post_processing_topn(cls, logits, id2label, topn=1, **kwargs):
        if topn < 1:
            topn = 1
        pred_ids = np.argsort(logits, axis=1)[:, ::-1][:, :topn]
        return [[id2label[i_sort] for i_sort in p_sort] for p_sort in pred_ids]
