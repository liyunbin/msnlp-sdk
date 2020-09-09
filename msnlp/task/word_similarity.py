import logging
from msnlp.config import TaskName
from msnlp.config import WORD_SIMILARITY_PATH
from msnlp.config import WORD_SIMILARITY_PARAMS
import torch
from transformers import  AutoModel, AutoConfig, AutoTokenizer, AlbertForMaskedLM, BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import numpy as np
import os
import sys

logger = logging.getLogger(__name__)


class WordSimilarity:

    # 词相似相似计算任务
    # token_dict: dict
    # graph: Graph
    # tokenizer: Tokenizer
    # model: Model

    def __init__(
            self,
            task_name: str = TaskName.word_similarity,
            is_train=False,
            model_path: str=WORD_SIMILARITY_PATH,
            model_params: str=WORD_SIMILARITY_PARAMS,
    ):
        """
            词相似计算构造函数
        :param task_name: NLP 任务名
        :param is_train: 不训练只预测
        :param model_path: 词相似计算模型路径
        :param model_params: 词相似计算配置参数
        """
        super().__init__()
        self.is_train = is_train
        self.task_name = task_name
        self.model_path = model_path
        self.model_params = model_params
        self._contextual_model_init()  # 基于预训练模型的上下文词相似
        self._word_model_init()  # 基于基础词相似

    def predict(self, texts):
        """
            nlp_predictor 接口适配
        :param texts:
            根据相似类型调用不同的相似计算接口，包含 similarity_type similarity_pairs 两个参数的字典
            比如：{
                'similarity_type': 'word_similarity',
                'similarity_pairs': [[],...]
            }
            dict_param similarity_type:
                similarity_type="word_similarity": 表示使用词相似接口 get_word_similarities
                similarity_type="contextual_word_similarity":
                    表示使用上下文词相似接口 get_contextual_word_similarities
            dict_param similarity_pairs:
                根据类型不同 similarity_type 调用不同参数，详见具体参数类型接口定义
        :return:
            {'data': similarity_probabilities}
            例如
            {'data': [0.67, 0.32, 0.01, ……]}
        """
        if 'similarity_type' not in texts or 'similarity_pairs' not in texts:
            ret = '参数错误，必须包含 similarity_type 和 similarity_pairs'
            return {'data': ret}

        similarity_type = texts['similarity_type']
        similarity_pairs = texts['similarity_pairs']
        if len(similarity_pairs) == 0:
            ret = '参数错误，similarity_pairs 没有值'
            return {'data': ret}

        if similarity_type == 'word_similarity':
            if len(similarity_pairs[0]) != 2:
                ret = '参数错误，word_similarity 的 similarity_pairs ' \
                      '每一个元素是由两个词组成的 list 或 tuple' \
                      '比如 ' + """[ ('菠萝', '凤梨'), ('苹果', '电脑'), ]"""
                return {'data': ret}
            ret = self.get_word_similarities(similarity_pairs)

        elif similarity_type == 'contextual_word_similarity':
            if len(similarity_pairs[0]) != 6:
                ret = '参数错误，contextual_word_similarity 的 similarity_pairs ' \
                      '每一个元素是由两个背景句子+词位置组成的 list 或 tuple' \
                      '比如 ' + """
                    [ ['菠萝很好吃', 0, 1, '凤梨很好吃', 0, 1], 
                    ['我要买苹果手机', 3, 4, '苹果电脑快', 0, 1]]"""
                return {'data': ret}
            ret = self.get_contextual_word_similarities(similarity_pairs)

        else:
            ret = '未知相似计算类型，' \
                  'similarity_type 只能是 word_similarity 或者 contextual_word_similarity'

        return {'data': ret}

    # 基于上下文的词相似计算
    def _contextual_model_init(self):
        """  基于上个下文的词相似计算初始化，加载词典，模型
        :return: 无
        """
        pretrain_name = self.model_path + self.model_params['pre_train_model_path']
        logging.info('pretrain_name', pretrain_name)
        if 'albert' in pretrain_name:
            self._contextual_model = AlbertForMaskedLM.from_pretrained(pretrain_name)
            self._contextual_tokenizer = BertTokenizer.from_pretrained(pretrain_name)
        elif 'ernie' in pretrain_name or 'roberta' in pretrain_name:
            self._contextual_tokenizer = BertTokenizer.from_pretrained(pretrain_name)
            self._contextual_model = BertModel.from_pretrained(pretrain_name)
        else:
            # elif 'bert' in pretrain_name:
            self._contextual_tokenizer = AutoTokenizer.from_pretrained(pretrain_name)
            model_config = AutoConfig.from_pretrained(pretrain_name)
            self._contextual_model = AutoModel.from_pretrained(pretrain_name, config=model_config)

    def _get_model_predict(self, sentences):
        """ 根据模型获取所有句向量
        :param sentences: 所有待预测句向量的句子
        :return: 所有句子的句向量 batch_size * sequence_length * embedding_length
        """

        # 句子预测获取预测向量
        encoding_dict = self._contextual_tokenizer(
            list(sentences), max_length=self.model_params['pre_train_max_seq_len'],
            padding="max_length", truncation=True)
        tokens = encoding_dict['input_ids']
        segments = encoding_dict['token_type_ids']
        input_masks = encoding_dict['attention_mask']

        # 转换成PyTorch tensors
        tokens_tensor = torch.tensor(tokens)
        segments_tensors = torch.tensor(segments)
        input_masks_tensors = torch.tensor(input_masks)

        output = self._contextual_model(
            tokens_tensor, token_type_ids=segments_tensors, attention_mask=input_masks_tensors)

        return output[0].detach().numpy()

    def get_contextual_word_similarities(self, context_word_pairs):
        """ 根据句子计算单词相似度，相似度绝对值无意义，相对值有意义
        :param context_word_pairs: 每一个上下文词对是 list 内容为：
            sentence1, word1_start, word1_end, sentence2, word2_start, word2_end，比如:
            比如参数：
            [
                ['菠萝很好吃', 0, 1, '凤梨很好吃', 0, 1], # 菠萝 凤梨
                ['我要买苹果手机', 3, 4, '苹果电脑快', 0, 1], # 苹果 苹果
                ['我要吃苹果', 3, 4, '苹果电脑快', 0, 1], # 苹果 苹果
                ['我要吃苹果', 3, 4, '苹果电脑快', 2, 3], # 苹果 电脑
            ]
        :return: 返回每一个词对的概率列表，超纲词返回-1
            [0.79189042 0.75348656 0.6831773  0.65359073]
        """
        s1_i = 0  # sentence1 index
        s1_s = 1  # word1_start
        s1_e = 2  # word1_end
        s2_i = 3  # sentence2 index
        # S2_s = 4  # word2_start
        s2_e = 5  # word2_end

        num_pairs = len(context_word_pairs)

        context_word_pairs = np.array(context_word_pairs)

        # sentence1, word1_start, word1_end, sentence2, word2_start, word2_end
        # word_1_data 取 sentence1, word1_start, word1_end
        # word_2_data 取 sentence2, word2_start, word2_end
        word_1_data = context_word_pairs[:, s1_i: s1_e + 1]
        word_2_data = context_word_pairs[:, s2_i: s2_e + 1]

        # 行拼接，形成 shape = 2 * num_sentence , 3 的矩阵
        # sentence1, word1_start, word1_end
        # sentence2, word2_start, word2_end
        word_1_2_data_concatenate = np.concatenate((word_1_data, word_2_data), axis=0)

        # 分割成三个一维向量，方便后续使用
        sentences = word_1_2_data_concatenate[:, s1_i]
        word_starts = word_1_2_data_concatenate[:, s1_s]
        word_ends = word_1_2_data_concatenate[:, s1_e]

        predicts = self._get_model_predict(sentences)
        sentence_length = predicts.shape[1]

        # 对于 word 字所在位子标记为 1 ，其它标记为 0 ，后续只对 1 标记字向量做平均操作
        word_masks = []
        for i in range(num_pairs * 2):
            word_mask = np.repeat([0], sentence_length)
            # CLS 占一个位置
            word_mask[int(word_starts[i]) + 1: int(word_ends[i]) + 2] = 1
            word_masks.append(word_mask)

        word_masks = np.array(word_masks)

        # 相当于pool，采用的是https://github.com/terrifyzhao/bert-utils/blob/master/graph.py

        """ 利用 mask 把所有 0 的字向量清 0，保留 mask 为 1 的词向量
        :param: x 表示句向量  num_sentence * sentence_len * embedding_size
             m 表示词 mask 向量  num_sentence * sentence_len
        :return: mask 为 1 的词向量 num_sentence * sentence_len * embedding_size
        """
        mul_mask = lambda x, m: x * np.expand_dims(m, axis=-1)

        # x 表示句向量， m 表示词 mask， 计算 word 中 mask 向量为 1 的字向量平均值
        """ 计算 word 中 mask 向量为 1 的字向量平均值
        :param: x 表示句向量  num_sentence * sentence_len * embedding_size
             m 表示词 mask 向量  num_sentence * sentence_len
        :return: mask 为 1 的平均词向量 num_sentence * embedding_size
        """
        masked_reduce_mean = lambda x, m: np.sum(mul_mask(x, m), axis=1) / (np.sum(m, axis=1, keepdims=True) + 1e-9)

        word_embedding = masked_reduce_mean(predicts, word_masks)

        # logger.info('word_embedding shape', word_embedding.shape)

        # 计算两个词向量距离 =  1 - 余弦相似度，之前拼接 word1 和 word2 所以 word2 从 num_pairs 开始
        word_pair_coses = []
        for i in range(num_pairs):
            word_embedding1 = word_embedding[i]
            word_embedding2 = word_embedding[i + num_pairs]
            word_pair_coses.append(1 - cosine(word_embedding1, word_embedding2))

        return np.array(word_pair_coses)

    # 基于基础词相似计算
    def _word_model_init(self):
        """ 基于词向量初始化函数，加载词向量
        :return: 无
        """
        path = self.model_path + self.model_params['word_embedding_file_name']
        assert os.path.isfile(path), "{} is not a file.".format(path)
        self.word_vector_dict = {}
        total = -1
        embedding_dim = -1
        with open(path, encoding='utf-8') as f:
            for line in f:
                line_split = line.strip().split(' ')
                if len(line_split) == 1:
                    embedding_dim = line_split[0]
                    break
                elif len(line_split) == 2:
                    total = line_split[0]
                    embedding_dim = line_split[1]
                    break
                else:
                    embedding_dim = len(line_split) - 1
                    break

        sys.stdout.write("\rstart read embedding")

        with open(path, encoding='utf-8') as f:
            index = 0
            for line in f:
                values = line.strip().split(' ')
                if len(values) == 1 or len(values) == 2:
                    continue
                if len(values) != int(embedding_dim) + 1:
                    logger.info("\nWarning {} -line.".format(index + 1))
                    continue
                self.word_vector_dict[values[0]] = np.array(list(map(float, values[1:])))
                if index % 2000 == 0:
                    sys.stdout.write("\rHandling with the {} lines. total {}".format(index + 1, total))
                index = index + 1
            sys.stdout.write("\rHandling with the {} lines. total {}".format(index + 1, total))
        logger.info("\nembedding words {}, embedding dim {}.".format(len(self.word_vector_dict), embedding_dim))

    def get_word_similarities(self, word_pairs):
        """ 根据单词对获取单词相似度，绝对值无意义，相对值有意义
        :param word_pairs:
            单词对里列表比如：[
                ('菠萝', '凤梨'),
                ('苹果', '电脑'),
            ]
        :return: 返回每一个词对的概率列表，超纲词返回 -1
        """
        pre_sim_list = []
        vector_dict = self.word_vector_dict
        for w1, w2 in word_pairs:

            hybrid = -1
            w1_is_not_in_dict = w1 not in vector_dict
            w2_is_not_in_dict = w2 not in vector_dict
            if w1_is_not_in_dict and w2_is_not_in_dict:
                logger.info(w1, w2, "都不在词表中")
            elif w1_is_not_in_dict:
                logger.info(w1, "不在词表中")
            elif w2_is_not_in_dict:
                logger.info(w2, "不在词表中")
            else:
                hybrid = 1 - cosine(vector_dict[w1], vector_dict[w2])

            pre_sim_list.append(hybrid)
        return pre_sim_list


def test():
    ws = WordSimilarity()
    word_coses = ws.get_contextual_word_similarities([
        ['菠萝很好吃', 0, 1, '凤梨很好吃', 0, 1],  # 菠萝 凤梨
        ['我要买苹果手机', 3, 4, '苹果电脑快', 0, 1],  # 苹果 苹果
        ['我要吃苹果', 3, 4, '苹果电脑快', 0, 1],  # 苹果 苹果
        ['我要吃苹果', 3, 4, '苹果电脑快', 2, 3],  # 苹果 电脑
    ])
    logging.info("contextual_word_similarities ret", word_coses)

    word_coses = ws.get_word_similarities([
        ('菠萝', '凤梨'),
        ('苹果', '电脑'),
    ])
    logging.info("word_similarities ret", word_coses)


if __name__ == '__main__':
    test()
