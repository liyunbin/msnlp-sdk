import os

from msnlp.data.metrics import word_similarity as word_similarity_metrics
from msnlp import NLPPredictor
import unittest
from msnlp.config import TaskName

import logging
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = ""

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class TestWordSimilarity(unittest.TestCase):

    def setUp(self):
        logging.info("enter WordSimilarity test")

        self.ws_nlp_predictor = NLPPredictor(
            TaskName.word_similarity,
            model_path=BASE_DIR + '/test_model/',
            model_params={
                # 上下文词相似计算
                'pre_train_model_path': 'voidful-albert_chinese_tiny/',
                'pre_train_max_seq_len': 64,

                # 基本词相似计算 词典随意构造，相似结果不可用
                'word_embedding_file_name': 'word_embedding.txt'
            })

    def tearDown(self):
        logging.info("out WordSimilarity test")

    @staticmethod
    def change_to_nlp_predictor_params(similarity_type, similarity_pairs):
        nlp_predictor_params = {
            'text': {
                'similarity_type': similarity_type,
                'similarity_pairs': similarity_pairs
            }
        }
        return nlp_predictor_params

    def test_error_input(self):
        out_self = self

        def error_input_and_check_error(ype_key, type_value, pairs_key, pairs_value, tips):
            error_ret = out_self.ws_nlp_predictor.predict_rest({
                'text': {
                    ype_key: pairs_key,
                    type_value: pairs_value
                }
            })
            out_self.assertEqual('参数错误', error_ret['data'][0:4], tips)

        contextual_word_pairs = [
            ['菠萝很好吃', 0, 1, '凤梨很好吃', 0, 1],  # 菠萝 凤梨
            ['我要买苹果手机', 3, 4, '苹果电脑快', 0, 1],  # 苹果 苹果
            ['我要吃苹果', 3, 4, '苹果电脑快', 0, 1],  # 苹果 苹果
            ['我要吃苹果', 3, 4, '苹果电脑快', 2, 3],  # 苹果 电脑
        ]
        word_pairs = [
            ['足球', '世纪'],
            ['黄瓜', '教授'],
        ]

        error_type = 'eroor_type'
        error_key = 'eroor_key'

        similarity_type_key = 'similarity_type'
        similarity_pairs_key = 'similarity_pairs'
        contextual_word_similarity_value = 'contextual_word_similarity'
        word_similarity_value = 'word_similarity'

        error_input_and_check_error(error_type, contextual_word_similarity_value,
                                    similarity_pairs_key, contextual_word_pairs,
                                    'similarity_type_key 错误，但未得到期望错误提示')
        error_input_and_check_error(similarity_type_key, contextual_word_similarity_value,
                                    error_key, contextual_word_pairs,
                                    'similarity_pairs_key 错误，但未得到期望错误提示')
        error_input_and_check_error(similarity_type_key, error_type,
                                    similarity_pairs_key, contextual_word_pairs,
                                    'similarity_type_value 错误，但未得到期望错误提示')
        error_input_and_check_error(similarity_type_key, contextual_word_similarity_value,
                                    similarity_pairs_key, word_pairs,
                                    'contextual_word_similarity_value 和 word_pairs 不匹配'
                                    '，但未得到期望错误提示')
        error_input_and_check_error(similarity_type_key, word_similarity_value,
                                    similarity_pairs_key, contextual_word_pairs,
                                    'word_similarity_value 和 contextual_word_pairs 不匹配'
                                    '，但未得到期望错误提示')

    def test_contextual_word_similarity(self):
        nlp_predictor_params = self.change_to_nlp_predictor_params('contextual_word_similarity', [
            ['菠萝很好吃', 0, 1, '凤梨很好吃', 0, 1],  # 菠萝 凤梨
            ['我要买苹果手机', 3, 4, '苹果电脑快', 0, 1],  # 苹果 苹果
            ['我要吃苹果', 3, 4, '苹果电脑快', 0, 1],  # 苹果 苹果
            ['我要吃苹果', 3, 4, '苹果电脑快', 2, 3],  # 苹果 电脑
        ])
        word_coses_data = self.ws_nlp_predictor.predict_rest(nlp_predictor_params)
        logging.info(word_coses_data)
        self.assertGreater(word_coses_data['data'][0], -1, '语义关系计算错误')

    def test_word_similarity(self):
        nlp_predictor_params = self.change_to_nlp_predictor_params('word_similarity', [
            ['足球', '世纪'],
            ['黄瓜', '教授'],
        ])
        word_coses_data = self.ws_nlp_predictor.predict_rest(nlp_predictor_params)
        logging.info(word_coses_data)
        self.assertGreater(word_coses_data['data'][0], -1, '词关系计算错误')

    def test_word_similarity_file(self):
        test_file_name = 'wordsim-10.txt'
        base_dir = os.path.dirname(os.path.realpath(__file__))
        metrics = word_similarity_metrics.eval_file(self.ws_nlp_predictor.nlptask,
                                                    base_dir + '/data/task_samples/word_similarity/' + test_file_name)
        logging.info(metrics)
        self.assertGreater(metrics['spearman'], -1,
                           test_file_name + '值小于-1，实际为' + str(metrics['spearman']))
        self.assertLess(metrics['spearman'], 1,
                        test_file_name + '值大于1，实际为' + str(metrics['spearman']))
