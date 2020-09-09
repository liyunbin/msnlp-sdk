from scipy import stats
import numpy as np
import logging
import os
logger = logging.getLogger(__name__)


def read_eval_file(file_path):
    """ 读取词相似文件, 格式 w1 w2 sim ，方便批量测试
    :return: 分别返回散三列数据
        {'word_a_list': word_a_list, 'word_b_list': word_b_list, 'sim_list': sim_list}
    """
    word_a_list = []
    word_b_list = []
    sim_list = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if "" == line:
                continue
            line_split = line.split()
            word_a_list.append(line_split[0])
            word_b_list.append(line_split[1])
            if len(line_split) > 2:
                sim_list.append(float(line_split[2]))
    return {'word_a_list': word_a_list, 'word_b_list': word_b_list, 'sim_list': sim_list}


def eval_file(model, file_path):
    test_file_name = os.path.basename(file_path)
    word_pairs = read_eval_file(file_path)
    w1 = word_pairs['word_a_list']
    w2 = word_pairs['word_b_list']
    sims = word_pairs['sim_list']

    pre_sims = model.get_word_similarities(zip(w1, w2))

    p_score = stats.pearsonr(pre_sims, sims)[0]
    sp_score = stats.spearmanr(pre_sims, sims).correlation
    logging.info(test_file_name, 'pearson', p_score, 'spearman', sp_score)  # 打印皮尔逊相关系数

    domain_sign = np.array(pre_sims) != -1

    know_pre_sim_list = np.array(pre_sims)[domain_sign]
    know_sim_list = np.array(sims)[domain_sign]

    p_score = stats.pearsonr(know_pre_sim_list, know_sim_list)[0]
    sp_score = stats.spearmanr(know_pre_sim_list, know_sim_list).correlation
    logging.info(test_file_name, '去掉未知 pearson', p_score, 'spearman', sp_score)
    logging.info('DataSet', 'found', 'not found', 'pearson', 'spearman')
    logging.info(test_file_name, len(domain_sign) - np.sum(domain_sign), np.sum(domain_sign), p_score, sp_score)
    return {'pearson': p_score, 'spearman': sp_score}
