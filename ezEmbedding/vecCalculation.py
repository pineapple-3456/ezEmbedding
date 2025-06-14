import numpy as np
from itertools import combinations


def cos_similarity(vec_1, vec_2):
    inner_product = np.dot(vec_1, vec_2)
    length = np.linalg.norm(vec_1) * np.linalg.norm(vec_2)

    cos = inner_product / length

    return cos


def SCWEAT(target, attribute_1, attribute_2, sig=False):
    associations_1 = np.array([cos_similarity(target, a) for a in attribute_1])
    mean_association_1 = np.mean(associations_1)

    associations_2 = np.array([cos_similarity(target, a) for a in attribute_2])
    mean_association_2 = np.mean(associations_2)

    associations_conjunction = np.concatenate((associations_1, associations_2))
    stdev = np.std(associations_conjunction)

    d = (mean_association_1 - mean_association_2) / stdev

    if not sig:
        result = {"effect_size": d}

        return result

    else:
        null_distribution = []

        associations_1_perm_list = list(combinations(associations_conjunction, round(len(associations_conjunction)/2)))

        for associations_1_perm in associations_1_perm_list:
            associations_1_perm = np.array(associations_1_perm)
            associations_2_perm = np.array([c for c in associations_conjunction if c not in associations_1_perm])

            mean_association_1_perm = np.mean(associations_1_perm)
            mean_association_2_perm = np.mean(associations_2_perm)

            null_distribution.append((mean_association_1_perm - mean_association_2_perm) / stdev)

        p = len([i for i in null_distribution if i > d]) / len(null_distribution)

        result = {"effect_size": d, "p_value": p}

        return result

if __name__ == '__main__':
    from ezEmbedding.Embedding import bertEmbedding

    '''cos_similarity用例'''

    Embed = bertEmbedding("../bert_localpath", "./bert_localpath/",
                          from_encoder="last4", pooling_method="mean")

    text_vec_1 = Embed.get_sentence_embedding("王冕死了父亲")
    text_vec_2 = Embed.get_sentence_embedding("王冕的父亲死了")

    # 计算余弦相似度，返回一个float
    # 注意输入的是两个词语或句子的向量(数组)，而非字符串
    sim = cos_similarity(text_vec_1, text_vec_2)

    '''SCWEAT用例'''

    attribute_1_words = ["高兴", "开心", "积极", "美丽"]
    attribute_2_words = ["悲伤", "伤心", "消极", "丑陋"]
    target_word = "蜘蛛"

    # 这里为了例子简洁，直接把词语当成句子，用bert生成向量。实际上这么做是错误的，应该使用词向量
    attribute_1_vecs = np.array([Embed.get_sentence_embedding(word) for word in attribute_1_words])
    attribute_2_vecs = np.array([Embed.get_sentence_embedding(word) for word in attribute_2_words])
    target_vec = Embed.get_sentence_embedding(target_word)

    # 单类别词嵌入联系测验，需要输入目标词和两种属性词所对应的向量
    # 目标词是一个单维数组，两种属性词分别是两个二维数组，第一个维度是词语，第二个维度是词向量
    # 返回一个字典，["effect_size"]是效应量，["p_value"]是p值
    # 默认不计算p值，因为置换检验非常耗时。如果要计算p值，请设置sig=True
    SCWEAT = SCWEAT(target_vec, attribute_1_vecs, attribute_2_vecs, sig=True)
    effect_size = SCWEAT["effect_size"]
    p_value = SCWEAT["p_value"]

    p_value




