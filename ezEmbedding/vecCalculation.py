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




