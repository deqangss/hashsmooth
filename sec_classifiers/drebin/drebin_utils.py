import os
from tools import utils

K_V_SPLITER = '_:'

def dump_feature(data_dict, new_path):
    if not os.path.exists(os.path.dirname(new_path)):
        utils.mkdir(os.path.dirname(new_path))

    if not isinstance(data_dict, dict):
        raise TypeError("Not 'dict' format")

    with open(new_path, 'w') as f:
        for k, v in data_dict.items():
            for _v in v:
                f.write(str(k) + K_V_SPLITER + str(_v) + '\n')
    return


def load_feature(drebin_feature_path):
    """
    load feature for given path
    :rtype list
    """
    if os.path.isfile(drebin_feature_path):
        return utils.read_txt(drebin_feature_path)
    else:
        raise ValueError("Invalid path.")