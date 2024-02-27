"""
@author: Guo Shiguang
@software: PyCharm
@file: custom_tools.py
@time: 2023/9/18 13:55
"""
import os
import random

import numpy as np
# import torch


def set_seed(param):
    random.seed(param)
    np.random.seed(param)
    os.environ['PYTHONHASHSEED'] = str(param)
    # torch.manual_seed(param)
    # torch.cuda.manual_seed(param)
    # torch.cuda.manual_seed_all(param)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
