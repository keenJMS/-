from __future__ import absolute_import
from collections import defaultdict
import numpy as np

import torch
from torch.utils.data.sampler import Sampler

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
    """
    def __init__(self,data_source,num_instances=4):
        self.data_source=data_source
        self.num_instances=num_instances
        self.index_dic=defaultdict(list)
        for index,(_,pid,_)in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids=list(self.index_dic.keys())
        self.num_instances=len(self.pids)
        pass

    def __iter__(self):
        pass

    def __len__(self):
        pass
if __name__=='__main__':
    from data.mydataset_manager import Market1501
    dataset=Market1501(root='D:\\engineer\ReIDDataset\Market')
    sampler=RandomIdentitySampler(dataset.train)