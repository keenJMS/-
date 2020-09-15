import os
import numpy as np
import os.path as osp
from IPython import embed
from torch.utils.data import Dataset
import glob
import re
class Market1501(object):
    dataset_dir='Market'
    def __init__(self,root):
    #第一步，生成路径，检查路径是否存在
        self.train_dir=osp.join(root,'bounding_box_train')
        self.test_dir=osp.join(root,'bounding_box_test')
        self.query_dir=osp.join(root,'query')
        dir_to_check=[self.train_dir,self.test_dir,self.query_dir]
        check_dir(dir_to_check)
        self.train,self.train_num_pids,self.train_num_imgs=self._process_dir(self.train_dir,relabel=True)
        self.test,self.test_num_pids,self.test_num_imgs =self._process_dir(self.test_dir, relabel=False)
        self.query,self.query_num_pids,self.query_num_imgs=self._process_dir(self.query_dir,relabel=False)
        pass
    def _process_dir(self,dir,relabel=False):
        #获取所有的图片路径，通过模式得到行人ID，再把ID变成1-751
        img_paths=glob.glob(osp.join(dir,'*.jpg'))
        pattern=re.compile(r'([\d]+)_c(\d)')
        pid_container=set()
        for img_path in img_paths:
            pid,cid=map(int,pattern.search(img_path).groups())
            if pid == -1: continue
            pid_container.add(pid)
        #把ID转换成LABEL
        plabel={pid:label for label,pid in enumerate(pid_container)}
        dataset=[]
        for img_path in img_paths:
            pid,cid=map(int,pattern.search(img_path).groups())
            if pid == -1: continue
            assert 0<=pid<=1501
            assert 1<=cid<=6
            cid-=1
            #对于测试集不需要relabel
            if relabel:
                pid=plabel[pid]
            dataset.append((img_path,pid,cid))
        num_pids=len(pid_container)
        num_imgs=len(img_paths)
        return dataset,num_pids,num_imgs




def check_dir(dir_to_check):
    for i,path in enumerate(dir_to_check):
        if not osp.exists(path):
            raise RuntimeError('{} is not exist'.format(path))
        print('{},no problem'.format(dir_to_check))
def print_file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print(root) #当前目录路径
        print(dirs) #当前路径下所有子目录
        print(files) #当前路径下所有非目录子文件
if __name__=='__main__':
    market=Market1501('/root/dataset/Market')

    pass