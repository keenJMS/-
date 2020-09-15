import os
from PIL import Image
import numpy as np
import os.path as osp
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision.transforms import transforms

transform_train_list = [
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((256,128), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_test_list = [
        transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

data_transform={'train':transform_train_list,
                'test':transform_test_list}
class ImageDataset(Dataset):
    def __init__(self,dataset,transformer=None):
        self.dataset=dataset
        self.transformer=transformer
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        img_path,pid,cid=self.dataset[index]
        img=read_image(img_path)
        if self.transformer is not None:
            img=self.transformer(img)
        return img,pid,cid


def read_image(img_path):
    if not osp.exists(img_path):
        raise IOError("{}DOES NOT EXIST".format(img_path))
    try:
        img=Image.open(img_path) .convert('RGB')
    except IOError:
        print("can't read {}".format(img_path))
    return img

if __name__=='__main__':
    from mydataset_manager import Market1501
    market=Market1501(root='/root/dataset/Market')
    train=market.train
    transform_train = transforms.Compose(data_transform['train'])
    train_loader=ImageDataset(train,transformer=transform_train)
    # from torchvision.transforms import transforms
    # transform=transforms.Compose([
    #     transforms.CenterCrop(100),
    #     transforms.RandomHorizontalFlip()
    #
    #
    # ])

    for batch,(img,pid,cid)in enumerate(train_loader):
        print(pid,cid)
        #Image._show(img_path)
        plt.figure(12)
        plt.subplot(121)
        plt.imshow(img.permute([1,2,0]))
        # img_t=transform(img)
        # plt.subplot(122)
        # plt.imshow(img_t.permute([1,2,0]))
        plt.show()