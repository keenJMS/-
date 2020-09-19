import torchvision
import torch.nn as nn
from torch.nn import functional as F
import torch

class ResNet50(nn.Module):
    def __init__(self,num_classes,loss='softmax',training=False,**kwargs):
        super(ResNet50,self).__init__()
        resnet50=torchvision.models.resnet50(pretrained=True)
        self.loss=loss
        self.base=nn.Sequential(*list(resnet50.children())[:-2])
        if not self.loss=='metric':
            self.classfier=nn.Linear(2048,num_classes)
        self.training=training
        pass
    def forward(self,x):
        x= self.base(x) #32*2048*8*4
        x=F.avg_pool2d(x,x.size()[2:])
        f=x.view(x.size(0),-1)
        if not self.training:
            return f
        if self.loss=='softmax':
            y=self.classfier(f)
            return y
        if self.loss=='metric':
            return f
        if self.loss=='softmax,metric':
            y = self.classfier(f)
            return y, f
        else:
            raise RuntimeError('LOSS SETTING ERROR')

if __name__=='__main__':
    model=ResNet50(num_classes=751,training=True)
    input=torch.rand(1,3,256,128)
    output=model(input)
    pass