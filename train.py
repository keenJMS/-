from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import argparse
import sys
from utils import *
import mydataset_manager
from dataset_loader import ImageDataset
from ResNet import ResNet50
import torchvision.transforms as T

parser = argparse.ArgumentParser(description='Train image model with center loss')

parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")

parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval-step', type=int, default=-1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()
def main():
    #config
    config=get_config('/root/proj/JMSreid/config/config.yaml')
    train_epoch=config['train_epoch']

    #gpu_setting
    use_gpu=torch.cuda.is_available()
    pin_memory=True if use_gpu else False
    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")
    #logs
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    #dataset 3 loader
    dataset=mydataset_manager.Market1501(root=config['dataset_root'])
    transform_train = T.Compose([
        T.Resize((config['height'], config['width']),interpolation=3),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((config['height'], config['width']),interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_loader=DataLoader(
        ImageDataset(dataset.train,transformer=transform_train),
        batch_size=config['train_batch'],
        pin_memory=pin_memory,
        shuffle=True,
        num_workers=config['workers'],
        drop_last=True
    )
    query_loader=DataLoader(
        ImageDataset(dataset.query,transformer=transform_test),
        batch_size=config['test_batch'],
        pin_memory=pin_memory,
        shuffle=False,
        num_workers=config['workers'],
        drop_last=False
    )
    gallery_loader=DataLoader(
        ImageDataset(dataset.test, transformer=transform_test),
        batch_size=config['test_batch'],
        pin_memory=pin_memory,
        shuffle=False,
        num_workers=config['workers'],
        drop_last=False
    )
    print("Initializing model: {}".format(config['arch']))
    if config['arch']=='ResNet50':
        model = ResNet50(num_classes=dataset.train_num_pids,training=True,loss='softmax')
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    #criterion
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    if config['step_size'] > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])

    start_epoch = args.start_epoch
    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        test(model, query_loader, gallery_loader, use_gpu)
        return 0

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")

    for epoch in range(start_epoch,train_epoch):
        start_train_time = time.time()
        train(epoch,model,criterion,optimizer,train_loader,use_gpu)
        if config['step_size']: scheduler.step()
def train(epoch,model,criterion,optimizer,train_loader,use_gpu):
    for batch_id,(imgs,pids,_) in enumerate(train_loader):
        if use_gpu:
            imgs,pids=imgs.cuda(),pids.cuda()
        prediction =model(imgs)
        loss =criterion(prediction,pids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(epoch,batch_id,loss)
    pass

def test():
    def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
        batch_time = AverageMeter()

        model.eval()

        with torch.no_grad():
            qf, q_pids, q_camids = [], [], []
            for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
                if use_gpu: imgs = imgs.cuda()

                end = time.time()
                features = model(imgs)
                batch_time.update(time.time() - end)

                features = features.data.cpu()
                qf.append(features)
                q_pids.extend(pids)
                q_camids.extend(camids)
            qf = torch.cat(qf, 0)
            q_pids = np.asarray(q_pids)
            q_camids = np.asarray(q_camids)

            print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

            gf, g_pids, g_camids = [], [], []
            for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
                if use_gpu: imgs = imgs.cuda()

                end = time.time()
                features = model(imgs)
                batch_time.update(time.time() - end)

                features = features.data.cpu()
                gf.append(features)
                g_pids.extend(pids)
                g_camids.extend(camids)
            gf = torch.cat(gf, 0)
            g_pids = np.asarray(g_pids)
            g_camids = np.asarray(g_camids)

            print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

        print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))

        m, n = qf.size(0), gf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.numpy()

        print("Computing CMC and mAP")
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

        print("Results ----------")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")

        return cmc[0]
    pass
def evaluate():
    pass











if __name__=='__main__':
    main()